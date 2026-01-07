import torch
import torch.nn as nn
from transformers import AutoModel

from pldm.models.encoders.base_class import SequenceBackbone
from pldm.models.encoders.enums import BackboneOutput
from pldm.models.utils import Expander2D


class VJEPA2Backbone(SequenceBackbone):
    def __init__(
        self,
        repo: str,
        out_obs_channels: int = 16,     # 視覚側（z_obs）のch
        total_channels: int = 30,       # predictor と合わせたい最終ch (= repr_dim)
        out_hw: int = 26,
        freeze: bool = True,
        img_size: int = 64,
        propio_dim: int | None = None,
        propio_encoder_arch: str | None = "id",  # 例: "75-64-14" とか
        chunk_size: int = 2,
    ):
        super().__init__()
        self.freeze = freeze
        self.out_hw = out_hw
        self.img_size = img_size

        self.out_obs_channels = out_obs_channels
        self.total_channels = total_channels
        assert total_channels >= out_obs_channels
        self.out_propio_channels = total_channels - out_obs_channels  # ここが「14」に相当

        self.propio_dim = propio_dim
        self.propio_encoder_arch = propio_encoder_arch

        self.vjepa2 = AutoModel.from_pretrained(repo)

        if self.freeze:
            self.vjepa2.eval()
            for p in self.vjepa2.parameters():
                p.requires_grad_(False)


        # --- obs adapter（VJEPA2 token D -> out_obs_channels）---
        self.obs_adapter = self._build_obs_adapter_with_dummy_forward(out_obs_channels)

        # --- propio encoder（propio -> out_propio_channels -> spatial）---
        if self.propio_dim and self.out_propio_channels > 0:
            self.propio_encoder = self._build_propio_encoder()
        else:
            self.propio_encoder = None
        
        self.chunk_size = chunk_size

    def _build_obs_adapter_with_dummy_forward(self, out_obs_channels: int) -> nn.Linear:
        # dummy: (B,T,C,H,W) で まずは T=1 で軽く通す（T=64はVRAM重い）
        p = next(self.vjepa2.parameters())
        dev, dtype = p.device, p.dtype

        dummy = torch.zeros(
            1, 1, 3, self.img_size, self.img_size,
            device=dev,
            dtype=dtype if dtype.is_floating_point else torch.float32,
        ).clamp(0, 1)

        was_training = self.vjepa2.training
        self.vjepa2.eval()
        with torch.no_grad():
            out = self.vjepa2(pixel_values_videos=dummy)
        if was_training and not self.freeze:
            self.vjepa2.train()

        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        D = h.shape[-1]
        return nn.Linear(D, out_obs_channels).to(dev)

    def _build_propio_encoder(self) -> nn.Module:
        # MeNet6 と同じノリ：
        # (BS, propio_dim) -> (BS, out_propio_channels) -> Expander2D(out_hw,out_hw)

        # 「id」でも propio_dim != out_propio_channels なら Linear が必要
        if (self.propio_encoder_arch is None) or (self.propio_encoder_arch == "id"):
            if self.propio_dim == self.out_propio_channels:
                mlp = nn.Identity()
            else:
                mlp = nn.Linear(self.propio_dim, self.out_propio_channels)
            return nn.Sequential(mlp, Expander2D(w=self.out_hw, h=self.out_hw))

        # 例: "75-64-14" みたいな文字列を想定
        layer_dims = [int(x) for x in self.propio_encoder_arch.split("-")]
        # もし arch の末尾が out_propio_channels じゃないなら合わせちゃう（安全）
        if layer_dims[-1] != self.out_propio_channels:
            layer_dims = layer_dims[:-1] + [self.out_propio_channels]

        layers = []
        in_dim = self.propio_dim
        for d in layer_dims:
            layers.append(nn.Linear(in_dim, d))
            layers.append(nn.ReLU())
            in_dim = d
        layers.pop()  # last ReLU remove

        return nn.Sequential(*layers, Expander2D(w=self.out_hw, h=self.out_hw))

    def forward(self, x, propio=None):
        """
        SequenceBackbone.forward_multiple が (T,B,...) を畳んでくるので、ここは基本 4D:
          x: (BS, C, H, W)
          propio: (BS, propio_dim)  (渡される時)
        """
        if x.dim() != 4:
            raise ValueError(f"expects (BS,C,H,W), got {tuple(x.shape)}")
        # print("VJEPA2Backbone.forward x:", x.shape)

        BS, C, H, W = x.shape

        # (BS,1,3,H,W) を作って VJEPA2 に入れる（T=1で軽量）
        x_vid = x.unsqueeze(1).clamp(0, 1)

        # dev = x_vid.device
        # if self.obs_adapter.weight.device != dev:
        #     self.obs_adapter = self.obs_adapter.to(dev)
        # if self.propio_encoder is not None:
        #     self.propio_encoder = self.propio_encoder.to(dev)
        # # vjepa2本体も必要ならデバイス合わせ（※すでに .to(device) 済みなら不要）
        # self.vjepa2 = self.vjepa2.to(dev)

        with torch.no_grad() if self.freeze else torch.enable_grad():
            out = self.vjepa2(pixel_values_videos=x_vid)

        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        pooled = h.mean(dim=1)                    # (BS, D)
        z_obs = self.obs_adapter(pooled)          # (BS, out_obs_channels)

        z_obs = z_obs.view(BS, self.out_obs_channels, 1, 1)
        z_obs = z_obs.expand(BS, self.out_obs_channels, self.out_hw, self.out_hw)

        if self.propio_encoder is not None and (propio is not None):
            z_prop = self.propio_encoder(propio)  # (BS, out_propio_channels, out_hw, out_hw)
            enc = torch.cat([z_obs, z_prop], dim=1)  # (BS, total_channels, out_hw, out_hw)
        else:
            z_prop = None
            enc = z_obs

        return BackboneOutput(
            encodings=enc,
            obs_component=z_obs,
            propio_component=z_prop,
        )

    def forward_multiple(self, x, propio=None):
        """
        Override to avoid flattening (T*BS) into a huge batch for ViT-style models.
        x: (T, BS, C, H, W) or (BS, C, H, W)
        propio: (T, BS, propio_dim) or (BS, propio_dim)
        """
        # No time dimension -> default behavior
        if x.dim() == 2 or x.dim() == 4:
            return self.forward(x, propio) if propio is not None else self.forward(x)

        T, BS = x.shape[:2]
        state = x.flatten(0, 1)  # (T*BS, C, H, W)

        if propio is not None:
            propio = propio.flatten(0, 1)  # (T*BS, propio_dim)

        outs_enc = []
        outs_obs = []
        outs_prop = []

        N = state.shape[0]
        for i in range(0, N, self.chunk_size):
            s = state[i:i + self.chunk_size]
            p = propio[i:i + self.chunk_size] if propio is not None else None

            out = self.forward(s, p) if p is not None else self.forward(s)

            outs_enc.append(out.encodings)
            outs_obs.append(out.obs_component)
            outs_prop.append(out.propio_component)

        enc = torch.cat(outs_enc, dim=0).reshape(T, BS, *outs_enc[0].shape[1:])

        obs_component = None
        if outs_obs[0] is not None:
            obs_component = torch.cat(outs_obs, dim=0).reshape(T, BS, *outs_obs[0].shape[1:])

        propio_component = None
        if outs_prop[0] is not None:
            propio_component = torch.cat(outs_prop, dim=0).reshape(T, BS, *outs_prop[0].shape[1:])

        return BackboneOutput(
            encodings=enc,
            obs_component=obs_component,
            propio_component=propio_component,
        )