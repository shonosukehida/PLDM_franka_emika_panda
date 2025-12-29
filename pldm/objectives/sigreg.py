# pldm/objectives/sigreg.py
from dataclasses import dataclass
from typing import NamedTuple, List

import torch
from torch import nn

from pldm.configs import ConfigBase
from pldm.models.jepa import ForwardResult
from pldm.models.utils import flatten_conv_output


class SIGRegLossInfo(NamedTuple):
    total_loss: torch.Tensor
    sigreg_loss: torch.Tensor
    loss_name: str = "sigreg"
    name_prefix: str = ""

    def build_log_dict(self):
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_loss": self.sigreg_loss.item(),
        }


@dataclass
class SIGRegObjectiveConfig(ConfigBase):
    coeff: float = 1.0        # LeJEPA の λ 相当
    num_slices: int = 256
    num_t: int = 17
    pred_attr: str = "obs"    # "state" / "obs" / "propio" 


def sigreg_core(x: torch.Tensor,
                num_slices: int = 256,
                num_t: int = 17) -> torch.Tensor:
    """
    x: (N, D) の latent
    """
    device = x.device
    N, D = x.shape

    #(1) ランダム 1D プロジェクションを num_slices 本サンプル
    A = torch.randn(D, num_slices, device=device)
    A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)  # (D, M)

    # N×D → N×M に射影
    z = x @ A  # (N, M)

    #(2) 積分点 t
    t = torch.linspace(-5.0, 5.0, num_t, device=device)  # (T,)
    exp_f = torch.exp(-0.5 * t ** 2)                     # 理論 CF (N(0,1))

    #(3) empirical CF
    #    z: (N, M) --> (N, M, T) で各 t に対する CF を計算
    z_t = z.unsqueeze(-1) * t  # (N, M, T)
    ecf = (1j * z_t).exp().mean(dim=0)  # (M, T), complex

    #(4) weighted L2 距離
    # |ecf - exp_f|^2 * exp_f を t で積分
    err = (ecf - exp_f).abs().square() * exp_f  # (M, T)
    T_val = torch.trapz(err, t, dim=-1)        # (M,)

    # スライス平均
    return T_val.mean() * N    # N 掛け: 論文のスケール
                            
                               

class SIGRegObjective(nn.Module):
    def __init__(
        self,
        config: SIGRegObjectiveConfig,
        repr_dim,              
        name_prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix

    def __call__(self, _batch, results: List[ForwardResult]) -> SIGRegLossInfo:
        result = results[-1]

        if self.config.pred_attr == "state":
            encodings = result.backbone_output.encodings  # (T,B,C,H,W) 
        elif self.config.pred_attr == "obs":
            encodings = result.backbone_output.obs_component
        elif self.config.pred_attr == "propio":
            encodings = result.backbone_output.propio_component
        else:
            raise NotImplementedError

        # (T,B,...) --> flatten --> (N, D)
        flat = flatten_conv_output(encodings)   # (T,B,D)
        x = flat.reshape(-1, flat.shape[-1])    # (N, D)

        sigreg_loss = sigreg_core(
            x,
            num_slices=self.config.num_slices,
            num_t=self.config.num_t,
        )

        total = self.config.coeff * sigreg_loss

        return SIGRegLossInfo(
            total_loss=total,
            sigreg_loss=sigreg_loss,
            name_prefix=self.name_prefix,
        )
