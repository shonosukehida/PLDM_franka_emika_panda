from typing import NamedTuple

import torch
from tqdm import tqdm

# STATS = {
#     "WallDataset": {
#         # NOTE: these values are in fact not correct but they work cuz random search
#         # was performed using these. When using correct values performance got worse.
#         # but should just perform random search over again on correct values
#         "state_mean": torch.tensor([0.0002, 0.0014]),
#         "state_std": torch.tensor([0.0034, 0.0112]),
#         "action_mean": torch.tensor([0.0120, -0.0074]),
#         "action_std": torch.tensor([0.7543, 0.7424]),
#         "location_mean": torch.tensor([31.1224, 31.3396]),
#         "location_std": torch.tensor([16.3134, 16.6708]),
#         "propio_pos_mean": torch.tensor([0, 1]),
#         "propio_pos_std": torch.tensor([0, 1]),
#         "propio_vel_mean": torch.tensor([0, 1]),
#         "propio_vel_std": torch.tensor([0, 1]),
#     }
# }


def get_nth_percentile(tensor, percentile):
    assert len(tensor.shape) == 1
    k = int(tensor.shape[0] * percentile)
    return tensor.kthvalue(k).values.item()


class Sample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]


class Normalizer:
    def __init__(
        self,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        location_mean: torch.Tensor,
        location_std: torch.Tensor,
        propio_pos_mean: torch.Tensor,
        propio_pos_std: torch.Tensor,
        propio_vel_mean: torch.Tensor,
        propio_vel_std: torch.Tensor,
        bluebox_locs_mean: torch.Tensor,
        bluebox_locs_std: torch.Tensor,
        state_min: torch.Tensor,
        state_max: torch.Tensor,
        action_min: torch.Tensor,
        action_max: torch.Tensor,
        location_min: torch.Tensor,
        location_max: torch.Tensor,
        propio_pos_min: torch.Tensor,
        propio_pos_max: torch.Tensor,
        propio_vel_min: torch.Tensor,
        propio_vel_max: torch.Tensor,
        bluebox_locs_min: torch.Tensor,
        bluebox_locs_max: torch.Tensor,
        backbone_arch: str = "menet6",
        normalize_mode: str = "minmax",
        normalize_actions_mode: str | None = None,  # ★ 追加
        min_states_val: float = 0.0,
        max_states_val: float = 1.0,
        min_actions_val: float = 0.0,
        max_actions_val: float = 1.0,
        min_locations_val: float = 0.0,
        max_locations_val: float = 1.0,
        min_propio_pos_val: float = 0.0,
        max_propio_pos_val: float = 1.0,
        min_propio_vel_val: float = 0.0,
        max_propio_vel_val: float = 1.0,
        min_bluebox_locs_val: float = 0.0,
        max_bluebox_locs_val: float = 1.0,
    ):
        self.normalize_mode = normalize_mode.lower()
        if normalize_actions_mode is None:
            normalize_actions_mode = normalize_mode  # ★ デフォルトで同じにする
        self.normalize_actions_mode = normalize_actions_mode.lower()
        
        self.backbone_arch = backbone_arch
        
        assert self.normalize_mode in ("minmax", "zscore"), "normalize_mode must be 'minmax' or 'zscore'"
        assert self.normalize_actions_mode in ("minmax", "zscore"), "normalize_actions_mode must be 'minmax' or 'zscore'"
        
        self.state_mean = state_mean
        self.state_std = state_std
        self.action_mean = action_mean
        self.action_std = action_std
        self.location_mean = location_mean
        self.location_std = location_std
        self.propio_pos_mean = propio_pos_mean
        self.propio_pos_std = propio_pos_std
        self.propio_vel_mean = propio_vel_mean
        self.propio_vel_std = propio_vel_std
        self.bluebox_locs_mean = bluebox_locs_mean
        self.bluebox_locs_std = bluebox_locs_std
        
        
        
        self.state_min = state_min
        self.state_max = state_max
        self.action_min = action_min
        self.action_max = action_max
        self.location_min = location_min
        self.location_max = location_max
        self.propio_pos_min = propio_pos_min
        self.propio_pos_max = propio_pos_max
        self.propio_vel_min = propio_vel_min
        self.propio_vel_max = propio_vel_max
        self.bluebox_locs_min = bluebox_locs_min
        self.bluebox_locs_max = bluebox_locs_max
        
        
        
        self.min_states_val = min_states_val
        self.max_states_val = max_states_val
        
        self.min_actions_val = min_actions_val
        self.max_actions_val = max_actions_val
        
        self.min_locations_val = min_locations_val 
        self.max_locations_val = max_locations_val
        
        self.min_propio_pos_val = min_propio_pos_val
        self.max_propio_pos_val = max_propio_pos_val 
        
        self.min_propio_vel_val = min_propio_vel_val 
        self.max_propio_vel_val = max_propio_vel_val 
        
        self.min_bluebox_locs_val = min_bluebox_locs_val
        self.max_bluebox_locs_val = max_bluebox_locs_val

    @staticmethod
    def _has_attr(sample, attr):
        return (
            hasattr(sample, attr)
            and getattr(sample, attr) is not None
            and bool(getattr(sample, attr).shape[-1])
        )


    @classmethod
    def build_normalizer(
        cls,
        dataset,
        n_samples: int = 100,
        backbone_arch: str = "menet6",
        normalize_mode: str = "minmax",
        normalize_actions_mode: str | None = None,  # ★ 追加
        normalizer_hardset: bool = False,
        min_states_val: float = 0.0,
        max_states_val: float = 1.0,
        min_actions_val: float = 0.0,
        max_actions_val: float = 1.0,
        min_locations_val: float = 0.0,
        max_locations_val: float = 1.0,
        min_propio_pos_val: float = 0.0,
        max_propio_pos_val: float = 1.0,
        min_propio_vel_val: float = 0.0,
        max_propio_vel_val: float = 1.0,
        min_bluebox_locs_val: float = 0.0,
        max_bluebox_locs_val: float = 1.0,
    ):
        if normalize_actions_mode is None:
            normalize_actions_mode = normalize_mode
        normalize_mode = normalize_mode.lower()
        normalize_actions_mode = normalize_actions_mode.lower()

        # DataLoader 経由でも Dataset 直接でも config を取れるようにする
        if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "config"):
            config = dataset.dataset.config
        elif hasattr(dataset, "config"):
            config = dataset.config
        else:
            config = None


        # ---- オンライン（逐次）集計に切り替え：CPU / no-grad ----
        import contextlib
        device_cpu = torch.device("cpu")

        # 初期化フラグ
        init = False

        # 総サンプル数（各フィールドで別管理）
        n_state = n_action = n_loc = n_ppos = n_pvel = n_blue = 0

        with torch.inference_mode():
            it2 = iter(dataset)
            for _i in tqdm(range(n_samples), desc="Estimating normalizer stats"):
                try:
                    sample = next(it2)
                except StopIteration:
                    it2 = iter(dataset)
                    sample = next(it2)

                # --- STATES ---
                if cls._has_attr(sample, "states"):
                    st = sample.states.float()  # (B,T,C,H,W) or (B,feat) etc.
                    if st.ndim == 5:
                        B, T, C, H, W = st.shape
                        st = st.reshape(B*T, C*H*W)
                    elif st.ndim == 3:  # (B, T, D)
                        B, T, D = st.shape
                        st = st.reshape(B*T, D)
                    elif st.ndim == 2:  # (B, D)
                        pass
                    else:
                        # 想定外はスキップ（元実装と同様の振る舞いに合わせるなら調整）
                        st = None

                    if st is not None:
                        st = st.to(device_cpu, dtype=torch.float32, copy=False)
                        # バッチ統計
                        nb = st.shape[0]
                        b_mean = st.mean(0)
                        # M2b = sum((xi - b_mean)^2) = var(unbiased=False) * nb
                        b_M2 = st.var(dim=0, unbiased=False) * nb

                        if not init:
                            # 走査の初回はそのままセット
                            state_mean = b_mean.clone()
                            state_M2   = b_M2.clone()
                            # min/max
                            state_min  = st.min(0).values
                            state_max  = st.max(0).values
                            n_state    = nb
                        else:
                            # Welford の並列合成
                            delta      = b_mean - state_mean
                            tot        = n_state + nb
                            state_mean = state_mean + delta * (nb / tot)
                            state_M2   = state_M2 + b_M2 + delta.pow(2) * (n_state * nb / tot)
                            n_state    = tot
                            # min/max 更新
                            state_min  = torch.minimum(state_min, st.min(0).values)
                            state_max  = torch.maximum(state_max, st.max(0).values)

                # --- ACTIONS ---
                act = sample.actions
                if (
                    config is not None
                    and getattr(config, "chunked_actions", False)
                    and getattr(config, "substitute_action", None) != "direction"
                ):
                    bs, T, chunk_sz, ad = act.shape
                    act = act.view(bs * T * chunk_sz, ad)
                else:
                    bs, T, ad = act.shape
                    act = act.view(bs * T, ad)

                act = act.to(device_cpu, dtype=torch.float32, copy=False)
                nb = act.shape[0]
                a_mean = act.mean(0); a_M2 = act.var(0, unbiased=False) * nb
                if not init:
                    action_mean, action_M2 = a_mean.clone(), a_M2.clone()
                    action_min = act.min(0).values; action_max = act.max(0).values
                    n_action = nb
                else:
                    delta = a_mean - action_mean; tot = n_action + nb
                    action_mean = action_mean + delta * (nb / tot)
                    action_M2   = action_M2 + a_M2 + delta.pow(2) * (n_action * nb / tot)
                    n_action    = tot
                    action_min  = torch.minimum(action_min, act.min(0).values)
                    action_max  = torch.maximum(action_max, act.max(0).values)

                # --- LOCATIONS ---
                loc = sample.locations.view(-1, sample.locations.shape[-1]).to(device_cpu, dtype=torch.float32, copy=False)
                nb = loc.shape[0]
                l_mean = loc.mean(0); l_M2 = loc.var(0, unbiased=False) * nb
                if not init:
                    location_mean, location_M2 = l_mean.clone(), l_M2.clone()
                    location_min = loc.min(0).values; location_max = loc.max(0).values
                    n_loc = nb
                else:
                    delta = l_mean - location_mean; tot = n_loc + nb
                    location_mean = location_mean + delta * (nb / tot)
                    location_M2   = location_M2 + l_M2 + delta.pow(2) * (n_loc * nb / tot)
                    n_loc         = tot
                    location_min  = torch.minimum(location_min, loc.min(0).values)
                    location_max  = torch.maximum(location_max, loc.max(0).values)

                # --- PROPIO_POS ---
                if cls._has_attr(sample, "propio_pos"):
                    ppos = sample.propio_pos.view(-1, sample.propio_pos.shape[-1]).to(device_cpu, dtype=torch.float32, copy=False)
                else:
                    ppos = torch.zeros([1, 2], dtype=torch.float32)
                nb = ppos.shape[0]
                ppos_mean_b = ppos.mean(0); ppos_M2_b = ppos.var(0, unbiased=False) * nb
                if not init:
                    propio_pos_mean, propio_pos_M2 = ppos_mean_b.clone(), ppos_M2_b.clone()
                    propio_pos_min = ppos.min(0).values; propio_pos_max = ppos.max(0).values
                    n_ppos = nb
                else:
                    delta = ppos_mean_b - propio_pos_mean; tot = n_ppos + nb
                    propio_pos_mean = propio_pos_mean + delta * (nb / tot)
                    propio_pos_M2   = propio_pos_M2 + ppos_M2_b + delta.pow(2) * (n_ppos * nb / tot)
                    n_ppos          = tot
                    propio_pos_min  = torch.minimum(propio_pos_min, ppos.min(0).values)
                    propio_pos_max  = torch.maximum(propio_pos_max, ppos.max(0).values)

                # --- PROPIO_VEL ---
                if cls._has_attr(sample, "propio_vel"):
                    pvel = sample.propio_vel.view(-1, sample.propio_vel.shape[-1]).to(device_cpu, dtype=torch.float32, copy=False)
                else:
                    pvel = torch.zeros([1, 2], dtype=torch.float32)
                nb = pvel.shape[0]
                pvel_mean_b = pvel.mean(0); pvel_M2_b = pvel.var(0, unbiased=False) * nb
                if not init:
                    propio_vel_mean, propio_vel_M2 = pvel_mean_b.clone(), pvel_M2_b.clone()
                    propio_vel_min = pvel.min(0).values; propio_vel_max = pvel.max(0).values
                    n_pvel = nb
                else:
                    delta = pvel_mean_b - propio_vel_mean; tot = n_pvel + nb
                    propio_vel_mean = propio_vel_mean + delta * (nb / tot)
                    propio_vel_M2   = propio_vel_M2 + pvel_M2_b + delta.pow(2) * (n_pvel * nb / tot)
                    n_pvel          = tot
                    propio_vel_min  = torch.minimum(propio_vel_min, pvel.min(0).values)
                    propio_vel_max  = torch.maximum(propio_vel_max, pvel.max(0).values)

                # --- BLUEBOX_LOCS ---
                if cls._has_attr(sample, "bluebox_locs"):
                    bl = sample.bluebox_locs.view(-1, sample.bluebox_locs.shape[-1]).to(device_cpu, dtype=torch.float32, copy=False)
                else:
                    bl = torch.zeros([1, 3], dtype=torch.float32)
                nb = bl.shape[0]
                bl_mean_b = bl.mean(0); bl_M2_b = bl.var(0, unbiased=False) * nb
                if not init:
                    bluebox_locs_mean, bluebox_locs_M2 = bl_mean_b.clone(), bl_M2_b.clone()
                    bluebox_locs_min = bl.min(0).values; bluebox_locs_max = bl.max(0).values
                    n_blue = nb
                    init = True
                else:
                    delta = bl_mean_b - bluebox_locs_mean; tot = n_blue + nb
                    bluebox_locs_mean = bluebox_locs_mean + delta * (nb / tot)
                    bluebox_locs_M2   = bluebox_locs_M2 + bl_M2_b + delta.pow(2) * (n_blue * nb / tot)
                    n_blue            = tot
                    bluebox_locs_min  = torch.minimum(bluebox_locs_min, bl.min(0).values)
                    bluebox_locs_max  = torch.maximum(bluebox_locs_max, bl.max(0).values)

        # 最終std（不偏に合わせたい場合は (n-1) で割るが、元実装は .std(dim=0) → デフォルト不偏=True なので合わせる）
        eps = 1e-8
        total_state_mean = state_mean
        total_state_std  = torch.sqrt(state_M2 / max(n_state-1, 1) + eps)

        total_action_mean = action_mean
        total_action_std  = torch.sqrt(action_M2 / max(n_action-1, 1) + eps)

        total_location_mean = location_mean
        total_location_std  = torch.sqrt(location_M2 / max(n_loc-1, 1) + eps)

        total_propio_pos_mean = propio_pos_mean
        total_propio_pos_std  = torch.sqrt(propio_pos_M2 / max(n_ppos-1, 1) + eps)

        total_propio_vel_mean = propio_vel_mean
        total_propio_vel_std  = torch.sqrt(propio_vel_M2 / max(n_pvel-1, 1) + eps)

        total_bluebox_locs_mean = bluebox_locs_mean
        total_bluebox_locs_std  = torch.sqrt(bluebox_locs_M2 / max(n_blue-1, 1) + eps)

        # minmax のレンジ
        if normalize_mode == "minmax":
            total_state_min, total_state_max = state_min, state_max
            total_action_min, total_action_max = action_min, action_max
            total_location_min, total_location_max = location_min, location_max
            total_propio_pos_min, total_propio_pos_max = propio_pos_min, propio_pos_max
            total_propio_vel_min, total_propio_vel_max = propio_vel_min, propio_vel_max
            total_bluebox_locs_min, total_bluebox_locs_max = bluebox_locs_min, bluebox_locs_max
        else:
            total_state_min = torch.zeros_like(total_state_mean)
            total_state_max = torch.ones_like(total_state_mean)
            total_action_min = torch.zeros_like(total_action_mean)
            total_action_max = torch.ones_like(total_action_mean)
            total_location_min = torch.zeros_like(total_location_mean)
            total_location_max = torch.ones_like(total_location_mean)
            total_propio_pos_min = torch.zeros_like(total_propio_pos_mean)
            total_propio_pos_max = torch.ones_like(total_propio_pos_mean)
            total_propio_vel_min = torch.zeros_like(total_propio_vel_mean)
            total_propio_vel_max = torch.ones_like(total_propio_vel_mean)
            total_bluebox_locs_min = torch.zeros_like(total_bluebox_locs_mean)
            total_bluebox_locs_max = torch.ones_like(total_bluebox_locs_mean)
        
        if normalize_actions_mode == "minmax":
            total_action_min, total_action_max = action_min, action_max
        else:
            total_action_min = torch.zeros_like(total_action_mean)
            total_action_max = torch.ones_like(total_action_mean)
            
        # return の直前に追加
        device = torch.device("cuda", 0)     # 使っているGPUに合わせて
        dtype  = torch.float32

        def _fix(t):
            return t.to(device=device, dtype=dtype, non_blocking=True).contiguous()

        total_state_mean = _fix(total_state_mean)
        total_state_std  = torch.clamp(_fix(total_state_std), min=1e-6)   # ゼロ割り対策

        total_action_mean = _fix(total_action_mean)
        total_action_std  = torch.clamp(_fix(total_action_std), min=1e-6)

        total_location_mean = _fix(total_location_mean)
        total_location_std  = torch.clamp(_fix(total_location_std), min=1e-6)

        total_propio_pos_mean = _fix(total_propio_pos_mean)
        total_propio_pos_std  = torch.clamp(_fix(total_propio_pos_std), min=1e-6)

        total_propio_vel_mean = _fix(total_propio_vel_mean)
        total_propio_vel_std  = torch.clamp(_fix(total_propio_vel_std), min=1e-6)

        total_bluebox_locs_mean = _fix(total_bluebox_locs_mean)
        total_bluebox_locs_std  = torch.clamp(_fix(total_bluebox_locs_std), min=1e-6)

        # minmax 用の下限上限も同様に
        total_state_min, total_state_max = _fix(total_state_min), _fix(total_state_max)
        total_action_min, total_action_max = _fix(total_action_min), _fix(total_action_max)
        total_location_min, total_location_max = _fix(total_location_min), _fix(total_location_max)
        total_propio_pos_min, total_propio_pos_max = _fix(total_propio_pos_min), _fix(total_propio_pos_max)
        total_propio_vel_min, total_propio_vel_max = _fix(total_propio_vel_min), _fix(total_propio_vel_max)
        total_bluebox_locs_min, total_bluebox_locs_max = _fix(total_bluebox_locs_min), _fix(total_bluebox_locs_max)


        return cls(
            total_state_mean,
            total_state_std,
            total_action_mean,
            total_action_std,
            total_location_mean,
            total_location_std,
            total_propio_pos_mean,
            total_propio_pos_std,
            total_propio_vel_mean,
            total_propio_vel_std,
            total_bluebox_locs_mean,
            total_bluebox_locs_std,
            total_state_min,
            total_state_max,
            total_action_min,
            total_action_max,
            total_location_min,
            total_location_max,
            total_propio_pos_min,
            total_propio_pos_max,
            total_propio_vel_min,
            total_propio_vel_max,
            total_bluebox_locs_min,
            total_bluebox_locs_max,
            backbone_arch = backbone_arch,
            normalize_mode = normalize_mode,
            normalize_actions_mode=normalize_actions_mode,  # ★ ここで渡す
            min_states_val=min_states_val,
            max_states_val=max_states_val,
            min_actions_val=min_actions_val,
            max_actions_val=max_actions_val,
            min_locations_val=min_locations_val,
            max_locations_val=max_locations_val,
            min_propio_pos_val=min_propio_pos_val,
            max_propio_pos_val=max_propio_pos_val,
            min_propio_vel_val=min_propio_vel_val,
            max_propio_vel_val=max_propio_vel_val,
            min_bluebox_locs_val=min_bluebox_locs_val,
            max_bluebox_locs_val=max_bluebox_locs_val,
        )


    @classmethod
    def build_id_normalizer(cls):
        z = torch.zeros(1, device=torch.device("cuda", 0), dtype=torch.float32)
        o = torch.ones(1, device=torch.device("cuda", 0), dtype=torch.float32)
        return cls(
            state_mean=z, state_std=o,
            action_mean=z, action_std=o,
            location_mean=z, location_std=o,
            propio_pos_mean=z, propio_pos_std=o,
            propio_vel_mean=z, propio_vel_std=o,
            bluebox_locs_mean=z, bluebox_locs_std=o,
            state_min=z, state_max=o,
            action_min=z, action_max=o,
            location_min=z, location_max=o,
            propio_pos_min=z, propio_pos_max=o,
            propio_vel_min=z, propio_vel_max=o,
            bluebox_locs_min=z, bluebox_locs_max=o,
            normalize_mode="zscore",  # or "minmax"
            normalize_actions_mode="zscore",  # ★ 追加（同じにしておけばOK）
            # minmax の出力レンジ
            min_states_val=0.0, max_states_val=1.0,
            min_actions_val=0.0, max_actions_val=1.0,
            min_locations_val=0.0, max_locations_val=1.0,
            min_propio_pos_val=0.0, max_propio_pos_val=1.0,
            min_propio_vel_val=0.0, max_propio_vel_val=1.0,
            min_bluebox_locs_val=0.0, max_bluebox_locs_val=1.0,
        )



    def _normalize(self, x, min_val, max_val, mean, std, min_range, max_range, mode: str | None = None, val_type: str = ""):
        # print("[DBG][pldm_envs/utils/normalizer.py] self.backbone_arch:", self.backbone_arch)
        if self.backbone_arch == "vjepa2" and val_type == "state": 
            # print("[DBG][pldm_envs/utils/normalizer.py] return original x if arch = vjepa2")
            return x
        
        
        if mode is None:
            mode = self.normalize_mode  # デフォルトは全体モード

        if mode == "minmax":
            denom = (max_val - min_val).to(x.device)
            denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
            x_norm = (x - min_val.to(x.device)) / denom
            x_norm = x_norm * (max_range - min_range) + min_range
            return x_norm.clamp(min_range, max_range)
        elif mode == "zscore":
            denom = std.to(x.device)
            denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
            return (x - mean.to(x.device)) / denom
        else:
            raise ValueError(f"Unknown normalize mode: {mode}")

    def _unnormalize(self, x_norm, min_val, max_val, mean, std, min_range, max_range, mode: str | None = None, val_type: str = ""):
        # print("[DBG][pldm_envs/utils/normalizer.py] return original x_norm if arch = vjepa2")
        if self.backbone_arch == "vjepa2" and val_type == "state": 
            return x_norm
        if mode is None:
            mode = self.normalize_mode

        if mode == "minmax":
            denom = (max_val - min_val).to(x_norm.device)
            denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
            x = (x_norm - min_range) / (max_range - min_range)
            return x * denom + min_val.to(x_norm.device)
        elif mode == "zscore":
            return x_norm * std.to(x_norm.device) + mean.to(x_norm.device)
        else:
            raise ValueError(f"Unknown normalize mode: {mode}")


    # --- 各フィールド用 ---
    # def normalize_state(self, state):
    #     orig_shape = state.shape
    #     state_flat = state.flatten(start_dim=2).view(-1, state.flatten(start_dim=2).shape[-1])
    #     state_norm = self._normalize(
    #         state_flat, self.state_min, self.state_max,
    #         self.state_mean, self.state_std,
    #         self.min_states_val, self.max_states_val
    #     )
    #     return state_norm.view(orig_shape)


    def normalize_state(self, state: torch.Tensor):
        """
        Input shape must be one of:
        (B,T,C,H,W) / (B,C,H,W) / (C,H,W) / (H,W).
        Normalizes over the flattened last 3 dims (C,H,W).
        """
        assert isinstance(state, torch.Tensor), "normalize_state expects a torch.Tensor"

        

        # (H,W) --> (1,1,H,W)
        if state.ndim == 2:
            x = state.unsqueeze(0).unsqueeze(0)
        # (C,H,W) --> (1,C,H,W)
        elif state.ndim == 3:
            x = state.unsqueeze(0)
        # (B,C,H,W) --> as is
        elif state.ndim == 4:
            x = state
        # (B,T,C,H,W) --> flatten time: (B*T,C,H,W)
        elif state.ndim == 5:
            B, T, C, H, W = state.shape
            x = state.reshape(B * T, C, H, W)   # ← view ではなく reshape
        else:
            raise ValueError(f"Unsupported state shape {state.shape}, ndim={state.ndim}")

        # ここで連続化しておくと安全
        x = x.contiguous()

        # Flatten to (N, D)
        N, C, H, W = x.shape
        x_flat = x.reshape(N, -1)   # ← view ではなく reshape

        # stats も 1D へ
        state_min  = self.state_min.reshape(-1)
        state_max  = self.state_max.reshape(-1)
        state_mean = self.state_mean.reshape(-1)
        state_std  = self.state_std.reshape(-1)

        assert x_flat.shape[-1] == state_min.numel(), \
            f"[normalize_state] D mismatch: x {x_flat.shape[-1]} vs stats {state_min.numel()}"

        # normalize
        x_norm = self._normalize(
            x_flat, state_min, state_max, state_mean, state_std,
            self.min_states_val, self.max_states_val, val_type = "state",
        ).reshape_as(x)   # ← view_as ではなく reshape_as

        # --- restore ---
        if state.ndim == 5:
            out = x_norm.reshape(B, T, C, H, W)
        elif state.ndim == 4:
            out = x_norm
        elif state.ndim == 3:
            out = x_norm[0]
        elif state.ndim == 2:
            out = x_norm[0, 0]
        else:
            out = x_norm

        return out



    def unnormalize_state(self, state_norm: torch.Tensor):
        """
        Input shape must be one of:
        (B,T,C,H,W) / (B,C,H,W) / (C,H,W) / (H,W).
        Inverts the normalization done over the flattened last 3 dims (C,H,W).
        """
        assert isinstance(state_norm, torch.Tensor), "unnormalize_state expects a torch.Tensor"
        if self.backbone_arch == "vjepa2":
            return state_norm

        if state_norm.ndim == 2:
            x = state_norm.unsqueeze(0).unsqueeze(0)
        elif state_norm.ndim == 3:
            x = state_norm.unsqueeze(0)
        elif state_norm.ndim == 4:
            x = state_norm
        elif state_norm.ndim == 5:
            B, T, C, H, W = state_norm.shape
            x = state_norm.reshape(B * T, C, H, W)   # ← reshape
        else:
            raise ValueError(f"Unsupported state_norm shape {state_norm.shape}, ndim={state_norm.ndim}")

        x = x.contiguous()

        N, C, H, W = x.shape
        x_flat = x.reshape(N, -1)   # ← reshape

        state_min  = self.state_min.reshape(-1)
        state_max  = self.state_max.reshape(-1)
        state_mean = self.state_mean.reshape(-1)
        state_std  = self.state_std.reshape(-1)

        assert x_flat.shape[-1] == state_min.numel(), \
            f"[unnormalize_state] D mismatch: x {x_flat.shape[-1]} vs stats {state_min.numel()}"

        x_unnorm = self._unnormalize(
            x_flat, state_min, state_max, state_mean, state_std,
            self.min_states_val, self.max_states_val, val_type = "state",
        ).reshape_as(x)   # ← reshape_as

        if state_norm.ndim == 5:
            out = x_unnorm.reshape(B, T, C, H, W)
        elif state_norm.ndim == 4:
            out = x_unnorm
        elif state_norm.ndim == 3:
            out = x_unnorm[0]
        elif state_norm.ndim == 2:
            out = x_unnorm[0, 0]
        else:
            out = x_unnorm

        return out


    def normalize_action(self, action):
        return self._normalize(
            action,
            self.action_min, self.action_max,
            self.action_mean, self.action_std,
            self.min_actions_val, self.max_actions_val,
            mode=self.normalize_actions_mode,   # ★ ここだけ別モード！
            val_type = "action",
        )

    def unnormalize_action(self, action_norm):
        return self._unnormalize(
            action_norm,
            self.action_min, self.action_max,
            self.action_mean, self.action_std,
            self.min_actions_val, self.max_actions_val,
            mode=self.normalize_actions_mode,
            val_type = "action",
        )

    def normalize_location(self, location):
        return self._normalize(location, self.location_min, self.location_max,
                               self.location_mean, self.location_std,
                               self.min_locations_val, self.max_locations_val, val_type = "location",)

    def unnormalize_location(self, location_norm):
        return self._unnormalize(location_norm, self.location_min, self.location_max,
                                 self.location_mean, self.location_std,
                                 self.min_locations_val, self.max_locations_val, val_type = "location",)

    def normalize_propio_pos(self, propio_pos):
        return self._normalize(propio_pos, self.propio_pos_min, self.propio_pos_max,
                               self.propio_pos_mean, self.propio_pos_std,
                               self.min_propio_pos_val, self.max_propio_pos_val, val_type = "propio_pos",)

    def unnormalize_propio_pos(self, propio_pos_norm):
        return self._unnormalize(propio_pos_norm, self.propio_pos_min, self.propio_pos_max,
                                 self.propio_pos_mean, self.propio_pos_std,
                                 self.min_propio_pos_val, self.max_propio_pos_val, val_type = "propio_pos",)

    def normalize_propio_vel(self, propio_vel):
        return self._normalize(propio_vel, self.propio_vel_min, self.propio_vel_max,
                               self.propio_vel_mean, self.propio_vel_std,
                               self.min_propio_vel_val, self.max_propio_vel_val, val_type = "propio_vel",)

    def unnormalize_propio_vel(self, propio_vel_norm):
        return self._unnormalize(propio_vel_norm, self.propio_vel_min, self.propio_vel_max,
                                 self.propio_vel_mean, self.propio_vel_std,
                                 self.min_propio_vel_val, self.max_propio_vel_val, val_type = "propio_vel",)

    def normalize_bluebox_locs(self, bluebox_locs):
        return self._normalize(bluebox_locs, self.bluebox_locs_min, self.bluebox_locs_max,
                               self.bluebox_locs_mean, self.bluebox_locs_std,
                               self.min_bluebox_locs_val, self.max_bluebox_locs_val, val_type = "bluebox_locs",)

    def unnormalize_bluebox_locs(self, bluebox_locs_norm):
        return self._unnormalize(bluebox_locs_norm, self.bluebox_locs_min, self.bluebox_locs_max,
                                 self.bluebox_locs_mean, self.bluebox_locs_std,
                                 self.min_bluebox_locs_val, self.max_bluebox_locs_val, val_type = "bluebox_locs",)




    def normalize_sample(self, sample):
        replaced = {}
        if self._has_attr(sample, "states"):
            
            replaced["states"] = self.normalize_state(sample.states)
            
            # print("=== states sample ===")
            # print(replaced['states'][:10].shape)  # 先頭10要素のチャンネル表示
            # print("mean:", replaced['states'].mean().item())
            # print("std :", replaced['states'].std().item())
            
        if self._has_attr(sample, "locations"):
            # print('=== before normalize locations sample ====')
            # print('shape:', sample.locations.shape)
            # print('mean:', sample.locations.mean().item())
            # print('std:', sample.locations.std().item())
            
            replaced["locations"] = self.normalize_location(sample.locations)

            # print("=== locations sample ===")
            # print(replaced['locations'][:10].shape)  # 先頭10要素のチャンネル表示
            # print("mean:", replaced['locations'].mean().item())
            # print("std :", replaced['locations'].std().item())
    
        if self._has_attr(sample, "actions"):
            replaced["actions"] = self.normalize_action(sample.actions)

            # print("=== actions sample ===")
            # print(replaced['actions'][:10].shape)  # 先頭10要素のチャンネル表示
            # print("mean:", replaced['actions'].mean().item())
            # print("std :", replaced['actions'].std().item())
        

        if self._has_attr(sample, "goal"):
            # print('=== before normalize locations sample ====')
            # print('shape:', sample.goal.shape)
            # print('mean:', sample.goal.mean().item())
            # print('std:', sample.goal.std().item())
            
            replaced["goal"] = self.normalize_location(sample.goal)
            
            # print("=== goal sample ===")
            # print(replaced['goal'][:10].shape)  # 先頭10要素のチャンネル表示
            # print("mean:", replaced['goal'].mean().item())
            # print("std :", replaced['goal'].std().item())
            
        if self._has_attr(sample, "propio_pos"):
            replaced["propio_pos"] = self.normalize_propio_pos(sample.propio_pos)

            # print("=== propio_pos sample ===")
            # print(replaced['propio_pos'][:10].shape)
            # print("mean:", replaced['propio_pos'].mean().item())
            # print("std :", replaced['propio_pos'].std().item())
            
        if self._has_attr(sample, "propio_vel"):
            replaced["propio_vel"] = self.normalize_propio_vel(sample.propio_vel)
            
            # print("=== propio_vel sample ===")
            # print(replaced['propio_vel'][:10].shape)
            # print("mean:", replaced['propio_vel'].mean().item())
            # print("std :", replaced['propio_vel'].std().item())

        if self._has_attr(sample, "bluebox_locs"):
            replaced["bluebox_locs"] = self.normalize_bluebox_locs(sample.bluebox_locs)
            
        if self._has_attr(sample, "chunked_locations"):
            replaced["chunked_locations"] = self.normalize_location(
                sample.chunked_locations
            )
        if self._has_attr(sample, "chunked_propio_pos"):
            replaced["chunked_propio_pos"] = self.normalize_propio_pos(
                sample.chunked_propio_pos
            )
        if self._has_attr(sample, "chunked_propio_vel"):
            replaced["chunked_propio_vel"] = self.normalize_propio_vel(
                sample.chunked_propio_vel
            )

        return sample._replace(**replaced)

    @torch.no_grad()
    def unnormalize_mse(self, mse, attribute="locations"):
        if self.normalize_mode == "zscore":
            std_mapper = {
                "locations": self.location_std,
                "propio_pos": self.propio_pos_std,
                "propio_vel": self.propio_vel_std,
                "bluebox_locs": self.bluebox_locs_std,
            }
            scale = std_mapper[attribute].to(mse.device)
            return mse * (scale ** 2)
        else:  # minmax
            # x_norm = (x - min) / (range) * (max_range - min_range) + min_range
            # ⇒ dx/dx_norm = range / (max_range - min_range)
            if attribute == "locations":
                rng = (self.location_max - self.location_min).to(mse.device)
                out_rng = (self.max_locations_val - self.min_locations_val)
            elif attribute == "propio_pos":
                rng = (self.propio_pos_max - self.propio_pos_min).to(mse.device)
                out_rng = (self.max_propio_pos_val - self.min_propio_pos_val)
            elif attribute == "propio_vel":
                rng = (self.propio_vel_max - self.propio_vel_min).to(mse.device)
                out_rng = (self.max_propio_vel_val - self.min_propio_vel_val)
            elif attribute == "bluebox_locs":
                rng = (self.bluebox_locs_max - self.bluebox_locs_min).to(mse.device)
                out_rng = (self.max_bluebox_locs_val - self.min_bluebox_locs_val)
            else:
                raise ValueError(f"unknown attribute {attribute}")

            rng = torch.where(rng < 1e-6, torch.ones_like(rng), rng)
            scale = rng / out_rng
            return mse * (scale ** 2)


    def to(self, device):
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        self.action_mean = self.action_mean.to(device)
        self.action_std = self.action_std.to(device)
        self.location_mean = self.location_mean.to(device)
        self.location_std = self.location_std.to(device)
        self.propio_pos_mean = self.propio_pos_mean.to(device)
        self.propio_pos_std = self.propio_pos_std.to(device)
        self.propio_vel_mean = self.propio_vel_mean.to(device)
        self.propio_vel_std = self.propio_vel_std.to(device)
        self.bluebox_locs_mean = self.bluebox_locs_mean.to(device)
        self.bluebox_locs_std = self.bluebox_locs_std.to(device)
        
        
        self.state_min = self.state_min.to(device)
        self.state_max = self.state_max.to(device)
        self.action_min = self.action_min.to(device)
        self.action_max = self.action_max.to(device)
        self.location_min = self.location_min.to(device)
        self.location_max = self.location_max.to(device)
        self.propio_pos_min = self.propio_pos_min.to(device)
        self.propio_pos_max = self.propio_pos_max.to(device)
        self.propio_vel_min = self.propio_vel_min.to(device)
        self.propio_vel_max = self.propio_vel_max.to(device)
        self.bluebox_locs_min = self.bluebox_locs_min.to(device)
        self.bluebox_locs_max = self.bluebox_locs_max.to(device)


    def save(self, path):
        torch.save(
            {
                "normalize_mode": self.normalize_mode,
                "normalize_actions_mode": self.normalize_actions_mode,  # ★ 追加
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                "action_mean": self.action_mean,
                "action_std": self.action_std,
                "location_mean": self.location_mean,
                "location_std": self.location_std,
                "propio_pos_mean": self.propio_pos_mean,
                "propio_pos_std": self.propio_pos_std,
                "propio_vel_mean": self.propio_vel_mean,
                "propio_vel_std": self.propio_vel_std,
                "bluebox_locs_mean": self.bluebox_locs_mean,
                "bluebox_locs_std": self.bluebox_locs_std,
                
                # add min/max
                "state_min": self.state_min,
                "state_max": self.state_max,
                "action_min": self.action_min,
                "action_max": self.action_max,
                "location_min": self.location_min,
                "location_max": self.location_max,
                "propio_pos_min": self.propio_pos_min,
                "propio_pos_max": self.propio_pos_max,
                "propio_vel_min": self.propio_vel_min,
                "propio_vel_max": self.propio_vel_max,
                "bluebox_locs_min": self.bluebox_locs_min,
                "bluebox_locs_max": self.bluebox_locs_max,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu")
        return cls(
            state["state_mean"],
            state["state_std"],
            state["action_mean"],
            state["action_std"],
            state["location_mean"],
            state["location_std"],
            state["propio_pos_mean"],
            state["propio_pos_std"],
            state["propio_vel_mean"],
            state["propio_vel_std"],
            state["bluebox_locs_mean"],
            state["bluebox_locs_std"],
            
            state["state_min"],
            state["state_max"],
            state["action_min"],
            state["action_max"],
            state["location_min"],
            state["location_max"],
            state["propio_pos_min"],
            state["propio_pos_max"],
            state["propio_vel_min"],
            state["propio_vel_max"],
            state["bluebox_locs_min"],
            state["bluebox_locs_max"],
            normalize_mode=state.get("normalize_mode", "minmax"),
            normalize_actions_mode=state.get("normalize_actions_mode", state.get("normalize_mode", "minmax")),
        )

    def state_dict(self):
        return {
            "normalize_mode": self.normalize_mode,
            "normalize_actions_mode": self.normalize_actions_mode,  # ★ 追加
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "location_mean": self.location_mean,
            "location_std": self.location_std,
            "propio_pos_mean": self.propio_pos_mean,
            "propio_pos_std": self.propio_pos_std,
            "propio_vel_mean": self.propio_vel_mean,
            "propio_vel_std": self.propio_vel_std,
            "bluebox_locs_mean": self.bluebox_locs_mean,
            "bluebox_locs_std": self.bluebox_locs_std,

            "state_min": self.state_min,
            "state_max": self.state_max,
            "action_min": self.action_min,
            "action_max": self.action_max,
            "location_min": self.location_min,
            "location_max": self.location_max,
            "propio_pos_min": self.propio_pos_min,
            "propio_pos_max": self.propio_pos_max,
            "propio_vel_min": self.propio_vel_min,
            "propio_vel_max": self.propio_vel_max,
            "bluebox_locs_min": self.bluebox_locs_min,
            "bluebox_locs_max": self.bluebox_locs_max,
        }

    def load_state_dict(self, state):
        self.state_mean = torch.tensor(state["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(state["state_std"], dtype=torch.float32)
        self.action_mean = torch.tensor(state["action_mean"], dtype=torch.float32)
        self.action_std = torch.tensor(state["action_std"], dtype=torch.float32)
        self.location_mean = torch.tensor(state["location_mean"], dtype=torch.float32)
        self.location_std = torch.tensor(state["location_std"], dtype=torch.float32)
        self.propio_pos_mean = torch.tensor(state["propio_pos_mean"], dtype=torch.float32)
        self.propio_pos_std = torch.tensor(state["propio_pos_std"], dtype=torch.float32)
        self.propio_vel_mean = torch.tensor(state["propio_vel_mean"], dtype=torch.float32)
        self.propio_vel_std = torch.tensor(state["propio_vel_std"], dtype=torch.float32)
        self.bluebox_locs_mean = torch.tensor(state["bluebox_locs_mean"], dtype=torch.float32)
        self.bluebox_locs_std = torch.tensor(state["bluebox_locs_std"], dtype=torch.float32)

        self.state_min = torch.tensor(state["state_min"], dtype=torch.float32)
        self.state_max = torch.tensor(state["state_max"], dtype=torch.float32)
        self.action_min = torch.tensor(state["action_min"], dtype=torch.float32)
        self.action_max = torch.tensor(state["action_max"], dtype=torch.float32)
        self.location_min = torch.tensor(state["location_min"], dtype=torch.float32)
        self.location_max = torch.tensor(state["location_max"], dtype=torch.float32)
        self.propio_pos_min = torch.tensor(state["propio_pos_min"], dtype=torch.float32)
        self.propio_pos_max = torch.tensor(state["propio_pos_max"], dtype=torch.float32)
        self.propio_vel_min = torch.tensor(state["propio_vel_min"], dtype=torch.float32)
        self.propio_vel_max = torch.tensor(state["propio_vel_max"], dtype=torch.float32)
        self.bluebox_locs_min = torch.tensor(state["bluebox_locs_min"], dtype=torch.float32)
        self.bluebox_locs_max = torch.tensor(state["bluebox_locs_max"], dtype=torch.float32)
        self.normalize_mode = state.get("normalize_mode", "minmax")
        self.normalize_actions_mode = state.get(
            "normalize_actions_mode",
            self.normalize_mode,  # ★ 互換性のため normalize_mode を fallback に
        )