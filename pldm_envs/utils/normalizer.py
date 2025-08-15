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
        normalize_mode: str = "minmax",
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
        #モードチェック
        self.normalize_mode = normalize_mode.lower()
        assert self.normalize_mode in ("minmax", "zscore"), "normalize_mode must be 'minmax' or 'zscore'"
        
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
        normalize_mode: str = "minmax",
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
        all_actions = []
        all_locations = []
        all_states = []
        all_propio_pos = []
        all_propio_vel = []
        all_bluebox_locs = []

        all_states_min = []
        all_states_max = []
        all_actions_min = []
        all_actions_max = []
        all_locations_min = []
        all_locations_max = []
        all_propio_pos_min = []
        all_propio_pos_max = []
        all_propio_vel_min = []
        all_propio_vel_max = []
        
        all_bluebox_locs_min = []
        all_bluebox_locs_max = []

        config = (
            dataset.dataset.config if hasattr(dataset, "dataset") else dataset.config
        )

        it = iter(dataset)
        for _i in tqdm(range(n_samples), desc="Estimating normalizer stats"):
            try:
                sample = next(it)
            except StopIteration:
                it = iter(dataset)
                sample = next(it)

            # --- STATES ---
            if cls._has_attr(sample, "states"):
                if len(sample.states.shape) == 5:
                    states = sample.states.float()

                    # flatten state to (B*T, C*H*W)
                    states_flat = states.flatten(start_dim=2)   # (B, T, C*H*W)
                    states_flat = states_flat.view(-1, states_flat.shape[-1])  # (B*T, C*H*W)

                    all_states.append(states_flat)  # (N, D)

                    if normalize_mode == "minmax":
                        all_states_min.append(states_flat.min(dim=0).values)
                        all_states_max.append(states_flat.max(dim=0).values)

                else:
                    # proprio
                    states = sample.states
                    states_flat = states.view(-1, states.shape[-1])
                    all_states.append(states_flat)
                    if normalize_mode == "minmax":
                        all_states_min.append(states_flat.min(dim=0).values)
                        all_states_max.append(states_flat.max(dim=0).values)
            else:
                # dummy zeros
                all_states.append(torch.zeros((1, 1)))

            # --- ACTIONS ---
            actions = sample.actions
            if config.chunked_actions and not config.substitute_action == "direction":
                bs, T, chunk_size, action_dim = actions.shape
            else:
                bs, T, action_dim = actions.shape
            actions_flat = actions.view(-1, action_dim)
            all_actions.append(actions_flat)

            if normalize_mode == "minmax":
                all_actions_min.append(actions_flat.min(dim=0).values)
                all_actions_max.append(actions_flat.max(dim=0).values)

            # --- LOCATIONS ---
            locations = sample.locations.view(-1, sample.locations.shape[-1])
            all_locations.append(locations)
            if normalize_mode == "minmax":
                all_locations_min.append(locations.min(dim=0).values)
                all_locations_max.append(locations.max(dim=0).values)

            # --- PROPIO_POS ---
            if cls._has_attr(sample, "propio_pos"):
                propio_pos = sample.propio_pos.view(-1, sample.propio_pos.shape[-1])
            else:
                propio_pos = torch.zeros([1, 2])
            all_propio_pos.append(propio_pos)
            if normalize_mode == "minmax":
                all_propio_pos_min.append(propio_pos.min(dim=0).values)
                all_propio_pos_max.append(propio_pos.max(dim=0).values)

            # --- PROPIO_VEL ---
            if cls._has_attr(sample, "propio_vel"):
                propio_vel = sample.propio_vel.view(-1, sample.propio_vel.shape[-1])
            else:
                propio_vel = torch.zeros([1, 2])
            all_propio_vel.append(propio_vel)
            if normalize_mode == "minmax":
                all_propio_vel_min.append(propio_vel.min(dim=0).values)
                all_propio_vel_max.append(propio_vel.max(dim=0).values)

            # --- BLUEBOX_LOCS ---
            if cls._has_attr(sample, "bluebox_locs"):
                bluebox_locs = sample.bluebox_locs.view(-1, sample.bluebox_locs.shape[-1])
            else:
                bluebox_locs = torch.zeros([1, 3])
            all_bluebox_locs.append(bluebox_locs)
            if normalize_mode == "minmax":
                all_bluebox_locs_min.append(bluebox_locs.min(dim=0).values)
                all_bluebox_locs_max.append(bluebox_locs.max(dim=0).values)

        if hasattr(dataset, "config") and normalizer_hardset: #False
            ds_stats = STATS[dataset.__class__.__name__]
            total_state_mean = ds_stats["state_mean"].to(locations.device)
            total_state_std = ds_stats["state_std"].to(locations.device)
            total_action_mean = ds_stats["action_mean"].to(locations.device)
            total_action_std = ds_stats["action_std"].to(locations.device)
            total_location_mean = ds_stats["location_mean"].to(locations.device)
            total_location_std = ds_stats["location_std"].to(locations.device)
            total_propio_pos_mean = ds_stats["propio_pos_mean"].to(locations.device)
            total_propio_pos_std = ds_stats["propio_pos_std"].to(locations.device)
            total_propio_vel_mean = ds_stats["propio_vel_mean"].to(locations.device)
            total_propio_vel_std = ds_stats["propio_vel_std"].to(locations.device)
            total_bluebox_locs_mean = ds_stats["bluebox_locs_mean"].to(locations.device)
            total_bluebox_locs_std = ds_stats["bluebox_locs_std"].to(locations.device)

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
        else: ##
            total_state = torch.cat(all_states, dim=0)
            total_state_mean = total_state.mean(dim=0)
            total_state_std = total_state.std(dim=0)

            total_action = torch.cat(all_actions)
            total_action_mean = total_action.mean(dim=0)
            total_action_std = total_action.std(dim=0)

            total_location = torch.cat(all_locations)
            total_location_mean = total_location.mean(dim=0)
            total_location_std = total_location.std(dim=0)

            total_propio_pos = torch.cat(all_propio_pos)
            total_propio_pos_mean = total_propio_pos.mean(dim=0)
            total_propio_pos_std = total_propio_pos.std(dim=0)

            total_propio_vel = torch.cat(all_propio_vel)
            total_propio_vel_mean = total_propio_vel.mean(dim=0)
            total_propio_vel_std = total_propio_vel.std(dim=0)

            total_bluebox_locs = torch.cat(all_bluebox_locs)
            total_bluebox_locs_mean = total_bluebox_locs.mean(dim=0)
            total_bluebox_locs_std = total_bluebox_locs.std(dim=0)
            
            
            if normalize_mode == "minmax":
                total_state_min = torch.stack(all_states_min).min(dim=0).values
                total_state_max = torch.stack(all_states_max).max(dim=0).values

                total_action_min = torch.stack(all_actions_min).min(dim=0).values
                total_action_max = torch.stack(all_actions_max).max(dim=0).values

                total_location_min = torch.stack(all_locations_min).min(dim=0).values
                total_location_max = torch.stack(all_locations_max).max(dim=0).values

                total_propio_pos_min = torch.stack(all_propio_pos_min).min(dim=0).values
                total_propio_pos_max = torch.stack(all_propio_pos_max).max(dim=0).values

                total_propio_vel_min = torch.stack(all_propio_vel_min).min(dim=0).values
                total_propio_vel_max = torch.stack(all_propio_vel_max).max(dim=0).values
                
                total_bluebox_locs_min = torch.stack(all_bluebox_locs_min).min(dim=0).values
                total_bluebox_locs_max = torch.stack(all_bluebox_locs_max).max(dim=0).values
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
            normalize_mode = normalize_mode,
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
        z = torch.zeros(1)
        o = torch.ones(1)
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
            normalize_mode="zscore",  # あるいは "minmax"
            # minmax の出力レンジ（必要なら）
            min_states_val=0.0, max_states_val=1.0,
            min_actions_val=0.0, max_actions_val=1.0,
            min_locations_val=0.0, max_locations_val=1.0,
            min_propio_pos_val=0.0, max_propio_pos_val=1.0,
            min_propio_vel_val=0.0, max_propio_vel_val=1.0,
            min_bluebox_locs_val=0.0, max_bluebox_locs_val=1.0,
        )



   # --- 共通ヘルパ ---
    def _normalize(self, x, min_val, max_val, mean, std, min_range, max_range):
        if self.normalize_mode == "minmax":
            denom = (max_val - min_val).to(x.device)
            denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
            x_norm = (x - min_val.to(x.device)) / denom
            x_norm = x_norm * (max_range - min_range) + min_range
            return x_norm.clamp(min_range, max_range)
        else:  # z-score
            denom = std.to(x.device)
            denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
            return (x - mean.to(x.device)) / denom

    def _unnormalize(self, x_norm, min_val, max_val, mean, std, min_range, max_range):
        if self.normalize_mode == "minmax":
            denom = (max_val - min_val).to(x_norm.device)
            denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
            x = (x_norm - min_range) / (max_range - min_range)
            return x * denom + min_val.to(x_norm.device)
        else:  # z-score
            return x_norm * std.to(x_norm.device) + mean.to(x_norm.device)

    # --- 各フィールド用 ---
    def normalize_state(self, state):
        orig_shape = state.shape
        state_flat = state.flatten(start_dim=2).view(-1, state.flatten(start_dim=2).shape[-1])
        state_norm = self._normalize(
            state_flat, self.state_min, self.state_max,
            self.state_mean, self.state_std,
            self.min_states_val, self.max_states_val
        )
        return state_norm.view(orig_shape)

    def unnormalize_state(self, state_norm):
        orig_shape = state_norm.shape
        state_flat = state_norm.flatten(start_dim=2).view(-1, state_norm.flatten(start_dim=2).shape[-1])
        state_unnorm = self._unnormalize(
            state_flat, self.state_min, self.state_max,
            self.state_mean, self.state_std,
            self.min_states_val, self.max_states_val
        )
        return state_unnorm.view(orig_shape)

    def normalize_action(self, action):
        return self._normalize(action, self.action_min, self.action_max,
                               self.action_mean, self.action_std,
                               self.min_actions_val, self.max_actions_val)

    def unnormalize_action(self, action_norm):
        return self._unnormalize(action_norm, self.action_min, self.action_max,
                                 self.action_mean, self.action_std,
                                 self.min_actions_val, self.max_actions_val)

    def normalize_location(self, location):
        return self._normalize(location, self.location_min, self.location_max,
                               self.location_mean, self.location_std,
                               self.min_locations_val, self.max_locations_val)

    def unnormalize_location(self, location_norm):
        return self._unnormalize(location_norm, self.location_min, self.location_max,
                                 self.location_mean, self.location_std,
                                 self.min_locations_val, self.max_locations_val)

    def normalize_propio_pos(self, propio_pos):
        return self._normalize(propio_pos, self.propio_pos_min, self.propio_pos_max,
                               self.propio_pos_mean, self.propio_pos_std,
                               self.min_propio_pos_val, self.max_propio_pos_val)

    def unnormalize_propio_pos(self, propio_pos_norm):
        return self._unnormalize(propio_pos_norm, self.propio_pos_min, self.propio_pos_max,
                                 self.propio_pos_mean, self.propio_pos_std,
                                 self.min_propio_pos_val, self.max_propio_pos_val)

    def normalize_propio_vel(self, propio_vel):
        return self._normalize(propio_vel, self.propio_vel_min, self.propio_vel_max,
                               self.propio_vel_mean, self.propio_vel_std,
                               self.min_propio_vel_val, self.max_propio_vel_val)

    def unnormalize_propio_vel(self, propio_vel_norm):
        return self._unnormalize(propio_vel_norm, self.propio_vel_min, self.propio_vel_max,
                                 self.propio_vel_mean, self.propio_vel_std,
                                 self.min_propio_vel_val, self.max_propio_vel_val)

    def normalize_bluebox_locs(self, bluebox_locs):
        return self._normalize(bluebox_locs, self.bluebox_locs_min, self.bluebox_locs_max,
                               self.bluebox_locs_mean, self.bluebox_locs_std,
                               self.min_bluebox_locs_val, self.max_bluebox_locs_val)

    def unnormalize_bluebox_locs(self, bluebox_locs_norm):
        return self._unnormalize(bluebox_locs_norm, self.bluebox_locs_min, self.bluebox_locs_max,
                                 self.bluebox_locs_mean, self.bluebox_locs_std,
                                 self.min_bluebox_locs_val, self.max_bluebox_locs_val)




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
            normalize_mode=state.get("normalize_mode", "minmax")
        )

    def state_dict(self):
        return {
            "normalize_mode": self.normalize_mode,
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