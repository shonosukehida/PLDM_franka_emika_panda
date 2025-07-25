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
        min_max_normalize: bool = True,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ):
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
        
        self.min_max_normalize = min_max_normalize
        
        self.min_val = min_val
        self.max_val = max_val

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
        min_max_normalize: bool = False,
        normalizer_hardset: bool = False,
        min_val: float = 0.0,
        max_val: float = 1.0,
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

                    if min_max_normalize:
                        all_states_min.append(states_flat.min(dim=0).values)
                        all_states_max.append(states_flat.max(dim=0).values)

                else:
                    raise NotImplementedError(
                        "min_max_state not implemented for propio"
                    )
                    # proprio
                    states = sample.states
                    states_flat = states.view(-1, states.shape[-1])
                    all_states.append(states_flat)
                    if min_max_normalize:
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

            if min_max_normalize:
                all_actions_min.append(actions_flat.min(dim=0).values)
                all_actions_max.append(actions_flat.max(dim=0).values)

            # --- LOCATIONS ---
            locations = sample.locations.view(-1, sample.locations.shape[-1])
            all_locations.append(locations)
            if min_max_normalize:
                all_locations_min.append(locations.min(dim=0).values)
                all_locations_max.append(locations.max(dim=0).values)

            # --- PROPIO_POS ---
            if cls._has_attr(sample, "propio_pos"):
                propio_pos = sample.propio_pos.view(-1, sample.propio_pos.shape[-1])
            else:
                propio_pos = torch.zeros([1, 2])
            all_propio_pos.append(propio_pos)
            if min_max_normalize:
                all_propio_pos_min.append(propio_pos.min(dim=0).values)
                all_propio_pos_max.append(propio_pos.max(dim=0).values)

            # --- PROPIO_VEL ---
            if cls._has_attr(sample, "propio_vel"):
                propio_vel = sample.propio_vel.view(-1, sample.propio_vel.shape[-1])
            else:
                propio_vel = torch.zeros([1, 2])
            all_propio_vel.append(propio_vel)
            if min_max_normalize:
                all_propio_vel_min.append(propio_vel.min(dim=0).values)
                all_propio_vel_max.append(propio_vel.max(dim=0).values)

            # --- BLUEBOX_LOCS ---
            if cls._has_attr(sample, "bluebox_locs"):
                bluebox_locs = sample.bluebox_locs.view(-1, sample.bluebox_locs.shape[-1])
            else:
                bluebox_locs = torch.zeros([1, 3])
            all_bluebox_locs.append(bluebox_locs)
            if min_max_normalize:
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
        else:
            total_state = torch.cat(all_states, dim=-1)
            total_state_mean = total_state.mean(dim=-1)
            total_state_std = total_state.std(dim=-1)

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
            
            
            if min_max_normalize:
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
            min_max_normalize=min_max_normalize,
            min_val=min_val,
            max_val=max_val,
        )

    @classmethod
    def build_id_normalizer(cls):
        return cls(
            state_mean=torch.zeros(1),
            state_std=torch.ones(1),
            action_mean=torch.zeros(1),
            action_std=torch.ones(1),
            location_mean=torch.zeros(1),
            location_std=torch.ones(1),
            propio_pos_mean=torch.zeros(1),
            propio_pos_std=torch.ones(1),
            propio_vel_mean=torch.zeros(1),
            propio_vel_std=torch.ones(1),
            bluebox_locs_mean=torch.zeros(1),
            bluebox_locs_std=torch.ones(1),
            min_max_normalize=False,
        )



    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Min-max normalize state to [0,1] range
        """
        orig_shape = state.shape
        state_flat = state.flatten(start_dim=2)   # (B, T, C*H*W)
        state_flat = state_flat.view(-1, state_flat.shape[-1])   # (B*T, C*H*W)

        denom = (self.state_max - self.state_min).to(state.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)

        state_norm_flat = (state_flat - self.state_min.to(state.device)) / denom * (self.max_val - self.min_val) + self.min_val
        state_norm_flat = state_norm_flat.clamp(self.min_val, self.max_val)

        state_norm = state_norm_flat.view(orig_shape)
        return state_norm

    def unnormalize_state(self, state_norm: torch.Tensor) -> torch.Tensor:
        """
        Undo min-max normalization for state
        """
        orig_shape = state_norm.shape
        state_norm_flat = state_norm.flatten(start_dim=2)   # (B, T, C*H*W)
        state_norm_flat = state_norm_flat.view(-1, state_norm_flat.shape[-1])   # (B*T, C*H*W)

        denom = (self.state_max - self.state_min).to(state_norm.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)

        state_unnorm_flat = (state_norm_flat - self.min_val) / (self.max_val - self.min_val) * denom + self.state_min.to(state_norm.device)
        state_unnorm = state_unnorm_flat.view(orig_shape)

        return state_unnorm



    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        denom = (self.action_max - self.action_min).to(action.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        
        action_norm = ((action - self.action_min.to(action.device)) / denom) * (self.max_val - self.min_val) + self.min_val
        action_norm = action_norm.clamp(self.min_val, self.max_val)
        return action_norm
    
    def unnormalize_action(self, action_norm: torch.Tensor) -> torch.Tensor:
        denom = (self.action_max - self.action_min).to(action_norm.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        
        action_unnorm = (action_norm - self.min_val) / (self.max_val - self.min_val) * denom + self.action_min.to(action_norm.device)
        return action_unnorm



    def normalize_location(self, location: torch.Tensor) -> torch.Tensor:
        denom = (self.location_max - self.location_min).to(location.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        loc_norm = ((location - self.location_min.to(location.device)) / denom) * (self.max_val - self.min_val) + self.min_val
        loc_norm = loc_norm.clamp(self.min_val, self.max_val)
        return loc_norm

    def unnormalize_location(self, location_norm: torch.Tensor) -> torch.Tensor:
        denom = (self.location_max - self.location_min).to(location_norm.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        
        loc_unnorm = (location_norm - self.min_val) / (self.max_val - self.min_val) * denom + self.location_min.to(location_norm.device)
        return loc_unnorm

    def normalize_propio_pos(self, propio_pos: torch.Tensor) -> torch.Tensor:
        denom = (self.propio_pos_max - self.propio_pos_min).to(propio_pos.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        propio_pos_norm = (propio_pos - self.propio_pos_min.to(propio_pos.device)) / denom * (self.max_val - self.min_val) + self.min_val
        propio_pos_norm = propio_pos_norm.clamp(self.min_val, self.max_val)
        return propio_pos_norm

    def unnormalize_propio_pos(self, propio_pos_norm: torch.Tensor) -> torch.Tensor:
        denom = (self.propio_pos_max - self.propio_pos_min).to(propio_pos_norm.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        propio_pos_unnorm = (propio_pos_norm - self.min_val) / (self.max_val - self.min_val) * denom + self.propio_pos_min.to(propio_pos_norm.device)
        return propio_pos_unnorm

    def normalize_propio_vel(self, propio_vel: torch.Tensor) -> torch.Tensor:
        denom = (self.propio_vel_max - self.propio_vel_min).to(propio_vel.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        propio_vel_norm = (propio_vel - self.propio_vel_min.to(propio_vel.device)) / denom * (self.max_val - self.min_val) + self.min_val
        propio_vel_norm = propio_vel_norm.clamp(self.min_val, self.max_val)
        return propio_vel_norm

    def unnormalize_propio_vel(self, propio_vel_norm: torch.Tensor) -> torch.Tensor:
        denom = (self.propio_vel_max - self.propio_vel_min).to(propio_vel_norm.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        propio_vel_unnorm = (propio_vel_norm - self.min_val) / (self.max_val - self.min_val) * denom + self.propio_vel_min.to(propio_vel_norm.device)
        return propio_vel_unnorm

    def normalize_bluebox_locs(self, bluebox_locs: torch.Tensor) -> torch.Tensor:
        denom = (self.bluebox_locs_max - self.bluebox_locs_min).to(bluebox_locs.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        
        bluebox_locs_norm = (bluebox_locs - self.bluebox_locs_min.to(bluebox_locs.device)) / denom * (self.max_val - self.min_val) + self.min_val
        bluebox_locs_norm = bluebox_locs_norm.clamp(self.min_val, self.max_val)
        return bluebox_locs_norm

    def unnormalize_bluebox_locs(self, bluebox_locs_norm: torch.Tensor) -> torch.Tensor:
        denom = (self.bluebox_locs_max - self.bluebox_locs_min).to(bluebox_locs_norm.device)
        denom = torch.where(denom < 1e-6, torch.ones_like(denom), denom)
        
        bluebox_locs_unnorm = (bluebox_locs_norm - self.min_val) / (self.max_val - self.min_val) * denom + self.bluebox_locs_min.to(bluebox_locs_norm.device)
        return bluebox_locs_unnorm



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
        # unnormalize locations mse
        std_mapper = {
            "locations": self.location_std,
            "propio_pos": self.propio_pos_std,
            "propio_vel": self.propio_vel_std,
            "bluebox_locs": self.bluebox_locs_std,
        }

        return mse * std_mapper[attribute].to(mse.device) ** 2

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
                "min_max_normalize": self.min_max_normalize,
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
            state.get("min_max_normalize", False),
        )

    def state_dict(self):
        return {
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
            "min_max_normalize": self.min_max_normalize,
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
        self.blubox_locs_min = torch.tensor(state["bluebox_locs_min"], dtype=torch.float32)
        self.blubox_locs_max = torch.tensor(state["bluebox_locs_max"], dtype=torch.float32)
        self.min_max_normalize = state.get("min_max_normalize", False)