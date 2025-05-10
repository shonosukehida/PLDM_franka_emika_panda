import torch
from torch.utils.data import Dataset
import numpy as np
from pldm_envs.franka.enums import FrankaSample


class FrankaDataset(Dataset):
    def __init__(self, config, images_tensor=None):
        self.config = config
        self.sample_length = config.sample_length
        self.stack_states = config.stack_states
        self.use_images = config.images_path is not None

        print("loading saved dataset from", config.path)
        self.splits = torch.load(config.path, map_location="cpu", weights_only=False)

        if self.use_images:
            if images_tensor is None:
                self.images_tensor = np.load(config.images_path, mmap_mode="r")
            else:
                self.images_tensor = images_tensor
            print("shape of images is:", self.images_tensor.shape)
        else:
            print("states will contain proprioceptive info")

        self.episode_lengths = [len(d["observations"]) for d in self.splits]
        self.cum_obs_counts = np.cumsum(self.episode_lengths)

        self.flattened_indices = []
        for ep_idx, d in enumerate(self.splits):
            usable = self.episode_lengths[ep_idx] - self.sample_length - (self.stack_states - 1)
            for t in range(usable):
                self.flattened_indices.append((ep_idx, t))

    def __len__(self):
        return len(self.flattened_indices)

    def __getitem__(self, idx):
        ep_idx, start_idx = self.flattened_indices[idx]
        episode = self.splits[ep_idx]

        length = self.sample_length + self.stack_states - 1
        end_idx = start_idx + length

        obs = episode["observations"][start_idx:end_idx]  # (L, D)
        actions = episode["actions"][start_idx:end_idx - 1]  # (L-1, A)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        qpos = obs[:, :7]
        qvel = obs[:, 7:]
        propio_pos = qpos
        propio_vel = qvel
        locations = qpos[:, 9:11]

        if self.use_images:
            if ep_idx == 0:
                img_start = start_idx
            else:
                img_start = self.cum_obs_counts[ep_idx - 1] + start_idx
            images = torch.from_numpy(self.images_tensor[img_start:img_start + length])
            images = images.permute(0, 3, 1, 2).float()
            states = images
        else:
            states = obs

        if self.stack_states > 1:
            states = torch.stack([
                states[i:i + self.stack_states] for i in range(self.sample_length)
            ], dim=0)
            states = states.flatten(1, 2)
            actions = actions[(self.stack_states - 1):]
            locations = locations[(self.stack_states - 1):]
            propio_pos = propio_pos[(self.stack_states - 1):]
            propio_vel = propio_vel[(self.stack_states - 1):]

        return FrankaSample(
            states=states,
            actions=actions,
            locations=locations,
            indices=idx,
            propio_pos=propio_pos,
            propio_vel=propio_vel,
        )
