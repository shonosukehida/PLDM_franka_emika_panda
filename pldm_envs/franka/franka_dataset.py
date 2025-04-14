import torch
from torch.utils.data import Dataset
import numpy as np
from pldm_envs.franka.enums import FrankaSample


class FrankaDataset(Dataset):
    def __init__(self, config, images_tensor=None):
        self.config = config

        print("loading saved dataset from", config.path)
        self.data = torch.load(config.path, map_location="cpu", weights_only=False)

        if config.images_path is not None:
            print("states will contain images")
            if images_tensor is None:
                self.images_tensor = np.load(config.images_path, mmap_mode="r")
            else:
                self.images_tensor = images_tensor
            print("shape of images is:", self.images_tensor.shape)
        else:
            print("states will contain proprioceptive info")

        self.T = self.data[0]["actions"].shape[0]  # each episode has T steps
        self.T_plus_1 = self.T + 1
        self.image_shape = self.images_tensor.shape[1:] if config.images_path else (31,)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]

        observations = torch.tensor(d["observations"], dtype=torch.float32)  # (T+1, 31)
        actions = torch.tensor(d["actions"], dtype=torch.float32)  # (T, 7)
        goal_obs = torch.tensor(d["goal_obs"], dtype=torch.float32)

        if self.config.images_path is not None:
            image_start_index = idx * self.T_plus_1
            image_end_index = image_start_index + self.T_plus_1
            images = torch.from_numpy(self.images_tensor[image_start_index:image_end_index])
            images = images.permute(0, 3, 1, 2).float()  # (T+1, C, H, W)
            states = images  # [T+1, C, H, W]
        else:
            states = observations  # fallback to proprio

        locations = observations[:, :2]  # (T+1, 2)

        return FrankaSample(
            states=states,  # (T+1, C, H, W) or (T+1, D)
            actions=actions,  # (T, 7)
            locations=locations,  # (T+1, 2)
            indices=idx,
            propio_vel=torch.empty(0),
            propio_pos=torch.empty(0),
        )
