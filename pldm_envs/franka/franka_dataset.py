# pldm/data/franka_dataset.py
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class FrankaDataset(Dataset):
    def __init__(self, config):
        data_dir = config["data_dir"]
        self.data = torch.load(os.path.join(data_dir, "data.p"), map_location="cpu")
        self.images = np.load(os.path.join(data_dir, "images.npy"))
        self.goal_images = np.load(os.path.join(data_dir, "goal_images.npy")) if config.get("use_goal_images", True) else None

        self.T = self.data[0]["actions"].shape[0]
        self.image_shape = self.images.shape[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ep = self.data[idx]
        obs = ep["observations"][:-1]         # shape: (T, 31)
        next_obs = ep["observations"][1:]     # shape: (T, 31)
        actions = ep["actions"]               # shape: (T, 7)

        img_start = idx * (self.T + 1)
        imgs = self.images[img_start:img_start + self.T + 1]  # (T+1, H, W, C)
        imgs = imgs.transpose(0, 3, 1, 2) / 255.0              # (T+1, C, H, W)

        sample = {
            "obs": torch.tensor(obs, dtype=torch.float32),
            "next_obs": torch.tensor(next_obs, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "images": torch.tensor(imgs[:-1], dtype=torch.float32),
            "next_images": torch.tensor(imgs[1:], dtype=torch.float32),
        }

        if self.goal_images is not None:
            goal_img = self.goal_images[idx].transpose(2, 0, 1) / 255.0  # (C, H, W)
            sample["goal_image"] = torch.tensor(goal_img, dtype=torch.float32)

        return sample
