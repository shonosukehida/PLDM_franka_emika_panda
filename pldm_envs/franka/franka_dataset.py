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
        
        # T+1 長の観測と T 長の行動
        observations = torch.tensor(d["observations"], dtype=torch.float32)  # (T+1, D)
        actions = torch.tensor(d["actions"], dtype=torch.float32)  # (T, action_dim)
        goal_obs = torch.tensor(d["goal_obs"], dtype=torch.float32)

        # スライス範囲（T+1 → T にする）
        start = 0
        end = start + self.config.sample_length + 1  # sample_length = T
        T = self.config.sample_length

        # proprio 情報を位置/速度で分離
        qpos = observations[:, :7]   # (T+1, 7)
        qvel = observations[:, 7:]   # (T+1, 7)

        propio_pos = qpos[start:end-1]  # (T, 7)
        propio_vel = qvel[start:end-1]  # (T, 7)

        # 青boxの位置情報（必要なら）
        locations = qpos[start:end-1, 9:11]  # (T, 2)

        # 画像のスライス
        if self.config.images_path is not None:
            image_start_index = idx * self.T_plus_1
            image_end_index = image_start_index + self.T_plus_1
            images = torch.from_numpy(self.images_tensor[image_start_index:image_end_index])
            images = images.permute(0, 3, 1, 2).float()  # (T+1, C, H, W)
            states = images[start:end-1]  # (T, C, H, W)
        else:
            states = observations[start:end-1]  # (T, D)

        return FrankaSample(
            states=states,             # (T, ...)
            actions=actions[start:end-1],  # (T, 7)
            locations=locations,       # (T, 2)
            indices=idx,
            propio_pos=propio_pos,     # (T, 7)
            propio_vel=propio_vel,     # (T, 7)
        )
