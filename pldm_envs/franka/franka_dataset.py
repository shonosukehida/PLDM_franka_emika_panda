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
        T = self.config.sample_length  # 15

        obs_np = d["observations"][:T]  # ← T個だけ
        act_np = d["actions"][:T-1]     # ← T-1個だけ
        goal_obs = torch.tensor(d["goal_obs"], dtype=torch.float32)

        # 整合性チェック
        assert obs_np.shape[0] == T, f"Expected T obs, got {obs_np.shape[0]}"
        assert act_np.shape[0] == T - 1, f"Expected T-1 actions, got {act_np.shape[0]}"

        observations = torch.tensor(obs_np, dtype=torch.float32)
        actions = torch.tensor(act_np, dtype=torch.float32)

        # proprio 情報を分離
        qpos = observations[:, :7]
        qvel = observations[:, 7:]
        propio_pos = qpos
        propio_vel = qvel
        locations = qpos[:, 9:11]

        # 画像も T枚でOK
        if self.config.images_path is not None:
            image_start_index = idx * self.T_plus_1  # このままでOK（保存形式がT+1固定なら）
            images = self.images_tensor[image_start_index:image_start_index + T]
            images = torch.from_numpy(images).permute(0, 3, 1, 2).float()  # (T, C, H, W)
            states = images
        else:
            states = observations  # (T, D)

        return FrankaSample(
            states=states,
            actions=actions,
            locations=locations,
            indices=idx,
            propio_pos=propio_pos,
            propio_vel=propio_vel,
        )


