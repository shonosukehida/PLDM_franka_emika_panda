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
        self.cum_obs_counts = np.cumsum(self.episode_lengths) #各エピソードの画像の「開始インデックス」を計算するための前準備

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
        actions = episode["actions"][start_idx:end_idx - 1]  # (L-1, D)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        qpos = obs[:, :7]
        qvel = obs[:, 7:14]
        ee_xyz = obs[:, 14:17]
        bluebox_xyz = obs[:, 17:20]
        
        propio_pos = qpos
        propio_vel = qvel
        locations = ee_xyz
        bluebox_locs = bluebox_xyz

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
        # print('images:', images.shape)
        # if img_start + length > self.images_tensor.shape[0]:
        #     raise ValueError(
        #         f"Invalid image slice: {img_start=} + {length=} > total={self.images_tensor.shape[0]}"
        #     )

        if self.stack_states > 1: #x
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
            bluebox_locs=bluebox_locs,
            indices=idx,
            propio_pos=propio_pos,
            propio_vel=propio_vel,
        )






##PC可視化用, episode単位で__getitem__

class FrankaEpisodeDataset(Dataset):
    """
    1 episode -> 1 sample だけ返す Dataset.
    sliding window (flattened_indices) を使わず、
    episode_idx だけで取り出す.

    - pick_mode:
        "first"  : start_idx = 0
        "middle" : episodeの中央寄りのstart_idx
        "random" : episode内からランダムstart_idx（seed固定したければ外からnp.random.seed）
        "last"   : 可能な最大start_idx
    """
    def __init__(self, config, images_tensor=None, pick_mode: str = "first"):
        self.config = config
        self.sample_length = config.sample_length
        self.stack_states = config.stack_states
        self.use_images = config.images_path is not None
        self.pick_mode = pick_mode

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

        # 各episodeで「このsample_lengthが取れるか」を事前チェック
        self.usable_starts = []
        for ep_idx, L in enumerate(self.episode_lengths):
            # 元実装と同じ定義
            usable = L - self.sample_length - (self.stack_states - 1)
            if usable <= 0:
                # 取れないepisodeはスキップ（落とさないならraiseにしてもOK）
                continue
            self.usable_starts.append((ep_idx, usable))

        if len(self.usable_starts) == 0:
            raise ValueError("No usable episodes found: sample_length/stack_states may be too large.")

        print(f"[FrankaEpisodeDataset] usable episodes: {len(self.usable_starts)} / total {len(self.splits)}")

    def __len__(self):
        # episode単位（= usable episode数）
        return len(self.usable_starts)

    def _choose_start(self, usable: int) -> int:
        # usable は「start_idx の候補数」（0..usable-1）
        if self.pick_mode == "random":
            return int(np.random.randint(0, usable))
        if self.pick_mode == "middle":
            return int((usable - 1) // 2)
        if self.pick_mode == "last":
            return int(usable - 1)
        # default: first
        return 0

    def __getitem__(self, idx):
        ep_idx, usable = self.usable_starts[idx]
        start_idx = self._choose_start(usable)

        episode = self.splits[ep_idx]

        length = self.sample_length + self.stack_states - 1
        end_idx = start_idx + length

        obs = episode["observations"][start_idx:end_idx]      # (L, D)
        actions = episode["actions"][start_idx:end_idx - 1]   # (L-1, D)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)

        qpos = obs[:, :7]
        qvel = obs[:, 7:14]
        ee_xyz = obs[:, 14:17]
        bluebox_xyz = obs[:, 17:20]

        propio_pos = qpos
        propio_vel = qvel
        locations = ee_xyz
        bluebox_locs = bluebox_xyz

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

        # indices には元idxではなく episode index 情報も欲しいのでタプルにしておく
        return FrankaSample(
            states=states,
            actions=actions,
            locations=locations,
            bluebox_locs=bluebox_locs,
            indices=(ep_idx, start_idx),   # ★ここが “episode単位” である証拠になる
            propio_pos=propio_pos,
            propio_vel=propio_vel,
        )
