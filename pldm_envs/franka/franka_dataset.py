import torch
from torch.utils.data import Dataset
import numpy as np
from pldm_envs.franka.enums import FrankaSample

from transformers import AutoVideoProcessor
import os
import time


def _acquire_lock(lock_path: str, wait_sec: int = 600):
    """複数プロセスが同時に前処理を走らせないための簡易ロック"""
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - t0 > wait_sec:
                raise TimeoutError(f"lock wait timeout: {lock_path}")
            time.sleep(1)

def _release_lock(lock_path: str):
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


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
                # (N, H, W, C) uint8 の想定
                self.images_tensor = np.load(config.images_path, mmap_mode="r")
            else:
                self.images_tensor = images_tensor

            print("shape of images is:", self.images_tensor.shape)  # (N,64,64,3)

            if self.config.backbone_arch == "vjepa2":
                save_dir = os.path.dirname(config.path)
                preproc_path = os.path.join(save_dir, "vjepa2_preprocessed_images.npy")
                lock_path = preproc_path + ".lock"

                if not os.path.exists(preproc_path):
                    print("[VJEPA2] preproc cache not found. building:", preproc_path)

                    _acquire_lock(lock_path)
                    try:
                        # すでに別プロセスが作った可能性を再チェック
                        if not os.path.exists(preproc_path):
                            self._build_vjepa2_preproc_cache(
                                preproc_path,
                                chunk=getattr(config, "vjepa2_preproc_chunk", 128),
                            )
                    finally:
                        _release_lock(lock_path)

                # ★作ったキャッシュを mmap で読む（RAMに載せない）
                self.images_tensor = np.load(preproc_path, mmap_mode="r")
                print("[VJEPA2] loaded preproc cache:", self.images_tensor.shape, self.images_tensor.dtype)

            else:
                # VJEPA2以外ならそのまま。必要なら CHW にする
                # (N,H,W,C) -> (N,C,H,W)
                self.images_tensor = self.images_tensor.transpose(0, 3, 1, 2)

        else:
            print("states will contain proprioceptive info")

        self.episode_lengths = [len(d["observations"]) for d in self.splits]
        self.cum_obs_counts = np.cumsum(self.episode_lengths)

        self.flattened_indices = []
        for ep_idx, d in enumerate(self.splits):
            usable = self.episode_lengths[ep_idx] - self.sample_length - (self.stack_states - 1)
            for t in range(usable):
                self.flattened_indices.append((ep_idx, t))

    def _build_vjepa2_preproc_cache(self, preproc_path: str, chunk: int = 128):
        """
        images.npy (memmap) をチャンクで processor に通し、
        vjepa2_preprocessed_images.npy を memmap として生成する
        """
        processor = AutoVideoProcessor.from_pretrained(self.config.vjepa2_repo)

        N = self.images_tensor.shape[0]  # frames
        # まず1チャンクだけ通して出力shapeを確定
        s0, e0 = 0, min(N, max(1, chunk))
        x0 = np.array(self.images_tensor[s0:e0], copy=True)  # (chunk,H,W,C) を局所copy（RAM爆発防止）
        x0t = torch.from_numpy(x0).permute(0, 3, 1, 2) 
        print("x0t.shape:", x0t.shape)
        

        with torch.inference_mode():
            out0 = processor(x0t, return_tensors="pt")["pixel_values_videos"]
            print("out0.shape:", out0.shape)

        # out0 の shape を想定しないで整形（あなたの元コードに寄せる）
        # よくある: (1, T, C, H, W) or (1, T, H, W, C) etc
        out0 = out0.squeeze(0)

        # CHW に揃える（もし最後がCなら permute）
        if out0.dim() == 4:
            # (T, ?, ?, ?) のどこがCか推測
            # 典型: (T, C, H, W)
            if out0.shape[1] in (1, 3) and out0.shape[2] >= 32:
                out0_chw = out0
            # 典型: (T, H, W, C)
            elif out0.shape[-1] in (1, 3):
                out0_chw = out0.permute(0, 3, 1, 2)
            else:
                raise RuntimeError(f"[VJEPA2] unexpected processor output shape: {tuple(out0.shape)}")
        else:
            raise RuntimeError(f"[VJEPA2] unexpected processor output dim: {out0.dim()}, shape={tuple(out0.shape)}")

        out0_chw = out0_chw.to(torch.float16).cpu().numpy()  # (T, C, H, W), float16
        _, C, H, W = out0_chw.shape

        print(f"[VJEPA2] processor output sample: (C,H,W)=({C},{H},{W}), dtype=float16")

        # memmap 作成（全体: (N, C, H, W)）
        out_mm = np.lib.format.open_memmap(
            preproc_path, mode="w+", dtype=np.float16, shape=(N, C, H, W)
        )

        # 先に out0 を書く
        out_mm[s0:e0] = out0_chw

        # 残りをチャンクで回す
        with torch.inference_mode():
            for s in range(e0, N, chunk):
                e = min(N, s + chunk)
                x = np.array(self.images_tensor[s:e], copy=True)      # (chunk,H,W,C)
                xt = torch.from_numpy(x).permute(0, 3, 1, 2)                            # uint8
                out = processor(xt, return_tensors="pt")["pixel_values_videos"]
                out = out.squeeze(0)

                if out.dim() != 4:
                    raise RuntimeError(f"[VJEPA2] unexpected output: {tuple(out.shape)}")

                if out.shape[1] in (1, 3) and out.shape[2] >= 32:
                    out_chw = out
                elif out.shape[-1] in (1, 3):
                    out_chw = out.permute(0, 3, 1, 2)
                else:
                    raise RuntimeError(f"[VJEPA2] unexpected output shape: {tuple(out.shape)}")

                out_mm[s:e] = out_chw.to(torch.float16).cpu().numpy()

        out_mm.flush()
        print("[VJEPA2] saved preproc cache:", preproc_path)

    def __len__(self):
        return len(self.flattened_indices)

    def __getitem__(self, idx):
        ep_idx, start_idx = self.flattened_indices[idx]
        episode = self.splits[ep_idx]

        length = self.sample_length + self.stack_states - 1
        end_idx = start_idx + length

        obs = episode["observations"][start_idx:end_idx]
        actions = episode["actions"][start_idx:end_idx - 1]

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
            img_start = start_idx if ep_idx == 0 else self.cum_obs_counts[ep_idx - 1] + start_idx

            # ★mmap slice は非writeableになりがちなので、局所 copy してから torch 化
            arr = np.array(self.images_tensor[img_start:img_start + length], copy=True)  # (L,C,H,W)
            images = torch.from_numpy(arr)  # float16

            states = images  # 必要なら .float() だけど、まずは型を維持（RAM節約）
        else:
            states = obs

        if self.stack_states > 1:
            states = torch.stack([states[i:i + self.stack_states] for i in range(self.sample_length)], dim=0)
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





##PCA可視化用, episode単位で__getitem__

class FrankaEpisodeDataset(Dataset):
    """
    PCA可視化用: 1 episode -> 1 sample だけ返す Dataset.

    - pick_mode:
        "first"  : start_idx = 0
        "middle" : episodeの中央寄り
        "random" : episode内ランダム
        "last"   : 最大start_idx
    """
    def __init__(self, config, images_tensor=None, pick_mode: str = "first"):
        self.config = config
        self.sample_length = config.sample_length
        self.stack_states = config.stack_states
        self.use_images = config.images_path is not None
        self.pick_mode = pick_mode

        print("loading saved dataset from", config.path)
        self.splits = torch.load(config.path, map_location="cpu", weights_only=False)

        # --- images ---
        if self.use_images:
            if images_tensor is None:
                # (N,H,W,C) uint8 を想定
                self.images_tensor = np.load(config.images_path, mmap_mode="r")
            else:
                self.images_tensor = images_tensor

            print("shape of images is:", self.images_tensor.shape)  # (N,64,64,3)

            if self.config.backbone_arch == "vjepa2":
                save_dir = os.path.dirname(config.path)
                preproc_path = os.path.join(save_dir, "vjepa2_preprocessed_images.npy")
                lock_path = preproc_path + ".lock"

                if not os.path.exists(preproc_path):
                    print("[VJEPA2] preproc cache not found. building:", preproc_path)
                    _acquire_lock(lock_path)
                    try:
                        # すでに別プロセスが作った可能性を再チェック
                        if not os.path.exists(preproc_path):
                            self._build_vjepa2_preproc_cache(
                                preproc_path,
                                chunk=getattr(config, "vjepa2_preproc_chunk", 128),
                            )
                    finally:
                        _release_lock(lock_path)

                # ★キャッシュを mmap で読む（RAMに載せない）
                self.images_tensor = np.load(preproc_path, mmap_mode="r")
                print("[VJEPA2] loaded preproc cache:", self.images_tensor.shape, self.images_tensor.dtype)

            else:
                # VJEPA2以外: (N,H,W,C)->(N,C,H,W)
                self.images_tensor = self.images_tensor.transpose(0, 3, 1, 2)

        else:
            print("states will contain proprioceptive info")

        # --- episode index mapping ---
        self.episode_lengths = [len(d["observations"]) for d in self.splits]
        self.cum_obs_counts = np.cumsum(self.episode_lengths)

        # 各episodeで「このsample_lengthが取れるか」を事前チェック
        self.usable_starts = []
        for ep_idx, L in enumerate(self.episode_lengths):
            usable = L - self.sample_length - (self.stack_states - 1)
            if usable <= 0:
                continue
            self.usable_starts.append((ep_idx, usable))

        if len(self.usable_starts) == 0:
            raise ValueError("No usable episodes found: sample_length/stack_states may be too large.")

        print(f"[FrankaEpisodeDataset] usable episodes: {len(self.usable_starts)} / total {len(self.splits)}")

    def _build_vjepa2_preproc_cache(self, preproc_path: str, chunk: int = 128):
        """
        images.npy (memmap) をチャンクで processor に通し、
        vjepa2_preprocessed_images.npy を memmap として生成する
        """
        processor = AutoVideoProcessor.from_pretrained(self.config.vjepa2_repo)

        N = self.images_tensor.shape[0]  # frames

        # まず1チャンクだけ通して出力shapeを確定
        s0, e0 = 0, min(N, max(1, chunk))
        x0 = np.array(self.images_tensor[s0:e0], copy=True)          # (chunk,H,W,C)
        x0t = torch.from_numpy(x0).permute(0, 3, 1, 2)               # (chunk,C,H,W)
        print("x0t.shape:", x0t.shape)

        with torch.inference_mode():
            out0 = processor(x0t, return_tensors="pt")["pixel_values_videos"]
            print("out0.shape:", out0.shape)

        out0 = out0.squeeze(0)  # (T,?,?,?)

        if out0.dim() != 4:
            raise RuntimeError(f"[VJEPA2] unexpected processor output dim: {out0.dim()}, shape={tuple(out0.shape)}")

        # (T,C,H,W) or (T,H,W,C)
        if out0.shape[1] in (1, 3) and out0.shape[2] >= 32:
            out0_chw = out0
        elif out0.shape[-1] in (1, 3):
            out0_chw = out0.permute(0, 3, 1, 2)
        else:
            raise RuntimeError(f"[VJEPA2] unexpected processor output shape: {tuple(out0.shape)}")

        out0_chw = out0_chw.to(torch.float16).cpu().numpy()
        _, C, H, W = out0_chw.shape
        print(f"[VJEPA2] processor output sample: (C,H,W)=({C},{H},{W}), dtype=float16")

        # memmap 作成（全体: (N,C,H,W)）
        out_mm = np.lib.format.open_memmap(
            preproc_path, mode="w+", dtype=np.float16, shape=(N, C, H, W)
        )

        # 先に out0 を書く
        out_mm[s0:e0] = out0_chw

        # 残りをチャンクで回す
        with torch.inference_mode():
            for s in range(e0, N, chunk):
                e = min(N, s + chunk)
                x = np.array(self.images_tensor[s:e], copy=True)      # (chunk,H,W,C)
                xt = torch.from_numpy(x).permute(0, 3, 1, 2)          # (chunk,C,H,W)

                out = processor(xt, return_tensors="pt")["pixel_values_videos"]
                out = out.squeeze(0)

                if out.dim() != 4:
                    raise RuntimeError(f"[VJEPA2] unexpected output: {tuple(out.shape)}")

                if out.shape[1] in (1, 3) and out.shape[2] >= 32:
                    out_chw = out
                elif out.shape[-1] in (1, 3):
                    out_chw = out.permute(0, 3, 1, 2)
                else:
                    raise RuntimeError(f"[VJEPA2] unexpected output shape: {tuple(out.shape)}")

                out_mm[s:e] = out_chw.to(torch.float16).cpu().numpy()

        out_mm.flush()
        print("[VJEPA2] saved preproc cache:", preproc_path)

    def __len__(self):
        return len(self.usable_starts)

    def _choose_start(self, usable: int) -> int:
        if self.pick_mode == "random":
            return int(np.random.randint(0, usable))
        if self.pick_mode == "middle":
            return int((usable - 1) // 2)
        if self.pick_mode == "last":
            return int(usable - 1)
        return 0  # first

    def __getitem__(self, idx):
        ep_idx, usable = self.usable_starts[idx]
        start_idx = self._choose_start(usable)

        episode = self.splits[ep_idx]

        length = self.sample_length + self.stack_states - 1
        end_idx = start_idx + length

        obs = episode["observations"][start_idx:end_idx]
        actions = episode["actions"][start_idx:end_idx - 1]

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
            img_start = start_idx if ep_idx == 0 else self.cum_obs_counts[ep_idx - 1] + start_idx

            # ★mmap slice は非writeableになりがちなので局所 copy
            arr = np.array(self.images_tensor[img_start:img_start + length], copy=True)  # (L,C,H,W) or (L,C,H,W)
            images = torch.from_numpy(arr)  # float16

            states = images
        else:
            states = obs

        if self.stack_states > 1:
            states = torch.stack(
                [states[i:i + self.stack_states] for i in range(self.sample_length)],
                dim=0
            )
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
            indices=(ep_idx, start_idx),  # episode単位である証拠
            propio_pos=propio_pos,
            propio_vel=propio_vel,
        )