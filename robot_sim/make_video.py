import os
import torch
import numpy as np
import imageio
from tqdm import tqdm

# ==== CONFIG ====
DATA_PATH = "robot_sim/dataset/pairs_5_ep_1"  
SAVE_DIR = "robot_sim/analyze/video"
FPS = 5

# ==== LOAD ====
print("📦 Loading dataset...")
data = torch.load(os.path.join(DATA_PATH, "data.p"), map_location="cpu", weights_only=False)
images = np.load(os.path.join(DATA_PATH, "images.npy"))

# ==== SETUP ====
os.makedirs(SAVE_DIR, exist_ok=True)
frames_per_episode = len(data[0]["observations"])  # T+1
print(f"🎞️ Frames per episode: {frames_per_episode}")
print(f"📁 Saving videos to: {SAVE_DIR}")

# ==== GENERATE VIDEOS ====
start_idx = 0
for i, episode in enumerate(tqdm(data, desc="🎬 Saving episodes as videos")):
    end_idx = start_idx + frames_per_episode
    episode_frames = images[start_idx:end_idx]
    save_path = os.path.join(SAVE_DIR, f"episode_{i:03d}.mp4")
    imageio.mimsave(save_path, episode_frames, fps=FPS)
    start_idx = end_idx

print("✅ All videos saved.")
