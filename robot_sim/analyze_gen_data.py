import torch
import numpy as np
import os
from PIL import Image
import imageio
from tqdm import tqdm

def check_data(data_path):
    data_p_path = os.path.join(data_path, "data.p")
    images_path = os.path.join(data_path, "images.npy")
    goal_img_path = os.path.join(data_path, "goal_images.npy")

    print(f"=== Checking dataset in: {data_path} ===\n")

    # --- Check data.p ---
    print("[1] Checking data.p...")
    try:
        data = torch.load(data_p_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load data.p: {e}")
        return None, None

    if not isinstance(data, list):
        print("❌ data.p is not a list.")
        return None, None

    num_episodes = len(data)
    print(f"✅ Loaded data.p with {num_episodes} episodes.")

    example = data[0]
    print("\n[1.1] Keys and shapes in one episode:")
    for k, v in example.items():
        if isinstance(v, np.ndarray):
            print(f" - {k}: shape {v.shape}")
        else:
            print(f" - {k}: type {type(v)}")

    actions = example["actions"]
    observations = example["observations"]

    if observations.shape[0] != actions.shape[0] + 1:
        print("❌ Mismatch: observations should have one more timestep than actions.")
    else:
        print("✅ actions and observations length match (T+1 vs T).")

    if "goal_obs" in example:
        goal_obs = example["goal_obs"]
        if goal_obs.shape != observations[0].shape:
            print(f"❌ goal_obs shape mismatch: got {goal_obs.shape}, expected {observations[0].shape}")
        else:
            print("✅ goal_obs is present and shape is valid.")

    # --- Check images.npy ---
    print("\n[2] Checking images.npy...")
    try:
        images = np.load(images_path)
    except Exception as e:
        print(f"❌ Failed to load images.npy: {e}")
        return data, None

    T_plus_1 = observations.shape[0]
    expected_images = num_episodes * T_plus_1

    if images.shape[0] != expected_images:
        print(f"❌ images.npy has {images.shape[0]} images, expected {expected_images}")
    else:
        print("✅ images.npy shape is consistent with data.p")

    # --- Save first 5 images ---
    print("\n[3] Saving first 5 images from images.npy...")
    save_dir = "robot_sim/analyze/operation"
    os.makedirs(save_dir, exist_ok=True)
    for i in range(min(5, images.shape[0])):
        Image.fromarray(images[i]).save(os.path.join(save_dir, f"img_{i}.png"))
    print(f"✅ Saved first 5 images to {save_dir}")

    # --- Save goal images ---
    if os.path.exists(goal_img_path):
        print("\n[4] Saving goal_images.npy as images...")
        try:
            goal_imgs = np.load(goal_img_path)
            save_goal_dir = "robot_sim/analyze/goal"
            os.makedirs(save_goal_dir, exist_ok=True)

            if goal_imgs.ndim == 4:
                for i in range(min(5, len(goal_imgs))):
                    Image.fromarray(goal_imgs[i]).save(os.path.join(save_goal_dir, f"goal_img_{i}.png"))
                print(f"✅ Saved goal images to {save_goal_dir}")
            else:
                print(f"❌ Unexpected goal_images shape: {goal_imgs.shape}")
        except Exception as e:
            print(f"❌ Failed to save goal image: {e}")
    else:
        print("⚠️ goal_images.npy not found.")

    print("\n✅ Dataset format check complete.")
    return data, images


def save_videos(data, images, save_dir="robot_sim/analyze/video", fps=5):
    print("\n🎞️ Generating videos...")
    os.makedirs(save_dir, exist_ok=True)

    frames_per_episode = len(data[0]["observations"])  # T+1
    print(f"🎞️ Frames per episode: {frames_per_episode}")
    print(f"📁 Saving videos to: {save_dir}")

    start_idx = 0
    for i, episode in enumerate(tqdm(data, desc="🎬 Saving episodes as videos")):
        end_idx = start_idx + frames_per_episode
        episode_frames = images[start_idx:end_idx]
        save_path = os.path.join(save_dir, f"episode_{i:03d}.mp4")
        imageio.mimsave(save_path, episode_frames, fps=fps)
        start_idx = end_idx

    print("✅ All videos saved.")


if __name__ == "__main__":
    data_path = "pldm_envs/franka/presaved_datasets/5pr_1ep_100t"
    data, images = check_data(data_path)

    if data is not None and images is not None:
        save_videos(data, images)
