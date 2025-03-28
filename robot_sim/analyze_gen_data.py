import torch
import numpy as np
import os

def check_data(data_path):
    data_p_path = os.path.join(data_path, "data.p")
    images_path = os.path.join(data_path, "images.npy")

    print(f"=== Checking dataset in: {data_path} ===\n")

    # --- Check data.p ---
    print("[1] Checking data.p...")
    try:
        data = torch.load(data_p_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load data.p: {e}")
        return

    if not isinstance(data, list):
        print("❌ data.p is not a list.")
        return

    num_episodes = len(data)
    print(f"✅ Loaded data.p with {num_episodes} episodes.")

    example = data[0]
    required_keys = ["actions", "observations", "map_idx"]
    for key in required_keys:
        if key not in example:
            print(f"❌ Missing key in episode dict: {key}")
            return

    actions = example["actions"]
    observations = example["observations"]
    print(f"→ actions shape: {actions.shape}")
    print(f"→ observations shape: {observations.shape}")

    if observations.shape[0] != actions.shape[0] + 1:
        print("❌ Mismatch: observations should have one more timestep than actions.")
    else:
        print("✅ actions and observations length match (T+1 vs T).")

    # --- Check images.npy ---
    print("\n[2] Checking images.npy...")
    try:
        images = np.load(images_path)
    except Exception as e:
        print(f"❌ Failed to load images.npy: {e}")
        return

    print(f"✅ Loaded images.npy with shape: {images.shape}")
    T_plus_1 = observations.shape[0]
    expected_images = num_episodes * T_plus_1

    if images.shape[0] != expected_images:
        print(f"❌ images.npy has {images.shape[0]} images, but expected {expected_images} ({num_episodes} episodes × {T_plus_1} steps)")
    else:
        print("✅ images.npy shape is consistent with data.p")

    print("\n✅ Dataset format looks good!")

if __name__ == "__main__":
    data_path = "robot_sim/dataset"
    check_data(data_path)
