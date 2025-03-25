import mujoco
import numpy as np
import torch
import os
import time
from datetime import datetime
from PIL import Image

# ===========================
# Config
# ===========================
SAVE_PATH = "robot_sim/dataset"
EPISODES = 10000
STEPS_PER_EPISODE = 100
IMAGE_SIZE = (64, 64)

os.makedirs(SAVE_PATH, exist_ok=True)

# ===========================
# Load Model
# ===========================
os.environ["MUJOCO_GL"] = "egl"
model_path = "mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view")

ctrlrange = model.actuator_ctrlrange
obs_dim = model.nq + model.nv
act_dim = model.nu

data_list = []
all_images = []

for ep in range(EPISODES):
    print(f"Episode {ep+1}/{EPISODES}")
    mujoco.mj_resetData(model, data)
    episode_obs = []
    episode_actions = []
    
    # 初期観測
    obs = np.concatenate([data.qpos[:], data.qvel[:]])
    episode_obs.append(obs.copy())
    
    # 🔽 初期画像の取得・保存を追加
    renderer.update_scene(data, camera=camera_id)
    img = renderer.render()
    all_images.append(img)

    for t in range(STEPS_PER_EPISODE):
        # ランダム行動
        action = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])
        data.ctrl[:] = action
        mujoco.mj_step(model, data)

        # 観測
        obs = np.concatenate([data.qpos[:], data.qvel[:]])
        episode_obs.append(obs.copy())
        episode_actions.append(action.copy())

        # 画像取得
        renderer.update_scene(data, camera=camera_id)
        img = renderer.render()
        all_images.append(img)

    # エピソード保存
    episode_dict = {
        "observations": np.array(episode_obs),       # shape (T+1, obs_dim)
        "actions": np.array(episode_actions),        # shape (T, act_dim)
        "map_idx": ep                                # 適当に ep 番号を ID に
    }
    data_list.append(episode_dict)

# ===========================
# Save data
# ===========================

data_p_path = os.path.join(SAVE_PATH, "data.p")
images_npy_path = os.path.join(SAVE_PATH, "images.npy")

torch.save(data_list, data_p_path)
np.save(images_npy_path, np.array(all_images, dtype=np.uint8))

print("Saved:")
print("  data.p       ->", data_p_path)
print("  images.npy   ->", images_npy_path)
print("Episode数:", len(data_list))
print("画像数:", len(all_images))
