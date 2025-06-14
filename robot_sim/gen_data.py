import mujoco
import numpy as np
import torch
import os
from tqdm import tqdm
import yaml
import pandas as pd

import imageio

# ==== CONFIG ====
with open("robot_sim/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PAIRS = config["pairs"]
MIN_DIST = config["min_dist"]
EPISODES_PER_PAIR = config["episodes_per_pair"]
STEPS_PER_EPISODE = config["steps_per_episode"]
IMAGE_SIZE = tuple(config["image_size"])
MODEL_PATH = config["model_path"]
CAMERA_NAME = config.get("camera_name", "")  # default: ""
IS_VAL = config["is_val"]
if not IS_VAL:
    SAVE_PATH = f"pldm_envs/franka/presaved_datasets/pairs_{PAIRS}_ep_{EPISODES_PER_PAIR}_timestep_{STEPS_PER_EPISODE}"
else:
    SAVE_PATH = f"pldm_envs/franka/presaved_datasets/val_pairs_{PAIRS}_ep_{EPISODES_PER_PAIR}_timestep_{STEPS_PER_EPISODE}"

#ゴール位置の範囲
X_RANGE = tuple(config["x_range"])
Y_RANGE = tuple(config["y_range"])
Z_RANGE = tuple(config["z_range"])

# ==== SETUP ====
os.makedirs(SAVE_PATH, exist_ok=True)
os.environ["MUJOCO_GL"] = "egl"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)

obs_dim = model.nq + model.nv #.nq:位置成分の次元数, .nv: 速度成分の次元数
act_dim = model.nu
ctrlrange = model.actuator_ctrlrange

# ==== DATA ====
data_list = []
all_images = []
goal_obs_list = []
pair_list = []

# ==== UTIL ====
def reset_and_place(pos, start_marker_pos=None, goal_marker_pos=None):
    mujoco.mj_resetData(model, data)
    data.ctrl[:] = 0
    
    start_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'start_marker')
    goal_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_joint_blue_box")
    start_idx = model.jnt_qposadr[joint_id]
    
    
    #青box の位置設定
    data.qpos[start_idx:start_idx+3] = pos
    data.qpos[start_idx+3:start_idx+7] = np.array([1, 0, 0, 0])
    data.qvel[start_idx:start_idx+6] = 0

    # スタートマーカーを動かす
    if start_marker_pos is not None:
        model.geom_pos[start_geom_id][:3] = start_marker_pos

    # ゴールマーカーを動かす
    if goal_marker_pos is not None:
        model.geom_pos[goal_geom_id][:3] = goal_marker_pos

    mujoco.mj_forward(model, data)

# def get_ee_position():
#     body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
#     return data.xpos[body_id].copy()  # shape: (3,)s

def get_ee_position(model, data):
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand") 
    return data.xpos[hand_id].copy()



def sample_random_position(x_range=X_RANGE, y_range=Y_RANGE, z_range =Z_RANGE):
    return np.array([
        np.random.uniform(*x_range),
        np.random.uniform(*y_range),
        np.random.uniform(*z_range),
    ])
    
# 登録されている全ての body と site を表示
print("\n💡 Bodies:")
for i in range(model.nbody):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

print("\n💡 Sites:")
for i in range(model.nsite):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))


# ==== Generate Start-Goal Pairs ====
pairs = []
while len(pairs) < PAIRS:
    start = sample_random_position()
    goal = sample_random_position()

    # 距離チェック
    dist = np.linalg.norm(start - goal)
    if dist >= MIN_DIST:
        pairs.append((start, goal))
        pair_list.append((start.copy(), goal.copy()))


# ==== Precompute Goal Observations ====
print("📸 Generating goal observations for each pair...")
for start_pos, goal_pos in pairs:
    reset_and_place(goal_pos, start_marker_pos=start_pos, goal_marker_pos=goal_pos)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_id)
    goal_img = renderer.render()
    ee_pos = get_ee_position(model, data)
    goal_obs = np.concatenate([data.qpos[:7], data.qvel[:7], ee_pos])
    goal_obs_list.append((goal_obs.copy(), goal_img.copy()))

# ==== MAIN LOOP ====
print("🎬 Generating episodes...")
for pair_idx, (start_pos, goal_pos) in enumerate(tqdm(pairs)):
    # goal_obs, goal_img = goal_obs_list[pair_idx]

    for _ in range(EPISODES_PER_PAIR):
        reset_and_place(start_pos, start_marker_pos=start_pos, goal_marker_pos=goal_pos)
        mujoco.mj_forward(model, data)

        episode_obs = []
        episode_actions = []

        franka_qpos = data.qpos[:7]
        franka_qvel = data.qvel[:7]
        ee_pos = get_ee_position(model, data)
        obs = np.concatenate([franka_qpos, franka_qvel, ee_pos])
        episode_obs.append(obs.copy())
        renderer.update_scene(data, camera=camera_id)
        all_images.append(renderer.render())

        for _ in range(STEPS_PER_EPISODE):
            action = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])
            data.ctrl[:] = action
            for _ in range(10): #1 STEPS_PER_EPISODE あたりに進めるステップ数
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                # print("xpos[9]", data.xpos[9])
            
            franka_qpos = data.qpos[:7]  # 関節角度: joint1〜joint7
            franka_qvel = data.qvel[:7]  # 関節速度: joint1〜joint7
            ee_pos = get_ee_position(model, data)
            # ee_pos = data.xpos[9]
            
            obs = np.concatenate([franka_qpos, franka_qvel, ee_pos])
            episode_obs.append(obs.copy())
            episode_actions.append(action.copy())
            renderer.update_scene(data, camera=camera_id)
            all_images.append(renderer.render())

        data_list.append({
            "observations": np.array(episode_obs),
            "actions": np.array(episode_actions),
            "goal_obs": goal_obs.copy(),
            "map_idx": pair_idx,
        })

# ==== SAVE ====
torch.save(data_list, os.path.join(SAVE_PATH, "data.p"))
np.save(os.path.join(SAVE_PATH, "images.npy"), np.array(all_images, dtype=np.uint8))
goal_imgs = np.stack([g[1] for g in goal_obs_list])
np.save(os.path.join(SAVE_PATH, "goal_images.npy"), goal_imgs)
torch.save({"pair_list": pair_list}, os.path.join(SAVE_PATH, "pair_info.p"))

print("✅ Done!")
print(f"  Total episodes: {len(data_list)}")
print(f"  Total images:   {len(all_images)}")
print(f"  Pairs saved to: {os.path.join(SAVE_PATH, 'pair_info.p')}")



# ====CONFIRM DATA====
FILE = 'data.p'
file_path = os.path.join(SAVE_PATH, FILE)
data = torch.load(file_path, weights_only=False)
obs = data[0]["observations"]  # shape: (T+1, 31)

joint_angles = obs[:, :7]
joint_vel = obs[:, 7:14]
xyz_pos = obs[:, 14:]

output_dir = "./robot_sim/data_value"
os.makedirs(output_dir, exist_ok=True)

# 保存処理
df = pd.DataFrame(joint_angles, columns=[f"joint_{i}" for i in range(1, 8)])
df.to_csv(os.path.join(output_dir, "franka_joint_angles.csv"), index=False)

df = pd.DataFrame(joint_vel, columns=[f"joint_{i}" for i in range(1, 8)])
df.to_csv(os.path.join(output_dir, "franka_joint_vel.csv"), index=False)

df = pd.DataFrame(xyz_pos, columns=[f"pos_{i}" for i in range(1, 4)])
df.to_csv(os.path.join(output_dir, "franka_xyz_pos.csv"), index=False)

print("successfully saved!!")


# ==== MAKE VIDEO ====
# ==== CONFIG ==== 
DATA_PATH = SAVE_PATH
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