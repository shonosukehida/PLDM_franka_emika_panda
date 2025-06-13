import mujoco
import numpy as np
import os
import torch
import yaml
from tqdm import tqdm
from ikpy.chain import Chain
import imageio
import pandas as pd


# ==== CONFIG ====
with open("robot_sim/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PAIRS = config["pairs"]
MIN_DIST = config["min_dist"]
EPISODES_PER_PAIR = config["episodes_per_pair"]
STEPS_PER_EPISODE = config["steps_per_episode"]
IMAGE_SIZE = tuple(config["image_size"])
MODEL_PATH = config["model_path"]
CAMERA_NAME = config.get("camera_name", "")
IS_VAL = config["is_val"]
GOAL_X_RANGE = tuple(config["goal_x_range"])
GOAL_Y_RANGE = tuple(config["goal_y_range"])
GOAL_Z_RANGE = tuple(config["goal_z_range"])

X_RANGE = tuple(config["x_range"])
Y_RANGE = tuple(config["y_range"])
Z_RANGE = tuple(config["z_range"])

SAVE_PATH = (
    f"pldm_envs/franka/presaved_datasets/val_xyz_ep_{EPISODES_PER_PAIR}_timestep_{STEPS_PER_EPISODE}"
    if IS_VAL else
    f"pldm_envs/franka/presaved_datasets/xyz_ep_{EPISODES_PER_PAIR}_timestep_{STEPS_PER_EPISODE}"
)
os.makedirs(SAVE_PATH, exist_ok=True)
os.environ["MUJOCO_GL"] = "egl"

# ==== MUJOCO SETUP ====
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1])
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)

for i in range(model.ncam):
    print(f"{i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)}") 
# ==== IKPY SETUP ====
franka_chain = Chain.from_urdf_file(
    "panda.urdf",
    base_elements=["panda_link0"],
    active_links_mask=[False, True, True, True, True, True, True, True, False]
)
INITIAL_POSITION = [
    0.0,  # base
    0.0,  # joint1
    0.0,  # joint2
    0.0,  # joint3
    -0.1,  # joint4
    0.0,  # joint5
    0.0,  # joint6
    0.0,  # joint7
    0.0   # fixed link
]

# 登録されている全ての body と site を表示
print("\n💡 Bodies:")
for i in range(model.nbody):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i))

print("\n💡 Sites:")
for i in range(model.nsite):
    print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i))


def move_franka_to_xyz(xyz):
    ik_solution = franka_chain.inverse_kinematics(
        target_position=xyz,
        initial_position=INITIAL_POSITION
    )
    joint_angles = ik_solution[1:8]
    data.qpos[:7] = joint_angles
    mujoco.mj_forward(model, data)
    return joint_angles

def get_ee_position():
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return data.xpos[hand_id].copy()

def sample_goal_xyz():
    return np.array([
        np.random.uniform(*GOAL_X_RANGE),
        np.random.uniform(*GOAL_Y_RANGE),
        np.random.uniform(*GOAL_Z_RANGE),
    ])

def sample_xyz():
    return np.array([
        np.random.uniform(*X_RANGE),
        np.random.uniform(*Y_RANGE),
        np.random.uniform(*Z_RANGE),
    ])


def reset_and_place_all(box_pos, start_marker_pos=None, goal_marker_pos=None):
    mujoco.mj_resetData(model, data)

    # Blue box (自由動作体)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "free_joint_blue_box")
    start_idx = model.jnt_qposadr[joint_id]
    data.qpos[start_idx:start_idx+3] = box_pos
    data.qpos[start_idx+3:start_idx+7] = np.array([1, 0, 0, 0])
    data.qvel[start_idx:start_idx+6] = 0

    # Start/goal marker
    if start_marker_pos is not None:
        start_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'start_marker')
        model.geom_pos[start_geom_id][:3] = start_marker_pos

    if goal_marker_pos is not None:
        goal_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_marker")
        model.geom_pos[goal_geom_id][:3] = goal_marker_pos

    mujoco.mj_forward(model, data)


# ==== MAIN ==== 
data_list = []
all_images = []
goal_obs_list = []
pairs = []

# ==== SAMPLE XYZ GOAL POSITIONS ====
print("📦 Sampling XYZ pairs...")
while len(pairs) < PAIRS:
    start = sample_goal_xyz()
    goal = sample_goal_xyz()
    if np.linalg.norm(start - goal) >= MIN_DIST:
        pairs.append((start, goal))

# ==== PRECOMPUTE GOAL OBS ====
print("🎯 Computing goal observations...")
for start_pos, goal_pos in pairs:
    reset_and_place_all(box_pos=goal_pos, start_marker_pos=start_pos, goal_marker_pos=goal_pos)
    move_franka_to_xyz(goal_pos)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera_id)
    img = renderer.render()
    goal_obs = np.concatenate([data.qpos[:7], data.qvel[:7], get_ee_position()])
    goal_obs_list.append((goal_obs.copy(), img.copy()))


# ==== GENERATE EPISODES ====
print("🎬 Generating episodes...")
for pair_idx, (start_xyz, goal_xyz) in enumerate(tqdm(pairs)):
    goal_obs, goal_img = goal_obs_list[pair_idx]

    for _ in range(EPISODES_PER_PAIR):
        mujoco.mj_resetData(model, data)
        reset_and_place_all(box_pos=start_xyz, start_marker_pos=start_xyz, goal_marker_pos=goal_xyz)
        # move_franka_to_xyz(start_xyz)
        mujoco.mj_forward(model, data)

        episode_obs = []
        episode_actions = []

        franka_qpos = data.qpos[:7]
        franka_qvel = data.qvel[:7]
        ee_pos = get_ee_position()
        obs = np.concatenate([franka_qpos, franka_qvel, ee_pos])
        episode_obs.append(obs.copy())
        renderer.update_scene(data, camera=camera_id)
        all_images.append(renderer.render())

        for _ in range(STEPS_PER_EPISODE):
            # action = np.random.uniform(model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            
            target_xyz = sample_xyz()
            # print('target_xyz:', target_xyz)
            try:
                joint_angles = move_franka_to_xyz(target_xyz)
            except Exception as e:
                print(f'IK failed for target {target_xyz}, skipping step.')
                continue 
            action = joint_angles.copy()
            for _ in range(300):
                mujoco.mj_step(model, data)
                # mujoco.mj_forward(model, data)
            franka_qpos = data.qpos[:7]
            franka_qvel = data.qvel[:7]
            ee_pos = get_ee_position()
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

print("✅ Done generating dataset!")
print(f"Saved to: {SAVE_PATH}")


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