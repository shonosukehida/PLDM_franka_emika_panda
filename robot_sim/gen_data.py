import mujoco
import numpy as np
import torch
import os
from tqdm import tqdm

# ==== CONFIG ====
PAIRS = 5
MIN_DIST = 0.4
EPISODES_PER_PAIR = 1
STEPS_PER_EPISODE = 100
SAVE_PATH = f"pldm_envs/franka/presaved_datasets/pairs_{PAIRS}_ep_{EPISODES_PER_PAIR}_timestep_{STEPS_PER_EPISODE}"
IMAGE_SIZE = (64, 64)
MODEL_PATH = "mujoco_menagerie/franka_emika_panda/scene.xml"
CAMERA_NAME = "top_view" # "": default

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

    # スタートマーカーを動かす（あれば）
    if start_marker_pos is not None:
        model.geom_pos[start_geom_id][:3] = start_marker_pos

    # ゴールマーカーを動かす（あれば）
    if goal_marker_pos is not None:
        model.geom_pos[goal_geom_id][:3] = goal_marker_pos


    mujoco.mj_forward(model, data)


def sample_random_position(x_range=(0.1, 0.3), y_range=(-0.3, 0.3), z=0.05):
    return np.array([
        np.random.uniform(*x_range),
        np.random.uniform(*y_range),
        z
    ])

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
    goal_obs = np.concatenate([data.qpos[:7], data.qvel[:7]])
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

        franka_qpos = data.qpos[:7]  # 関節角度: joint1〜joint7
        franka_qvel = data.qvel[:7]  # 関節速度: joint1〜joint7
        obs = np.concatenate([franka_qpos, franka_qvel])

        episode_obs.append(obs.copy())
        renderer.update_scene(data, camera=camera_id)
        all_images.append(renderer.render())

        for _ in range(STEPS_PER_EPISODE):
            action = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])
            data.ctrl[:] = action
            for _ in range(10): #1 STEPS_PER_EPISODE あたりに進めるステップ数
                mujoco.mj_step(model, data)

            obs = np.concatenate([franka_qpos, franka_qvel])
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
