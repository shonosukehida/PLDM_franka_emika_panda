import mujoco
import mujoco.viewer
import numpy as np
import imageio
import os
import time
from datetime import datetime

os.environ["MUJOCO_GL"] = "glfw"

model_path = "mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

ctrlrange = model.actuator_ctrlrange
mujoco.mj_resetData(model, data)


# Rendererを作成（幅と高さは任意）
renderer = mujoco.Renderer(model, height=480, width=640)
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "top_view")

# 動画保存のための writer を作成
SAVE_VIDEO = True
output_path = 'robot_sim/output'
os.makedirs(output_path, exist_ok=True)  # ディレクトリが存在しない場合は作成する
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f'panda_sim_{timestamp}.mp4'
video_path = os.path.join(output_path, video_filename)


if SAVE_VIDEO:
    writer = imageio.get_writer(video_path, fps=30)
frame_count = 0
max_frames = 300  # 10秒（30fps）なら300フレーム

# 初期の目標
target_ctrl = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])

update_interval = 5  # 秒ごとに目標を変更
last_update_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and frame_count < max_frames:
        current_time = time.time()
        if current_time - last_update_time > update_interval:
            
            target_ctrl = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])

            last_update_time = current_time

        data.ctrl[:] = target_ctrl
        mujoco.mj_step(model, data)


        # 画像を取得
        renderer.update_scene(data, camera=camera_id)
        rgb_image = renderer.render()

        # 動画に追加
        if SAVE_VIDEO:
            writer.append_data(rgb_image)
        
        print("qpos (関節角度):", np.round(data.qpos[:model.nq], 3), flush=True)
        print("qvel (関節速度):", np.round(data.qvel[:model.nv], 3), flush=True)
        print("contacts:", data.ncon)
        print("---")
        
        viewer.sync()
        time.sleep(0.01)
        frame_count += 1

if SAVE_VIDEO:
    writer.close()
print('successfully saved video!!' if SAVE_VIDEO else 'simulation finished (no video).')
