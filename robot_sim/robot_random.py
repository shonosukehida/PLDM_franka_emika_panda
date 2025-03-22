import mujoco
import mujoco.viewer
import numpy as np
import imageio
import os
import time

os.environ["MUJOCO_GL"] = "glfw"

model_path = "mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

ctrlrange = model.actuator_ctrlrange
mujoco.mj_resetData(model, data)


# Rendererを作成（幅と高さは任意）
renderer = mujoco.Renderer(model, height=480, width=640)
# 動画保存のための writer を作成
output_path = '/robot_sim/output'
writer = imageio.get_writer("panda_sim.mp4", fps=30)
frame_count = 0
max_frames = 300  # 10秒（30fps）なら300フレーム

# 初期の目標
target_ctrl = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])

update_interval = 10  # 秒ごとに目標を変更
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
        renderer.update_scene(data)
        rgb_image = renderer.render()

        # 動画に追加
        writer.append_data(rgb_image)
        
        print("qpos (関節角度):", np.round(data.qpos[:model.nq], 3), flush=True)
        print("qvel (関節速度):", np.round(data.qvel[:model.nv], 3), flush=True)
        print("---")
        
        viewer.sync()
        time.sleep(0.01)
        frame_count += 1

writer.close() 
renderer.close() 
print('successfully saved video!!')