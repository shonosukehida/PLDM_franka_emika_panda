import mujoco
import mujoco.viewer
import os
import numpy as np
import time

os.environ["MUJOCO_GL"] = "glfw"

model_path = "mujoco_menagerie/franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

ctrlrange = model.actuator_ctrlrange
mujoco.mj_resetData(model, data)

# 初期の目標
target_ctrl = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])

update_interval = 10  # 秒ごとに目標を変更
last_update_time = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        current_time = time.time()
        if current_time - last_update_time > update_interval:
            
            target_ctrl = np.random.uniform(ctrlrange[:, 0], ctrlrange[:, 1])

            last_update_time = current_time

        data.ctrl[:] = target_ctrl
        mujoco.mj_step(model, data)
        
        print("qpos (関節角度):", np.round(data.qpos[:model.nq], 3), flush=True)
        print("qvel (関節速度):", np.round(data.qvel[:model.nv], 3), flush=True)
        print("---")
        
        viewer.sync()
        time.sleep(0.01)