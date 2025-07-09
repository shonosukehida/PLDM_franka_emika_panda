import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import yaml
import pandas as pd
import imageio
from tqdm import tqdm
from dm_control import mujoco
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from scipy.spatial.transform import Rotation as R

class FrankaSimEnv:
    def __init__(self, config):
        self.config = config

        self.MODEL_PATH = config["model_path"]
        self.CAMERA_NAME = config.get("camera_name", "")
        self.IMAGE_SIZE = tuple(config["image_size"])

        self.CONFIRM_IK = config["confirm_ik_result"]

        self.STEPS = config["steps"]
        self.TOL = float(config["tol"])


        os.environ["MUJOCO_GL"] = "egl"
        self.physics = mujoco.Physics.from_xml_path(self.MODEL_PATH)

        if self.CAMERA_NAME == 'default':
            self.camera_id = -1
        else:
            self.camera_id = self.physics.model.name2id(self.CAMERA_NAME, mujoco.mjtObj.mjOBJ_CAMERA)
        
        self.sim_steps = config['steps']
        self.target_sampling_step = config['target_sampling_step']

        


    def reset_and_place_all(self, box_pos, start_marker_pos=None, goal_marker_pos=None, init_position=None):
        self.physics.reset()
        
        if init_position is not None:
            self.physics.data.qpos[:7] = init_position
            self.physics.data.qvel[:7] = 0

        joint_id = self.physics.model.name2id("free_joint_blue_box", "joint")
        start_idx = self.physics.model.jnt_qposadr[joint_id]
        self.physics.data.qpos[start_idx:start_idx+3] = box_pos
        self.physics.data.qpos[start_idx+3:start_idx+7] = np.array([1, 0, 0, 0])
        self.physics.data.qvel[start_idx:start_idx+6] = 0

        if start_marker_pos is not None:
            model_id = self.physics.model.name2id('start_marker', 'geom')
            self.physics.model.geom_pos[model_id][:3] = start_marker_pos
        if goal_marker_pos is not None:
            model_id = self.physics.model.name2id('goal_marker', 'geom')
            self.physics.model.geom_pos[model_id][:3] = goal_marker_pos
        
        self.physics.forward()
    
    def calc_inverse_kinematic(self, target_xyz, target_rotmat=None, rot_weight=1.0):
        target_quat = None
        if target_rotmat is not None:
            target_quat = R.from_matrix(target_rotmat).as_quat(scalar_first=True)
        # print('target_quat:', target_quat)
        
        joint_names = [f"joint{i}" for i in range(1, 8)]
        result = qpos_from_site_pose(
            self.physics,
            site_name="ee_target",
            target_pos=target_xyz,
            target_quat=target_quat,
            joint_names=joint_names,
            rot_weight=rot_weight
        )
        return result
    
    
    def get_ee_position(self):
        return self.physics.named.data.site_xpos['ee_target'].copy()

        
    def step_xyz(self, target_pos, target_rotmat=None, steps=200, tol=1e-3, rot_weight=1.0):
        result = self.calc_inverse_kinematic(
            target_pos, 
            target_rotmat=target_rotmat, 
            rot_weight=rot_weight,
        )
        if not result.success:
            raise ValueError("IK failed!")
        joint_angles = result.qpos[:7]
        
        alpha = self.config['alpha']
        objective_reached = False
        dist_steps = []
        
        for _ in range(steps):
            q_current = self.physics.data.qpos[:7]
            error = joint_angles - q_current
            ctrl = q_current + alpha * error
            self.physics.data.ctrl[:7] = ctrl
            self.physics.step()
            
            ee_pos = self.get_ee_position()
            dist_steps.append(np.abs(ee_pos - target_pos))
            
            dist = np.linalg.norm(ee_pos - target_pos)
            if dist < tol:
                objective_reached = True
                break

        site_pos = ee_pos.copy()
        return joint_angles, site_pos, dist_steps, objective_reached

    def set_xyz(self, target_pos, target_rotmat=None, rot_weight=0.1):
        result = self.calc_inverse_kinematic(
            target_pos, 
            target_rotmat=target_rotmat, 
            rot_weight=rot_weight,
        )
        if not result.success:
            raise ValueError("IK failed!")

        # --- 計算結果をそのまま反映 ---
        self.physics.data.qpos[:7] = result.qpos[:7]
        self.physics.data.qvel[:7] = 0
        self.physics.forward()

        ee_pos = self.get_ee_position()
        return result.qpos[:7], ee_pos


    
    def render_image(self, size = (64, 64)):
        return self.physics.render(*size, camera_id = self.camera_id)
    

    def get_obs(self):
        qpos = self.physics.data.qpos[:7]
        qvel = self.physics.data.qvel[:7]
        ee = self.get_ee_position()
        return np.concatenate([qpos, qvel, ee])


    def check_ik_accuracy(self, target_xyz):
        self.physics.reset()
        
        # target_xyz = self.sample_uniform_xyz()
        result = self.calc_inverse_kinematic(target_xyz)
        
        if not result.success:
            print("⚠️ IK失敗しました!")
            return None

        self.physics.data.qpos[:7] = result.qpos[:7]
        self.physics.data.qvel[:7] = 0
        self.physics.forward() 

        ee_pos = self.get_ee_position()

        dx, dy, dz = ee_pos - target_xyz
        dist = np.linalg.norm([dx, dy, dz])
        tol = 1e-4
        flag = dist < tol


        self.physics.reset()
        return flag
