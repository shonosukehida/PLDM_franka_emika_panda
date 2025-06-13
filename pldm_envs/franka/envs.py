#Franka 実験環境
#未検証

import torch
import mujoco
import numpy as np
from pldm_envs.utils.normalizer import Normalizer

class FrankaSimEnv:
    def __init__(
        self, 
        model_path: str, 
        image_size=(64, 64), 
        camera_name="top_view", 
        goal_noise=0.01,
        normalizer:Normalizer=None
        ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[0], width=image_size[1])
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


        self.ctrlrange = self.model.actuator_ctrlrange
        self.goal_noise = goal_noise

        #青オブジェクト
        self.joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_joint_blue_box")
        self.start_idx = self.model.jnt_qposadr[self.joint_id]

        self.t = 0
        self.max_episode_steps = 100
        self.goal_obs = None
        
        self.start_pos = None 
        self.goal_pos = None
        
        self.use_normalize = True
        self.normalizer = normalizer


        
        #環境情報表示
        self.disp_data_info_flag = False
        if self.disp_data_info_flag:
            print("qpos size =", self.model.nq)
            print("qvel size =", self.model.nv)
            print("qpos adr (start idx of joints):", self.model.jnt_qposadr)
            print("joint names:")
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                start_idx = self.model.jnt_qposadr[i]
                size = 7 if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE else 1
                print(f"  {i}: {name} (type: {self.model.jnt_type[i]}) → qpos[{start_idx}:{start_idx+size}]")


    def reset(self, start_pos=None, goal_pos=None):
        mujoco.mj_resetData(self.model, self.data)
        self.t = 0

        if start_pos is None:
            start_pos = np.random.uniform(low=[0.1, -0.3, 0.05], high=[0.3, 0.3, 0.05])
        if goal_pos is None:
            goal_pos = np.random.uniform(low=[0.1, -0.3, 0.05], high=[0.3, 0.3, 0.05])

        self.start_pos = start_pos
        self.goal_pos = goal_pos
        
        
        # 初期位置設定
        self.data.qpos[self.start_idx:self.start_idx+3] = start_pos
        self.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
        self.data.qvel[self.start_idx:self.start_idx+6] = 0

        mujoco.mj_forward(self.model, self.data)
        self.goal_obs = self._get_goal_obs_vec(goal_pos)

        # return self._get_obs_vec()
        return self.get_obs()

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)

        self.t += 1
        image_obs = self.get_obs()
        object_obs = self.get_object_position()
        done = self.t >= self.max_episode_steps
        reward = self._is_success(object_obs)

        truncated = False  # 必要なら時間制限等で設定
        info = self.get_info()

        return image_obs, reward, done, truncated, info

    #現在のpropio
    def _get_obs_vec(self):
        qpos = self.data.qpos[:7]
        qvel = self.data.qvel[:7]
        return np.concatenate([qpos, qvel])

    #現在の観測画像
    def get_obs(self):
        img = self.renderer.render()  
        self.renderer.update_scene(self.data, camera=self.camera_id) 
        img = self.renderer.render()  
        
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        if self.use_normalize: ##
            # img = img.astype(np.float32) / 255.0
            
            # img = (img.astype(np.float32) - 0) / 1
            img = self.normalizer.normalize_state(img)

        # print('img.mean:', img.mean())
        # print('img.std:', img.std())

        return img

    def _get_goal_obs_vec(self, goal_pos):
        self.data.qpos[self.start_idx:self.start_idx+3] = goal_pos
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs_vec()

    def _is_success(self, object_pos):
        return False
        return float(np.linalg.norm(object_pos - self.goal_pos) < 0.05)

    def get_target(self,):
        '''ゴール座標'''
        return self.goal_pos

    def get_target_obs(self):
        # ゴール位置にオブジェクトを置く
        self.data.qpos[self.start_idx:self.start_idx+3] = self.goal_pos
        self.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
        self.data.qvel[self.start_idx:self.start_idx+6] = 0
        mujoco.mj_forward(self.model, self.data)

        img = self.renderer.render()
        img = np.transpose(img, (2, 0, 1))  # [H, W, C] → [C, H, W]
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        if self.use_normalize:
            # img = img.astype(np.float32) / 255.0
            # img = (img.astype(np.float32) - 0) / 1  
            img = self.normalizer.normalize_state(img)


        print('target_img.mean:', img.mean())
        print('target_img.std:', img.std())

        return img

    def get_ee_position(self):
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        pos = self.data.xpos[ee_body_id].copy()  # [x, y, z]
        return pos

    def get_info(self):
        return {
            "location": self.get_ee_position(),               # エンドエフェクタの位置 (x, y, z)
            "qpos": self.data.qpos[:7].copy(),                # 関節角度（7自由度）
            "qvel": self.data.qvel[:7].copy(),                # 関節速度
        }


    def get_propio_pos(self):
        qpos = torch.from_numpy(self.data.qpos[:7].copy()).float()
        if self.use_normalize:
            qpos = self.normalizer.normalize_propio_pos(qpos)
        return qpos

    def get_propio_vel(self):
        qvel = torch.from_numpy(self.data.qvel[:7].copy()).float()
        if self.use_normalize:
            qvel = self.normalizer.normalize_propio_vel(qvel)
        return qvel

    #オブジェクトの現在の位置
    def get_object_position(self):
        return self.data.qpos[self.start_idx : self.start_idx + 3].copy()


