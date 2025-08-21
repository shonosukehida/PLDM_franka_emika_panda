# envs_dc.py など新ファイルにしてもOK
import os
os.environ["MUJOCO_GL"] = "egl"   # ★ import前に！train.py 最上段にも入れてね
os.environ.pop("DISPLAY", None)   # ★ヘッドレスなら DISPLAY は邪魔なので消す
import numpy as np
import torch
from dm_control import mujoco as dm_mj
from pldm_envs.utils.normalizer import Normalizer

class FrankaSimEnv:
    def __init__(
        self,
        model_path: str,
        image_size=(64, 64),
        camera_name="top_view",
        goal_noise=0.01,
        normalizer: Normalizer = None,
        success_thresh=0.05,    # しきい値 m
        substeps=200,           # 1 env.step で進める物理ステップ数（dataset に合わせて 200）
    ):
        self.model_path = model_path
        self.image_size = tuple(image_size)
        self.camera_name = camera_name
        self.goal_noise = goal_noise
        self.normalizer = normalizer
        self.use_normalize = normalizer is not None
        self.success_thresh = success_thresh
        self.substeps = substeps

        # dm_control の Physics を使う（オフスクリーン描画OK）
        self.physics = dm_mj.Physics.from_xml_path(self.model_path)

        # カメラ
        if self.camera_name == "default":
            self.camera_id = -1
        else:
            self.camera_id = self.physics.model.name2id(self.camera_name, dm_mj.mjtObj.mjOBJ_CAMERA)

        # 青箱 joint / qpos index
        self.joint_id = self.physics.model.name2id("free_joint_blue_box", "joint")
        self.start_idx = self.physics.model.jnt_qposadr[self.joint_id]  # pos(xyz)=3, quat=4

        # actuator 範囲（行動クリップ用）
        self.ctrlrange = self.physics.model.actuator_ctrlrange.copy()

        self.t = 0
        self.max_episode_steps = 100
        self.start_pos = None
        self.goal_pos = None

    # ========== 基本I/O ==========
    def reset(self, start_pos=None, goal_pos=None):
        self.physics.reset()
        self.t = 0

        if start_pos is None:
            start_pos = np.random.uniform(low=[0.365, -0.15, 0.05], high=[0.665, 0.15, 0.05])
        if goal_pos is None:
            goal_pos = np.random.uniform(low=[0.365, -0.15, 0.05], high=[0.665, 0.15, 0.05])

        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # 青箱を start に置く
        self.physics.data.qpos[self.start_idx:self.start_idx+3] = start_pos
        self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
        self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
        self.physics.forward()

        # 目標の proprio ベクトルを事前計算（使わないなら残してOK）
        _ = self._get_goal_obs_vec(goal_pos)

        return self.get_obs()

    def step(self, action):
        # 行動クリップ（dm_control でも ctrlrange は同じ形状）
        low, high = self.ctrlrange[:, 0], self.ctrlrange[:, 1]
        action = np.clip(np.asarray(action), low, high)
        self.physics.data.ctrl[:] = action

        # 1制御で 200 サブステップ（XML timestep=0.001 → 0.2s/step）
        for _ in range(self.substeps):
            self.physics.step()

        self.t += 1
        image_obs = self.get_obs()
        object_obs = self.get_object_position()
        done = self.t >= self.max_episode_steps
        reward = self._is_success(object_obs)

        truncated = False
        info = self.get_info()
        return image_obs, reward, done, truncated, info

    # ========== 観測 ==========
    def get_obs(self):
        # 画像レンダ（dm_controlは offscreen 対応済）
        img = self.physics.render(height=self.image_size[0], width=self.image_size[1], camera_id=self.camera_id)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # [C,H,W]
        img = torch.from_numpy(img).contiguous()
        if self.use_normalize:
            img = self.normalizer.normalize_state(img)
        return img

    def get_target_obs(self):
        # ゴール位置のシーンを作って 1枚レンダ
        self.physics.data.qpos[self.start_idx:self.start_idx+3] = self.goal_pos
        self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
        self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
        self.physics.forward()
        img = self.physics.render(height=self.image_size[0], width=self.image_size[1], camera_id=self.camera_id)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).contiguous()
        if self.use_normalize:
            img = self.normalizer.normalize_state(img)
        return img

    def _get_goal_obs_vec(self, goal_pos):
        # proprio ベクトル版（必要なら）
        self.physics.data.qpos[self.start_idx:self.start_idx+3] = goal_pos
        self.physics.forward()
        qpos = self.physics.data.qpos[:7]
        qvel = self.physics.data.qvel[:7]
        return np.concatenate([qpos, qvel])

    # ========== 補助 ==========
    def _is_success(self, object_pos):
        return float(np.linalg.norm(object_pos - self.goal_pos) < self.success_thresh)

    def get_target(self):
        return self.goal_pos

    def get_ee_position(self):
        hand_body_id = self.physics.model.name2id("panda_hand", "body")
        return self.physics.data.xpos[hand_body_id].copy()

    def get_object_position(self):
        return self.physics.data.qpos[self.start_idx : self.start_idx + 3].copy()

    def get_info(self):
        return {
            "location": self.get_ee_position(),
            "qpos": self.physics.data.qpos[:7].copy(),
            "qvel": self.physics.data.qvel[:7].copy(),
        }

    def get_propio_pos(self):
        qpos = torch.from_numpy(self.physics.data.qpos[:7].copy()).float()
        if self.use_normalize:
            qpos = self.normalizer.normalize_propio_pos(qpos)
        return qpos

    def get_propio_vel(self):
        qvel = torch.from_numpy(self.physics.data.qvel[:7].copy()).float()
        if self.use_normalize:
            qvel = self.normalizer.normalize_propio_vel(qvel)
        return qvel
