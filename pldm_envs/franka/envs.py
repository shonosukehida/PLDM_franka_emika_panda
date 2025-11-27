import os
os.environ["MUJOCO_GL"] = "egl"   
os.environ.pop("DISPLAY", None)   
import numpy as np
import torch
from dm_control import mujoco as dm_mj
# from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from pldm_envs.franka.ik_with_limits import qpos_from_site_pose
from scipy.spatial.transform import Rotation as R
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
        substeps=200,           
    ):
        self.model_path = model_path
        self.image_size = tuple(image_size)
        self.camera_name = camera_name
        self.goal_noise = goal_noise
        self.normalizer = normalizer
        self.use_normalize = normalizer is not None
        self.success_thresh = success_thresh
        self.substeps = substeps

        self.physics = dm_mj.Physics.from_xml_path(self.model_path)

        if self.camera_name == "default":
            self.camera_id = -1
        else:
            self.camera_id = self.physics.model.name2id(self.camera_name, dm_mj.mjtObj.mjOBJ_CAMERA)

        # 青箱 joint / qpos index
        self.joint_id = self.physics.model.name2id("free_joint_blue_box", "joint")
        self.start_idx = self.physics.model.jnt_qposadr[self.joint_id]  # pos(xyz)=3, quat=4

        # actuator 範囲（行動クリップ用）
        # self.ctrlrange = self.physics.model.actuator_ctrlrange.copy()
        
        arm_ids = []
        for i in range(1, 8):
            arm_ids.append(self.physics.model.name2id(f"actuator{i}", "actuator"))
        self.arm_actuator_ids = np.array(arm_ids, dtype=int)
        self.ctrlrange = self.physics.model.actuator_ctrlrange[self.arm_actuator_ids].copy()
        self.n_arm_act = len(self.arm_actuator_ids) 
        
        self.t = 0
        self.max_episode_steps = 100
        self.start_pos = None
        self.goal_pos = None
        
        self.MAX_DQ = 1000_000_000.0
        
        self.control_dt = float(self.physics.model.opt.timestep) * int(self.substeps) 

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
            tol=1e-4,
            rot_weight=rot_weight
        )
        return result

    # ========== 基本I/O ==========
    def reset(self, start_pos=None, goal_pos=None, robot_only = False):
        self.physics.reset()
        self.t = 0
        
        
        target_rotmat = None
        # x = np.array([1, 0, 0])         
        # y = np.array([0, -1, 0])
        # z = np.array([0, 0, -1])        
        # target_rotmat = np.stack([x, y, z], axis=1)
        target_rot_weight = 0.1
        
        init_xyz = np.array([0.515, 0.0, 0.1])
        result = self.calc_inverse_kinematic(init_xyz, target_rotmat=target_rotmat, rot_weight = target_rot_weight)
        init_joint = result.qpos[:7]
        self.physics.forward()
        
        
        self.physics.data.qpos[:7] = init_joint
        self.physics.data.qvel[:7] = 0.0
        self.physics.forward()
        
        self.physics.data.ctrl[:] = 0.0
        self.physics.data.ctrl[self.arm_actuator_ids] = init_joint

        ##### IK計算デバッグ
        sid = self.physics.model.name2id("ee_target", "site")
        ee_now = self.physics.data.site_xpos[sid].copy()
        print("[reset] IK target:", init_xyz, "actual:", ee_now, "err:", np.linalg.norm(ee_now-init_xyz))
        print("IK success:", result.success)
        #####

        #時間を回して落ち着かせる
        # for _ in range(50):
        #     self.physics.step()

        if start_pos is None:
            start_pos = np.random.uniform(low=[0.415, -0.10, 0.05], high=[0.615, 0.10, 0.05])
        if goal_pos is None:
            goal_pos = np.random.uniform(low=[0.415, -0.10, 0.05], high=[0.615, 0.10, 0.05])

        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # 青箱を start に置く
        if not robot_only:
            self.physics.data.qpos[self.start_idx:self.start_idx+3] = start_pos
            self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
            self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
            self.physics.forward()
            
            
            _ = self._get_goal_obs_vec(goal_pos)
        else:
            self.physics.data.qpos[self.start_idx:self.start_idx+3] = np.array([100.0, 100.0, 0.06])
            self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
            self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
            self.physics.forward()
            


        

        return self.get_obs()

    def step(self, action, max_dq = 0.01):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_arm_act:
            raise ValueError(f"expected action dim {self.n_arm_act} but got {action.shape}")
        if not np.all(np.isfinite(action)):
            action = np.zeros_like(action)

        
        qpos = self.physics.data.qpos[:7].copy()
        
        self.MAX_DQ = max_dq #1step あたりの最大増分rad
        dq = np.clip(action - qpos, -self.MAX_DQ, self.MAX_DQ)
        target = qpos + dq #qpos + dq 


        low, high = self.ctrlrange[:, 0], self.ctrlrange[:, 1]
        target = np.clip(target, low, high)


        self.physics.data.ctrl[:] = 0.0
        self.physics.data.ctrl[self.arm_actuator_ids] = target


        for _ in range(self.substeps):
            self.physics.step()

        self.t += 1
        image_obs = self.get_obs()
        if isinstance(image_obs, torch.Tensor):
            image_obs = image_obs.detach().cpu().numpy()
        object_obs = self.get_object_position()
        done = self.t >= self.max_episode_steps
        reward = self._is_success(object_obs)

        truncated = False
        info = self.get_info()
        return image_obs, reward, done, truncated, info

    def set_xyz(self, target_pos, target_rotmat=None, rot_weight=0.1,
                settle_steps=10000, sync_ctrl=True):
        result = self.calc_inverse_kinematic(
            target_pos, target_rotmat=target_rotmat, rot_weight=rot_weight,
        )
        if not result.success:
            raise ValueError("IK failed!")

        q_des = result.qpos[:7].copy()
        with self.physics.reset_context():
            self.physics.data.qpos[:7] = q_des
            self.physics.data.qvel[:]  = 0.0          #速度ゼロ
            self.physics.data.act[:]   = 0.0          #アクチュエータゼロ
            self.physics.data.qacc_warmstart[:] = 0.0 #ソルば初期化
            self.physics.forward()

            # self.physics.data.qpos[:7] = q_des
            # self.physics.data.qvel[:7] = 0.0
            # self.physics.forward()
        
        #デバッグ
        sid = self.physics.model.name2id("ee_target", "site")
        ee_ik = self.physics.data.site_xpos[sid].copy()
        err_ik = float(np.linalg.norm(ee_ik - target_pos))
        print("[IK only] pos:", ee_ik, "err:", err_ik)
        #######

        #接触判定
        print("ncon BEFORE =", int(self.physics.data.ncon))
        for i in range(int(self.physics.data.ncon)):
            c = self.physics.data.contact[i]
            g1 = self.physics.model.id2name(c.geom1, 'geom')
            g2 = self.physics.model.id2name(c.geom2, 'geom')
            print(i, g1, g2, "dist=", c.dist)  # dist<0 ならめり込み
        
        #関節限界チェック
        q = self.physics.data.qpos[:7].copy()
        lo = self.physics.model.jnt_range[:7,0]
        hi = self.physics.model.jnt_range[:7,1]
        print("near_limit joints:", np.where((q<lo+1e-3)|(q>hi-1e-3))[0])


        

        if sync_ctrl:
            low, high = self.ctrlrange[:, 0], self.ctrlrange[:, 1]
            # self.physics.data.ctrl[:] = 0.0
            self.physics.data.ctrl[self.arm_actuator_ids] = q_des


        for _ in range(settle_steps):
            self.physics.step()
            
        #デバッグ
        ee_after = self.physics.data.site_xpos[sid].copy()
        err_after = float(np.linalg.norm(ee_after - target_pos))
        print("[after step] pos:", ee_after, "err:", err_after)
        #############

        ee_pos = self.get_ee_position()
        return q_des, ee_pos
    
    def set_joint(self, joint_angle):

        q = np.asarray(joint_angle, dtype=np.float32).reshape(-1)
        if q.shape[0] != 7:
            raise ValueError(f"expected (7,) but got {q.shape}")
        self.physics.data.qpos[:7] = q
        self.physics.data.qvel[:7] = 0.0
        self.physics.forward()
        
        return 



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
        # backup
        qpos_bk = self.physics.data.qpos.copy()
        qvel_bk = self.physics.data.qvel.copy()

        # 一時的にゴール配置にしてレンダ
        self.physics.data.qpos[self.start_idx:self.start_idx+3] = self.goal_pos
        self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
        self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
        self.physics.forward()
        img = self.physics.render(height=self.image_size[0], width=self.image_size[1], camera_id=self.camera_id)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).contiguous()
        if self.use_normalize:
            img = self.normalizer.normalize_state(img)

        # restore
        self.physics.data.qpos[:] = qpos_bk
        self.physics.data.qvel[:] = qvel_bk
        self.physics.forward()
        return img

    def _get_goal_obs_vec(self, goal_pos):
        # --- backup ---
        qpos_bk = self.physics.data.qpos.copy()
        qvel_bk = self.physics.data.qvel.copy()
        try:
            # 一時的にゴール配置にして順運動学の値を取る
            self.physics.data.qpos[self.start_idx:self.start_idx+3] = goal_pos
            self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
            self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
            self.physics.forward()

            qpos = self.physics.data.qpos[:7].copy()
            qvel = self.physics.data.qvel[:7].copy()
            return np.concatenate([qpos, qvel])
        finally:
            # --- restore ---
            self.physics.data.qpos[:] = qpos_bk
            self.physics.data.qvel[:] = qvel_bk
            self.physics.forward()



    def _is_success(self, object_pos):
        return float(np.linalg.norm(object_pos - self.goal_pos) < self.success_thresh)

    def get_target(self):
        return self.goal_pos

    def get_ee_position(self):
        try:
            sid = self.physics.model.name2id("ee_target", "site")
            return self.physics.data.site_xpos[sid].copy()
        except Exception:
            pass

        bid = self.physics.model.name2id("hand", "body")
        return self.physics.data.xpos[bid].copy()

    def get_object_position(self):
        return self.physics.data.qpos[self.start_idx : self.start_idx + 3].copy()

    def get_info(self):
        return {
            "location": self.get_ee_position(),
            "qpos": self.physics.data.qpos[:7].copy(),
            "qvel": self.physics.data.qvel[:7].copy(),
            "object_pos": self.get_object_position().copy(),
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
