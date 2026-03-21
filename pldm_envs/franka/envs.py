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
from transformers import AutoVideoProcessor
import mujoco


class FrankaSimEnv:
    def __init__(
        self,
        model_path: str,
        image_size=(64, 64),
        camera_name="top_view",
        goal_noise=0.01,
        normalizer: Normalizer = None,
        success_thresh=0.05,    # しきい値 m
        substeps=20,       
        max_dq=0.01,
        task_cfg = None,
        task_name="push_to_goal",
        max_reset_tries=500,
        min_start_goal_dist=0.12,
    ):
        self.model_path = model_path
        self.image_size = tuple(image_size)
        self.camera_name = camera_name
        self.goal_noise = goal_noise
        self.normalizer = normalizer
        self.use_normalize = normalizer is not None
        self.success_thresh = success_thresh
        self.substeps = substeps
        print("[DBG][pldm_envs/franka/envs.py] self.use_normalize:", self.use_normalize)

        self.physics = dm_mj.Physics.from_xml_path(self.model_path)

        if self.camera_name == "default":
            self.camera_id = -1
        else:
            self.camera_id = self.physics.model.name2id(self.camera_name, dm_mj.mjtObj.mjOBJ_CAMERA)

        # 青箱 joint / qpos index
        self.joint_id = self.physics.model.name2id("free_joint_blue_box", "joint")
        self.start_idx = self.physics.model.jnt_qposadr[self.joint_id]  # pos(xyz)=3, quat=4
        
        self.max_reset_tries = max_reset_tries
        self.min_start_goal_dist = min_start_goal_dist

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
        
        self.MAX_DQ = max_dq
        print("[dbg][pldm_envs/franka/envs.py] self.MAX_DQ:", self.MAX_DQ)
        
        self.control_dt = float(self.physics.model.opt.timestep) * int(self.substeps) 

        self.task_cfg = task_cfg
        self.task_name = task_name

        self.ee_goal_pos = None
        self.box_init_pos = None
        self.robot_only = False
        
        if task_name == "push_to_goal":
            self.use_box = True
        elif task_name == "reach_no_touch":
            tc = getattr(self.task_cfg, "reach_no_touch", None)
            self.use_box = bool(getattr(tc, "use_box", True))
        else:
            # 新タスクは基本箱ありにする、など
            self.use_box = True

        self.vjepa2_processor = None
        if getattr(self.task_cfg, "backbone_arch", None) == "vjepa2":
            self.vjepa2_processor = AutoVideoProcessor.from_pretrained(self.task_cfg.vjepa2_repo)
        

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
    def reset(self, start_pos=None, goal_pos=None, robot_only = False, init_qpos=None, init_qvel=None, sync_ctrl=True):
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


        if init_qpos is not None:
            self.physics.data.qpos[:7] = np.asarray(init_qpos, dtype=np.float32)
        if init_qvel is not None:
            self.physics.data.qvel[:7] = np.asarray(init_qvel, dtype=np.float32)
        else:
            if init_qpos is not None:
                self.physics.data.qvel[:7] = 0.0

        if sync_ctrl and (init_qpos is not None):
            self.physics.data.ctrl[:] = 0.0
            self.physics.data.ctrl[self.arm_actuator_ids] = self.physics.data.qpos[:7].copy()
        self.physics.forward()

        ##### IK計算デバッグ
        sid = self.physics.model.name2id("ee_target", "site")
        ee_now = self.physics.data.site_xpos[sid].copy()
        print("[reset] IK target:", init_xyz, "actual:", ee_now, "err:", np.linalg.norm(ee_now-init_xyz))
        print("IK success:", result.success)
        #####


        if (start_pos is None) or (goal_pos is None):
            for _ in range(self.max_reset_tries):
                sp = start_pos if start_pos is not None else np.random.uniform(
                    low=[0.415, -0.10, 0.05], high=[0.615, 0.10, 0.05]
                )
                gp = goal_pos if goal_pos is not None else np.random.uniform(
                    low=[0.415, -0.10, 0.05], high=[0.615, 0.10, 0.05]
                )

                # 距離チェック（zは固定なので xy でも3DでもOK。ここは3Dで）
                if np.linalg.norm(sp - gp) < self.min_start_goal_dist:
                    continue

                start_pos = sp
                goal_pos  = gp
                break
            else:
                raise RuntimeError("reset: failed to sample (start_pos, goal_pos) far enough")

        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # ---- task specific goal params ----
        print("[DBG][pldm_envs/franka/envs.py] task_name:", self.task_name)
        if self.task_name == "reach_no_touch":
            tc = getattr(self.task_cfg, "reach_no_touch", None)
            if tc is None:
                raise ValueError("task_cfg.reach_no_touch is missing")  # もしくはデフォルト生成

            self.ee_goal_pos = np.random.uniform(low=[0.415, -0.10, 0.05], high=[0.615, 0.10, 0.05])

            # 箱の「初期位置」(保持したい位置)
            if tc.box_init_xyz is None:
                self.box_init_pos = self.start_pos.copy()
            else:
                self.box_init_pos = np.array(tc.box_init_xyz, dtype=np.float32)
            robot_only = robot_only or  not self.use_box
            self.robot_only = robot_only

        else:
            # push_to_goal
            self.ee_goal_pos = None
            self.box_init_pos = None




        # 青箱を start に置く
        if not self.robot_only:
            self.physics.data.qpos[self.start_idx:self.start_idx+3] = start_pos
            self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
            self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
            self.physics.forward()
            

            if self.task_name == "push_to_goal":
                _ = self._get_goal_obs_vec(goal_pos)
            elif self.task_name == "reach_no_touch":
                _ = self._get_goal_obs_vec(self.start_pos)  # or box_init_pos
                
        else:
            self.physics.data.qpos[self.start_idx:self.start_idx+3] = np.array([100.0, 100.0, 0.06])
            self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
            self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
            self.physics.forward()
            

        #関節名チェック
        # for i in range(self.physics.model.njnt):
        #     name = mujoco.mj_id2name(self.physics.model.ptr, mujoco.mjtObj.mjOBJ_JOINT, i)
        #     print(i, name)


        return self.get_obs()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.n_arm_act:
            raise ValueError(f"expected action dim {self.n_arm_act} but got {action.shape}")
        if not np.all(np.isfinite(action)):
            action = np.zeros_like(action)

        
        qpos = self.physics.data.qpos[:7].copy()
        delta = action - qpos
        dq = np.clip(delta, -self.MAX_DQ, self.MAX_DQ)
        target = qpos + dq #qpos + dq 

        # ===== debug (最初の数ステップだけ) =====
        # if self.t < 5:
        #     print("[dbg] t=", self.t)
        #     print("[dbg] action:", action)
        #     print("[dbg] qpos  :", qpos)
        #     print("[dbg] delta :", delta)
        #     print("[dbg] |delta|:", np.abs(delta))
        #     print("[dbg] dq    :", dq)
        #     print("[dbg] saturated:", (np.abs(delta) > self.MAX_DQ))
        # =====================================

        
        low, high = self.ctrlrange[:, 0], self.ctrlrange[:, 1]
        target = np.clip(target, low, high)


        if self.t < 5:
            print("[pldm_envs/franka/envs.py]ctrlrange low/high:", low, high)
            print("[pldm_envs/franka/envs.py]target(before clip):", qpos + dq)
            print("[pldm_envs/franka/envs.py]target(after  clip):", target)
            print("[pldm_envs/franka/envs.py]clipped?:", np.any((qpos + dq) != target))


        self.physics.data.ctrl[:] = 0.0
        self.physics.data.ctrl[self.arm_actuator_ids] = target




        for _ in range(self.substeps):
            # self.physics.data.qpos[7] = 0.0 
            # self.physics.data.qpos[8] = 0.0 
            # self.physics.data.qvel[7] = 0.0
            # self.physics.data.qvel[8] = 0.0
            self.physics.step()
        

        qfrc = self.physics.data.qfrc_actuator[:7].copy()      # 関節へ入った actuator トルク
        afrc_all = self.physics.data.actuator_force.copy()  # (nu,)
        afrc_arm = afrc_all[self.arm_actuator_ids].copy()   # (7,)


        self.t += 1
        image_obs = self.get_obs()
        if isinstance(image_obs, torch.Tensor):
            image_obs = image_obs.detach().cpu().numpy()
        object_obs = self.get_object_position()
        done = self.t >= self.max_episode_steps
        reward = self._is_success(object_obs)

        truncated = False
        info = self.get_info()
        info["qfrc_actuator"] = qfrc
        info["actuator_force"] = afrc_arm



        # fj1 = self.physics.data.qpos[7]
        # fj2 = self.physics.data.qpos[8]
        # print(f"[DBG][pldm_envs/franka/envs.py] finger qpos: {fj1:.6f}, {fj2:.6f}")
        

        
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
    def get_obs(self, normalize:bool = True):
        img = self.physics.render(
            height=self.image_size[0],
            width=self.image_size[1],
            camera_id=self.camera_id
        )  # (H,W,3) uint8

        # (C,H,W) float32
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img).contiguous()
        
        if not normalize: return img


        if self.task_cfg is not None:
            if self.task_cfg.backbone_arch == "vjepa2":
                x = img.unsqueeze(0)  # (1,3,H,W)

                pv = self.vjepa2_processor(x, return_tensors="pt")["pixel_values_videos"]
                pv = pv.squeeze(0).squeeze(0)  # -> (3,256,256)

                return pv

        # vjepa2以外は従来通り normalizer
        if self.use_normalize:
            img = self.normalizer.normalize_state(img)
        return img

    def get_target_obs(self, normalize:bool = True):
        # backup
        qpos_bk = self.physics.data.qpos.copy()
        qvel_bk = self.physics.data.qvel.copy()
        ctrl_bk = self.physics.data.ctrl.copy()

        try:
            if self.task_name == "push_to_goal":
                # 既存：箱を goal_pos に置いてレンダ
                self.physics.data.qpos[self.start_idx:self.start_idx+3] = self.goal_pos
                self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
                self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
                self.physics.forward()

            elif self.task_name == "reach_no_touch":
                # reach_no_touch：箱は初期位置のまま、EEはゴールへ
                tc = getattr(self.task_cfg, "reach_no_touch", None)
                ee_eps = float(tc.ee_success_eps)
                box_eps = float(tc.box_hold_eps)

                if self.use_box:
                    box_pos = self.box_init_pos if self.box_init_pos is not None else self.start_pos
                    self.physics.data.qpos[self.start_idx:self.start_idx+3] = box_pos
                    self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
                    self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0
                else:
                    self.physics.data.qpos[self.start_idx:self.start_idx+3] = np.array([100.0, 100.0, 0.05], dtype=np.float32)
                    self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0], dtype=np.float32)
                    self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0.0


                # EEをゴールへ（IK→qpos）
                result = self.calc_inverse_kinematic(self.ee_goal_pos, rot_weight=0.1)
                q_des = result.qpos[:7].copy()
                self.physics.data.qpos[:7] = q_des
                self.physics.data.qvel[:7] = 0.0

                # ctrlも同期しておく（レンダ時の姿勢を安定させる）
                self.physics.data.ctrl[:] = 0.0
                self.physics.data.ctrl[self.arm_actuator_ids] = q_des

                self.physics.forward()

            # render
            img = self.physics.render(
                height=self.image_size[0],
                width=self.image_size[1],
                camera_id=self.camera_id
            )
            img = np.transpose(img, (2, 0, 1)).astype(np.float32)
            img = torch.from_numpy(img).contiguous()
            
            if not normalize: return img

            if self.task_cfg.backbone_arch == "vjepa2":
                x = img.unsqueeze(0)  # (1,3,H,W)
                pv = self.vjepa2_processor(x, return_tensors="pt")["pixel_values_videos"]
                pv = pv.squeeze(0).squeeze(0)  # (3,256,256)
                return pv

            if self.use_normalize:
                img = self.normalizer.normalize_state(img)
            return img

        finally:
            # restore
            self.physics.data.qpos[:] = qpos_bk
            self.physics.data.qvel[:] = qvel_bk
            self.physics.data.ctrl[:] = ctrl_bk
            self.physics.forward()


    def get_target(self):
        if self.task_name == "reach_no_touch":
            return self.ee_goal_pos
        return self.goal_pos

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

    def _get_goal_propio_reach_no_touch(self):
        # --- backup ---
        qpos_bk = self.physics.data.qpos.copy()
        qvel_bk = self.physics.data.qvel.copy()
        ctrl_bk = self.physics.data.ctrl.copy()
        try:
            # 箱の扱いは get_target_obs() と揃える（必要なら）
            if self.use_box and (not self.robot_only):
                box_pos = self.box_init_pos if self.box_init_pos is not None else self.start_pos
                self.physics.data.qpos[self.start_idx:self.start_idx+3] = box_pos
                self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0])
                self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0.0
            else:
                self.physics.data.qpos[self.start_idx:self.start_idx+3] = np.array([100.0, 100.0, 0.05], dtype=np.float32)
                self.physics.data.qpos[self.start_idx+3:self.start_idx+7] = np.array([1, 0, 0, 0], dtype=np.float32)
                self.physics.data.qvel[self.start_idx:self.start_idx+6] = 0.0

            # EEをゴールへ（IK）
            result = self.calc_inverse_kinematic(self.ee_goal_pos, rot_weight=0.1)
            q_des = result.qpos[:7].copy()

            self.physics.data.qpos[:7] = q_des
            self.physics.data.qvel[:7] = 0.0  # goal速度は0が自然
            self.physics.data.ctrl[:] = 0.0
            self.physics.data.ctrl[self.arm_actuator_ids] = q_des
            self.physics.forward()

            qpos = self.physics.data.qpos[:7].copy()
            qvel = self.physics.data.qvel[:7].copy()
            return np.concatenate([qpos, qvel]).astype(np.float32)

        finally:
            # --- restore ---
            self.physics.data.qpos[:] = qpos_bk
            self.physics.data.qvel[:] = qvel_bk
            self.physics.data.ctrl[:] = ctrl_bk
            self.physics.forward()

       
    def get_target_propio(self):
        """
        Returns goal proprio in the SAME normalization space as training.
        shape: (14,) = [qpos(7), qvel(7)]
        """
        # goal を仮想的に配置して、そのときの関節状態を取得
        if self.task_name == "push_to_goal":
            vec = self._get_goal_obs_vec(self.goal_pos)   # (14,) numpy
        elif self.task_name == "reach_no_touch":
            # reach_no_touch は「EEをゴールへ」なので、
            # get_target_obs() と同じく IK した姿勢から qpos/qvel を作るのが筋
            vec = self._get_goal_propio_reach_no_touch()
        else:
            raise NotImplementedError(f"get_target_propio not implemented for {self.task_name}")

        # torchへ
        vec = torch.from_numpy(vec).float()

        # ここで正規化（学習と同じ空間に合わせる）
        if self.use_normalize:
            qpos = self.normalizer.normalize_propio_pos(vec[:7])
            qvel = self.normalizer.normalize_propio_vel(vec[7:])
            vec = torch.cat([qpos, qvel], dim=0)

        return vec





    def _is_success(self, object_pos):
        if self.task_name == "reach_no_touch":
            tc = self.task_cfg.reach_no_touch
            ee_eps = float(tc.ee_success_eps)
            box_eps = float(tc.box_hold_eps)

            ee = self.get_ee_position()
            ok_ee = np.linalg.norm(ee - self.ee_goal_pos) < ee_eps

            # デフォルトは箱条件なし（robot_only or use_box==False を安全に吸収）
            ok_box = True
            if (not self.robot_only) and self.use_box:
                ok_box = np.linalg.norm(object_pos - self.box_init_pos) < box_eps

            return float(ok_ee and ok_box)
        return float(np.linalg.norm(object_pos - self.goal_pos) < self.success_thresh)





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
        qfrc = self.physics.data.qfrc_actuator[:7].copy()
        afrc = self.physics.data.actuator_force[self.arm_actuator_ids].copy()
        return {
            "location": self.get_ee_position(),
            "qpos": self.physics.data.qpos[:7].copy(),
            "qvel": self.physics.data.qvel[:7].copy(),
            "object_pos": self.get_object_position().copy(),
            "qfrc_actuator": qfrc,
            "actuator_force": afrc,
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
