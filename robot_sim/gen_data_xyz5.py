import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import yaml
import pandas as pd
import imageio
from tqdm import tqdm
from dm_control import mujoco

from robot_sim.franka_envs import FrankaSimEnv
from PIL import Image 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glob


class FrankaDatasetGenerator:
    def __init__(self, config):
        self.config = config
        self.PAIRS = config["pairs"]
        self.MIN_DIST = config["min_dist"]
        self.EPISODES_PER_PAIR = config["episodes_per_pair"]
        self.STEPS_PER_EPISODE = config["steps_per_episode"]
        self.IMAGE_SIZE = tuple(config["image_size"])
        self.MODEL_PATH = config["model_path"]
        self.CAMERA_NAME = config.get("camera_name", "")
        self.IS_VAL = config["is_val"]
        self.START_GOAL_X_RANGE = tuple(config["start_goal_x_range"])
        self.START_GOAL_Y_RANGE = tuple(config["start_goal_y_range"])
        self.START_GOAL_Z_RANGE = tuple(config["start_goal_z_range"])
        self.X_RANGE = tuple(config["x_range"])
        self.Y_RANGE = tuple(config["y_range"])
        self.Z_RANGE = tuple(config["z_range"])
        self.CONFIRM_IK = config["confirm_ik_result"]
        self.CONFIRM_DIST = config["confirm_dist"]
        self.STEPS = config["steps"]
        
        self.SAMPLE_METHOD = config['sample_method']
        self.specify_init_position = config['specify_init_position']
        

        self.SAVE_PATH = (
            f"pldm_envs/franka/presaved_datasets/val_pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}"
            if self.IS_VAL else
            f"pldm_envs/franka/presaved_datasets/pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}"
        )
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        os.environ["MUJOCO_GL"] = "egl"
        self.env = FrankaSimEnv(config)


        if self.CAMERA_NAME == 'default':
            self.camera_id = -1
        else:
            self.camera_id = self.env.physics.model.name2id(self.CAMERA_NAME, mujoco.mjtObj.mjOBJ_CAMERA)
        
        self.count_objective_reached = 0 #手先が目標位置に到達した回数をカウント
        self.count_command_robot = 0 #ロボットに目標位置を指定した回数をカウント
        self.pair_list = self.get_start_goal_pairs()
        self.goal_obs_list = self.get_goal_obs_list()
        self.all_images = []
        self.data_list = []
        self.data_enums = {'target_pos':[], 'contact_count':[]} #目標直交座標を格納
        
        self.mjc_steps = config['steps']
        self.target_sampling_step = config['target_sampling_step']
        self.rot_weight = config['rot_weight']
        
        self.margin_ratio = self.config['margin_ratio']
        self.mgn_x_range = self.shrink_range(self.X_RANGE, ratio=self.margin_ratio)
        self.mgn_y_range = self.shrink_range(self.Y_RANGE, ratio=self.margin_ratio)
        self.mgn_z_range = self.shrink_range(self.Z_RANGE, ratio=self.margin_ratio)
        
        
        self.target_rotmat = None
        print('freeze_quat:', self.config["freeze_quat"])
        if self.config["freeze_quat"]:
            # 姿勢制御: hand を下に向けたい
            x = np.array(self.config['ee_x'])         
            y = np.array(self.config['ee_y'])
            z = np.array(self.config['ee_z'])        

            # 回転行列を構成（各軸を列に並べる）
            self.target_rotmat = np.stack([x, y, z], axis=1)
        
        #ロボットの周波数
        physic_timestep = self.env.physics.model.opt.timestep
        freq = 1 / (self.STEPS * physic_timestep * self.target_sampling_step)
        print('robot frequency:', freq)
        
        #逆運動学計算の確認
        loop = 100
        success_cnt = 0
        for _ in range(loop):
            target_xyz = self.sample_uniform_xyz(self.X_RANGE, self.Y_RANGE, self.Z_RANGE)
            success_cnt += int(self.env.check_ik_accuracy(target_xyz))
        success_rate = success_cnt / loop * 100 
        print(f'ik-calculation success rate: {success_rate:2f}')
        
        self.bluebox_geom_id = self.env.physics.model.name2id("blue_box", mujoco.mjtObj.mjOBJ_GEOM)
        
        self.franka_geom_ids = [
            self.env.physics.model.name2id("link0_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link1_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link2_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link3_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link4_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link5_c0", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link5_c1", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link5_c2", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link6_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link7_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("hand_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("left_finger_0", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("right_finger_0", mujoco.mjtObj.mjOBJ_GEOM),
            # fingertip pads も入れる
            self.env.physics.model.name2id("fingertip_pad_collision_1", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_2", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_3", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_4", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_5", mujoco.mjtObj.mjOBJ_GEOM),
        ]
        
        self.episode_chunk_size = config['episode_chunk_size']
        self.chunk_idx = 0


    def _get_center_of_cube(self):
        x_center = (self.X_RANGE[0] + self.X_RANGE[1]) / 2
        y_center = (self.Y_RANGE[0] + self.Y_RANGE[1]) / 2
        z_center = (self.Z_RANGE[0] + self.Z_RANGE[1]) / 2
        return np.array([x_center, y_center, z_center])
    
    def generate(self):
        #ik の計算結果の確認
        ik_idx = 0
        d_idx = 0
        count_all_timestep = 0
        out_of_bound_count = 0
        for pair_idx, (start_xyz, goal_xyz) in enumerate(tqdm(self.pair_list)):
            ep_idx = 0
            
            pbar = tqdm(total=self.EPISODES_PER_PAIR + 1, desc="Episode Progress")
            while ep_idx < self.EPISODES_PER_PAIR + 1:
                # print('ep_idx:', ep_idx)
                valid_episode = True
                
                init_xyz = self.sample_uniform_xyz(self.mgn_x_range, self.mgn_y_range, self.mgn_z_range)
                init_xyz = self._get_center_of_cube()

                try:
                    result = self.env.calc_inverse_kinematic(init_xyz, target_rotmat=self.target_rotmat)
                except Exception as e:
                    print(f'IK失敗: {e}')
                    valid_episode = False 
                    continue
                
                init_joint = result.qpos[:7]
                self.env.reset_and_place_all(
                    box_pos=start_xyz, 
                    start_marker_pos=start_xyz, 
                    goal_marker_pos=goal_xyz, 
                    init_position=init_joint if self.specify_init_position else None
                    )
                self.env.physics.forward()
                for _ in range(10):
                    self.env.physics.forward()

                episode_images = []
                episode_obs = []
                episode_actions = []
                ik_log = []
                dist_log = []
                dist_xyz_log = []
                
                episode_target_xyz = [] #指定直交座標を記録
                bluebox_contact_count = 0 #blue-box との衝突回数
                
                if ep_idx != 0:
                    obs = np.concatenate(
                            [
                                self.env.physics.data.qpos[:7], 
                                self.env.physics.data.qvel[:7], 
                                self.env.get_ee_position()
                                ]
                        )
                    episode_obs.append(obs.copy())
                    img = self.env.render_image(size = self.IMAGE_SIZE)
                    # self.all_images.append(img)
                    episode_images.append(img)
                    
                    episode_target_xyz.append(self.env.get_ee_position())
                
                cur_yaw = None #self.SAMPLE_METHOD == 'direction'でvonmises分布の場合の水平角の初期化
                for idx in range(self.STEPS_PER_EPISODE):
                    if idx % self.target_sampling_step == 0:
                        self.count_command_robot += 1
                        if idx == 0: #初期IKの
                            target_xyz = init_xyz.copy()
                        else:
                            if self.SAMPLE_METHOD == 'uniform':
                                target_xyz = self.sample_uniform_xyz(self.mgn_x_range, self.mgn_y_range, self.mgn_z_range)
                                
                            elif self.SAMPLE_METHOD == 'direction':
                                current_pos = self.env.get_ee_position()
                                dist_range = tuple(self.config['sample_direction']['dist_range'])
                                max_loop = self.config['sample_direction']['max_loop']
                                freeze_z = self.config['sample_direction']['freeze_z']
                                kappa = self.config['sample_direction']['vonmises']['kappa']

                                target_xyz, cur_yaw = self.sample_direc_xyz(current_pos, dist_range, freeze_z=freeze_z, max_loop=max_loop, prev_yaw=cur_yaw, kappa=kappa)
                                
                            elif self.SAMPLE_METHOD == 'uniform_constrain':
                                current_pos = self.env.get_ee_position()
                                
                                max_dist = self.config['sample_uniform_constrain']['max_dist']
                                max_loop = self.config['sample_uniform_constrain']['max_loop']
                                target_xyz = self.sample_uniform_constrain_xyz(current_pos, max_dist, max_loop = max_loop)
                            else:
                                raise ValueError(f"Unknown SAMPLE_METHOD '{self.SAMPLE_METHOD}'. Expected 'uniform' or 'direction'.")
                    
                    try:
                        mjc_tol = float(self.config['tol'])
                        joint_angles, ee_pos, dist_steps, objective_reached = self.env.step_xyz(
                            target_xyz, 
                            target_rotmat=self.target_rotmat,
                            steps=self.mjc_steps, 
                            tol=mjc_tol,
                            rot_weight=self.rot_weight
                            )
                        # if not self.is_within_bounds(ee_pos, self.X_RANGE, self.Y_RANGE):
                        #     print('OOOOOO')
                        #     valid_episode = False
                    except Exception as e:
                        print(f'IK失敗: {e}')
                        valid_episode = False
                        if idx % self.target_sampling_step == 0:
                            self.count_command_robot -= 1
                        continue
                    
                    #衝突をカウント
                    ncon = self.env.physics.data.ncon
                    for i in range(ncon):
                        contact = self.env.physics.data.contact[i]
                        geom1 = contact.geom1
                        geom2 = contact.geom2

                        if (
                            (geom1 == self.bluebox_geom_id and geom2 in self.franka_geom_ids) or
                            (geom2 == self.bluebox_geom_id and geom1 in self.franka_geom_ids)
                        ):
                            bluebox_contact_count += 1
                    

                    
                    # 目標到達回数をカウント(/ 目標指示回数)
                    if (idx + 1) % self.target_sampling_step == 0:
                        if objective_reached:
                            self.count_objective_reached += 1
                            
                    if ep_idx != 0 and valid_episode:
                        command_target = int(idx % self.target_sampling_step == 0)
                        ik_log.append({'target_xyz': target_xyz, 'ee_pos': ee_pos})
                        dist_log.append(
                            {
                                'dist_start' : np.linalg.norm(dist_steps[0]), 
                                'dist_end' : np.linalg.norm(dist_steps[-1]), 
                                'command_target' : command_target
                                }
                            )
                        dist_xyz_log.append(
                                {
                                    'dist_start_x' : dist_steps[0][0], 
                                    'dist_goal_x' : dist_steps[-1][0], 
                                    'dist_start_y' : dist_steps[0][1],
                                    'dist_goal_y' : dist_steps[-1][1],
                                    'dist_start_z' : dist_steps[0][2],
                                    'dist_goal_z' : dist_steps[-1][2],
                                }
                            )

                        #手先が範囲外に出る回数をカウント
                        out_of_bound_count += int(not self.is_within_bounds(self.env.get_ee_position(), self.X_RANGE, self.Y_RANGE))
                        count_all_timestep += 1
                        
                        
                        action = joint_angles.copy()
                        obs = np.concatenate(
                                [
                                    self.env.physics.data.qpos[:7], 
                                    self.env.physics.data.qvel[:7], 
                                    self.env.get_ee_position()
                                ]
                            )
                        
                        episode_obs.append(obs.copy())
                        episode_actions.append(action.copy())
                        img = self.env.physics.render(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], camera_id=self.camera_id)
                        # self.all_images.append(img)
                        episode_images.append(img)
                        
                        episode_target_xyz.append(target_xyz.copy())
                        
                if ep_idx != 0 and valid_episode:
                    self.data_list.append(
                                {
                                "observations": np.array(episode_obs),
                                "actions": np.array(episode_actions),
                                "goal_obs": self.goal_obs_list[pair_idx][0].copy(),
                                "map_idx": pair_idx,
                            }
                        )
                    self.all_images.extend(episode_images)
                    
                    if (
                        self.episode_chunk_size is not None
                        and len(self.data_list) >= self.episode_chunk_size
                    ):
                        self._save_chunk()

                    
                    self.data_enums['target_pos'].append(np.array(episode_target_xyz))
                    self.data_enums['contact_count'].append(bluebox_contact_count)
                
                    if self.CONFIRM_IK:
                        self.confirm_target_actual_pos(ik_log, ik_idx)
                        ik_idx += 1
                    if self.CONFIRM_DIST:
                        self.confirm_target_actual_dist(dist_log, d_idx)
                        self.confirm_target_actual_dist_xyz(dist_xyz_log, d_idx)
                        d_idx += 1
                if ep_idx == 0: ep_idx += 1
                else:
                    if valid_episode: 
                        ep_idx += 1
                        pbar.update(1)
                    else: 
                        valid_episode = True
            pbar.close()



        reach_success_rate = (
            self.count_objective_reached / self.count_command_robot * 100 
            if self.count_command_robot > 0 
            else 0
            )
        out_of_bound_rate = (
            out_of_bound_count / count_all_timestep * 100
            if count_all_timestep > 0 
            else 0
        )
        print(f'percentage of targets reached : {reach_success_rate:2f}')
        print(f'percentage of out-of-bound : {out_of_bound_rate:2f}')

        
    def get_start_goal_pairs(self):
        print("📦 Sampling XYZ pairs...")
        pair_list = []
        
        while len(pair_list) < self.PAIRS:
            start = self.sample_goal_xyz()
            goal = self.sample_goal_xyz()
            if np.linalg.norm(start - goal) >= self.MIN_DIST:
                pair_list.append((start, goal))
        
        return pair_list

    def get_goal_obs_list(self):
        print("🎯 Computing goal observations...")
        goal_obs_list = []
        for start_pos, goal_pos in self.pair_list:
            self.env.reset_and_place_all(box_pos=goal_pos, start_marker_pos=start_pos, goal_marker_pos=goal_pos)
            
            offset = np.array(self.config['goal_offset'])
            self.env.set_xyz(goal_pos + offset)
            img = self.env.render_image(size=self.IMAGE_SIZE)
            goal_obs = np.concatenate([self.env.physics.data.qpos[:7], self.env.physics.data.qvel[:7], self.env.get_ee_position()])
            goal_obs_list.append((goal_obs.copy(), img.copy()))

        return goal_obs_list

    def sample_xyz(self, range_tuple):
        return np.array([np.random.uniform(*r) for r in range_tuple])
    
    def sample_uniform_xyz(self, x_range, y_range, z_range):
        return self.sample_xyz([x_range, y_range, z_range])

    def sample_start_xyz(self):
        return self.sample_xyz([self.START_GOAL_X_RANGE, self.START_GOAL_Y_RANGE, self.START_GOAL_Z_RANGE])
    
    def sample_goal_xyz(self):
        return self.sample_xyz([self.START_GOAL_X_RANGE, self.START_GOAL_Y_RANGE, self.START_GOAL_Z_RANGE])
    
    def sample_direc_xyz(self, current_pos, dist_range, freeze_z=True, max_loop=100, prev_yaw=None, kappa=3.0):
        prob_dist = self.config['sample_direction']['prob_dist']

        flag = False
        for _ in range(max_loop):
            
            if prob_dist == 'uniform':
                cur_yaw = np.random.uniform(- np.pi, np.pi)
            elif prob_dist == 'vonmises':
                if prev_yaw is None:
                    cur_yaw = np.random.uniform(- np.pi, np.pi)
                else:
                    cur_yaw = np.random.vonmises(mu=prev_yaw, kappa=kappa)
            else:
                raise ValueError(f"Unknown prob_dist '{prob_dist}'. Expected 'uniform' or 'vonmises'.")
            
            # cur_yaw = (cur_yaw + np.pi) % (2 * np.pi) - np.pi
            
            if not freeze_z:
                pitch = np.random.uniform(- np.pi / 2, np.pi / 2)
            else:
                pitch = 0
            
            
            dist = np.random.uniform(*dist_range)
            
            dx = dist * np.cos(pitch) * np.cos(cur_yaw)
            dy = dist * np.cos(pitch) * np.sin(cur_yaw)
            dz = dist * np.sin(pitch)
            
            delta = np.array([dx, dy, dz])
            new_pos = current_pos + delta
            

            if self.is_within_bounds(new_pos, self.mgn_x_range, self.mgn_y_range, self.mgn_z_range):
                flag = True
                return new_pos, cur_yaw

        if not flag:
            # print('max_loop reached in sample_direc_xyz')
            if not self.is_within_bounds(new_pos, self.X_RANGE, self.Y_RANGE):
                cur_yaw = cur_yaw + np.pi 
                cur_yaw = (cur_yaw + np.pi) % (2 * np.pi) - np.pi
                # cur_yaw = np.random.uniform(- np.pi, np.pi)
                pass
            x = np.clip(new_pos[0], *self.mgn_x_range)
            y = np.clip(new_pos[1], *self.mgn_y_range)
            z = np.clip(new_pos[2], *self.mgn_z_range)
            new_pos = np.array([x, y, z])
        
        return new_pos, cur_yaw

    def shrink_range(self, range_tuple, ratio=0.8):
        min_val, max_val = range_tuple
        center = (min_val + max_val) / 2 
        half_width = (max_val - min_val) / 2
        new_half_width = half_width * ratio 
        return (center - new_half_width, center + new_half_width)

    
    def sample_uniform_constrain_xyz(self, current_pos, max_dist = 0.1, max_loop = 100):
        for _ in range(max_loop):
            target_xyz = self.sample_xyz([self.X_RANGE, self.Y_RANGE, self.Z_RANGE])
            dist = np.linalg.norm(target_xyz - current_pos)
            if dist < max_dist:
                break
            else:
                target_xyz = current_pos.copy()
                x = np.clip(target_xyz[0], *self.X_RANGE)
                y = np.clip(target_xyz[1], *self.Y_RANGE)
                z = np.clip(target_xyz[2], *self.Z_RANGE)
                target_xyz = np.array([x, y, z])
            return target_xyz

    
    def is_within_bounds(self, pos, x_range, y_range, z_range=None):
        x, y, z = pos 
        if z_range is not None:
            return (x_range[0] <= x <= x_range[1] and
                    y_range[0] <= y <= y_range[1] and
                    z_range[0] <= z <= z_range[1])
        else:
            return (x_range[0] <= x <= x_range[1] and
                    y_range[0] <= y <= y_range[1])

    
    def confirm_target_actual_pos(self, ik_log, ik_idx):
        os.makedirs('robot_sim/data_value/ik_value', exist_ok=True)
        df = pd.DataFrame(ik_log)
        df.to_csv(f'robot_sim/data_value/ik_value/ik_log{ik_idx}.csv', index=False)
    
    
    def confirm_target_actual_dist(self, dist_log, d_idx):
        os.makedirs('robot_sim/data_value/dist_value', exist_ok=True)
        df = pd.DataFrame(dist_log)
        df.to_csv(f'robot_sim/data_value/dist_value/dist_log{d_idx}.csv', index=False)
    
    def confirm_target_actual_dist_xyz(self, dist_xyz_log, d_idx):
        os.makedirs('robot_sim/data_value/dist_xyz_value', exist_ok=True)
        df = pd.DataFrame(dist_xyz_log)
        df.to_csv(f'robot_sim/data_value/dist_xyz_value/dist_xyz_log{d_idx}.csv', index=False)
        
    # def data_save(self):
    #     torch.save(self.data_list, os.path.join(self.SAVE_PATH, "data.p"))
    #     np.save(os.path.join(self.SAVE_PATH, "images.npy"), np.array(self.all_images, dtype=np.uint8))
    #     goal_imgs = np.stack([g[1] for g in self.goal_obs_list])
    #     np.save(os.path.join(self.SAVE_PATH, "goal_images.npy"), goal_imgs)
    #     torch.save({"pair_list": self.pair_list}, os.path.join(self.SAVE_PATH, "pair_info.p"))
    #     print("✅ Done generating dataset!")
    
    


    # def data_save(self, chunk_size=1000):
    #     torch.save(self.data_list, os.path.join(self.SAVE_PATH, "data.p"))

    #     # --- chunk 保存 ---
    #     chunk_dir = os.path.join(self.SAVE_PATH, "image_chunks")
    #     os.makedirs(chunk_dir, exist_ok=True)
        
    #     num_images = len(self.all_images)
    #     print(f"Saving {num_images} images in chunks...")

    #     for i in range(0, num_images, chunk_size):
    #         file_path = os.path.join(chunk_dir, f"images_chunk_{i//chunk_size}.npy")
    #         chunk = self.all_images[i:i+chunk_size]
    #         chunk_arr = np.array(chunk, dtype=np.uint8)
    #         np.save(file_path, chunk_arr)
        
    #     print("✅ Image chunks saved.")

    #     # goal images
    #     goal_imgs = np.stack([g[1] for g in self.goal_obs_list])
    #     np.save(os.path.join(self.SAVE_PATH, "goal_images.npy"), goal_imgs)

    #     torch.save({"pair_list": self.pair_list}, os.path.join(self.SAVE_PATH, "pair_info.p"))

    #     print("✅ Done generating dataset!")

    # def merge_chunks(self):
        
    #     chunk_dir = os.path.join(self.SAVE_PATH, "image_chunks")
    #     chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "images_chunk_*.npy")))

    #     print(f"Found {len(chunk_files)} chunk files.")

    #     arrays = []
    #     for f in chunk_files:
    #         arr = np.load(f)
    #         arrays.append(arr)
    #         print(f"Loaded {f} with shape {arr.shape}")

    #     images = np.concatenate(arrays, axis=0)
    #     np.save(os.path.join(self.SAVE_PATH, "images.npy"), images)

    #     print("✅ Merged all chunks into images.npy")


    def _save_chunk(self):
        chunk_dir = os.path.join(self.SAVE_PATH, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        data_chunk_dir = os.path.join(chunk_dir, "data")
        os.makedirs(data_chunk_dir, exist_ok=True)
        image_chunk_dir = os.path.join(chunk_dir, "image")
        os.makedirs(image_chunk_dir, exist_ok=True)
        

        # --- save data_list ---
        data_path = os.path.join(
            data_chunk_dir, f"data_chunk_{self.chunk_idx}.pt"
        )
        torch.save(self.data_list, data_path)

        # --- save images ---
        if len(self.all_images) > 0:
            images_arr = np.array(self.all_images, dtype=np.uint8)
            img_path = os.path.join(
                image_chunk_dir, f"images_chunk_{self.chunk_idx}.npy"
            )
            np.save(img_path, images_arr)

        print(f"✅ Saved chunk {self.chunk_idx} ({len(self.data_list)} episodes)")

        # --- flush ---
        self.data_list = []
        self.all_images = []
        self.chunk_idx += 1



    def merge_chunks(self):
        chunk_dir = os.path.join(self.SAVE_PATH, "chunks")
        data_chunk_dir = os.path.join(chunk_dir, "data")
        image_chunk_dir = os.path.join(chunk_dir, "image")

        # --- data を読み込み ---
        data_files = sorted(glob.glob(os.path.join(data_chunk_dir, "data_chunk_*.pt")))
        merged_data_list = []
        for f in data_files:
            chunk_data = torch.load(f, weights_only=False)
            merged_data_list.extend(chunk_data)
            print(f"Loaded {f} with {len(chunk_data)} episodes")

        # --- image を読み込み ---
        image_files = sorted(glob.glob(os.path.join(image_chunk_dir, "images_chunk_*.npy")))
        images_list = []
        for f in image_files:
            arr = np.load(f)
            images_list.append(arr)
            print(f"Loaded {f} with shape {arr.shape}")

        if len(images_list) > 0:
            merged_images = np.concatenate(images_list, axis=0)
        else:
            merged_images = np.array([], dtype=np.uint8)

        # --- 保存 ---
        torch.save(merged_data_list, os.path.join(self.SAVE_PATH, "data.p"))
        np.save(os.path.join(self.SAVE_PATH, "images.npy"), merged_images)

        # goal_images はそのまま
        goal_imgs = np.stack([g[1] for g in self.goal_obs_list])
        np.save(os.path.join(self.SAVE_PATH, "goal_images.npy"), goal_imgs)

        torch.save({"pair_list": self.pair_list}, os.path.join(self.SAVE_PATH, "pair_info.p"))

        print("✅ All chunks merged and final dataset saved!")



    
    def confirm_data(self):
        FILE = 'data.p'
        file_path = os.path.join(self.SAVE_PATH, FILE)
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

        images = np.load(os.path.join(self.SAVE_PATH, "images.npy"))
        pixels = [img.mean(axis=(0, 1)) for img in images]  # 各画像の平均 [R,G,B]
        df = pd.DataFrame(pixels, columns=["mean_R", "mean_G", "mean_B"])
        df.to_csv(os.path.join(output_dir, "image_pixels_summary.csv"), index=False)
        
        
        contact_counts = self.data_enums["contact_count"]
        steps_per_episode_list = [self.STEPS_PER_EPISODE] * len(contact_counts)

        contact_per_timestep = [
            cnt / self.STEPS_PER_EPISODE for cnt in contact_counts
        ]

        # DataFrame にまとめる
        df = pd.DataFrame({
            "bluebox_contact_count": contact_counts,
            "timestep": steps_per_episode_list,
            "contact_per_timestep": contact_per_timestep
        })

        # CSV に保存
        df.to_csv(os.path.join(output_dir, "bluebox_contact_count.csv"), index=False)
        print("✅ Saved bluebox contact count CSV with steps and per-timestep values!")



        
        
    def make_video(self):
        DATA_PATH = self.SAVE_PATH
        SAVE_DIR = "robot_sim/analyze/video"

        timestep = self.env.physics.model.opt.timestep
        FPS = 1 / (timestep * self.mjc_steps)

        #LOAD
        print("📦 Loading dataset...")
        data = torch.load(os.path.join(DATA_PATH, "data.p"), map_location="cpu", weights_only=False)
        images = np.load(os.path.join(DATA_PATH, "images.npy"))

        #SETUP
        os.makedirs(SAVE_DIR, exist_ok=True)
        frames_per_episode = len(data[0]["observations"])  # T+1
        print(f"🎞️ Frames per episode: {frames_per_episode}")
        print(f"📁 Saving videos to: {SAVE_DIR}")

        #GENERATE VIDEOS
        start_idx = 0
        for i, episode in enumerate(tqdm(data, desc="🎬 Saving episodes as videos")):
            end_idx = start_idx + frames_per_episode
            episode_frames = images[start_idx:end_idx]
            save_path = os.path.join(SAVE_DIR, f"episode_{i:03d}.mp4")
            imageio.mimsave(save_path, episode_frames, fps=FPS)
            start_idx = end_idx

        print("✅ All videos are saved.")
        
        
    def confirm_data_architecture(self):
        data_p_path = os.path.join(self.SAVE_PATH, "data.p")
        images_path = os.path.join(self.SAVE_PATH, "images.npy")
        goal_img_path = os.path.join(self.SAVE_PATH, "goal_images.npy")
        
        log_dir = 'robot_sim/analyze/data_architecture'
        os.makedirs(log_dir, exist_ok = True)
        log_path = os.path.join(log_dir, 'data_architecture.txt')

        
        with open(log_path, 'w') as logfile:
            print(f"=== Checking dataset in: {self.SAVE_PATH} ===\n", file=logfile)

            # --- Check data.p ---
            print("[1] Checking data.p...", file=logfile)
            try:
                data = torch.load(data_p_path, map_location="cpu", weights_only=False)
            except Exception as e:
                print(f"❌ Failed to load data.p: {e}", file=logfile)
                return None, None

            if not isinstance(data, list):
                print("❌ data.p is not a list.", file=logfile)
                return None, None

            num_episodes = len(data)
            print(f"✅ Loaded data.p with {num_episodes} episodes.", file=logfile)

            example = data[0]
            print("\n[1.1] Keys and shapes in one episode:", file=logfile)
            for k, v in example.items():
                if isinstance(v, np.ndarray):
                    print(f" - {k}: shape {v.shape}", file=logfile)
                else:
                    print(f" - {k}: type {type(v)}", file=logfile)


            actions = example["actions"]
            observations = example["observations"]

            if observations.shape[0] != actions.shape[0] + 1:
                print("❌ Mismatch: observations should have one more timestep than actions.", file=logfile)
            else:
                print("✅ actions and observations length match (T+1 vs T).", file=logfile)

            if "goal_obs" in example:
                goal_obs = example["goal_obs"]
                if goal_obs.shape != observations[0].shape:
                    print(f"❌ goal_obs shape mismatch: got {goal_obs.shape}, expected {observations[0].shape}", file=logfile)
                else:
                    print("✅ goal_obs is present and shape is valid.", file=logfile)


            # --- EE位置の確認 ---
            print("\n[1.3] Checking end-effector xyz values:", file=logfile)

            ee_xyz_all = []

            for ep in data:
                obs = ep["observations"]  # shape: (T+1, obs_dim)
                if obs.shape[1] >= 17:  # 確保のため
                    ee_xyz = obs[:, -3:]  # 最後の3次元が xyz
                    ee_xyz_all.append(ee_xyz)
                else:
                    print("⚠️ obs dim too small to include ee xyz:", obs.shape[1], file=logfile)

            if ee_xyz_all:
                ee_xyz_all = np.concatenate(ee_xyz_all, axis=0)  # 全時刻全エピソードの xyz

                print(f" - Total ee_xyz samples: {ee_xyz_all.shape[0]}", file=logfile)
                print(f" - x: mean={ee_xyz_all[:,0].mean():.3f}, min={ee_xyz_all[:,0].min():.3f}, max={ee_xyz_all[:,0].max():.3f}", file=logfile)
                print(f" - y: mean={ee_xyz_all[:,1].mean():.3f}, min={ee_xyz_all[:,1].min():.3f}, max={ee_xyz_all[:,1].max():.3f}", file=logfile)
                print(f" - z: mean={ee_xyz_all[:,2].mean():.3f}, min={ee_xyz_all[:,2].min():.3f}, max={ee_xyz_all[:,2].max():.3f}", file=logfile)
            else:
                print("❌ Could not extract any ee xyz values.", file=logfile)




            # --- Check images.npy ---
            print("\n[2] Checking images.npy...", file=logfile)
            try:
                images = np.load(images_path)
                print(f"📐 images shape: {images.shape}", file=logfile)

            except Exception as e:
                print(f"❌ Failed to load images.npy: {e}", file=logfile)
                return data, None

            T_plus_1 = observations.shape[0]
            expected_images = num_episodes * T_plus_1

            if images.shape[0] != expected_images:
                print(f"❌ images.npy has {images.shape[0]} images, expected {expected_images}", file=logfile)
            else:
                print("✅ images.npy shape is consistent with data.p", file=logfile)

            # --- Save first 5 images ---
            print("\n[3] Saving first 5 images from images.npy...", file=logfile)
            save_dir = "robot_sim/analyze/operation"
            os.makedirs(save_dir, exist_ok=True)
            for i in range(min(5, images.shape[0])):
                Image.fromarray(images[i]).save(os.path.join(save_dir, f"img_{i}.png"))
            print(f"✅ Saved first 5 images to {save_dir}", file=logfile)

            # --- Save goal images ---
            if os.path.exists(goal_img_path):
                print("\n[4] Saving goal_images.npy as images...", file=logfile)
                try:
                    goal_imgs = np.load(goal_img_path)
                    print(f"📐 goal_images shape: {goal_imgs.shape}", file=logfile)

                    save_goal_dir = "robot_sim/analyze/goal"
                    os.makedirs(save_goal_dir, exist_ok=True)

                    if goal_imgs.ndim == 4:
                        for i in range(min(5, len(goal_imgs))):
                            Image.fromarray(goal_imgs[i]).save(os.path.join(save_goal_dir, f"goal_img_{i}.png"))
                        print(f"✅ Saved goal images to {save_goal_dir}", file=logfile)
                    else:
                        print(f"❌ Unexpected goal_images shape: {goal_imgs.shape}", file=logfile)
                except Exception as e:
                    print(f"❌ Failed to save goal image: {e}", file=logfile)
            else:
                print("⚠️ goal_images.npy not found.", file=logfile)




    def confirm_endeffector_trajectory(self, axes: str, visualize_target_trj=True):
        axes_to_num = {'x':0, 'y':1, 'z':2}
        axis_num = [axes_to_num[axes[0]], axes_to_num[axes[1]]]

        ranges = {
            'x' : self.config['x_range'],
            'y' : self.config['y_range'],
            'z' : self.config['z_range'],
        }

        xlim = ranges[axes[0]]
        if abs(xlim[0] - xlim[1]) < 1e-3:
            xlim[0] -= 0.2
            xlim[1] += 0.2
        ylim = ranges[axes[1]]
        if abs(ylim[0] - ylim[1]) < 1e-3:
            ylim[0] -= 0.2
            ylim[1] += 0.2



        for ep_idx, episode in enumerate(self.data_list):
            fig, ax = plt.subplots(figsize=(6, 6))
            
            if visualize_target_trj:
                target_xyz = self.data_enums['target_pos'][ep_idx]
                tx = target_xyz[:, axis_num[0]]
                ty = target_xyz[:, axis_num[1]]
                M = len(tx)
                t_norm_target = np.linspace(0, 1, M)

                ax.scatter(
                    tx, ty,
                    c=t_norm_target,
                    cmap='Reds', 
                    marker='x',
                    s=2,
                    label='Target Positions',
                    alpha=0.8,
                    zorder=4
                )
                ax.plot(
                    tx, ty,
                    color='gray',
                    linewidth=1,
                    alpha=0.8,
                    zorder=3,
                    label='Target Path'
                )
            
            
            obs = episode["observations"]
            ee_xyz = obs[:, -3:]


            x = ee_xyz[:, axis_num[0]]
            y = ee_xyz[:, axis_num[1]]
            

            N = len(x)
            t_norm = np.linspace(0, 1, N)

            scatter = ax.scatter(
                x, y,
                c=t_norm,
                cmap='viridis',
                s=10,
                alpha=0.8,
                zorder=2
            )

            ax.plot(
                x, y,
                color='gray',
                linewidth=1,
                alpha=0.5,
                zorder=1
            )
            

            
            
            # --- 矩形を追加 ---
            from matplotlib.patches import Rectangle
            if axes[0] == 'x':
                rect_width = self.mgn_x_range[1] - self.mgn_x_range[0]
            elif axes[0] == 'y':
                rect_width = self.mgn_y_range[1] - self.mgn_y_range[0]
            elif axes[0] == 'z':
                rect_width = self.mgn_z_range[1] - self.mgn_z_range[0]
                
            if axes[1] == 'x':
                rect_height = self.mgn_x_range[1] - self.mgn_x_range[0]
            elif axes[1] == 'y':
                rect_height = self.mgn_y_range[1] - self.mgn_y_range[0]
            elif axes[1] == 'z':
                rect_height = self.mgn_z_range[1] - self.mgn_z_range[0]
                
            rect = Rectangle(
                (self.mgn_x_range[0], self.mgn_y_range[0]),
                rect_width,
                rect_height,
                linewidth=1,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                alpha=0.3,
                zorder=3
            )
            ax.add_patch(rect)

            # カラーバーを追加
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Time Progress (normalized)")

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))

            ax.set_xlabel(axes[0].upper())
            ax.set_ylabel(axes[1].upper())
            ax.set_title(f"EE Trajectory - Episode {ep_idx}")
            ax.set_aspect('equal', adjustable='box')

            SAVE_DIR = f"robot_sim/analyze/endeffector_trajectory/{axes}"
            os.makedirs(SAVE_DIR, exist_ok=True)
            save_path = os.path.join(SAVE_DIR, f"ee_trajectory_ep{ep_idx}.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

        return






if __name__ == "__main__":
    
    with open("robot_sim/config.yaml", "r") as f:
        config = yaml.safe_load(f)


    dataset_generator = FrankaDatasetGenerator(config)
    dataset_generator.generate()
    # dataset_generator.data_save(chunk_size = config['chunk_size'])
    if len(dataset_generator.data_list) > 0:
        dataset_generator._save_chunk()
        
        
    dataset_generator.merge_chunks()
    
    dataset_generator.confirm_data_architecture()
    

    dataset_generator.confirm_data()
    if config['make_video']: 
        dataset_generator.make_video()
    if config['confirm_ee_trajectory']:
        dataset_generator.confirm_endeffector_trajectory('xy', config['visualize_target_trajectory'])
        dataset_generator.confirm_endeffector_trajectory('xz', config['visualize_target_trajectory'])