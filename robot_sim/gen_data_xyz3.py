import os
import numpy as np
import torch
import yaml
import pandas as pd
import imageio
from tqdm import tqdm
from dm_control import mujoco
from dm_control.utils.inverse_kinematics import qpos_from_site_pose

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
        self.TOL = float(config["tol"])
        
        self.SAMPLE_METHOD = config['sample_method']
        self.specify_init_position = config['specify_init_position']
        

        self.SAVE_PATH = (
            f"pldm_envs/franka/presaved_datasets/val_pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}"
            if self.IS_VAL else
            f"pldm_envs/franka/presaved_datasets/pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}"
        )
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        os.environ["MUJOCO_GL"] = "egl"
        self.physics = mujoco.Physics.from_xml_path(self.MODEL_PATH)

        if self.CAMERA_NAME == 'default':
            self.camera_id = -1
        else:
            self.camera_id = self.physics.model.name2id(self.CAMERA_NAME, mujoco.mjtObj.mjOBJ_CAMERA)
        
        self.count_objective_reached = 0 #手先が目標位置に到達した回数をカウント
        self.count_command_robot = 0 #ロボットに目標位置を指定した回数をカウント
        self.pair_list = self.get_start_goal_pairs()
        self.goal_obs_list = self.get_goal_obs_list()
        self.all_images = []
        self.data_list = []
        
        self.mjc_steps = config['steps']
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
    
    def calc_inverse_kinematic(self, target_xyz):
        joint_names = [f"joint{i}" for i in range(1, 8)]
        result = qpos_from_site_pose(
            self.physics,
            site_name="ee_target",
            target_pos=target_xyz,
            joint_names=joint_names
        )
        return result

    def get_ee_position(self):
        return self.physics.named.data.site_xpos['ee_target'].copy()
    
    def move_franka_to_xyz(self, target_xyz, steps=200, tol=1e-3):
        result = self.calc_inverse_kinematic(target_xyz)
        
        if not result.success:
            raise ValueError("IK failed!")
        # physics.data.qpos[:7] = result.qpos[:7]
        # physics.data.qvel[:7] = 0 
        
        joint_angles = result.qpos[:7]

        dist_steps = []
        ee_pos = self.physics.named.data.site_xpos['ee_target']
        dist = np.linalg.norm(ee_pos - target_xyz)
        dx, dy, dz = np.abs(ee_pos - target_xyz)
        dist_steps.append(
                {
                    'all': dist, 
                    'dx' : dx, 
                    'dy' : dy, 
                    'dz' : dz
                }
            )
        
        # self.count_command_robot += 1
        objective_reached = False
        for _ in range(steps):
            self.physics.data.ctrl[:7] = joint_angles.copy()
            self.physics.step()
            ee_pos = self.physics.named.data.site_xpos['ee_target']
            
            #速度計算
            # site_id = self.physics.model.name2id('ee_target', 'site')
            # body_id = self.physics.model.site_bodyid[site_id]
            # vel_ee = self.physics.data.cvel[body_id]  # [:3]:角速度, [3:]:線速度
            # linear_vel = vel_ee[3:]
            # abs_vel = np.linalg.norm(linear_vel)

            dist = np.linalg.norm(ee_pos - target_xyz)
            dx, dy, dz = np.abs(ee_pos - target_xyz)
            dist_steps.append(
                    {
                        'all': dist, 
                        'dx' : dx, 
                        'dy' : dy, 
                        'dz' : dz
                    }
                )
            
            
            if dist < tol:
                # print('objective reached!!')
                # self.count_objective_reached += 1
                objective_reached = True
                break

        site_pos = ee_pos.copy()
        return joint_angles, target_xyz, site_pos, dist_steps, objective_reached
    
    
    def generate_all_pairs(self):
        print(f'SAMPLE METHOD : {self.SAMPLE_METHOD}')
        
        
        #ik の計算結果の確認
        ik_idx = 0
        d_idx = 0
        
        mjc_tol = float(config['tol'])
    
        for pair_idx, (start_xyz, goal_xyz) in enumerate(tqdm(self.pair_list)):
            for _ in range(self.EPISODES_PER_PAIR):
                self.physics.reset()
                
                init_xyz = self.sample_uniform_xyz()
                result = self.calc_inverse_kinematic(init_xyz)
                
                init_joint = result.qpos[:7]
                self.reset_and_place_all(
                    box_pos=start_xyz, 
                    start_marker_pos=start_xyz, 
                    goal_marker_pos=goal_xyz, 
                    init_position=init_joint if self.specify_init_position else None
                    )
                self.physics.forward()


                episode_obs = []
                episode_actions = []
                ik_log = []
                dist_log = []
                dist_xyz_log = []
                
                obs = np.concatenate([self.physics.data.qpos[:7], self.physics.data.qvel[:7], self.get_ee_position()])
                episode_obs.append(obs.copy())
                img = self.physics.render(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], camera_id=self.camera_id)
                self.all_images.append(img)
                
                for idx in range(self.STEPS_PER_EPISODE):
                    if idx % self.target_sampling_step == 0:
                        self.count_command_robot += 1
                        
                        if self.SAMPLE_METHOD == 'uniform':
                            target_xyz = self.sample_uniform_xyz()
                        elif self.SAMPLE_METHOD == 'direction':
                            current_pos = self.get_ee_position()
                            dist_range = tuple(self.config['sample_direction']['dist_range'])
                            max_loop = self.config['sample_direction']['max_loop']
                            freeze_z = self.config['sample_direction']['freeze_z']

                            target_xyz = self.sample_direc_xyz(current_pos, dist_range, freeze_z=freeze_z, max_loop=max_loop)
                        elif self.SAMPLE_METHOD == 'uniform_constrain':
                            current_pos = self.get_ee_position()
                            
                            max_dist = self.config['sample_uniform_constrain']['max_dist']
                            max_loop = self.config['sample_uniform_constrain']['max_loop']
                            target_xyz = self.sample_uniform_constrain_xyz(current_pos, max_dist, max_loop = max_loop)
                        else:
                            raise ValueError(f"Unknown SAMPLE_METHOD '{self.SAMPLE_METHOD}'. Expected 'uniform' or 'direction'.")
                    
                    
                    try:
                        joint_angles, target_xyz, ee_pos, dist_seq, objective_reached = self.move_franka_to_xyz(target_xyz, self.mjc_steps, mjc_tol)
                    except Exception as e:
                        print(f'IK失敗: {e}')
                        continue
                    
                    # 目標到達回数をカウント(/ 目標指示回数)
                    if (idx + 1) % self.target_sampling_step == 0:
                        if objective_reached:
                            self.count_objective_reached += 1
                    
                    command_target = int(idx % self.target_sampling_step == 0)
                    ik_log.append({'target_xyz': target_xyz, 'ee_pos': ee_pos})
                    dist_log.append(
                        {
                            'dist_start' : dist_seq[0]['all'], 
                            'dist_end' : dist_seq[-1]['all'], 
                            'command_target' : command_target
                            }
                        )
                    dist_xyz_log.append(
                            {
                                'dist_start_x' : dist_seq[0]['dx'], 
                                'dist_goal_x' : dist_seq[-1]['dx'], 
                                'dist_start_y' : dist_seq[0]['dy'],
                                'dist_goal_y' : dist_seq[-1]['dy'],
                                'dist_start_z' : dist_seq[0]['dz'],
                                'dist_goal_z' : dist_seq[-1]['dz'],
                            }
                        )

                        
                    action = joint_angles.copy()
                    obs = np.concatenate([self.physics.data.qpos[:7], self.physics.data.qvel[:7], self.get_ee_position()])
                    episode_obs.append(obs.copy())
                    episode_actions.append(action.copy())
                    img = self.physics.render(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], camera_id=self.camera_id)
                    self.all_images.append(img)

                self.data_list.append({
                    "observations": np.array(episode_obs),
                    "actions": np.array(episode_actions),
                    "goal_obs": self.goal_obs_list[pair_idx][0].copy(),
                    "map_idx": pair_idx,
                })


                if self.CONFIRM_IK:
                    self.confirm_target_actual_pos(ik_log, ik_idx)
                    ik_idx += 1
                if self.CONFIRM_DIST:
                    self.confirm_target_actual_dist(dist_log, d_idx)
                    self.confirm_target_actual_dist_xyz(dist_xyz_log, d_idx)
                    d_idx += 1

        print(f'percentage of targets reached : {self.count_objective_reached / self.count_command_robot * 100 :2f}')


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
            self.reset_and_place_all(box_pos=goal_pos, start_marker_pos=start_pos, goal_marker_pos=goal_pos)
            self.move_franka_to_xyz(goal_pos)
            img = self.physics.render(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], camera_id=self.camera_id)
            goal_obs = np.concatenate([self.physics.data.qpos[:7], self.physics.data.qvel[:7], self.get_ee_position()])
            goal_obs_list.append((goal_obs.copy(), img.copy()))
        
        return goal_obs_list

    def sample_xyz(self, range_tuple):
        return np.array([np.random.uniform(*r) for r in range_tuple])
    
    def sample_uniform_xyz(self):
        return self.sample_xyz([self.X_RANGE, self.Y_RANGE, self.Z_RANGE])

    def sample_start_xyz(self):
        return self.sample_xyz([self.START_GOAL_X_RANGE, self.START_GOAL_Y_RANGE, self.START_GOAL_Z_RANGE])
    
    def sample_goal_xyz(self):
        return self.sample_xyz([self.START_GOAL_X_RANGE, self.START_GOAL_Y_RANGE, self.START_GOAL_Z_RANGE])
    
    def sample_direc_xyz(self, current_pos, dist_range, freeze_z=True, max_loop=100):
        for _ in range(max_loop):
            yaw = np.random.uniform(- np.pi, np.pi)
            if not freeze_z:
                pitch = np.random.uniform(- np.pi / 2, np.pi / 2)
            else:
                pitch = 0
            
            dist = np.random.uniform(*dist_range)
            
            dx = dist * np.cos(pitch) * np.cos(yaw)
            dy = dist * np.cos(pitch) * np.sin(yaw)
            dz = dist * np.sin(pitch)
            
            delta = np.array([dx, dy, dz])
            new_pos = current_pos + delta
            
            if self.is_withn_bounds(new_pos, self.X_RANGE, self.Y_RANGE, self.Z_RANGE):
                break
        else:
            x = np.clip(new_pos[0], *self.X_RANGE)
            y = np.clip(new_pos[1], *self.Y_RANGE)
            z = np.clip(new_pos[2], *self.Z_RANGE)
            new_pos = np.array([x, y, z])
        
        return new_pos 
    
    def sample_uniform_constrain_xyz(self, current_pos, max_dist = 0.1, max_loop = 100):
        for _ in range(max_loop):
            target_xyz = self.sample_xyz([self.X_RANGE, self.Y_RANGE, self.Z_RANGE])
            dist = np.linalg.norm(target_xyz - current_pos)
            if dist < max_dist:
                break
        else:
            x = np.clip(new_pos[0], *self.X_RANGE)
            y = np.clip(new_pos[1], *self.Y_RANGE)
            z = np.clip(new_pos[2], *self.Z_RANGE)
            new_pos = np.array([x, y, z])
        return target_xyz
    
    def is_withn_bounds(self, pos, x_range, y_range, z_range):
        x, y, z = pos 
        return (x_range[0] <= x <= x_range[1] and
                y_range[0] <= y <= y_range[1] and
                z_range[0] <= z <= z_range[1])
    
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
        
    def data_save(self):
        torch.save(self.data_list, os.path.join(self.SAVE_PATH, "data.p"))
        np.save(os.path.join(self.SAVE_PATH, "images.npy"), np.array(self.all_images, dtype=np.uint8))
        goal_imgs = np.stack([g[1] for g in self.goal_obs_list])
        np.save(os.path.join(self.SAVE_PATH, "goal_images.npy"), goal_imgs)
        torch.save({"pair_list": self.pair_list}, os.path.join(self.SAVE_PATH, "pair_info.p"))
        print("✅ Done generating dataset!")
    
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
        pixels = [img[0, 0] for img in images]  # 各画像の左上ピクセル [R,G,B]
        df = pd.DataFrame(pixels, columns=["R", "G", "B"])
        df.to_csv(os.path.join(output_dir, "image_pixels_summary.csv"), index=False)
        
        
    def make_video(self):
        DATA_PATH = self.SAVE_PATH
        SAVE_DIR = "robot_sim/analyze/video"

        timestep = self.physics.model.opt.timestep
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

    def check_ik_accuracy(self, verbose=True):
        print('check if ik-calculation is correct')
        self.physics.reset()
        
        target_xyz = self.sample_uniform_xyz()
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

        if verbose:
            print(f"target_xyz: {target_xyz}")
            print(f"actual_ee_pos: {ee_pos}")
            print(f"誤差: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}, norm={dist:.6f}")

        self.physics.reset()
        return {
            'target': target_xyz,
            'actual': ee_pos,
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'dist': dist
        }


if __name__ == "__main__":
    
    with open("robot_sim/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    dataset_generator = FrankaDatasetGenerator(config)
    dataset_generator.check_ik_accuracy()
    dataset_generator.generate_all_pairs()
    dataset_generator.data_save()
    dataset_generator.confirm_data()
    dataset_generator.make_video()
