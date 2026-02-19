from pldm.planning import objectives_v2
import torch
from pldm.models.jepa import JEPA
from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.planners.enums import PlannerType
from pldm.planning.planners.mppi_planner import MPPIPlanner
from pldm.planning.planners.sgd_planner import SGDPlanner
from pldm.planning.planners.cem_planner import CEMPlanner
from pldm.planning.utils import normalize_actions
from abc import ABC
from pldm.planning.enums import MPCResult, PooledMPCResult
import numpy as np
from pldm.models.utils import flatten_conv_output
from tqdm import tqdm
from pldm.planning.franka.env_wrapper import HoldUntilReachWrapper, ActionRepeatWrapper
import os
import imageio
from datetime import datetime
from pldm.data.enums import ProbingDatasets, DatasetType, Datasets
from typing import Optional
from pldm.planning.franka.enums import DesignateStartGoalPosConfig

class MPCEvaluator(ABC):
    def __init__(
        self,
        config,
        model: JEPA,
        prober: torch.nn.Module,
        normalizer: Normalizer,
        quick_debug: bool = False,
        prefix: str = "",
        pixel_mapper=None,
        image_based=True,
        train_ds: Optional[Datasets] = None,
    ):
        self.config = config
        self.model = model
        self.prober = prober
        self.normalizer = normalizer
        self.quick_debug = quick_debug
        self.prefix = prefix
        self.pixel_mapper = pixel_mapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_based = image_based
        self.train_ds = train_ds

    def close(self):
        pass

    def _infer_chunk_sizes(self):
        config = self.config

        if config.level1.planner_type == PlannerType.MPPI:
            # n_envs_batch_size = 500000 // (config.num_samples * config.max_plan_length)
            n_envs_batch_size = config.n_envs_batch_size
        else:
            n_envs_batch_size = config.n_envs
        print("[DBG][pldm/planning/mpc.py] n_envs_batch_size:", n_envs_batch_size)

        chunk_sizes = [n_envs_batch_size] * (config.n_envs // n_envs_batch_size) + (
            [config.n_envs % n_envs_batch_size]
            if config.n_envs % n_envs_batch_size != 0
            else []
        )

        return chunk_sizes

    def _construct_planner(self, n_envs: int):
        config = self.config
        print("[DBG][pldm/planning/mpc.py] config:", config)

        objective = objectives_v2.ReprTargetMPCObjective(
            model=self.model,
            propio_cost=config.level1.propio_cost,
            sum_all_diffs=config.level1.sum_all_diffs,
            loss_coeff_first=config.level1.loss_coeff_first,
            loss_coeff_last=config.level1.loss_coeff_last,
        )

        #公式実験
        # action_normalizer = lambda x: normalize_actions(
        #     x,
        #     min_norm=config.level1.min_step,
        #     max_norm=config.level1.max_step,
        #     xy_action=True,
        #     clamp_actions=config.level1.clamp_actions,
        # )
        
        #Franka だがここで不必要
        # action_normalizer = lambda x: normalize_actions(x)

        if config.level1.planner_type == PlannerType.MPPI:
            planner = MPPIPlanner(
                config.level1.mppi,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,  # [300, 13456]
                prober=self.prober,
                action_normalizer=None,
                n_envs=n_envs,
                projected_cost=config.level1.projected_cost,
            )
        elif config.level1.planner_type == PlannerType.SGD:
            planner = SGDPlanner(
                config.level1.sgd,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,
                prober=self.prober,
                action_normalizer=None, #diverse_maze の時は, action_normalizer を使用
            )
        elif config.level1.planner_type == PlannerType.CEM:
            planner = CEMPlanner(
                config.level1.cem,
                model=self.model,
                normalizer=self.normalizer,
                objective=objective,
                prober=self.prober,
                action_normalizer=None,
                n_envs=n_envs,
                projected_cost=config.level1.projected_cost,  # ← CEM側が受けるなら
            )
        else:
            raise NotImplementedError(
                f"Unknown planner type {config.level1.planner_type}"
            )

        return planner

    def _perform_mpc_in_chunks(self):
        """
        Divide it up in chunks in order to prevent OOM
        """
        chunk_sizes = self._infer_chunk_sizes()
        # print('CHUNK_SIZES:', chunk_sizes)

        mpc_data = PooledMPCResult()
        chunk_offset = 0

        for chunk_size in chunk_sizes:
            planner = self._construct_planner(n_envs=chunk_size)
            envs = self.envs[chunk_offset : chunk_offset + chunk_size]

            mpc_result = self._perform_mpc(
                planner=planner,
                envs=envs,
            )

            obs_c = mpc_result.observations
            location_history_c = mpc_result.locations
            action_history_c = mpc_result.action_history
            reward_history_c = mpc_result.reward_history
            pred_locations_c = mpc_result.pred_locations
            final_preds_dist_c = mpc_result.final_preds_dist
            targets_c = mpc_result.targets
            loss_history_c = mpc_result.loss_history
            qpos_history_c = mpc_result.qpos_history
            propio_history_c = mpc_result.propio_history
            object_history_c = getattr(mpc_result, "object_history", None)
            
            torque_history_c = getattr(mpc_result, "torque_history", None)
            actforce_history_c = getattr(mpc_result, "actforce_history", None)


            mpc_data.observations.append(obs_c)
            mpc_data.locations.append(location_history_c)
            mpc_data.action_history.append(action_history_c) #[num_chunk, task_timestep, num_env, plan_timestep, num_joint]
            mpc_data.reward_history.append(reward_history_c)
            mpc_data.pred_locations.append(pred_locations_c)
            mpc_data.final_preds_dist.append(final_preds_dist_c)
            mpc_data.targets.append(targets_c)
            mpc_data.loss_history.append(loss_history_c)
            mpc_data.qpos_history.append(qpos_history_c)
            mpc_data.propio_history.append(propio_history_c)
            if object_history_c is not None:                         
                mpc_data.object_history.append(object_history_c)
            if torque_history_c is not None:
                mpc_data.torque_history.append(torque_history_c)
            if actforce_history_c is not None:
                mpc_data.actforce_history.append(actforce_history_c)

            chunk_offset += chunk_size
        

        mpc_data.concatenate_chunks() #[task_timestep, num_env, plan_timestep, num_joint]

        return mpc_data

    def _perform_mpc(
        self,
        planner,
        envs,
    ):
        """
        Parameters:
            starts: (bs, 4)
            targets: (bs, 4)
        Outputs:
            observations: list of a_T (bs, 3, 64, 64) or (bs, 2)
            locations: list of a_T (bs, 2)
            action_history: list of a_T (bs, p_T, 2)
            reward_history: list of a_T (bs,)
            pred_locations: list of a_T (p_T, bs, 1, 2)
            targets: (bs, 4)
            loss_history: list of a_T (n_iters,)
        """
        
        
        envs = [HoldUntilReachWrapper(e, reach_eps=self.config.reach_eps, max_inner_steps=self.config.max_inner_steps) for e in envs]
        # for env in envs: print(type(env))
        
        physic_timestep = envs[0].physics.model.opt.timestep
        substeps = envs[0].substeps
        desired_freq = self.config.fps
        dt_target = 1.0 / desired_freq
        per_step_dt = physic_timestep * substeps
        
        repeat = max(1, round(dt_target / per_step_dt))

        envs = [ActionRepeatWrapper(e, repeat=repeat) for e in envs]
        
        effective_dt = per_step_dt * repeat
        print(f"[MPC] timestep={physic_timestep:.6f}s  substeps={substeps}  "
            f"repeat={repeat}  → effective_dt={effective_dt:.3f}s "
            f"({1.0/effective_dt:.2f} Hz)")

        
        
        dsg = self.config.designate_start_goal_pos
        if isinstance(dsg, dict): 
            dsg = DesignateStartGoalPosConfig(**dsg)

        if dsg and dsg.valid:
            box_start_pos = dsg.start_pos 
            if box_start_pos is not None: 
                box_start_pos = np.array(box_start_pos)
            box_goal_pos = dsg.goal_pos 
            if box_goal_pos is not None:
                box_goal_pos = np.array(box_goal_pos)
            
            if (box_start_pos is not None) and (box_goal_pos is not None):
                for e in envs: 
                    e.reset(start_pos=box_start_pos, goal_pos=box_goal_pos)
            else:
                for e in envs:
                    e.reset()            

        elif self.config.val_from_train_ds:
            t0 = 0
            threshold = getattr(self.config, "bluebox_move_threshold", 0.01)
            max_batches = getattr(self.config, "val_scan_max_batches", 200)  # 走査上限（無限防止）
            tg_cap = 40  # 今まで通り
            need = len(envs)

            assert hasattr(self.normalizer, "unnormalize_bluebox_locs"), \
                "Normalizer に unnormalize_bluebox_locs が無い！"

            def unnorm_traj(traj: torch.Tensor) -> torch.Tensor:
                """
                traj: (L,3) on CPU
                Normalizer が (L,3) を一括で受けられない場合に備えてフォールバックする
                """
                try:
                    out = self.normalizer.unnormalize_bluebox_locs(traj)
                    if isinstance(out, torch.Tensor) and out.shape == traj.shape:
                        return out
                except Exception:
                    pass
                return torch.stack([self.normalizer.unnormalize_bluebox_locs(traj[i]) for i in range(traj.shape[0])], dim=0)

            candidates = []  # (start_pos_np, goal_pos_np)

            for bi, ds_btc in enumerate(self.train_ds):
                blue = ds_btc.bluebox_locs.detach().cpu()  # (B,T,3)
                B, T, D = blue.shape
                assert D == 3

                tg = min(T - 1, tg_cap)

                # バッチ内の各軌跡をチェックして候補追加
                for b in range(B):
                    traj = blue[b, t0:tg+1]          # (L,3)
                    traj_u = unnorm_traj(traj)       # (L,3)

                    start_pos_t = traj_u[0]
                    dist = torch.norm(traj_u - start_pos_t.unsqueeze(0), dim=1)  # (L,)

                    idx = torch.nonzero(dist >= threshold, as_tuple=False).view(-1)
                    if idx.numel() == 0:
                        continue

                    k = int(idx[0].item())  # 最初に threshold 超えた時刻
                    goal_pos_t = traj_u[k]

                    candidates.append((start_pos_t.numpy(), goal_pos_t.numpy()))
                    if len(candidates) >= need:
                        break

                if len(candidates) >= need:
                    break

                if bi + 1 >= max_batches:
                    break

            # env に割り当て
            if len(candidates) >= need:
                for j, e in enumerate(envs):
                    start_pos, goal_pos = candidates[j]
                    e.reset(start_pos=start_pos, goal_pos=goal_pos)
                print(f"[MPC] sampled {need}/{need} tasks from train_ds (threshold={threshold}) ✅")
            else:
                # fallback（見つからなかった場合）
                print(f"[MPC] only {len(candidates)}/{need} tasks found (threshold={threshold}). fallback reset() ⚠️")
                for e in envs:
                    e.reset()

        else:
            for e in envs:
                e.reset()

        
        
        # ===== DEBUG: 初期画像を書き出す（timestamp付き）=====

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "debug_init_obs"
        os.makedirs(save_dir, exist_ok=True)

        # reset直後の初期観測を取得
        init_obs_t = torch.stack([e.get_obs() for e in envs]).to(self.device)  # (B,C,H,W)

        dbg = init_obs_t.detach().cpu()

        # 正規化されてるなら戻す（targetと揃える）
        try:
            dbg = self.normalizer.unnormalize_state(dbg)
        except Exception:
            pass

        for i in range(min(5, dbg.shape[0])):  # 先頭5環境だけ
            img = dbg[i].numpy()  # (C,H,W)
            img = np.transpose(img, (1, 2, 0))  # (H,W,C)

            if img.max() <= 1.0:
                img = (img * 255.0)

            img = np.clip(img, 0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, f"init_obs_env{i}_{ts}.png"), img)

        print(f"[DBG][pldm/planning/mpc.py] saved init_obs images to {save_dir}/")
        # =====================================================

        
            
        #ゴール位置
        targets = [e.get_target() for e in envs]
        targets = torch.from_numpy(np.stack(targets))

        #ゴール画像
        targets_t = torch.stack([e.get_target_obs() for e in envs]).to(self.device)
        targets_propio_t = torch.stack([e.get_target_propio() for e in envs]).to(self.device).float()

        
        
        # ===== DEBUG: 目標画像を書き出す（timestamp付き）=====


        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "debug_target_obs"
        os.makedirs(save_dir, exist_ok=True)

        # targets_t: (B, C, H, W) 想定
        dbg = targets_t.detach().cpu()

        # もし Normalizer がかかってるなら、ここで「戻す」方が見やすい
        try:
            dbg = self.normalizer.unnormalize_state(dbg)
        except Exception:
            pass

        for i in range(min(5, dbg.shape[0])):  # 先頭5環境だけ
            img = dbg[i].numpy()  # (C,H,W)
            img = np.transpose(img, (1, 2, 0))  # (H,W,C)

            # 0-1 の可能性もあるので保険（どっちでも破綻しにくい）
            if img.max() <= 1.0:
                img = (img * 255.0)

            img = np.clip(img, 0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(save_dir, f"target_obs_env{i}_{ts}.png"), img)

        print(f"[DBG][pldm/planning/mpc.py] saved target_obs images to {save_dir}/")
        # =====================================================
                        

        print('[DBG][pldm/planning/mpc.py] self.model.config.backbone.propio_dim is not None: ', self.model.config.backbone.propio_dim is not None)
        use_propio = True
        
        # encode target obs
        if self.model.config.backbone.propio_dim is not None:
            if not use_propio:
                # for target we don't care about the proprioceptive states. just make it zero.
                propio_states = torch.zeros(
                    (targets_t.shape[0], self.model.config.backbone.propio_dim)
                ).to(self.device)
            else:
                propio_states = targets_propio_t
                
            target_states = self.model.backbone(
                targets_t, propio=propio_states
            )
            
            targets_t = target_states.obs_component.detach()
            targets_propio_t = target_states.propio_component.detach()
        else:
            targets_t = self.model.backbone(targets_t).obs_component.detach()

        targets_t = flatten_conv_output(targets_t)
        print('[DBG][pldm/planning/mpc.py] targets_t.shape:', targets_t.shape)
        
        
        planner.reset_targets(targets_t, repr_input=True)
        print('[DBG][pldm/planning/mpc.py] planner.objective.target_enc.shape:', planner.objective.target_enc.shape)
        
        if targets_propio_t is not None:
            targets_propio_t = flatten_conv_output(targets_propio_t)
            planner.reset_targets_propio(targets_propio_t, targets_t, repr_input=True)
        print("[DBG][pldm/planning/mpc.py] planner.objective.target_propio_enc is not None", planner.objective.target_propio_enc is not None)
        if planner.objective.target_propio_enc is not None:
            print("[DBG][pldm/planning/mpc.py] planner.objective.target_propio_enc.shape: ", planner.objective.target_propio_enc.shape)
            

        observation_history = [torch.stack([e.get_obs() for e in envs])]

        obs_t = observation_history[0] #[5, 3, 64, 64]

        if self.image_based: ##
            obs_t = torch.cat([obs_t] * self.config.stack_states, dim=1)  # [5, 3, 64, 64]

        action_history = []
        reward_history = []
        location_history = []
        qpos_history = []
        propio_history = []

        pred_positions_history = []
        loss_history = []
        final_preds_dist_history = []
        
        object_history = [] 
        torque_history = []
        actforce_history = []


        init_infos = [e.get_info() for e in envs]
        if "location" in init_infos[0]:
            location_history.append(np.array([info["location"] for info in init_infos]))

        if "qpos" in init_infos[0]:
            qpos_history.append(np.array([info["qpos"] for info in init_infos]))

        if "propio" in init_infos[0]:
            propio_history.append(np.array([info["propio"] for info in init_infos]))
            
        if "object_pos" in init_infos[0]:
            object_history.append(np.array([info["object_pos"] for info in init_infos]))

        print("[DBG][pldm/planning/mpc.py]self.config.n_steps:", self.config.n_steps)
        print("[DBG][pldm/planning/mpc.py]self.config:", self.config)
        for i in tqdm(range(self.config.n_steps), desc="Planning steps"):
            if i % self.config.replan_every == 0:

                if planner.model.use_propio_pos:
                    curr_propio_pos = [e.get_propio_pos() for e in envs]
                    curr_propio_pos = torch.from_numpy(
                        np.stack(curr_propio_pos)
                    ).float()
                else:
                    curr_propio_pos = None

                if planner.model.use_propio_vel:
                    curr_propio_vel = [e.get_propio_vel() for e in envs]
                    curr_propio_vel = torch.from_numpy(
                        np.stack(curr_propio_vel)
                    ).float()
                else:
                    curr_propio_vel = None


                print("[DBG][pldm/planning/mpc.py]self.config.level1.max_plan_length:", self.config.level1.max_plan_length)
                planning_result = planner.plan(
                    obs_t,
                    curr_propio_pos=curr_propio_pos,
                    curr_propio_vel=curr_propio_vel,
                    plan_size=min(
                        self.config.n_steps - i, self.config.level1.max_plan_length, self.config.pred_steps
                    ),
                    repr_input=False,
                )

            last_pred_obs = flatten_conv_output(planning_result.pred_obs)

            pred_dist = torch.norm(last_pred_obs - targets_t.unsqueeze(0), dim=2).cpu()
            final_preds_dist_history.append(pred_dist)

            planned_actions = (
                planning_result.actions[:, i % self.config.replan_every :]
                .detach()
                .cpu()
            )
            
            
            print("[DBG][pldm/planning/plotting.py]", planned_actions.shape)
            dbg_ac = planned_actions.reshape(-1, 7)
            print("[DBG][pldm/planning/plotting.py]", "min:", dbg_ac.min(dim=0).values, "max:", dbg_ac.max(dim=0).values, "mean:", dbg_ac.mean(dim=0).values)

            if self.config.random_actions: #x
                results = [
                    envs[j].step(envs[0].action_space.sample())
                    for j in range(len(envs))
                ]
            else: ##
                results = [
                    envs[j].step(
                        planned_actions[j, 0].detach().cpu().contiguous().numpy()
                    )
                    for j in range(len(envs))
                ]

            
            d = envs[0].physics.data
            m = envs[0].physics.model
            print("ncon", int(d.ncon))
            for k in range(int(d.ncon)):
                c = d.contact[k]
                g1 = m.id2name(c.geom1, "geom")
                g2 = m.id2name(c.geom2, "geom")
                print(k, g1, g2, "dist=", float(c.dist))


            
            assert len(results[0]) == 5
            print("[DBG][pldm/planning/mpc.py] results.type:", type(results))
            print("[DBG][pldm/planning/mpc.py] results:", results)
            current_obs = torch.from_numpy(np.stack([r[0] for r in results])).float()
            rewards_t = torch.from_numpy(np.stack([r[1] for r in results])).float()
            infos = [r[4] for r in results]
            
            ################################################################
            # envs[0] で代表1本を見る（まずはこれでOK）
            inner_env = envs[0]
            while hasattr(inner_env, "env"):
                inner_env = inner_env.env  # Wrapper を剥がす

            qb2 = inner_env.physics.data.qfrc_bias[1]
            qa2 = inner_env.physics.data.qfrc_actuator[1]
            q2  = inner_env.physics.data.qpos[1]

            print(f"[i={i:03d}] q2={q2:+.3f}  bias={qb2:+.2f}  act={qa2:+.2f}")
            ################################################################
                        
            

            if i == 0:
                print("info keys:", infos[0].keys())
                print("location:", infos[0]["location"], "shape:", np.array(infos[0]["location"]).shape)
                

            action_history.append(planned_actions.detach().cpu()) #[task_timestep, env_idx, plan_timestep, num_joint]
            observation_history.append(current_obs)
            reward_history.append(rewards_t)

            if "location" in infos[0]:
                location_history.append(np.array([info["location"] for info in infos]))

            if "qpos" in infos[0]:
                qpos_history.append(np.array([info["qpos"] for info in infos]))

            if "propio" in infos[0]:
                propio_history.append(np.array([info["propio"] for info in infos]))
                
            if "object_pos" in infos[0]:
                object_history.append(np.array([info["object_pos"] for info in infos]))
                
            if "qfrc_actuator" in infos[0]:
                torque_history.append(np.array([info["qfrc_actuator"] for info in infos]))  # (B,7)

            if "actuator_force" in infos[0]:
                actforce_history.append(np.array([info["actuator_force"] for info in infos]))  # (B,7)


            if planning_result.locations is not None:
                pred_locations = planning_result.locations.detach().cpu()
                pred_locations = pred_locations.squeeze(2)
                pred_positions_history.append(pred_locations)

            # stack states if necessary for next iteration
            if self.config.stack_states == 1:
                obs_t = current_obs
            else:
                obs_t = torch.cat(
                    [obs_t[:, current_obs.shape[1] :], current_obs], dim=1
                )

            loss_history.append(planning_result.losses)

        observation_history = [
            self.normalizer.unnormalize_state(o) for o in observation_history
        ]
        print("len(torque_history):", len(torque_history))
        print("len(actforce_history):", len(actforce_history))


        return MPCResult(
            observations=observation_history,
            locations=[torch.from_numpy(x) for x in location_history],
            action_history=action_history,
            reward_history=reward_history,
            pred_locations=pred_positions_history,
            final_preds_dist=final_preds_dist_history,
            targets=targets,
            loss_history=loss_history,
            qpos_history=[torch.from_numpy(x) for x in qpos_history],
            propio_history=[torch.from_numpy(x) for x in propio_history],
            object_history=[torch.from_numpy(x) for x in object_history],
            torque_history=[torch.from_numpy(x) for x in torque_history],
            actforce_history=[torch.from_numpy(x) for x in actforce_history],
        )


