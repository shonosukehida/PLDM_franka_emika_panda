#未検証

# planning/franka/mpc.py
from typing import Optional
import torch
from pldm.models.jepa import JEPA
from pldm_envs.utils.normalizer import Normalizer
from pldm.planning.mpc import MPCEvaluator
from .enums import FrankaMPCConfig
from pldm.planning.enums import MPCResult, PooledMPCResult
from pldm.planning.utils import calc_avg_steps_to_goal
from pldm.planning.plotting import log_planning_plots, log_l1_planning_loss, log_planning_plots_split, log_planning_videos_split
from pldm.planning.plotting import log_planning_obs_plots_split, log_planning_traj_plots_split, log_planning_joint_angle_plots_split, log_planning_torque_plots_split
from pldm.planning.d4rl.enums import MPCReport  

from pldm.data.enums import ProbingDatasets, DatasetType, Datasets
from pldm_envs.franka.evaluation.envs_generator import FrankaEnvsGenerator


class FrankaMPCEvaluator(MPCEvaluator):
    def __init__(
        self,
        config: FrankaMPCConfig,
        normalizer: Normalizer,
        jepa: JEPA,
        prober: Optional[torch.nn.Module] = None,
        quick_debug: bool = False,
        prefix: str = "franka_",
        pixel_mapper = None,
        train_ds: Optional[Datasets] = None,
    ):
        super().__init__(
            config=config,
            model=jepa,
            prober=prober,
            normalizer=normalizer,
            quick_debug=quick_debug,
            prefix=prefix,
            pixel_mapper=pixel_mapper,
            train_ds=train_ds,
        )


        self.config.task.backbone_arch = config.backbone_arch
        self.config.task.vjepa2_repo = config.vjepa2_repo
        envs_generator = FrankaEnvsGenerator(
            model_path=config.model_path,  
            n_envs=config.n_envs,
            normalizer=normalizer,
            max_dq=self.config.max_dq,
            task_name=self.config.task.name,
            task_cfg=self.config.task,
            camera_name=self.config.camera_name
        )
        self.envs = envs_generator()
        for e in self.envs:
            e.reset()

    def _construct_report(self, data: PooledMPCResult):
        # D4RLの `MazeMPCEvaluator` に近い方式で評価
        T = len(data.reward_history)
        B = data.reward_history[0].shape[0]

        terminations = [T] * B
        for b_i in range(B):
            for t_i in range(T):
                if data.reward_history[t_i][b_i]:
                    terminations[b_i] = t_i
                    break

        successes = [int(x < T) for x in terminations]
        success_rate = sum(successes) / len(successes)
        avg_steps = calc_avg_steps_to_goal(data.reward_history)
        median_steps = calc_avg_steps_to_goal(data.reward_history, reduce_type="median")

        return MPCReport(
            success_rate=success_rate,
            success=torch.tensor(successes),
            avg_steps_to_goal=avg_steps,
            median_steps_to_goal=median_steps,
            terminations=terminations,
            one_turn_success_rate=-1,
            two_turn_success_rate=-1,
            three_turn_success_rate=-1,
            num_one_turns=0,
            num_two_turns=0,
            num_three_turns=0,
            num_turns=[0] * B,
            block_dists=[0.0] * B,
            ood_report={},
        )

    def evaluate(self):
        # print('MPC EVALUATE')
        mpc_data = self._perform_mpc_in_chunks()
        report = self._construct_report(mpc_data)
        log_l1_planning_loss(result=mpc_data, prefix=self.prefix)
        

        if self.config.visualize_planning:
            # log_planning_plots_split(
            #     result=mpc_data,
            #     report=report,
            #     idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
            #     plot_every=self.config.plot_every,
            #     plot_failure_only=self.config.plot_failure_only,
            #     world_xlim = [0.315, 0.715],
            #     world_ylim = [-0.2, 0.2],
            #     use_pixel_mapper=False,           
            #     pixel_mapper=self.pixel_mapper, 
            #     env = self.envs[0],
            # )
            log_planning_obs_plots_split(
                result=mpc_data,
                report=report,
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                plot_every=self.config.plot_every,
                plot_failure_only=self.config.plot_failure_only,
                world_xlim=[0.315, 0.715],
                world_ylim=[-0.2, 0.2],
                use_pixel_mapper=False,
                pixel_mapper=self.pixel_mapper,
                env=self.envs[0],
            )


            
            if self.config.task.name == "push_to_goal": 
                use_box = True
            elif self.config.task.name == "reach_no_touch":
                if self.config.task.reach_no_touch.use_box:
                    use_box = True 
                else:
                    use_box = False 
            else: 
                use_box = True
                
            #行動列の出力あり
            log_planning_traj_plots_split(
                result=mpc_data,
                report=report,
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                plot_every=self.config.plot_every,
                plot_failure_only=self.config.plot_failure_only,
                world_xlim=[0.315, 0.715],
                world_ylim=[-0.2, 0.2],
                use_pixel_mapper=False,
                pixel_mapper=self.pixel_mapper,
                env=self.envs[0],
                plot_action = True,
                use_box = use_box
            )

            #行動列の出力なし
            log_planning_traj_plots_split(
                result=mpc_data,
                report=report,
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                plot_every=self.config.plot_every,
                plot_failure_only=self.config.plot_failure_only,
                world_xlim=[0.315, 0.715],
                world_ylim=[-0.2, 0.2],
                use_pixel_mapper=False,
                pixel_mapper=self.pixel_mapper,
                env=self.envs[0],
                plot_action = False
            )

            log_planning_joint_angle_plots_split(
                result=mpc_data,
                report=report,
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                plot_every=self.config.plot_every,
                plot_failure_only=self.config.plot_failure_only,
                model_path = self.config.model_path
            )

            log_planning_torque_plots_split(
                result=mpc_data,
                report=report,
                env=self.envs[0], 
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                plot_failure_only=self.config.plot_failure_only,
            )


            
            if self.config.visualize_planning_videos:
                print("visualize_planning_videos!!")
                log_planning_videos_split(
                    result=mpc_data,
                    report=report,
                    idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                    plot_every=self.config.plot_every,
                    plot_failure_only=self.config.plot_failure_only,
                    world_xlim = [0.315, 0.715],
                    world_ylim = [-0.2, 0.2],
                    use_pixel_mapper=False,           
                    pixel_mapper=self.pixel_mapper, 
                    env = self.envs[0],
                )
                print("finished making video!!")






            # log_planning_plots(
            #     result=mpc_data,
            #     report=report,
            #     idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
            #     prefix=self.prefix,
            #     n_steps=self.config.n_steps,
            #     xy_action=True,
            #     plot_every=self.config.plot_every,
            #     quick_debug=self.quick_debug,
            #     pixel_mapper=self.pixel_mapper,
            #     plot_failure_only=self.config.plot_failure_only,
            #     log_pred_dist_every=self.config.log_pred_dist_every,
            #     mark_action=False,
            # )
            
        return mpc_data, report
