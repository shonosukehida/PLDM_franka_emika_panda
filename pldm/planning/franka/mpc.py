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
from pldm.planning.plotting import log_planning_plots, log_l1_planning_loss
from pldm.planning.d4rl.enums import MPCReport  


from pldm_envs.franka.envs import FrankaSimEnv
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
        pixel_mapper = None
    ):
        super().__init__(
            config=config,
            model=jepa,
            prober=prober,
            normalizer=normalizer,
            quick_debug=quick_debug,
            prefix=prefix,
            pixel_mapper=pixel_mapper
        )

        # ここで Franka の gym 環境を複数生成するコードを書く
        envs_generator = FrankaEnvsGenerator(
            model_path=config.model_path,  
            n_envs=config.n_envs,
            normalizer=normalizer
        )
        self.envs = envs_generator()

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
            log_planning_plots(
                result=mpc_data,
                report=report,
                idxs=list(range(self.config.n_envs)) if not self.quick_debug else [0],
                prefix=self.prefix,
                n_steps=self.config.n_steps,
                xy_action=True,
                plot_every=self.config.plot_every,
                quick_debug=self.quick_debug,
                pixel_mapper=self.pixel_mapper,
                plot_failure_only=self.config.plot_failure_only,
                log_pred_dist_every=self.config.log_pred_dist_every,
                mark_action=False,
            )
        return mpc_data, report
