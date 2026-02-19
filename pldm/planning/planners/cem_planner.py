from typing import Optional, Callable, List

import torch
from torch import nn

from .mppi_torch import MPPI
from pldm.models.utils import flatten_conv_output
from .planner import PlanningResult
from pldm.planning.planners.calc_dynamics import LearnedDynamics, RunningCost




class _CEMController:
    """
    1 environment 分の CEM。
    - state は (nx) でも (1, nx) でもOK（LearnedDynamics 側で整形される）
    - action は (H, A) を最終的に返す（full sequence）
    """
    def __init__(
        self,
        dynamics: Callable,             # LearnedDynamics
        running_cost: Callable,         # RunningCost
        action_dim: int,
        num_samples: int,
        num_elites: int,
        init_std: float = 1.0,
        n_iters: int = 5,
        alpha: float = 0.1,
        device: torch.device = torch.device("cuda"),
        eps_std: float = 1e-6,
        clamp_actions: bool = False,
        action_low: float = -1.0,
        action_high: float = 1.0,
        action_normalizer: Optional[Callable] = None,
        max_batch_size: int = 500,
        momentum_mean: float = 0.0,
        momentum_std: float = 0.0,
        max_norms: Optional[List[float]] = None,
        max_norm_dims: Optional[List[List[int]]] = None,
    ):

        self.F = dynamics
        self.cost_fn = running_cost
        self.A = action_dim

        self.K = int(num_samples)
        self.E = int(num_elites)
        self.init_std = float(init_std)
        self.n_iters = int(n_iters)
        self.alpha = float(alpha)
        self.device = device
        self.eps_std = float(eps_std)

        self.clamp_actions = clamp_actions
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.action_normalizer = action_normalizer

        # warm start 用（horizon ごとに shape が変わるので、plan 時に作り直す）
        self.mu: Optional[torch.Tensor] = None   # (H, A)
        self.std: Optional[torch.Tensor] = None  # (H, A)
        
        self.max_batch_size = int(max_batch_size)
        
        self.momentum_mean = float(momentum_mean)
        self.momentum_std = float(momentum_std)
        self.max_norms = max_norms 
        self.max_norm_dims = max_norm_dims        
         


    def _init_dist(self, H: int):
        if self.mu is None or self.mu.shape[0] != H:
            self.mu = torch.zeros(H, self.A, device=self.device)
        else:
            # horizon が同じなら warm start
            self.mu = self.mu.clone()

        if self.std is None or self.std.shape[0] != H:
            self.std = torch.ones(H, self.A, device=self.device) * self.init_std
        else:
            self.std = self.std.clone()

    def shift_nominal_trajectory(self):
        """
        MPCで replan しながら使う場合、1 step 実行した後に
        mu/std を 1つ前に詰めて最後を初期化すると安定しやすいです✨
        （MPPI の shift_nominal_trajectory 相当）
        """
        if self.mu is None or self.std is None:
            return
        self.mu = torch.roll(self.mu, shifts=-1, dims=0)
        self.std = torch.roll(self.std, shifts=-1, dims=0)
        self.mu[-1].zero_()
        self.std[-1].fill_(self.init_std)

    def change_horizon(self, H: int):
        # dist を horizon に合わせる
        self._init_dist(H)

    @torch.no_grad()
    def command(self, state: torch.Tensor, horizon: int, shift_nominal_trajectory: bool = False):
        """
        inputs:
            state: (C,H,W)
        
        Returns:
            U_best: (H, A)  full action sequence (normalized space)
        """

        if state.ndim == 3:
            state = state.unsqueeze(0).repeat(self.K, 1, 1, 1)  # (K,C,H,W)
        
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()

        self.change_horizon(horizon)
        mu = self.mu
        std = self.std

        for it in range(self.n_iters):
            # sample actions: (K, H, A)
            U = mu[None] + std[None] * torch.randn(self.K, mu.shape[0], self.A, device=self.device)
            U[0] = mu #前回の平均を含める
            
            
            if self.max_norms is not None and self.max_norm_dims is not None:
                for dims, maxnorm in zip(self.max_norm_dims, self.max_norms):
                    U[:, :, dims] = torch.clamp(U[:, :, dims], -maxnorm, maxnorm)

            # optional clamp / normalize
            if self.clamp_actions:
                U = torch.clamp(U, self.action_low, self.action_high)

            # action_normalizer (例えば diverse_maze の min_step/max_step 正規化など)
            if self.action_normalizer is not None:
                # normalizer は (K, A) を想定してることが多いので時間方向ループ
                for t in range(U.shape[1]):
                    U[:, t] = self.action_normalizer(U[:, t])

            # rollout in chunks to avoid OOM
            K = self.K
            H = U.shape[1]  # horizon
            cost = torch.zeros(K, device=self.device)

            chunk = self.max_batch_size
            for s in range(0, K, chunk):
                e = min(K, s + chunk)

                # (Kc, C, H, W)
                state_chunk = state[s:e]

                # (H, Kc, A)
                action_chunk = U[s:e].permute(1, 0, 2)

                pred_encs_c, pred_obs_c, pred_propio_c = self.F(
                    state=state_chunk,
                    action=action_chunk,
                    only_return_last=False,
                    flatten_output=False,
                )

                # pred_obs_c: (H+1, Kc, Cobs, 26, 26)
                # pred_propio_c: (H+1, Kc, Cprop, 26, 26)
                # ここでコストを時間方向に加算

                cost_c = torch.zeros(e - s, device=self.device)
                for t in range(1, H + 1):
                    c = self.cost_fn(
                        state_obs=pred_obs_c[t],
                        state_propio=pred_propio_c[t],
                        action=None,
                    )
                    # c: (Kc,)
                    while c.dim() > 1:
                        c = c.mean(dim=-1)
                    cost_c = cost_c + c

                cost[s:e] = cost_c

                # メモリ回収（かなり効きます）
                del pred_encs_c, pred_obs_c, pred_propio_c, state_chunk, action_chunk, cost_c


            elite_idx = torch.argsort(cost)[: self.E]
            elite = U[elite_idx]                     # (E, H, A)
            new_mu = elite.mean(dim=0)               # (H, A)
            new_std = elite.std(dim=0) + self.eps_std

            mu  = new_mu  * (1 - self.momentum_mean) + mu  * self.momentum_mean
            std = new_std * (1 - self.momentum_std)  + std * self.momentum_std
            
            print(
                f"[CEM][pldm/planning/planners/cem_planner.py] iter={it} | "
                f"std mean={std.mean().item():.4f} "
                f"min={std.min().item():.4f} "
                f"max={std.max().item():.4f}"
            )


        # warm start 保存
        self.mu = mu
        self.std = std
        return mu  # (H, A)


class CEMPlanner:
    """
    MPPIPlanner と同じ “上位プランナ” の形に合わせた CEM版 ✅
    """
    def __init__(
        self,
        config,
        model,
        normalizer,
        objective,
        prober: Optional[torch.nn.Module] = None,
        action_normalizer: Optional[Callable] = None,
        num_refinement_steps: int = 1,
        n_envs: int = None,
        l2: bool = False,
        projected_cost: bool = False,
    ):
        device = next(model.parameters()).device
        self.device = device

        self.model = model
        self.config = config
        self.normalizer = normalizer
        self.action_normalizer = action_normalizer
        self.prober = prober
        self.objective = objective

        # dynamics wrapper（MPPIと同じ）
        self.dynamics = LearnedDynamics(
            model,
            state_dim=model.spatial_repr_dim,
        )

        # cost per env（MPPIと同じ）
        # projected_cost=True なら obs_projector に prober を入れる（公式と同じ挙動）
        self.costs: List[RunningCost] = [
            RunningCost(
                objective,
                idx=i,
                obs_projector=prober if projected_cost else None,
                propio_projector=None,
                obs_coeff=getattr(config, "obs_coeff", 1.0),
                propio_coeff=getattr(config, "propio_coeff", 0.0),
            )
            for i in range(n_envs)
        ]

        # print("[DBG][pldm/planning/planners/cem_planner.py] init_std", getattr(config, "init_std", 1.0))
        # CEM controller per env
        # ※ action 空間の clamp は環境に合わせて調整してね（ここは一旦 -1..1）
        self.ctrls = [
            _CEMController(
                dynamics=self.dynamics,
                running_cost=self.costs[i],
                action_dim=model.predictor.action_dim,
                num_samples=getattr(config, "num_samples", 500),
                num_elites=getattr(config, "num_elites", 50),
                init_std=getattr(config, "init_std", 1.0),
                n_iters=getattr(config, "n_iters", 5),
                # alpha=getattr(config, "alpha", 0.1),
                device=device,
                clamp_actions=getattr(config, "clamp_actions", False),
                action_low=getattr(config, "action_low", -1.0),
                action_high=getattr(config, "action_high", 1.0),
                action_normalizer=action_normalizer,
                max_batch_size=getattr(config, "max_batch_size", 64),
                momentum_mean=getattr(config, "momentum_mean", 0.0),
                momentum_std=getattr(config, "momentum_std", 0.0),
                max_norms=getattr(config, "max_norms", None),
                max_norm_dims=getattr(config, "max_norm_dims", None),
            )
            for i in range(n_envs)
        ]

        self.last_plan_size = None
        self.num_refinement_steps = num_refinement_steps

    @torch.no_grad()
    def plan(
        self,
        current_state: torch.Tensor,
        plan_size: int,
        repr_input: bool = True,
        curr_propio_pos: Optional[torch.Tensor] = None,
        curr_propio_vel: Optional[torch.Tensor] = None,
        diff_loss_idx: Optional[torch.Tensor] = None,
    ):
        """
        Returns PlanningResult compatible with MPPIPlanner.plan ✅
        """
        batch_size = current_state.shape[0]
        self.dynamics.before_planning_callback()

        # repr_input=False のときは backbone で encoding へ
        if not repr_input:
            if self.model.backbone.config.propio_dim:
                # 現状のMPPIPlanner と同じロジック
                if curr_propio_vel is not None and curr_propio_pos is not None:
                    curr_propio_states = torch.cat([curr_propio_pos, curr_propio_vel], dim=-1)
                elif curr_propio_vel is not None:
                    curr_propio_states = curr_propio_vel
                elif curr_propio_pos is not None:
                    curr_propio_states = curr_propio_pos
                else:
                    raise ValueError("Need proprio states to plan")

                backbone_output = self.model.backbone(
                    current_state.to(self.device), propio=curr_propio_states.to(self.device)
                )
            else:
                backbone_output = self.model.backbone(current_state.to(self.device))

            current_state = backbone_output.encodings  # (B, C, 26, 26) etc.

        # actions (B, H, A) in normalized space (CEM samples in normalized space)
        actions = []
        for i in range(batch_size):
            # MPPIみたいに horizon が短くなった時の shift を入れるならここで
            if self.last_plan_size is not None and plan_size < self.last_plan_size:
                # 使うなら shift_nominal_trajectory を複数回呼ぶ
                for _ in range(self.last_plan_size - plan_size):
                    self.ctrls[i].shift_nominal_trajectory()

            actions.append(
                self.ctrls[i].command(
                    state=current_state[i],
                    horizon=plan_size,
                    shift_nominal_trajectory=False,
                )
            )

        actions = torch.stack(actions, dim=0)  # (B, H, A)

        # rollout to get full predicted sequence (T=H)
        pred_encs, pred_obs, pred_propio = self.dynamics(
            state=current_state,
            action=actions.permute(1, 0, 2),
            only_return_last=False,
            flatten_output=False,
        )

        # optional: action_normalizer の出力空間に合わせる（MPPIPlanner と同様の場所で）
        if self.action_normalizer is not None:
            # ここは「envへ出す直前の調整」をするなら使う
            pass

        # envへ渡す action は逆正規化
        actions_env = self.normalizer.unnormalize_action(actions)

        self.dynamics.after_planning_callback()
        self.last_plan_size = plan_size

        # locations（proberがあるときだけ）
        if self.prober is not None:
            pred_locs = torch.stack([self.prober(x) for x in pred_obs])
            unnormed_locations = self.normalizer.unnormalize_location(pred_locs).detach()
        else:
            unnormed_locations = None

        losses = [0]  # いったん MPPI と合わせてダミー
        print("actions (norm)  min/max/mean:", actions.reshape(-1,7).min(0).values, actions.reshape(-1,7).max(0).values, actions.reshape(-1,7).mean(0))
        print("actions_env(raw) min/max/mean:", actions_env.reshape(-1,7).min(0).values, actions_env.reshape(-1,7).max(0).values, actions_env.reshape(-1,7).mean(0))


        return PlanningResult(
            pred_encs=pred_encs,
            pred_obs=pred_obs,
            actions=actions_env,
            locations=unnormed_locations,
            losses=losses,
        )

    def reset_targets(self, targets: torch.Tensor, repr_input: bool = True):
        self.objective.set_target(targets, repr_input=repr_input)

    def reset_targets_propio(self, targets_propio: torch.Tensor, targets_obs: torch.Tensor, repr_input: bool = True):
        self.objective.set_target_propio(targets_propio, targets_obs, repr_input=repr_input)