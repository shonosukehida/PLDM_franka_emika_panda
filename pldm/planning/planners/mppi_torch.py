# copied from https://github.com/UM-ARM-Lab/pytorch_mppi

import logging
import time

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from arm_pytorch_utilities import handle_batch_input

# from planning.planner import Planner

logger = logging.getLogger(__name__)


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


class MPPI:
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(
        self,
        dynamics,
        running_cost,
        nx,
        noise_sigma,
        num_samples=100,
        horizon=15,
        device="cpu",
        terminal_state_cost=None,
        lambda_=1.0,
        noise_mu=None,
        action_normalizer=None,
        u_init=None,
        U_init=None,
        u_scale=1,
        u_per_command=1,
        latent_actions=False,
        z_reg_coeff=0.1,
        step_dependent_dynamics=False,
        rollout_samples=1,
        rollout_var_cost=0,
        rollout_var_discount=0.95,
        sample_null_action=False,
        noise_abs_cost=False,
    ):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS
        self.latent_actions = latent_actions
        self.z_reg_coeff = z_reg_coeff

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.action_normalizer = action_normalizer
        self.u_scale = u_scale
        self.u_per_command = u_per_command

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)

        ##############################################################
        # ===== Debug & Stabilize covariance =====
        def _symmetrize(A):
            return 0.5 * (A + A.transpose(-1, -2))

        def _add_jitter(A, eps):
            eye = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
            return A + eps * eye

        # 基本情報ログ
        print(f"[MPPI] device={self.d}, dtype={self.noise_sigma.dtype}, "
              f"nu={self.nu}, T(horizon)={self.T}, K(samples)={self.K}")

        # 形状・有限値
        assert self.noise_sigma.dim() == 2, "noise_sigma must be 2D"
        assert self.noise_sigma.shape[0] == self.noise_sigma.shape[1], "noise_sigma must be square"
        assert self.noise_sigma.shape[0] == self.nu, (
            f"noise_sigma shape {self.noise_sigma.shape} != (nu, nu) with nu={self.nu}"
        )
        if not torch.isfinite(self.noise_sigma).all():
            raise RuntimeError("[MPPI] noise_sigma has NaN/Inf")

        # 対称化
        self.noise_sigma = _symmetrize(self.noise_sigma)

        # 固有値チェック（CPUで倍精度）
        eigvals_cpu = torch.linalg.eigvalsh(self.noise_sigma.double().cpu())
        min_eig = float(eigvals_cpu.min().item())
        max_eig = float(eigvals_cpu.max().item())
        print(f"[MPPI] cov eigvals: min={min_eig:.3e}, max={max_eig:.3e}")

        # 正定値でない/ギリギリの時はジッター自動付与（段階的に増やす）
        if min_eig <= 0.0:
            for eps in [1e-10, 1e-8, 1e-6, 1e-4, 1e-3]:
                cov_try = _add_jitter(self.noise_sigma, eps)
                me = torch.linalg.eigvalsh(cov_try.double().cpu()).min().item()
                print(f"[MPPI] add jitter {eps:g} => min_eig={me:.3e}")
                if me > 0:
                    self.noise_sigma = cov_try
                    break
            # まだ非正定値なら、ここで即停止して原因を見たい
            eig_final = torch.linalg.eigvalsh(self.noise_sigma.double().cpu()).min().item()
            if eig_final <= 0:
                raise RuntimeError(f"[MPPI] covariance not PD even after jitter, min_eig={eig_final}")

        # ---- MultivariateNormal 生成（GPU Cholesky バイパス版）----
        # まずは CPU で Cholesky を試し、成功したら scale_tril を渡す
        try:
            L_cpu = torch.linalg.cholesky(self.noise_sigma.double().cpu())
            L = L_cpu.to(dtype=self.noise_sigma.dtype, device=self.d)
            self.noise_dist = MultivariateNormal(self.noise_mu, scale_tril=L)
            print("[MPPI] MultivariateNormal initialized with CPU cholesky -> scale_tril")
        except Exception as e:
            print(f"[MPPI] CPU cholesky failed: {e}")
            # 最終手段として、GPU でそのまま covariance_matrix を渡す（元の実装）
            # ここでまた落ちるなら本当に cuSolver 側の問題
            self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
            print("[MPPI] MultivariateNormal initialized with covariance_matrix on device (GPU)")
        ##############################################################
        

        try:
            self.noise_sigma_inv = torch.linalg.pinv(self.noise_sigma)
        except RuntimeError as e:
            print("WARNING: pinv failed on CUDA. Switching to CPU.")
            self.noise_sigma_inv = torch.linalg.pinv(self.noise_sigma.cpu()).to(self.noise_sigma.device)    
            
        # self.noise_dist = MultivariateNormal(
        #     self.noise_mu, covariance_matrix=self.noise_sigma
        # )
        
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    @handle_batch_input(n=2)
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input(n=2)
    def _running_cost(self, state, u, t):
        return (
            self.running_cost(state, u, t)
            if self.step_dependency
            else self.running_cost(state, u)
        )

    def shift_nominal_trajectory(self):
        """
        Shift the nominal trajectory forward one step
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def command(self, state, shift_nominal_trajectory=True):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
        if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :returns action: (nu) best action
        """
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()

        return self._command(state)

    def _command(self, state): #一本の行動列を作る
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        
        print("[pldm/planning/planners/mppi_torch.py, 242] self.state.shape:", self.state.shape)
        with torch.inference_mode():
            print("[pldm/planning/planners/mppi_torch.py, 244] before cost:", torch.cuda.memory_allocated()/1024**2, "MB")
            cost_total = self._compute_total_cost_batch()
            torch.cuda.empty_cache()
            beta = torch.min(cost_total)
            self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
            eta = torch.sum(self.cost_total_non_zero)
            self.omega = (1.0 / eta) * self.cost_total_non_zero
            perturbations = []
            for t in range(self.T):
                perturbations.append(
                    torch.sum(self.omega.view(-1, 1) * self.noise[:, t], dim=0)
                )
            perturbations = torch.stack(perturbations)
            self.U = self.U + perturbations
            if self.u_per_command == -1:
                # return all the actions
                return self.U
            action = self.U[: self.u_per_command]
            # reduce dimensionality if we only need the first command
            if self.u_per_command == 1:
                action = action[0]
            return action

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            self.U = torch.cat(
                (self.U, self.u_init.repeat(horizon - self.U.shape[0], 1))
            )
        self.T = horizon

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        torch.cuda.reset_peak_memory_stats()
        print("[MEM] start MB=", torch.cuda.memory_allocated()/1024**2)

        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        state = state.repeat(self.M, 1, 1)

        # states = []
        actions = []
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            state, state_obs = self._dynamics(state, u, t)
            c = self._running_cost(state_obs, u, t)
            cost_samples = cost_samples + c
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount**t)

            # Save total states/actions
            # states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        # states = torch.stack(states, dim=-2)
        states = None

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples = cost_samples + c

        if self.latent_actions:
            # we assume we're regularizing towards a standard uniform prior for now
            prior_mus = torch.zeros_like(actions[0])
            prior_vars = torch.ones_like(actions[0])
            prior_d = torch.distributions.Normal(loc=prior_mus, scale=prior_vars)
            z_reg = -prior_d.log_prob(actions[0]).mean(dim=(1, 2))
            z_reg = z_reg * self.z_reg_coeff

        cost_total = cost_total + cost_samples.mean(dim=0)
        cost_total = cost_total + cost_var * self.rollout_var_cost
        return cost_total, states, actions

    def _compute_total_cost_batch(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        
        ##########################################################
        loc   = self.noise_dist.loc
        L     = self.noise_dist._unbroadcasted_scale_tril  # ※ scale_tril を使っている前提
        A     = loc.shape[-1]

        print("[CHK] loc   :", tuple(loc.shape),  loc.dtype,  loc.device, "finite=", bool(torch.isfinite(loc).all()))
        print("[CHK] L     :", tuple(L.shape),    L.dtype,    L.device,   "finite=", bool(torch.isfinite(L).all()))
        diag = torch.diagonal(L, 0, -2, -1)   # これを一度計算して…
        print("[CHK] diag>0:", (diag > 0).all().item(),
            "min=", diag.min().item(),
            "max=", diag.max().item())

        print("[CHK] K,T,A :", self.K, self.T, A)

        # まず極小バッチで単発テスト（ここで落ちたら分布が壊れてます）
        # test = self.noise_dist.rsample((1,1))
        # print("[CHK] single rsample OK:", test.shape)
        ##########################################################
        
        
        
        
        
        
        # noise = self.noise_dist.rsample((self.K, self.T))

        with torch.no_grad():
            # try:
            # 一時的に CPU 側の分布を作ってサンプル → GPU へ
            if self.noise_dist._unbroadcasted_scale_tril is not None:
                L_cpu = self.noise_dist._unbroadcasted_scale_tril.detach().double().cpu()
                mu_cpu = self.noise_dist.loc.detach().double().cpu()
                cpu_dist = MultivariateNormal(mu_cpu, scale_tril=L_cpu)
            else:
                Sigma_cpu = self.noise_sigma.detach().double().cpu()
                mu_cpu    = self.noise_mu.detach().double().cpu()
                cpu_dist  = MultivariateNormal(mu_cpu, covariance_matrix=Sigma_cpu)

            noise = cpu_dist.sample((self.K, self.T)).to(self.d, dtype=self.dtype, non_blocking=True)
            # except RuntimeError:
            #     # 最終手段：対角近似で手動サンプル（min_eig≈max_eig=1e-2 なら等方と同等）
            #     std = torch.sqrt(torch.diagonal(self.noise_sigma, 0, -2, -1))
            #     noise = (torch.randn(self.K, self.T, self.nu, device=self.d, dtype=self.dtype) * std + self.noise_mu)

        
        
        
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + noise
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = (
                self.lambda_ * self.noise @ self.noise_sigma_inv
            )  # Like original paper

        rollout_cost, self.states, actions = self._compute_rollout_costs(
            self.perturbed_action
        )
        self.actions = actions / self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.action_normalizer is not None:
            for t in range(self.T):
                u = action[:, self._slice_control(t)]
                cu = self.action_normalizer(u)  # double check
                action[:, self._slice_control(t)] = cu
        return action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def get_rollouts(self, state, num_rollouts=1):
        """
        :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
        :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                             dynamics
        :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros(
            (num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device
        )
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(
                states[:, t].view(num_rollouts, -1),
                self.u_scale * self.U[t].tile(num_rollouts, 1),
                t,
            )
        print("[MEM] after rollout MB=", torch.cuda.max_memory_allocated()/1024**2)
        return states[:, 1:]


def run_mppi(
    mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True
):
    dataset = torch.zeros(
        (retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d
    )
    total_reward = 0
    for i in range(iter):
        state = env.unwrapped.state.copy()
        command_start = time.perf_counter()
        action = mppi.command(state)
        elapsed = time.perf_counter() - command_start
        res = env.step(action.cpu().numpy())
        s, r = res[0], res[1]
        total_reward += r
        logger.debug(
            "action taken: %.4f cost received: %.4f time taken: %.5fs",
            action,
            -r,
            elapsed,
        )
        if render:
            env.render()

        di = i % retrain_after_iter
        if di == 0 and i > 0:
            retrain_dynamics(dataset)
            # don't have to clear dataset since it'll be overridden, but useful for debugging
            dataset.zero_()
        dataset[di, : mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[di, mppi.nx :] = action
    return total_reward, dataset
