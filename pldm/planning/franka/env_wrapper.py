# pldm/planning/franka/env_wrapper.py
import numpy as np
from pldm_envs.franka.envs import FrankaSimEnv


class HoldUntilReachWrapper:
    """
    高レベルの 1 step で同じ action（目標関節角）を繰り返し送り続け、
    qpos がその action に十分近づいたら次のステップに進むラッパー。
    """

    def __init__(self, env: FrankaSimEnv, reach_eps: float = 1e-3, max_inner_steps: int = 50):
        self.env = env
        self.reach_eps = reach_eps
        self.max_inner_steps = max_inner_steps
        print("[HoldUntilReachWrapper] reach_eps=", self.reach_eps,
      "max_inner_steps=", self.max_inner_steps)


    # --- 未定義の属性は全部内側の env に委譲する ---
    def __getattr__(self, name):
        return getattr(self.env, name)

    
    

    def step(self, action):
        action = np.asarray(action, np.float32).reshape(-1)

        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        obs = None

        for _ in range(self.max_inner_steps):
            qpos = self.env.physics.data.qpos[:7].copy()

            if np.linalg.norm(action - qpos) < self.reach_eps:
                obs = self.env.get_obs()
                info = self.env.get_info()
                break

            obs, r, done, truncated, info = self.env.step(action)
            total_reward += r

            # if done or truncated:
            #     break

        # 念のため：max_inner_steps=0 などで obs が一度も更新されなかった場合
        if obs is None:
            obs = self.env.get_obs()
            info = self.env.get_info()

        return obs, total_reward, done, truncated, info


class ActionRepeatWrapper:
    def __init__(self, env, repeat):
        self.env = env
        self.repeat = int(repeat)
        # 物理dt
        self.dt = float(env.physics.model.opt.timestep) * getattr(env, "substeps", 1) * self.repeat
    def reset(self, *a, **kw): return self.env.reset(*a, **kw)
    def get_info(self): return self.env.get_info()
    def get_obs(self, normalize=True): return self.env.get_obs(normalize)
    def step(self, action):
        total_r, done, trunc, info = 0.0, False, False, None
        obs = None
        for _ in range(self.repeat):
            obs, r, done, trunc, info = self.env.step(action)
            total_r += r
            if done or trunc:
                break
        return obs, total_r, done, trunc, info
    def __getattr__(self, name):  # 既存属性に委譲
        return getattr(self.env, name)