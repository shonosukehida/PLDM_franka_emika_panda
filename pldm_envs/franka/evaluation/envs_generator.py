from pldm_envs.franka.envs import FrankaSimEnv
from pldm_envs.utils.normalizer import Normalizer

class FrankaEnvsGenerator:
    def __init__(
        self, 
        model_path: str, 
        n_envs: int = 10,
        normalizer: Normalizer = None,
        max_dq: float = 0.01,
        task_name: str = "push_to_goal",
        task_cfg = None,
                 ):
        self.model_path = model_path
        self.n_envs = n_envs
        self.normalizer = normalizer
        self.max_dq = max_dq
        
        self.task_name = task_name
        self.task_cfg = task_cfg

    def __call__(self):
        envs = []
        for _ in range(self.n_envs):
            env = FrankaSimEnv(
                model_path=self.model_path,
                normalizer=self.normalizer,
                max_dq = self.max_dq,
                task_name = self.task_name,
                task_cfg = self.task_cfg,
                )
            obs = env.reset()  # ← reset内でstart / goalをランダム決定＆保持
            envs.append(env)
        return envs
