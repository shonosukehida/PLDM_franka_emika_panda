from pldm_envs.franka.envs import FrankaSimEnv
from pldm_envs.utils.normalizer import Normalizer

class FrankaEnvsGenerator:
    def __init__(
        self, model_path: str, 
        n_envs: int = 10,
        normalizer: Normalizer = None,
                 ):
        self.model_path = model_path
        self.n_envs = n_envs
        self.normalizer = normalizer

    def __call__(self):
        envs = []
        for _ in range(self.n_envs):
            env = FrankaSimEnv(
                model_path=self.model_path,
                normalizer=self.normalizer
                )
            obs = env.reset()  # ← reset内でstart / goalをランダム決定＆保持
            envs.append(env)
        return envs
