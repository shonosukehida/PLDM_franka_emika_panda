# planning/franka/enums.py
from dataclasses import dataclass
from pldm.planning.enums import MPCConfig

@dataclass
class FrankaMPCConfig(MPCConfig):
    goal_pos_noise: float = 0.01
    use_ik: bool = True
    model_path: str = "mujoco_menagerie/franka_emika_panda/scene.xml"
    fps: float = 2.5
    max_dq: float = 0.01
    max_inner_steps: int = 50
    reach_eps: float = 1e-3
