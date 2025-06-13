# planning/franka/enums.py
from dataclasses import dataclass
from pldm.planning.enums import MPCConfig

@dataclass
class FrankaMPCConfig(MPCConfig):
    goal_pos_noise: float = 0.01
    use_ik: bool = True
    model_path: str = "mujoco_menagerie/franka_emika_panda/scene.xml"
