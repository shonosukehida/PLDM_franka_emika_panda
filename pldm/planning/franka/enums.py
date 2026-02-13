# planning/franka/enums.py
from dataclasses import dataclass, field
from pldm.planning.enums import MPCConfig
from typing import Optional, List



@dataclass
class DesignateStartGoalPosConfig:
    valid: bool = False
    start_pos: Optional[List[float]] = None
    goal_pos: Optional[List[float]] = None


@dataclass
class ReachNoTouchConfig:
    ee_goal_xyz: List[float] = (0.55, 0.0, 0.10)
    box_init_xyz: Optional[List[float]] = None
    box_hold_eps: float = 0.02
    ee_success_eps: float = 0.03
    use_box: bool = True


@dataclass
class PushToGoalConfig:
    success_thresh: float = 0.05


@dataclass
class TaskConfig:
    name: str = "push_to_goal"  # "push_to_goal" or "reach_no_touch"
    reach_no_touch: ReachNoTouchConfig = field(default_factory=ReachNoTouchConfig)
    push_to_goal: PushToGoalConfig = field(default_factory=PushToGoalConfig)

@dataclass
class FrankaMPCConfig(MPCConfig):
    goal_pos_noise: float = 0.01
    use_ik: bool = True
    model_path: str = "mujoco_menagerie/franka_emika_panda/scene.xml"
    fps: float = 2.5
    max_dq: float = 0.01
    max_inner_steps: int = 50
    reach_eps: float = 1e-3
    
    task: TaskConfig = TaskConfig()
    
    pred_steps: int = 50
    val_from_train_ds: bool = False
    designate_start_goal_pos: Optional[DesignateStartGoalPosConfig] = field(
        default_factory=DesignateStartGoalPosConfig
    )
    
    backbone_arch: str = 'menet6'
    vjepa2_repo: str = ''
    
