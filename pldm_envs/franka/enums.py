from dataclasses import dataclass
from typing import Optional

from typing import NamedTuple
import torch


class FrankaSample(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    locations: torch.Tensor
    indices: int
    propio_vel: Optional[torch.Tensor] = None
    propio_pos: Optional[torch.Tensor] = None


@dataclass
class FrankaConfig:
    env_name: Optional[str] = None
    path: Optional[str] = None
    images_path: Optional[str] = None
    goal_images_path: Optional[str] = None

    num_workers: int = 0
    batch_size: int = 64
    seed: int = 0
    normalize: bool = True
    quick_debug: bool = False
    val_fraction: float = 0.2
    train: bool = True
    sample_length: int = 15
    location_only: bool = False
    prioritized: bool = False
    alpha: float = 0.6
    beta: float = 0.4
    crop_length: Optional[int] = None
    stack_states: int = 1
    img_size: int = 64
    random_actions: bool = False

    use_goal_images: bool = True  # PLDM特有のオプション

    def __post_init__(self):
        pass  # 将来追加があればここで処理
