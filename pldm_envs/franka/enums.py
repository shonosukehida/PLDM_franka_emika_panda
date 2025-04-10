from dataclasses import dataclass
from typing import Optional

@dataclass
class FrankaConfig:
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    use_goal_images: bool = True

    def __post_init__(self):
        if self.train_dir is None:
            raise ValueError("FrankaConfig.train_dir must be specified.")

