from dataclasses import dataclass
from typing import Optional

@dataclass
class FrankaConfig:
    path: Optional[str] = None
    images_path: Optional[str] = None
    goal_images_path: Optional[str] = None
    use_goal_images: bool = True
