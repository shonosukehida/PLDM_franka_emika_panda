from dataclasses import dataclass

@dataclass
class FrankaConfig:
    data_dir: str
    use_goal_images: bool = True
