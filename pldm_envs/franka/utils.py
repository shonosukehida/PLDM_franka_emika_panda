import torch
import mujoco
import yaml

def get_xy_range_from_model():
    with open("robot_sim/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    MODEL_PATH = config["model_path"]
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_left")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_right")
    bottom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_bottom")
    top_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall_top")

    x_min = model.geom_pos[left_id][0] - model.geom_size[left_id][0]
    x_max = model.geom_pos[right_id][0] + model.geom_size[right_id][0]
    y_min = model.geom_pos[bottom_id][1] - model.geom_size[bottom_id][1]
    y_max = model.geom_pos[top_id][1] + model.geom_size[top_id][1]

    return (x_min, x_max), (y_min, y_max)


def franka_pixel_mapper(coords, image_size=64):
    
    """
    coords: Tensor[B, T, 1, 2] or [B, T, 2] or [T, 2]
    """

    coords = coords.clone()
    x_range, y_range = get_xy_range_from_model()
    
    # x_range = (-0.201, 0.701)
    # y_range = (-0.401, 0.401)


    x = (coords[..., 0] - x_range[0]) / (x_range[1] - x_range[0]) * (image_size - 1)
    y = (coords[..., 1] - y_range[0]) / (y_range[1] - y_range[0]) * (image_size - 1)

    # matplotlib 用に上下反転
    pixel = torch.stack([x, y], dim=-1)
    pixel[..., 1] = image_size - 1 - pixel[..., 1]

    return pixel


print(get_xy_range_from_model())