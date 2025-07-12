import torch
import mujoco
import yaml

def get_xy_range_from_model():
    with open("robot_sim/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    MODEL_PATH = config["model_path"]
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    def get_geom_bound(name, axis, sign):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id == -1:
            raise ValueError(f"{name} geom not found in model!")
        return model.geom_pos[geom_id][axis] + sign * model.geom_size[geom_id][axis]

    y_max = get_geom_bound("wall_left", 1, -1)
    y_min = get_geom_bound("wall_right", 1, +1)
    x_min = get_geom_bound("wall_bottom", 0, +1)
    x_max = get_geom_bound("wall_top", 0, -1)
    
    print('BOUND')
    print('X', x_min, x_max)
    print('Y', y_min, y_max)

    return (x_min, x_max), (y_min, y_max)

#保留
#関数内のx_range, y_range は, 現在のtop_view が捉える環境の範囲を目測で計算したもの
def franka_pixel_mapper(coords, image_size=64):
    
    """
    coords: Tensor[B, T, 1, 2] or [B, T, 2] or [T, 2]
    """

    coords = coords.clone()
    x_range, y_range = [-0.018333, 1.048333], [-0.533333,0.533333]


    y_pixel = (coords[:, :, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (image_size - 1)
    y_pixel = image_size - 1 - y_pixel
    
    x_pixel = (coords[:, :, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (image_size - 1)
    x_pixel = image_size - 1 - x_pixel
    
    # pixel = torch.stack([x_pixel, y_pixel], dim = -1)
    pixel = torch.stack([y_pixel, x_pixel], dim = -1)


    return pixel


print(get_xy_range_from_model())