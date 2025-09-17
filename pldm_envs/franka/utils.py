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
    return (x_min, x_max), (y_min, y_max)




def franka_pixel_mapper(
    coords,
    image_size: int = 64,
    x_range = (-0.018333, 1.048333),   # top-view の表示範囲（必要なら get_xy_range_from_model() で動的取得も可）
    y_range = (-0.533333, 0.533333),
    flip_x: bool = True,               # 画像座標に合わせて左右反転
    flip_y: bool = True,               # 画像座標に合わせて上下反転
    xyz_axes=(0, 1),                   # 3次元が来たときに使う2軸（デフォ: (x,y)）
):
    """
    coords  : [B,T,1,2] / [B,T,2] / [T,2] / [2]
              に加え、同型の **最後の次元=3** も許容（例: [B,T,3], [T,3], [3], [B,T,1,3]）
    return  : [B,T,2] の LongTensor（[x_pixel, y_pixel]）
    """
    coords = torch.as_tensor(coords)
    device, dtype = coords.device, coords.dtype

    # ---- (1) 最後の次元が3なら所望の2軸だけ抜く ----
    if coords.ndim >= 1 and coords.size(-1) == 3:
        xi, yi = xyz_axes
        if not (0 <= xi <= 2 and 0 <= yi <= 2 and xi != yi):
            raise ValueError(f"Invalid xyz_axes={xyz_axes}; must pick two distinct indices from {{0,1,2}}.")
        coords = coords[..., [xi, yi]]  # -> (..., 2)

    # ---- (2) 形を [B, T, 2] に正規化 ----
    if coords.ndim == 1 and coords.numel() == 2:          # [2] -> [1,1,2]
        coords = coords.view(1, 1, 2)
    elif coords.ndim == 2 and coords.size(-1) == 2:       # [T,2] -> [1,T,2]
        coords = coords.unsqueeze(0)
    elif coords.ndim == 3 and coords.size(-1) == 2:       # [B,T,2] -> そのまま
        pass
    elif coords.ndim == 4 and coords.size(-1) == 2:       # [B,T,1,2] -> [B,T,2]
        coords = coords.squeeze(-2)
    else:
        raise ValueError(
            f"Unexpected coords shape: {tuple(coords.shape)}. "
            "Expected [B,T,1,2] or [B,T,2] or [T,2] or [2] "
            "and the same forms with last-dim==3."
        )

    x0, x1 = x_range
    y0, y1 = y_range
    eps = 1e-12

    # world -> pixel（画像座標系に合わせて x/y を入れ替えている点に注意）
    x_world = coords[..., 0]
    y_world = coords[..., 1]

    # 画像yは上が0なので上下反転、xは左右反転のオプションあり
    y_pixel = (x_world - x0) / max((x1 - x0), eps) * (image_size - 1)
    x_pixel = (y_world - y0) / max((y1 - y0), eps) * (image_size - 1)

    if flip_y:
        y_pixel = (image_size - 1) - y_pixel
    if flip_x:
        x_pixel = (image_size - 1) - x_pixel

    x_pixel = x_pixel.clamp(0, image_size - 1).round().to(torch.long)
    y_pixel = y_pixel.clamp(0, image_size - 1).round().to(torch.long)

    pixel = torch.stack([x_pixel, y_pixel], dim=-1).to(device=device)
    return pixel





# def franka_pixel_mapper(
#     coords,
#     image_size: int = 64,
#     x_range = (-0.018333, 1.048333),   # world X の可視範囲（top_viewベース）
#     y_range = (-0.533333, 0.533333),   # world Y の可視範囲
#     flip_x: bool = True,               # 画像座標に合わせて左右反転するなら True
#     flip_y: bool = True,               # 画像座標に合わせて上下反転するなら True
# ):
#     """
#     coords: [B, T, 1, 2] or [B, T, 2] or [T, 2] or [2]
#     返り値: [B, T, 2]（[x_pixel, y_pixel] の順で返します）
#     """

#     # ---- 1) Tensor化 & デバイス/型維持 ----
#     coords = torch.as_tensor(coords)
#     device, dtype = coords.device, coords.dtype

#     # ---- 2) 形を [B, T, 2] に正規化 ----
#     if coords.ndim == 1 and coords.numel() == 2:          # [2] -> [1,1,2]
#         coords = coords.view(1, 1, 2)
#     elif coords.ndim == 2 and coords.size(-1) == 2:       # [T,2] -> [1,T,2]
#         coords = coords.unsqueeze(0)
#     elif coords.ndim == 3 and coords.size(-1) == 2:       # [B,T,2] -> そのまま
#         pass
#     elif coords.ndim == 4 and coords.size(-1) == 2:       # [B,T,1,2] -> [B,T,2]
#         coords = coords.squeeze(-2)
#     else:
#         raise ValueError(f"Unexpected coords shape: {tuple(coords.shape)}. "
#                          "Expected [B,T,1,2] or [B,T,2] or [T,2] or [2].")


#     x0, x1 = x_range
#     y0, y1 = y_range
#     eps = 1e-12
#     x_world = coords[..., 0]
#     y_world = coords[..., 1]


#     y_pixel = (x_world - x0) / (max(x1 - x0, eps)) * (image_size - 1)
#     x_pixel = (y_world - y0) / (max(y1 - y0, eps)) * (image_size - 1)

#     if flip_y:
#         y_pixel = (image_size - 1) - y_pixel
#     if flip_x:
#         x_pixel = (image_size - 1) - x_pixel


#     x_pixel = x_pixel.clamp(0, image_size - 1).round().to(torch.long)
#     y_pixel = y_pixel.clamp(0, image_size - 1).round().to(torch.long)


#     pixel = torch.stack([x_pixel, y_pixel], dim=-1).to(device=device)

#     return pixel




#保留
#関数内のx_range, y_range は, 現在のtop_view が捉える環境の範囲を目測で計算したもの
# def franka_pixel_mapper(coords, image_size=64):
    
#     """
#     coords: Tensor[B, T, 1, 2] or [B, T, 2] or [T, 2]
#     """

#     coords = coords.clone()
#     x_range, y_range = [-0.018333, 1.048333], [-0.533333,0.533333]


#     y_pixel = (coords[:, :, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (image_size - 1)
#     y_pixel = image_size - 1 - y_pixel
    
#     x_pixel = (coords[:, :, 1] - y_range[0]) / (y_range[1] - y_range[0]) * (image_size - 1)
#     x_pixel = image_size - 1 - x_pixel
    
#     # pixel = torch.stack([x_pixel, y_pixel], dim = -1)
#     pixel = torch.stack([y_pixel, x_pixel], dim = -1)


#     return pixel

