from pldm.logger import Logger
from typing import List
from matplotlib import pyplot as plt
from pldm.logger import Logger
import math
from tqdm import tqdm
import numpy as np
import torch
import sys

default_plot_idxs = list(range(100))

#直交座標 --> ピクセル座標
def log_planning_plots(
    result,
    report,
    idxs: List[int] = default_plot_idxs,
    prefix: str = "wall_",
    n_steps: int = 44,
    xy_action: bool = True,
    plot_every: int = 1,
    quick_debug: bool = False,
    pixel_mapper=None,
    plot_failure_only: bool = False,
    log_pred_dist_every: int = sys.maxsize,
    mark_action: bool = True,
):
    num_plots = math.ceil(len(result.locations) / plot_every)
    grid_size = max(4, math.ceil(math.sqrt(num_plots)))

    img_size = result.observations[0][0].shape[-1]

    #直交座標 --> pixel座標に変換
    if pixel_mapper is not None: ##
        # targets_pixels = torch.as_tensor(
        #     pixel_mapper.obs_coord_to_pixel_coord(result.targets)
        # )
        targets_pixels = torch.as_tensor(
            pixel_mapper(result.targets)
        )

        # locations_pixels = [
        #     torch.as_tensor(pixel_mapper.obs_coord_to_pixel_coord(x))
        #     for x in result.locations
        # ]
        locations_pixels = [
            torch.as_tensor(pixel_mapper(x)) for x in result.locations
        ]

        # pred_locations_pixels = [
        #     torch.as_tensor(pixel_mapper.obs_coord_to_pixel_coord(x))
        #     for x in result.pred_locations
        # ]
        pred_locations_pixels = [
            torch.as_tensor(pixel_mapper(x)) for x in result.pred_locations
        ]
    else: #x
        targets_pixels = result.targets
        locations_pixels = result.locations
        pred_locations_pixels = result.pred_locations

    #時系列で画像を描画
    if idxs is None:
        idxs = default_plot_idxs
    for idx in idxs:
        if plot_failure_only and report.success[idx]:
            continue

        fig = plt.figure(dpi=300)
        start_location = locations_pixels[0][idx].cpu()
        subplot_idx = 0
        for i in range(len(locations_pixels)):
            if i % plot_every:
                continue

            if i > report.terminations[idx]:
                break
            plt.subplot(grid_size, grid_size, subplot_idx + 1)
            
            

            if "wall" in prefix:
                img = -1 * result.observations[i][idx].sum(dim=0).detach().cpu()
            elif result.observations[i][idx].shape[0] > 1:
                # if multiple channels, need to convert to grayscale
                img = result.observations[i][idx].detach().cpu().numpy()  # (3, 64, 64)
                img = np.transpose(img, (1, 2, 0))
                # Convert to grayscale using the weighted sum of RGB channels
                img = (
                    0.2989 * img[:, :, 0]
                    + 0.5870 * img[:, :, 1]
                    + 0.1140 * img[:, :, 2]
                )
                
                # Normalize the grayscale image to the range [0, 1]
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = result.observations[i][idx][0]

            plt.imshow(
                img,
                cmap="gray",
            )
            current_location = locations_pixels[i][idx].detach().cpu()
            if i != len(locations_pixels) - 1:
                # skip last one as there's no action at the last timestep
                action = result.action_history[i][idx, 0].detach()
                if not xy_action:
                    action = DotDataset.polar_to_xy(action)
                action = action * 5  # for visibility

                if mark_action:
                    plt.arrow(
                        x=current_location[0],
                        y=current_location[1],
                        dx=action[0].cpu(),
                        dy=action[1].cpu(),
                        width=0.05,
                        color="#F77F00",
                        head_width=2,
                    )

                if pred_locations_pixels is not None:
                    plt.plot(
                        pred_locations_pixels[i][:, idx, 0].detach().cpu(),
                        pred_locations_pixels[i][:, idx, 1].detach().cpu(),
                        marker="o",
                        markersize=0.1,
                        linewidth=0.1,
                        c="red",
                        alpha=1,
                    )

                    if log_pred_dist_every < 999999:
                        final_pred_dists = result.final_preds_dist[i][:, idx].tolist()
                        # skip every nth. make sure to include first and last
                        last_dist = final_pred_dists[-1]
                        final_pred_dists = final_pred_dists[::log_pred_dist_every]
                        if last_dist != final_pred_dists[-1]:
                            final_pred_dists.append(last_dist)

                        plt.text(
                            1.05,
                            0.5,
                            "\n".join([f"{dist:.2f}" for dist in final_pred_dists]),
                            transform=plt.gca().transAxes,
                            fontsize=2.5,
                            verticalalignment="center",
                            horizontalalignment="left",
                            color="blue",
                        )

            plt.scatter(
                start_location[0],
                start_location[1],
                s=0.1,
                c="blue",
                marker="o",
                alpha=1,
            )

            plt.scatter(
                targets_pixels[idx, 0].cpu(),
                targets_pixels[idx, 1].cpu(),
                s=0.1,
                c="#F77F00",
                marker="o",
                alpha=1,
            )
            plt.xlim(0, img_size - 1)
            plt.ylim(img_size - 1, 0)
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            subplot_idx += 1


        log_name = f"mpc/{prefix}_{idx}"
        if plot_failure_only:
            start_x = result.locations[0][idx][0]
            start_y = result.locations[0][idx][1]
            target_x = result.targets[idx][0]
            target_y = result.targets[idx][1]
            log_name += (
                f"_{int(start_x)}_{int(start_y)}_{int(target_x)}_{int(target_y)}"
            )

        Logger.run().log_figure(fig, log_name)
        plt.close(fig)


def log_l1_planning_loss(result, prefix: str = "wall_"):
    logger = Logger.run()
    steps = [0, len(result.loss_history) // 2]

    for step in steps:
        losses = result.loss_history[step]
        for loss in losses:
            log_dict = {f"{prefix}_l1_step_{step}_plan_loss": loss}
            logger.log(log_dict)
        logger.log({f"{prefix}_l1_step_{step}_plan_iterations": len(losses)})



def log_planning_plots_split(
    result, report, idxs=None,
    plot_every=1, xy_action=True,
    plot_failure_only=False,
    world_xlim=None, world_ylim=None,
    use_pixel_mapper=False, pixel_mapper=None,
):

    if idxs is None:
        idxs = default_plot_idxs

    T = len(result.locations)
    B = result.observations[0].shape[0]
    H = result.observations[0][0].shape[-2]
    W = result.observations[0][0].shape[-1]
    
    ###############################################################################################
    print("[DBG] has object_history:", hasattr(result, "object_history"))
    if hasattr(result, "object_history"):
        print("[DBG] len(object_history):", len(result.object_history))
        if len(result.object_history) > 0:
            oh0 = result.object_history[0]
            try:
                print("[DBG] object_history[0] shape:", getattr(oh0, "shape", type(oh0)))
            except Exception as e:
                print("[DBG] object_history[0] type:", type(oh0), "err:", e)
    ###############################################################################################

    #図A:観測のみ
    num_panels = math.ceil(T / plot_every)
    grid = max(4, math.ceil(math.sqrt(num_panels)))

    for idx in idxs:
        if plot_failure_only and report.success[idx]:
            continue

        figA = plt.figure(dpi=250)
        p = 0
        for t in range(T):
            if t % plot_every: continue
            if t > report.terminations[idx]: break
            plt.subplot(grid, grid, p+1)

            obs = result.observations[t][idx]
            if obs.shape[0] > 1:  
                img = obs.detach().cpu().numpy().transpose(1,2,0)
                # img = 0.2989*img[...,0] + 0.5870*img[...,1] + 0.1140*img[...,2]
                img = (img - img.min())/(img.max()-img.min() + 1e-12)
            else:
                img = obs[0].detach().cpu().numpy()

            plt.imshow(img)
            plt.xticks([]); plt.yticks([])
            p += 1

        Logger.run().log_figure(figA, f"mpc/obs_seq_{idx}")
        plt.close(figA)

        #図B: 軌跡のみ
        starts = result.locations[0][idx].detach().cpu()        # (2,)
        targets = result.targets[idx].detach().cpu()            # (2,)
        traj = torch.stack([result.locations[t][idx] for t in range(T)], dim=0).detach().cpu()  # (T,2)

        # 予測列：各tで H 分の予測
        preds = []
        T_loc  = len(result.locations)         # 初期+各ステップ → T+1
        T_pred = len(result.pred_locations)    # 各ステップ → T
        t_term = getattr(report, "terminations", [T_loc-1])[idx]
        t_max = min(T_pred, T_loc - 1, t_term + 1)
        for t in range(t_max):
            if t > report.terminations[idx]: break
            preds.append(result.pred_locations[t][:, idx, :].detach().cpu())  # (H,2)


        obj_traj = None
        if getattr(result, "object_history", None):  
            T_obj = len(result.object_history)
            t_term = getattr(report, "terminations", [T - 1])[idx]  # report が無ければ T-1
            tt = min(T, T_obj, t_term + 1) 

            if tt > 0:
                obj_traj = torch.stack(
                    [result.object_history[t][idx] for t in range(tt)],
                    dim=0
                ).cpu()
                # xyz→xy
                if obj_traj.shape[-1] == 3:
                    obj_traj = obj_traj[:, :2]


            
        #pixel変換(使う予定なし)
        if use_pixel_mapper and pixel_mapper is not None:
            starts = pixel_mapper(starts).squeeze().float()
            targets = pixel_mapper(targets).squeeze().float()
            traj = pixel_mapper(traj).squeeze().float()
            preds = [pixel_mapper(p).squeeze().float() for p in preds]
            if obj_traj is not None:                        
                obj_traj = pixel_mapper(obj_traj).squeeze().float()

        figB = plt.figure(dpi=250)
        ax = plt.gca()


        # ax.scatter(starts[0],  starts[1],  s=12, c="black",  label="start")
        ax.scatter(targets[0], targets[1], s=12, c="tab:orange", label="goal", zorder=5)
        ax.scatter(traj[:,0],  traj[:,1],  s=8,  c="black",  alpha=0.9, label="end-effector", zorder=6)
        ax.text(traj[0, 0], traj[0, 1], "S", fontsize=10, color="black", ha="center", va="center", fontweight="bold", zorder=7)

        if preds:
            for t, P in enumerate(preds):
                if t % plot_every:  continue
                ax.plot(P[:,0], P[:,1], lw=0.6, alpha=0.6, c="red", zorder=2)
                ax.scatter(P[0, 0], P[0, 1], s=10, c="lime", marker="o", zorder=1)


        if obj_traj is not None and len(obj_traj) > 0:
            ax.plot(obj_traj[:,0], obj_traj[:,1], lw=1.2, c="tab:blue", label="bluebox", zorder=4)
            ax.scatter(obj_traj[0,0], obj_traj[0,1], s=14, c="tab:blue", marker="x", label="bluebox_start", zorder=3)
            


        # 環境grid可視化
        if world_xlim is not None and world_ylim is not None:
            xmin, xmax = world_xlim
            ymin, ymax = world_ylim


            if use_pixel_mapper and pixel_mapper is not None:
                box_xy = torch.tensor([[xmin, ymin],
                                    [xmax, ymin],
                                    [xmax, ymax],
                                    [xmin, ymax],
                                    [xmin, ymin]], dtype=torch.float32)
                box_xy = pixel_mapper(box_xy).squeeze().float().cpu().numpy()
                ax.plot(box_xy[:,0], box_xy[:,1],
                        linestyle="--", linewidth=1.0, alpha=0.6,
                        color="black",)
            else:
                from matplotlib.patches import Rectangle
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, linewidth=1.0, linestyle="--",
                                edgecolor="black", alpha=0.6, zorder=0)
                ax.add_patch(rect)

        ax.set_aspect("equal")
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        Logger.run().log_figure(figB, f"mpc/prediction_seq_{idx}")
        plt.close(figB)
