from pldm.logger import Logger
from typing import List
from matplotlib import pyplot as plt
from pldm.logger import Logger
import math
from tqdm import tqdm
import numpy as np
import torch
import sys
import gc
from matplotlib.patches import Rectangle
from pathlib import Path
import imageio.v2 as imageio
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
    env = None, #運動学計算用
    plot_action = True,
):

    if idxs is None:
        idxs = default_plot_idxs

    T = len(result.locations)
    B = result.observations[0].shape[0]
    H = result.observations[0][0].shape[-2]
    W = result.observations[0][0].shape[-1]


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
        single_step_preds = []
        T_loc  = len(result.locations)         # 初期+各ステップ → T+1
        # T_pred = len(result.pred_locations)    # 各ステップ → T
        T_act  = len(getattr(result, "action_history", []))
        t_term = getattr(report, "terminations", [T_loc-1])[idx]
        t_max = min(T_act, T_loc - 1, t_term + 1)
        
        env.reset()
        for t in range(t_max):
            if t > report.terminations[idx]: break
            # preds.append(result.pred_locations[t][:, idx, :].detach().cpu())  # (H,2)
            
            # print("[DBG: log_planning_plots_split/pldm/planning/plotting.py] action_history.len: ",  len(result.action_history)) #200 = T
            # print("[DBG: log_planning_plots_split/pldm/planning/plotting.py] action_history[0].shape: ",  result.action_history[0].shape) #(10, 100, 7)=(num_envs, H, dim)
            
            
            #運動学によるMPPI予測軌跡
            act_seq = result.action_history[t][idx].detach().cpu().numpy()   # (H_t, 7)
            a0 = act_seq[0]  
            # print("act.shape: ", act.shape)
            env.set_joint(a0)
            ee = env.get_ee_position()[:2]
            single_step_preds.append(ee)

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


            
        figB = plt.figure(dpi=250)
        ax = plt.gca()


        # ax.scatter(starts[0],  starts[1],  s=12, c="black",  label="start")
        ax.scatter(targets[0], targets[1], s=12, c="tab:orange", label="goal", zorder=5)
        ax.scatter(traj[:,0],  traj[:,1],  s=8,  c="black",  alpha=0.9, label="end-effector", zorder=6)
        ax.text(traj[0, 0], traj[0, 1], "S", fontsize=10, color="black", ha="center", va="center", fontweight="bold", zorder=7)
        ax.text(traj[-1, 0], traj[-1, 1], "G", fontsize=10, color="black", ha="center", va="center", fontweight="bold", zorder=7)


        if single_step_preds and plot_action:
            single_step_preds = np.asarray(single_step_preds, dtype=np.float32)

            for t, ee_pred in enumerate(single_step_preds):
                if t % plot_every: 
                    continue

                cur = traj[t]   # 実際の EE 位置 (result.locations[t][idx])

                # 矢印（現在 → 1ステップ後の予測位置）
                ax.plot([cur[0], ee_pred[0]], [cur[1], ee_pred[1]],
                        lw=0.5, alpha=0.7, c="red", zorder=2)

                # 1ステップ予測点
                ax.scatter(ee_pred[0], ee_pred[1], s=10, c="lime",
                        marker="o", zorder=1)



        if obj_traj is not None and len(obj_traj) > 0:
            ax.plot(obj_traj[:,0], obj_traj[:,1],
                    lw=1.2, c="tab:blue", label="bluebox_center", zorder=4)
            ax.scatter(obj_traj[0,0], obj_traj[0,1],
                    s=14, c="tab:blue", marker="x", label="bluebox_start", zorder=3)

            # ---- bluebox の大きさ（half extent）----
            hx, hy = 0.05, 0.05  # XML の size と対応

            # 現在（最後）の位置に矩形を描画
            cx, cy = obj_traj[-1, 0].item(), obj_traj[-1, 1].item()
            rect = Rectangle(
                (cx - hx, cy - hy),    # 左下
                2 * hx, 2 * hy,        # 幅・高さ
                fill=False,
                linewidth=1.2,
                edgecolor="tab:blue",
                alpha=0.8,
                zorder=4.5,
            )
            ax.add_patch(rect)


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
                
                rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                fill=False, linewidth=1.0, linestyle="--",
                                edgecolor="black", alpha=0.6, zorder=0)
                ax.add_patch(rect)

        ax.set_aspect("equal")
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        Logger.run().log_figure(figB, f"mpc/prediction_seq_{idx}")
        plt.close(figB)


        # figを必ず閉じる（ログ直後にやるのがベスト）
        try: plt.close(figA)
        except: pass
        try: plt.close(figB)
        except: pass


        for _n in ('ax', 'traj', 'obj_traj'):
            if _n in locals():
                try: del locals()[_n]
                except: pass


        if 'preds' in locals() and isinstance(preds, list):
            try: preds.clear()
            except: pass
            try: del preds
            except: pass
            
        if 'single_step_preds' in locals() and isinstance(single_step_preds, list):
            try: single_step_preds.clear()
            except: pass
            try: del single_step_preds
            except: pass

        for _n in ('img', 'obs', 'P'):
            if _n in locals():
                try: del locals()[_n]
                except: pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()




# def log_planning_traj_plots_split(
#     result, report, idxs=None,
#     plot_every=1, xy_action=True,
#     plot_failure_only=False,
#     world_xlim=None, world_ylim=None,
#     use_pixel_mapper=False, pixel_mapper=None,
#     env=None,  # 運動学計算用
#     plot_action = True,
# ):
#     """
#     図B: EE 軌跡 + 1ステップ予測 + bluebox + ワールド範囲 を可視化
#     """
#     if idxs is None:
#         idxs = default_plot_idxs

#     T = len(result.locations)
#     B = result.observations[0].shape[0]

#     for idx in idxs:
#         if plot_failure_only and getattr(report, "success", [False] * B)[idx]:
#             continue

#         starts = result.locations[0][idx].detach().cpu()        # (2,)
#         targets = result.targets[idx].detach().cpu()            # (2,)
#         traj = torch.stack(
#             [result.locations[t][idx] for t in range(T)],
#             dim=0
#         ).detach().cpu()                                        # (T,2)

#         # ---- 1step 予測列 ----
#         single_step_preds = []
#         T_loc  = len(result.locations)         # 初期+各ステップ → T+1
#         T_pred = len(result.pred_locations)    # 各ステップ → T
#         t_term = getattr(report, "terminations", [T_loc - 1])[idx]
#         t_max = min(T_pred, T_loc - 1, t_term + 1)

#         if env is not None:
#             env.reset()
#             for t in range(t_max):
#                 if t > report.terminations[idx]:
#                     break
#                 act_seq = result.action_history[t][idx].detach().cpu().numpy()  # (H_t,7)
#                 a0 = act_seq[0]
#                 env.set_joint(a0)
#                 ee = env.get_ee_position()[:2]
#                 single_step_preds.append(ee)

#         # ---- オブジェクト軌跡 (bluebox) ----
#         obj_traj = None
#         if getattr(result, "object_history", None):
#             T_obj = len(result.object_history)
#             t_term = getattr(report, "terminations", [T - 1])[idx]
#             tt = min(T, T_obj, t_term + 1)

#             if tt > 0:
#                 obj_traj = torch.stack(
#                     [result.object_history[t][idx] for t in range(tt)],
#                     dim=0
#                 ).cpu()
#                 if obj_traj.shape[-1] == 3:  # xyz → xy
#                     obj_traj = obj_traj[:, :2]

#         figB = plt.figure(dpi=250)
#         ax = plt.gca()

#         # EE 軌跡 + start/end
#         ax.scatter(targets[0], targets[1],
#                    s=12, c="tab:orange",
#                    label="goal", zorder=5)
#         ax.scatter(traj[:, 0], traj[:, 1],
#                    s=8, c="black", alpha=0.9,
#                    label="end-effector", zorder=6)
#         ax.text(traj[0, 0], traj[0, 1], "S",
#                 fontsize=10, color="black",
#                 ha="center", va="center",
#                 fontweight="bold", zorder=7)
#         ax.text(traj[-1, 0], traj[-1, 1], "G",
#                 fontsize=10, color="black",
#                 ha="center", va="center",
#                 fontweight="bold", zorder=7)

#         # ---- 1step 予測の可視化 ----
#         if len(single_step_preds) > 0 and plot_action:
#             single_step_preds_np = np.asarray(single_step_preds, dtype=np.float32)

#             for t, ee_pred in enumerate(single_step_preds_np):
#                 if t % plot_every:
#                     continue

#                 cur = traj[t]  # 実際の EE 位置
#                 ax.plot(
#                     [cur[0], ee_pred[0]],
#                     [cur[1], ee_pred[1]],
#                     lw=0.5, alpha=0.7, c="red", zorder=2
#                 )
#                 ax.scatter(
#                     ee_pred[0], ee_pred[1],
#                     s=10, c="lime", marker="o", zorder=1
#                 )

#         # ---- bluebox の中心軌跡 + box 枠 ----
#         if obj_traj is not None and len(obj_traj) > 0:
#             ax.plot(obj_traj[:, 0], obj_traj[:, 1],
#                     lw=1.2, c="tab:blue",
#                     label="bluebox_center", zorder=4)
#             ax.scatter(obj_traj[0, 0], obj_traj[0, 1],
#                        s=14, c="tab:blue",
#                        marker="x", label="bluebox_start", zorder=3)

#             hx, hy = 0.05, 0.05  # bluebox の半径（XML の size と対応）
#             cx, cy = obj_traj[-1, 0].item(), obj_traj[-1, 1].item()
#             rect_bb = Rectangle(
#                 (cx - hx, cy - hy),
#                 2 * hx, 2 * hy,
#                 fill=False,
#                 linewidth=1.2,
#                 edgecolor="tab:blue",
#                 alpha=0.8,
#                 zorder=4.5,
#             )
#             ax.add_patch(rect_bb)

#         # ---- ワークスペース枠 ----
#         if world_xlim is not None and world_ylim is not None:
#             xmin, xmax = world_xlim
#             ymin, ymax = world_ylim

#             rect_ws = Rectangle(
#                 (xmin, ymin),
#                 xmax - xmin, ymax - ymin,
#                 fill=False, linewidth=1.0, linestyle="--",
#                 edgecolor="black", alpha=0.6, zorder=0
#             )
#             ax.add_patch(rect_ws)

#         ax.set_aspect("equal")
#         ax.grid(True, ls=":", lw=0.5, alpha=0.5)
#         ax.legend(fontsize=8, loc="best")
        
#         ax.set_xlabel("X [m]")
#         ax.set_ylabel("Y [m]")

#         Logger.run().log_figure(figB, f"mpc/prediction_seq_{idx}")
#         plt.close(figB)

#         # メモリ掃除
#         try:
#             del figB, ax, traj, obj_traj, single_step_preds
#         except Exception:
#             pass

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#             torch.cuda.empty_cache()




#タスク時の俯瞰視点観測を並べたもの
def log_planning_obs_plots_split(
    result, report, idxs=None,
    plot_every=1, xy_action=True,
    plot_failure_only=False,
    world_xlim=None, world_ylim=None,
    use_pixel_mapper=False, pixel_mapper=None,
    env=None,  # 使わないけどシグネチャ合わせ
):
    """
    図A: 観測列 (result.observations) のみをグリッド表示して保存
    """
    if idxs is None:
        idxs = default_plot_idxs

    T = len(result.locations)
    B = result.observations[0].shape[0]
    H = result.observations[0][0].shape[-2]
    W = result.observations[0][0].shape[-1]

    num_panels = math.ceil(T / plot_every)
    grid = max(4, math.ceil(math.sqrt(num_panels)))

    for idx in idxs:
        if plot_failure_only and getattr(report, "success", [False] * B)[idx]:
            continue

        figA = plt.figure(dpi=250)
        p = 0
        for t in range(T):
            if t % plot_every:
                continue
            if t > report.terminations[idx]:
                break

            plt.subplot(grid, grid, p + 1)

            obs = result.observations[t][idx]
            if obs.shape[0] > 1:
                img = obs.detach().cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-12)
            else:
                img = obs[0].detach().cpu().numpy()

            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            p += 1

        Logger.run().log_figure(figA, f"mpc/obs_seq_{idx}")
        plt.close(figA)

        # メモリ掃除
        try:
            del figA, img, obs
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()



def log_planning_videos_split(
    result, report, idxs=None,
    plot_every=1, xy_action=True,
    plot_failure_only=False,
    world_xlim=None, world_ylim=None,
    use_pixel_mapper=False, pixel_mapper=None,
    env=None,  # 未使用
):
    """
    タスク実行時の観測列(result.observations)を動画(mp4)にして保存・ログする。
    """

    if idxs is None:
        idxs = default_plot_idxs

    T = len(result.observations)
    B = result.observations[0].shape[0]

    # 保存先ディレクトリを決める（Logger に output_dir があればそこを利用）
    run = Logger.run()
    if getattr(run, "output_path", None) is not None:
        base_dir = Path(run.output_path) / "media" / "mpc" / "mpc_videos"
    else:
        base_dir = Path("mpc_videos")
    base_dir.mkdir(parents=True, exist_ok=True)

    # terminations がないケースも一応ケア
    if hasattr(report, "terminations"):
        terminations = report.terminations
    else:
        terminations = [T - 1] * B

    for idx in idxs:
        # 失敗ケースのみ描画したいとき
        if plot_failure_only and getattr(report, "success", [False] * B)[idx]:
            continue

        t_term = terminations[idx]
        frames = []

        for t in range(T):
            if t % plot_every:
                continue
            if t > t_term:
                break

            obs = result.observations[t][idx]  # (C,H,W) or (1,H,W)
            # Tensor → numpy
            obs_np = obs.detach().cpu().numpy()

            # チャンネル次元を最後に持ってくる
            # C>1 (RGB 等) の場合: (C,H,W) → (H,W,C)
            # C==1 の場合: (1,H,W) → (H,W)
            if obs_np.ndim == 3 and obs_np.shape[0] > 1:
                img = obs_np.transpose(1, 2, 0)
            elif obs_np.ndim == 3 and obs_np.shape[0] == 1:
                img = obs_np[0]  # (H,W)
            else:
                # 既に (H,W,C) or (H,W) な場合も一応許容
                img = obs_np

            # 0〜1 に正規化 → 0〜255 uint8
            img = img.astype(np.float32)
            img_min = img.min()
            img_max = img.max()
            img = img - img_min
            if img_max - img_min > 1e-12:
                img = img / (img_max - img_min)
            img = (img * 255.0).clip(0, 255).astype(np.uint8)

            # グレースケールなら 3ch に拡張（動画可視化しやすくするため）
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)  # (H,W) → (H,W,3)

            frames.append(img)

        if len(frames) == 0:
            continue

        # ファイルパス決定
        video_path = base_dir / f"obs_seq_{idx}.mp4"

        # fps は好みで調整（ここでは 5fps）
        imageio.mimsave(video_path, frames, fps=5)

        # Logger にも動画としてログ（対応している場合のみ）
        if hasattr(run, "log_video"):
            try:
                run.log_video(str(video_path), name=f"mpc/obs_video_{idx}")
            except TypeError:
                # log_video のシグネチャが違う場合はここを環境に合わせて修正
                run.log_video(str(video_path))

        # 後始末
        del frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()



# =========================
# MuJoCo world (X,Y) -> Plot 座標への変換
# =========================
def world_to_plot_xy(xy):
    """
    MuJoCo world 座標 (X, Y) -> plot 座標 (px, py) に変換する。

    - world: 俯瞰視点で
        X: 上方向が +X
        Y: 左方向が +Y
    - plot:
        x 軸: 右が + (matplotlib のデフォルト)
        y 軸: 上が +

    これを「上 = +X」「左 = +Y」として表示したいので,
        px = -Y  (左が +Y → 右が -Y)
        py =  X  (上が +X)
    として変換する。
    """
    xy = np.asarray(xy)
    X = xy[..., 0]
    Y = xy[..., 1]
    px = -Y
    py = X
    return px, py


# =========================
# 図B: EE軌跡 + 1step予測 + bluebox + ワークスペース枠
# =========================
def log_planning_traj_plots_split(
    result, report, idxs=None,
    plot_every=1, xy_action=True,
    plot_failure_only=False,
    world_xlim=None, world_ylim=None,
    use_pixel_mapper=False, pixel_mapper=None,
    env=None,  # 運動学計算用
    plot_action=True, use_box=True,
):
    """
    図B: EE 軌跡 + 1ステップ予測 + bluebox + ワールド範囲 を可視化
    （MuJoCo 世界座標 (X: 上, Y: 左) を, plot 上で
     「上=+X, 左=+Y」になるように変換して描画する）
    """
    
    print("[DBG][pldm/planning/plotting.py] locations:", len(result.locations) if getattr(result, "locations", None) is not None else None)
    print("[DBG][pldm/planning/plotting.py] pred_locations:", len(getattr(result, "pred_locations", [])))
    print("[DBG][pldm/planning/plotting.py] action_history:", len(getattr(result, "action_history", [])))
    print("[DBG][pldm/planning/plotting.py] object_history:", len(getattr(result, "object_history", [])))

    if idxs is None:
        idxs = default_plot_idxs

    T = len(result.locations)
    B = result.observations[0].shape[0]

    for idx in idxs:
        if plot_failure_only and getattr(report, "success", [False] * B)[idx]:
            continue

        # ----- EE 軌跡などの準備 -----
        starts = result.locations[0][idx].detach().cpu()        # (2,)
        targets = result.targets[idx].detach().cpu()            # (2,)
        traj = torch.stack(
            [result.locations[t][idx] for t in range(T)],
            dim=0
        ).detach().cpu()                                        # (T,2)

        # numpy 化
        traj_np = traj.numpy()          # (T,2) [X,Y]
        targets_np = targets.numpy()    # (2,) [X,Y]

        # ---- 1step 予測列 ----
        single_step_preds = []
        T_loc  = len(result.locations)         # 初期+各ステップ → T+1
        # T_pred = len(result.pred_locations)
        T_act  = len(getattr(result, "action_history", []))
        t_term = getattr(report, "terminations", [T_loc - 1])[idx]
        t_max = min(T_act, T_loc - 1, t_term + 1)

        if env is not None:
            env.reset()
            for t in range(t_max):
                if t > report.terminations[idx]:
                    break
                act_seq = result.action_history[t][idx].detach().cpu().numpy()  # (H_t,7)
                a0 = act_seq[0]
                env.set_joint(a0)
                ee = env.get_ee_position()[:2]   # world (X,Y)
                single_step_preds.append(ee)

        # ---- オブジェクト軌跡 (bluebox) ----
        obj_traj = None
        if getattr(result, "object_history", None):
            T_obj = len(result.object_history)
            t_term = getattr(report, "terminations", [T - 1])[idx]
            tt = min(T, T_obj, t_term + 1)

            if tt > 0:
                obj_traj = torch.stack(
                    [result.object_history[t][idx] for t in range(tt)],
                    dim=0
                ).cpu()
                if obj_traj.shape[-1] == 3:  # xyz → xy
                    obj_traj = obj_traj[:, :2]

        figB = plt.figure(dpi=250)
        ax = plt.gca()

        # ====== EE軌跡 + start/end ======
        # ゴール点
        tx, ty = world_to_plot_xy(targets_np)  # (px,py)
        ax.scatter(tx, ty,
                   s=12, c="tab:orange",
                   label="goal", zorder=5)

        # EE 軌跡
        px_traj, py_traj = world_to_plot_xy(traj_np)  # (T,)
        ax.scatter(px_traj, py_traj,
                   s=8, c="black", alpha=0.9,
                   label="end-effector", zorder=6)
        ax.plot(px_traj, py_traj, c="black", linewidth=1.0, alpha=0.7, zorder=5)


        # S/G ラベル
        sx, sy = px_traj[0], py_traj[0]
        gx, gy = px_traj[-1], py_traj[-1]
        ax.text(sx, sy, "S",
                fontsize=15, color="black",
                ha="center", va="center",
                fontweight="bold", zorder=7)
        ax.text(gx, gy, "G",
                fontsize=15, color="black",
                ha="center", va="center",
                fontweight="bold", zorder=7)

        # ====== 1step 予測の可視化 ======
        if len(single_step_preds) > 0 and plot_action:
            single_step_preds_np = np.asarray(single_step_preds, dtype=np.float32)  # (T',2)

            for t, ee_pred in enumerate(single_step_preds_np):
                if t % plot_every:
                    continue

                # 実際の EE 位置 (world → plot)
                cur_world = traj_np[t]    # (2,)
                cx, cy = world_to_plot_xy(cur_world)
                px_pred, py_pred = world_to_plot_xy(ee_pred)

                ax.plot(
                    [cx, px_pred],
                    [cy, py_pred],
                    lw=0.5, alpha=0.7, c="red", zorder=2
                )
                ax.scatter(
                    px_pred, py_pred,
                    s=10, c="lime", marker="o", zorder=1
                )

        # ====== bluebox の中心軌跡 + box 枠（start/end）======
        if obj_traj is not None and len(obj_traj) > 0 and use_box:
            obj_np = obj_traj.numpy()            # (T_obj,2) world
            px_obj, py_obj = world_to_plot_xy(obj_np)

            ax.plot(px_obj, py_obj,
                    lw=1.2, c="tab:blue",
                    label="bluebox_center", zorder=4)
            ax.scatter(px_obj[0], py_obj[0],
                    s=14, c="tab:blue",
                    marker="x", label="bluebox_start", zorder=3)

            # box half-size (world coordinates)
            hx, hy = 0.05, 0.05

            # ---- start box ----
            sx_world, sy_world = obj_np[0, 0].item(), obj_np[0, 1].item()
            sx_plot, sy_plot = world_to_plot_xy([sx_world, sy_world])

            rect_bb_start = Rectangle(
                (sx_plot - hx, sy_plot - hy),
                2 * hx, 2 * hy,
                fill=False,
                linewidth=1.2,
                edgecolor="tab:blue",
                alpha=0.8,
                zorder=4.5,
            )
            ax.add_patch(rect_bb_start)

            ax.text(
                sx_plot, sy_plot, "S",
                fontsize=15, color="tab:blue",
                ha="center", va="center",
                fontweight="bold", zorder=6
            )

            # ---- end box ----
            gx_world, gy_world = obj_np[-1, 0].item(), obj_np[-1, 1].item()
            gx_plot, gy_plot = world_to_plot_xy([gx_world, gy_world])

            rect_bb_end = Rectangle(
                (gx_plot - hx, gy_plot - hy),
                2 * hx, 2 * hy,
                fill=False,
                linewidth=1.2,
                edgecolor="tab:blue",
                alpha=0.8,
                zorder=4.5,
            )
            ax.add_patch(rect_bb_end)

            ax.text(
                gx_plot, gy_plot, "G",
                fontsize=15, color="tab:blue",
                ha="center", va="center",
                fontweight="bold", zorder=6
            )

        # ====== ワークスペース枠 ======
        if world_xlim is not None and world_ylim is not None:
            xmin, xmax = world_xlim
            ymin, ymax = world_ylim

            corners_world = np.array([
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
            ])
            px_c, py_c = world_to_plot_xy(corners_world)
            px_min, px_max = px_c.min(), px_c.max()
            py_min, py_max = py_c.min(), py_c.max()

            rect_ws = Rectangle(
                (px_min, py_min),
                px_max - px_min, py_max - py_min,
                fill=False, linewidth=1.0, linestyle="--",
                edgecolor="black", alpha=0.6, zorder=0
            )
            ax.add_patch(rect_ws)

        ax.set_aspect("equal")
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8, loc="best")

        # 軸ラベル（MuJoCo座標との対応を明示）
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("X [m]")

        if plot_action:
            Logger.run().log_figure(figB, f"mpc/prediction_seq_{idx}_actionplot")
        else:
            Logger.run().log_figure(figB, f"mpc/prediction_seq_{idx}")
        plt.close(figB)

        # メモリ掃除
        try:
            del figB, ax, traj, traj_np, obj_traj, obj_np, single_step_preds
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()



def log_planning_joint_angle_plots_split(
    result,
    report,
    idxs=None,
    plot_every: int = 1,
    plot_failure_only: bool = False,
    joint_names=None,
):
    if idxs is None:
        idxs = default_plot_idxs

    if not hasattr(result, "qpos_history") or result.qpos_history is None or len(result.qpos_history) == 0:
        print("[WARN] result.qpos_history が存在しないため、関節角度の可視化をスキップします。")
        return

    T_q = len(result.qpos_history)      # 状態列の長さ（q_0, q_1, ..., q_{T_q-1})
    T_a = len(result.action_history)    # 行動列の長さ（a_0, ..., a_{T_a-1})

    if T_q <= 1 or T_a == 0:
        print("[WARN] qpos_history / action_history が短すぎるのでスキップします。")
        return

    # a_t と q_{t+1} を比較できるのは t = 0..min(T_a-1, T_q-2)
    T_cmp = min(T_a, T_q - 1)

    B = result.qpos_history[0].shape[0]

    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(7)]

    for idx in idxs:
        if idx >= B:
            print(f"[WARN] idx={idx} は batch 次元 B={B} を超えています。スキップ。")
            continue

        if plot_failure_only and getattr(report, "success", [False] * B)[idx]:
            continue

        # terminations は「状態インデックス（q_t）の最大値」を指していると仮定
        # すると有効な行動インデックスは 0..t_term-1
        t_term_states = getattr(report, "terminations", [T_q - 1])[idx]
        # t_max = min(T_cmp, t_term_states)  # t in [0, t_max-1] が有効
        t_max = T_cmp

        target_traj = []
        actual_traj = []

        for t in range(t_max):
            # ----- target: 出力行動 a_t -----
            act_t = result.action_history[t]

            if act_t.ndim == 3:
                a0 = act_t[idx, 0]   # (7,)
            elif act_t.ndim == 2:
                a0 = act_t[idx]
            else:
                raise ValueError(f"Unexpected action_history[{t}].shape={act_t.shape}")

            target_traj.append(a0.detach().cpu())

            # ----- actual: q_{t+1} -----
            qpos_t = result.qpos_history[t + 1][idx]  # (7,) を想定
            if qpos_t.numel() > 7:
                qpos_t = qpos_t[:7]
            actual_traj.append(qpos_t.detach().cpu())

        if len(target_traj) == 0:
            continue

        target_traj = torch.stack(target_traj, dim=0)  # (t_max, 7)
        actual_traj = torch.stack(actual_traj, dim=0)  # (t_max, 7)

        target_np = target_traj.numpy()
        actual_np = actual_traj.numpy()
        ts = np.arange(target_np.shape[0])
        
        # --- 関節ごとの min-max を求める ---
        min_j = np.minimum(target_np.min(axis=0), actual_np.min(axis=0))  # (7,)
        max_j = np.maximum(target_np.max(axis=0), actual_np.max(axis=0))  # (7,)

        range_j = max_j - min_j
        # max_j == min_j（＝その関節がずっと同じ値）のとき、ゼロ割りを防ぐ
        range_j[range_j == 0] = 1.0

        target_norm = 2.0 * (target_np - min_j) / range_j - 1.0
        actual_norm = 2.0 * (actual_np - min_j) / range_j - 1.0

        target_np = target_norm
        actual_np = actual_norm

        fig, axes = plt.subplots(
            nrows=7,
            ncols=1,
            figsize=(8, 10),
            dpi=200,
            sharex=True,
        )

        for j in range(7):
            ax = axes[j]

            ax.plot(
                ts,
                target_np[:, j],
                linewidth=1.0,
                label="target (action)",
            )

            ax.plot(
                ts,
                actual_np[:, j],
                linewidth=1.0,
                linestyle="--",
                label="actual (qpos)",
            )

            ax.set_ylabel(joint_names[j], fontsize=8)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

            if j == 0:
                ax.legend(fontsize=8, loc="best")

        axes[-1].set_xlabel("timestep (t, comparing a_t vs q_{t+1})", fontsize=9)
        fig.suptitle(f"Joint angle comparison (episode idx={idx})", fontsize=10)
        fig.tight_layout()

        Logger.run().log_figure(fig, f"mpc/joint_angle_comparison_ep{idx}")
        plt.close(fig)

        try:
            del fig, axes, target_traj, actual_traj, target_np, actual_np
        except Exception:
            pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            
def log_planning_torque_plots_split(
    result,
    report,
    env,                      # ★追加
    idxs=None,
    plot_failure_only=False,
    joint_names=None,
):
    if idxs is None:
        idxs = default_plot_idxs

    if not hasattr(result, "actforce_history") or len(result.actforce_history) == 0:
        print("[WARN] actforce_history が無いのでスキップ")
        return

    # ===== MuJoCo model を env から取得（dm_control版）=====
    model = env.physics.model   # ★ここがポイント

    # (n_actuator, 2) : [min, max]
    act_force_range = model.actuator_forcerange.copy()

    # Franka arm の actuator id を env 側で持っているので、それを使うのが安全
    arm_ids = getattr(env, "arm_actuator_ids", None)
    if arm_ids is None:
        # フォールバック：先頭7個（ただし危険）
        arm_ids = np.arange(7)

    torque_min = act_force_range[arm_ids, 0]
    torque_max = act_force_range[arm_ids, 1]
    torque_limit = np.maximum(np.abs(torque_min), np.abs(torque_max))  # (7,)

    T = len(result.actforce_history)
    B = result.actforce_history[0].shape[0]

    if joint_names is None:
        joint_names = [f"Joint {i+1}" for i in range(7)]

    for idx in idxs:
        if idx >= B:
            continue
        if plot_failure_only and getattr(report, "success", [False] * B)[idx]:
            continue

        if T <= 0:
            print(f"[WARN] actforce plot skip: T={T}, idx={idx}")
            continue

        # (T, 7)  ※actforce_history[t][idx] が (7,) を返す前提
        af = torch.stack(
            [result.actforce_history[t][idx] for t in range(T)],
            dim=0
        ).cpu().numpy()

        # ===== 正規化：|τ| / τ_max =====
        af_norm = af / torque_limit[None, :]
        af_norm = np.clip(af_norm, -1.1, 1.1)

        ts = np.arange(af_norm.shape[0])

        fig, axes = plt.subplots(7, 1, figsize=(8, 10), dpi=200, sharex=True)
        for j in range(7):
            ax = axes[j]
            ax.plot(ts, af_norm[:, j], linewidth=1.0, label="|τ| / τ_max")
            ax.axhline(1.0, linestyle="--", linewidth=0.6)
            ax.axhline(-1.0, linestyle="--", linewidth=0.6)
            ax.set_ylim(-1.1, 1.1)
            ax.set_ylabel(joint_names[j], fontsize=8)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
            if j == 0:
                ax.legend(fontsize=8, loc="best")

        axes[-1].set_xlabel("timestep", fontsize=9)
        fig.suptitle(f"Torque normalized by actuator_forcerange idx={idx}", fontsize=10)
        fig.tight_layout()
        Logger.run().log_figure(fig, f"mpc/torque_actforce_norm_limit_ep{idx}")
        plt.close(fig)

