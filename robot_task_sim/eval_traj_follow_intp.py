# evaluate_trajectory_following.py
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ.pop("DISPLAY", None)
from pathlib import Path
import numpy as np
import json, time
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from dm_control import mujoco as dm_mj
from pldm_envs.franka.envs import FrankaSimEnv

import logging
import sys
from tqdm import tqdm

import collections.abc as cabc

from box import Box
import yaml



logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("traj_eval")

def load_config(cfg_path="robot_task_sim/config.yaml") -> Box:
    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Box(raw, default_box=False, box_dots=True)

config = load_config("robot_task_sim/config.yaml")


OUTDIR = Path(config.paths.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "frames").mkdir(parents=True, exist_ok=True)


import shutil
frames_dir = OUTDIR / "frames"
if frames_dir.exists():
    shutil.rmtree(frames_dir)
frames_dir.mkdir(parents=True, exist_ok=True)


#SETTING
################################################################################################################################

#評価パラメータ
IMAGE_SIZE = tuple(config.eval.image_size)         # 保存フレーム解像度
CAMERA = str(config.camera)
SUCCESS_TOL = float(config.eval.success_tol)             # EE到達判定 (m)
STEP_SUBSTEPS = int(config.eval.step_substeps)              
MAX_STEPS_PER_WAYPOINT = int(config.eval.max_steps_per_waypoint)    # 各目標に対して許容する最大ステップ数
HOLD_STEPS_AT_TARGET = int(config.eval.hold_steps_at_target)       # 収束後に少し保持して撮影
SAVE_EVERY = int(config.eval.save_every)                # フレーム保存間隔
SEED = int(config.eval.seed)

NUM_PNTS = int(config.eval.num_points) #軌跡の分割ポイント数: 80


x_min = float(config.workspace.x_min)
x_max = float(config.workspace.x_max)
y_min = float(config.workspace.y_min)
y_max = float(config.workspace.y_max)
z_fixed = float(config.workspace.z_fixed) 
MARGIN = float(config.workspace.margin)

# ロボットパラメータ
Franka_FREQ = float(config.robot.franka_freq) #2.5
KP=None if config.robot.kp is None else list(config.robot.kp)#[4500, 4500, 3500, 3500, 2000, 2000, 2000] # 位置アクチュエータの比例ゲイン  (値を上げるほど追従が速くなるが、振動が起きやすい)
KD=None if config.robot.kd is None else list(config.robot.kd)#[450, 450, 350, 350, 200, 200, 200]
DOF_DAMPING=config.robot.dof_damping           # DOFダンピング（振動抑制）      (高いとブレーキがかかるよう動作する. 過剰だと応答が鈍くなる)
DOF_ARMATURE=config.robot.dof_armature          # DOFアーマチュア（慣性付加）    (大きくすると応用が重く安定する)
CTRL_LOW=config.robot.ctrl_low              # アーム用ctrl下限（7要素）     (学習時や制御ポリシー設計と一致しているか確認)
CTRL_HIGH=config.robot.ctrl_high            # アーム用ctrl上限（7要素）     (学習時や制御ポリシー設計と一致しているか確認)
SOLVER_ITERS=config.robot.solver_iters          # ソルバ反復                  (接触解決制度. 上げると接触安定性up, 速度down. デフォルトでは十分なことが多い)
LS_ITERS=config.robot.ls_iters            # ラインサーチ反復             (大抵は solver_iters に比べ影響小. 接触剛性が高い場合のみ確認)



ROT_WEIGHT = float(config.control.rot_weight)
TARGET_ROTMAT = None
rot = config.control.target_rotmat  # null → None
if rot is not None:
    TARGET_ROTMAT = np.asarray(rot, dtype=np.float32)
    if TARGET_ROTMAT.shape != (3,3):
        raise ValueError(f"target_rotmat must be 3x3, got {TARGET_ROTMAT.shape}")


MAX_DQ = float(config.control.max_dq)

N_INTERP = config.control.n_interp      # 10–20 gives smooth motion

################################################################################################################################





# def set_kp_per_joint(env, kp):
#     """
#     kp: 
#       - float（全関節を同一値に）
#       - 長さ=len(env.arm_actuator_ids) のシーケンス
#         その中に None を含めると、その関節は現状値を維持します
#     """
    
#     m = env.physics.model
#     arm_ids = list(env.arm_actuator_ids)
#     n = len(arm_ids)


#     if isinstance(kp, (int, float, np.floating)):
#         kp_list = [float(kp)] * n
#     else:
#         try:
#             kp_list = list(kp)
#         except TypeError:
#             raise TypeError("kp は float か、長さが関節数のシーケンスで指定してください")
#         assert len(kp_list) == n, f"kp の長さは {n} 要素（関節数）にしてください"


#     before = m.actuator_gainprm[arm_ids, 0].copy()
#     print("kp before:", before)


#     for aid, v in zip(arm_ids, kp_list):
#         if v is None:
#             continue
#         m.actuator_gainprm[aid, 0] = float(v)

#     env.physics.forward()
#     after = m.actuator_gainprm[arm_ids, 0]
#     print("kp after :", after)


# def set_kd_per_joint(env, kd):
#     """
#     kd:
#       - float（全関節を同一値に）
#       - 長さ=len(env.arm_actuator_ids) のシーケンス
#         その中に None を含めると、その関節は現状値を維持
#     備考: MuJoCoの一般アクチュエータ(affine)では kd = -biasprm[2]
#     """
#     m = env.physics.model
#     arm_ids = list(env.arm_actuator_ids)
#     n = len(arm_ids)

#     if isinstance(kd, (int, float, np.floating)):
#         kd_list = [float(kd)] * n
#     else:
#         try:
#             kd_list = list(kd)
#         except TypeError:
#             raise TypeError("kd は float か、長さが関節数のシーケンスで指定してください")
#         assert len(kd_list) == n, f"kd の長さは {n} 要素（関節数）にしてください"

    
#     before = -m.actuator_biasprm[arm_ids, 2].copy()
#     print("kd before:", before)

#     for aid, v in zip(arm_ids, kd_list):
#         if v is None:
#             continue
        
#         m.actuator_biasprm[aid, 2] = -float(v)

#     env.physics.forward()
#     after = -m.actuator_biasprm[arm_ids, 2]
#     print("kd after :", after)



def patch_franka_runtime(env, kp=None, kd=None, dof_damping=None, dof_armature=None,
                         ctrl_low=None, ctrl_high=None, solver_iters=None, ls_iters=None):
    m = env.physics.model

    # if kp is not None:
    #     set_kp_per_joint(env, kp)
    # if kd is not None:
    #     set_kd_per_joint(env, kd)


    if dof_damping is not None:
        arr = np.asarray(dof_damping, np.float32)
        assert arr.shape[0] == m.nv, "dof_damping の長さが nv と一致していません"
        print("\n[DEBUG] dof_damping before:", np.round(m.dof_damping[:10], 6), "...")
        m.dof_damping[:] = arr
        print("[DEBUG] dof_damping after :", np.round(m.dof_damping[:10], 6), "...")
        print("[DEBUG] total nv =", m.nv)


    if dof_armature is not None:
        arr = np.asarray(dof_armature, np.float32)
        assert arr.shape[0] == m.nv, "dof_armature の長さが nv と一致していません"
        print("\n[DEBUG] dof_armature before:", np.round(m.dof_armature[:10], 6), "...")
        m.dof_armature[:] = arr
        print("[DEBUG] dof_armature after :", np.round(m.dof_armature[:10], 6), "...")
        print("[DEBUG] total nv =", m.nv)


    if (ctrl_low is not None) or (ctrl_high is not None):
        lo = np.asarray(ctrl_low if ctrl_low is not None
                        else m.actuator_ctrlrange[env.arm_actuator_ids, 0], np.float32)
        hi = np.asarray(ctrl_high if ctrl_high is not None
                        else m.actuator_ctrlrange[env.arm_actuator_ids, 1], np.float32)
        assert lo.shape[0] == len(env.arm_actuator_ids)
        assert hi.shape[0] == len(env.arm_actuator_ids)

        print("\n[DEBUG] ctrlrange before:")
        print("  low :", np.round(m.actuator_ctrlrange[env.arm_actuator_ids, 0], 6))
        print("  high:", np.round(m.actuator_ctrlrange[env.arm_actuator_ids, 1], 6))

        m.actuator_ctrlrange[env.arm_actuator_ids, 0] = lo
        m.actuator_ctrlrange[env.arm_actuator_ids, 1] = hi
        env.ctrlrange = m.actuator_ctrlrange[env.arm_actuator_ids].copy()

        print("[DEBUG] ctrlrange after:")
        print("  low :", np.round(env.ctrlrange[:, 0], 6))
        print("  high:", np.round(env.ctrlrange[:, 1], 6))


    if solver_iters is not None:
        print(f"\n[DEBUG] solver_iters before: {m.opt.iterations}")
        m.opt.iterations = int(solver_iters)
        print(f"[DEBUG] solver_iters after : {m.opt.iterations}")

    if ls_iters is not None:
        print(f"\n[DEBUG] ls_iters before: {m.opt.ls_iterations}")
        m.opt.ls_iterations = int(ls_iters)
        print(f"[DEBUG] ls_iters after : {m.opt.ls_iterations}")

    env.physics.forward()
    print("\n[DEBUG] patch_franka_runtime complete ✅\n")



def set_control_frequency_by_substeps(env, control_hz: float):
    m = env.physics.model
    dt = float(m.opt.timestep)           
    sub = max(1, int(round((1.0 / control_hz) / dt)))
    env.substeps = sub
    actual_hz = 1.0 / (sub * dt)
    print(f"[CTRL FREQ] target={control_hz:.3f} Hz -> substeps={sub}, actual≈{actual_hz:.3f} Hz")
    return actual_hz

# === 軌跡プリセット ===
def make_trajectory(kind="rectangle", n_points=60, shuffle=False, seed=0):
    rng = np.random.RandomState(seed)
    x_lo, x_hi = x_min + MARGIN, x_max - MARGIN
    y_lo, y_hi = y_min + MARGIN, y_max - MARGIN

    if kind == "rectangle":
        xs_top = np.linspace(x_lo, x_hi, n_points//4, endpoint=False)
        ys_top = np.full_like(xs_top, y_lo)
        xs_right = np.full(n_points//4, x_hi)
        ys_right = np.linspace(y_lo, y_hi, n_points//4, endpoint=False)
        xs_bottom = np.linspace(x_hi, x_lo, n_points//4, endpoint=False)
        ys_bottom = np.full_like(xs_bottom, y_hi)
        xs_left = np.full(n_points - 3*(n_points//4), x_lo)
        ys_left = np.linspace(y_hi, y_lo, len(xs_left), endpoint=False)
        xs = np.concatenate([xs_top, xs_right, xs_bottom, xs_left])
        ys = np.concatenate([ys_top, ys_right, ys_bottom, ys_left])
        zs = np.full_like(xs, z_fixed)

    elif kind == "lawnmower":
        rows = int(np.sqrt(n_points))
        cols = max(2, n_points // rows)
        xs = np.linspace(x_lo, x_hi, cols)
        ys = np.linspace(y_lo, y_hi, rows)
        path = []
        for i, y in enumerate(ys):
            if i % 2 == 0:
                for x in xs:
                    path.append((x, y, z_fixed))
            else:
                for x in xs[::-1]:
                    path.append((x, y, z_fixed))
        arr = np.array(path, dtype=np.float32)
        xs, ys, zs = arr[:,0], arr[:,1], arr[:,2]

    elif kind == "lissajous":
        t = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        xs = (x_lo+x_hi)/2 + 0.45*(x_hi-x_lo)/2 * np.sin(2*t+0.3)
        ys = (y_lo+y_hi)/2 + 0.85*(y_hi-y_lo)/2 * np.sin(3*t)
        xs = np.clip(xs, x_lo, x_hi)
        ys = np.clip(ys, y_lo, y_hi)
        zs = np.full_like(xs, z_fixed)

    elif kind == "random":
        xs = rng.uniform(x_lo, x_hi, size=n_points)
        ys = rng.uniform(y_lo, y_hi, size=n_points)
        zs = np.full_like(xs, z_fixed)

    else:
        raise ValueError("unknown trajectory kind")

    traj = np.stack([xs, ys, zs], axis=1).astype(np.float32)
    if shuffle:
        rng.shuffle(traj)

    diffs = traj[1:] - traj[:-1]          # 各点の差分ベクトル
    dists = np.linalg.norm(diffs, axis=1) # ユークリッド距離（各区間の長さ）


    print(f"--- Trajectory '{kind}' ---")
    print(f"Total points: {len(traj)}")
    print(f"Segment distances (mean={dists.mean():.4f} m, std={dists.std():.4f} m):")
    
    return traj

def ee(env):
    return env.get_ee_position().astype(np.float32)

def render_rgb(env, size=IMAGE_SIZE, camera=CAMERA):
    rgb = env.physics.render(height=size[0], width=size[1],
                             camera_id=env.camera_id if camera!="default" else -1)

    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0  # -> 0..1 float
    else:
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return rgb


P_TASK_GAIN_FOR_PLOT = 1.0   # 距離誤差に掛ける可視化スケール
K_TASK_GAIN_FOR_PLOT = 1.0   # 速度に掛ける可視化スケール

def get_control_dt(env):
    """高レベル1ステップあたりの実時間 [sec] を返す"""
    m = env.physics.model
    return float(m.opt.timestep) * int(getattr(env, "substeps", 1))

def get_joint_pd_gains_from_model(env):
    """
    MuJoCoでは一般的なPDは actuator_gainprm[:,0] に Kp,
    actuator_biasprm[:,2] に -Kd（※負符号で格納）として入っているケースが多いです。
    既存の set_kd_per_joint のコメントとも整合します。
    """
    m = env.physics.model
    arm_ids = list(env.arm_actuator_ids)
    
    kp_vec = m.actuator_gainprm[arm_ids, 0].astype(np.float32).copy()     # Kp
    kd_vec = (-m.actuator_biasprm[arm_ids, 2]).astype(np.float32).copy()  # Kd（負符号を戻す）
    return kp_vec, kd_vec  # shape: (7,), (7,)




def run_follow(env, traj_xyz, name="rectangle"):
    ee_traj = []
    tgt_traj = []
    ee_traj_interp = []
    reached = []
    steps_used = []
    total_frames = 0

    frame_ee_xy = []
    frame_tgt_xy = []

    times = []
    dist_errs = []
    vel_mag = []
    vel_xyz = []


    dt_step = get_control_dt(env)
    t_accum = 0.0
    prev_cur = None

    kp_vec, kd_vec = get_joint_pd_gains_from_model(env)  # XML由来のPDゲイン
    pd_p_hist = []     # list of (7,) arrays   -> Kp*(q_des - q)
    pd_d_hist = []     # list of (7,) arrays   -> Kd*(qd_des - qd)  (qd_des=0)
    pd_tot_hist = []   # list of (7,) arrays   -> p + d

    q_err_hist = []    # q_des - q         [rad]
    qd_err_hist = []   # qd_des - qd (= -qd) [rad/s]
    qd_hist = []       # qd                [rad/s]


    log.info(f"[run] >>> start '{name}' | waypoints={len(traj_xyz)} tol={SUCCESS_TOL} substeps={STEP_SUBSTEPS}")
    t0 = time.time()


    env.reset(robot_only = True)
    q_des, ee_after_set = env.set_xyz(traj_xyz[0], target_rotmat=TARGET_ROTMAT, rot_weight = ROT_WEIGHT)
    print("EE after set:", ee_after_set, " target:", traj_xyz[0], " err:", np.linalg.norm(ee_after_set-traj_xyz[0]))
    frame_ee_xy.append(env.get_ee_position()[:2].copy())
    frame_tgt_xy.append(traj_xyz[0][:2].copy())


##############
    for i, target in enumerate(tqdm(traj_xyz[1:], desc="Waypoints")):
        log.info(f"[run] wp[{i+1}/{len(traj_xyz)}] target={np.round(target,4)}")

        target_rot_weight = ROT_WEIGHT
        target_rotmat = TARGET_ROTMAT

        ik = env.calc_inverse_kinematic(target, target_rotmat = target_rotmat, rot_weight = target_rot_weight)
        q_des = ik.qpos[:7].copy()
        q_cur = env.physics.data.qpos[:7].copy()

        # step_cnt = 0
        # while step_cnt < MAX_STEPS_PER_WAYPOINT:

        for j, alpha in enumerate(np.linspace(0, 1, N_INTERP, endpoint=False)):
            q_interp = q_cur + alpha * (q_des - q_cur)

            q_cur2 = env.physics.data.qpos[:7].copy()

            # print(f"[DEBUG] Waypoint {i}, Interpolation alpha={alpha:.3f}, q_interp={np.round(q_interp, 4)}, current_pose={np.round(q_cur2, 4)}")

            obs, reward, done, truncated, info = env.step(q_interp, max_dq = MAX_DQ)
            cur = ee(env)
            err = float(np.linalg.norm(cur - target))

            #dist_errs, vel_mag, vec_xyz ログ収集
            ##########################################
            if prev_cur is None:
                v = np.zeros(3, dtype=np.float32)
            else:
                v = (cur - prev_cur) / dt_step  # [m/s]
            prev_cur = cur.copy()

            times.append(t_accum)
            dist_errs.append(err)
            vel_mag.append(float(np.linalg.norm(v)))
            vel_xyz.append(v.copy())
            t_accum += dt_step


            q  = env.physics.data.qpos[:7].astype(np.float32).copy()
            qd = env.physics.data.qvel[:7].astype(np.float32).copy()

            qd_des = np.zeros_like(qd, dtype=np.float32)

            tau_p = kp_vec * (q_interp - q)
            tau_d = kd_vec * (qd_des - qd)
            tau_tot = tau_p + tau_d

            pd_p_hist.append(tau_p.copy())
            pd_d_hist.append(tau_d.copy())
            pd_tot_hist.append(tau_tot.copy())

            q_err = (q_interp - q)                  # [rad]
            qd_des = np.zeros_like(qd, dtype=np.float32)  # 目標速度は0とする
            qd_err = (qd_des - qd)               # [rad/s]

            q_err_hist.append(q_err.copy())
            qd_err_hist.append(qd_err.copy())
            qd_hist.append(qd.copy())
            
                
            #毎フレームのEE/targetをログ（XY）
            frame_ee_xy.append(cur[:2].copy())
            frame_tgt_xy.append(target[:2].copy())

            # env.physics.data.qpos[:7] = q_interp
            # env.physics.forward()
            ee_pos_interp = ee(env)    # get end-effector position (3D)
            ee_traj_interp.append(ee_pos_interp[:2].copy())  # store XY for plotting

            #VIDEO
            # if step_cnt % SAVE_EVERY == 0:
            if j % SAVE_EVERY == 0:  
                rgb = render_rgb(env)
                plt.imsave(OUTDIR / "frames" / f"{name}_{i:03d}_{j:04d}.png", rgb)
                total_frames += 1

        q_cur = env.physics.data.qpos[:7].copy()

        reached.append(True)
            
        # steps_used.append(step_cnt+HOLD_STEPS_AT_TARGET)
            # break

            # step_cnt += 1

        # if step_cnt >= MAX_STEPS_PER_WAYPOINT:
        #     reached.append(False)
        #     steps_used.append(step_cnt)

        ee_traj.append(cur)
        tgt_traj.append(target)
        
    print('reached: ', reached)
    dt = time.time() - t0
    ee_traj = np.array(ee_traj, dtype=np.float32)
    ee_traj_interp = np.array(ee_traj_interp, dtype=np.float32)
    tgt_traj = np.array(tgt_traj, dtype=np.float32)
    err_vec = np.linalg.norm(ee_traj - tgt_traj, axis=1)

    metrics = {
        "name": name,
        "n_waypoints": int(len(traj_xyz)),
        "reached_ratio": float(np.mean(reached)),
        "mean_err_m": float(np.mean(err_vec)),
        "median_err_m": float(np.median(err_vec)),
        "max_err_m": float(np.max(err_vec)),
        "p90_err_m": float(np.percentile(err_vec, 90)),
        "mean_steps_per_wp": float(np.mean(steps_used)),
        "frames_saved": int(total_frames),
        "elapsed_sec": float(dt),
        "tol_m": SUCCESS_TOL,
        "substeps": STEP_SUBSTEPS,
    }
    log.info(f"[run] <<< done '{name}' | reached={metrics['reached_ratio']:.3f} "
             f"mean_err={metrics['mean_err_m']:.4f} p90={metrics['p90_err_m']:.4f} "
             f"avg_steps={metrics['mean_steps_per_wp']:.1f} elapsed={dt:.2f}s")

    plt.figure(figsize=(5,5))
    plt.plot(tgt_traj[:,0], tgt_traj[:,1], linestyle="--", marker="o", markersize=8, label="target XY")
    plt.plot(ee_traj_interp[:,0], ee_traj_interp[:,1], linestyle="--", marker="o", markersize=1, label="interp XY")
    plt.plot(ee_traj[:,0],  ee_traj[:,1],  linestyle="-",  marker=".", markersize=5, label="executed XY")

    plt.text(tgt_traj[0,0], tgt_traj[0,1], "S", color="blue", fontsize=12, fontweight="bold", ha="center", va="center")
    plt.text(tgt_traj[-1,0], tgt_traj[-1,1], "G", color="blue", fontsize=12, fontweight="bold", ha="center", va="center")
    
    plt.text(ee_traj[0,0], ee_traj[0,1], "S", color="orange", fontsize=12, fontweight="bold", ha="center", va="center")
    plt.text(ee_traj[-1,0], ee_traj[-1,1], "G", color="orange", fontsize=12, fontweight="bold", ha="center", va="center")

    plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title(f"Trajectory: {name}")
    plt.axis("equal"); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"ee_traj_plot_{name}.png", dpi=150)
    plt.close()


    plt.figure(figsize=(6,3))
    plt.plot(err_vec)
    plt.xlabel("waypoint idx"); plt.ylabel("||EE - target|| [m]"); plt.title("Waypoint errors")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(OUTDIR / f"errors_{name}.png", dpi=150)
    plt.close()

    with open(OUTDIR / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)


    frame_ee_xy = np.array(frame_ee_xy, dtype=np.float32)
    frame_tgt_xy = np.array(frame_tgt_xy, dtype=np.float32)


    plt.figure(figsize=(5,5))
    plt.plot(frame_tgt_xy[:,0], frame_tgt_xy[:,1], 'g--', lw=1, marker="o",markersize=5, label='target XY (per frame)')
    plt.plot(ee_traj_interp[:,0], ee_traj_interp[:,1], linestyle="--", marker="o", markersize=1, label="interp XY (per frame)")
    plt.plot(frame_ee_xy[:,0],  frame_ee_xy[:,1],  'r-',  lw=1, marker="o",markersize=1,label='executed XY (per frame)')

    plt.text(frame_tgt_xy[0,0], frame_tgt_xy[0,1], "S", color="blue", fontsize=12, fontweight="bold", ha="center", va="center")
    plt.text(frame_tgt_xy[-1,0], frame_tgt_xy[-1,1], "G", color="blue", fontsize=12, fontweight="bold", ha="center", va="center")

    plt.text(frame_ee_xy[0,0], frame_ee_xy[0,1], "S", color="red", fontsize=12, fontweight="bold", ha="center", va="center")
    plt.text(frame_ee_xy[-1,0], frame_ee_xy[-1,1], "G", color="red", fontsize=12, fontweight="bold", ha="center", va="center")

    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.title(f"Frame-wise trajectory: {name}")
    plt.axis("equal"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"ee_traj_plot_framewise_{name}.png", dpi=150)
    plt.close()
    print(f"[SAVE] framewise plot -> ee_traj_plot_framewise_{name}.png")

    vel_xyz_arr = np.stack(vel_xyz, axis=0) if len(vel_xyz) > 0 else np.zeros((0,3), dtype=np.float32)
    plot_controller_diagnostics(
        times=times,
        dist_errs=dist_errs,
        vel_mag=vel_mag,
        vel_xyz=vel_xyz_arr,
        name=name,
        outdir=OUTDIR,
    )
    # pd_p = np.stack(pd_p_hist, axis=0) if len(pd_p_hist)>0 else np.zeros((0,7), dtype=np.float32)
    # pd_d = np.stack(pd_d_hist, axis=0) if len(pd_d_hist)>0 else np.zeros((0,7), dtype=np.float32)
    # pd_tot = np.stack(pd_tot_hist, axis=0) if len(pd_tot_hist)>0 else np.zeros((0,7), dtype=np.float32)

    # plot_joint_pd_time_series(times, pd_p,   title="Joint-wise P-term (Kp*(qdes - q))",        ylabel="P-term [~Nm]",         fname_prefix="joint_p_term",   name=name, outdir=OUTDIR)
    # plot_joint_pd_time_series(times, pd_d,   title="Joint-wise D-term (Kd*(qd_des - qd))",     ylabel="D-term [~Nm]",         fname_prefix="joint_d_term",   name=name, outdir=OUTDIR)
    # plot_joint_pd_time_series(times, pd_tot, title="Joint-wise Total torque (P + D, approx.)", ylabel="Total torque [~Nm]",   fname_prefix="joint_total",    name=name, outdir=OUTDIR)

    # save_joint_pd_csv(times, pd_p, pd_d, pd_tot, name=name, outdir=OUTDIR)

    q_err_arr   = np.stack(q_err_hist, axis=0) if q_err_hist   else np.zeros((0,7), np.float32)
    qd_err_arr  = np.stack(qd_err_hist, axis=0) if qd_err_hist else np.zeros((0,7), np.float32)
    qd_arr      = np.stack(qd_hist, axis=0)    if qd_hist      else np.zeros((0,7), np.float32)

    plot_joint_series(times, q_err_arr,  title="Joint Position Error (q_des - q)",   ylabel="Δq [rad]",     fname_prefix="joint_pos_error",  name=name, outdir=OUTDIR)
    plot_joint_series(times, qd_err_arr, title="Joint Velocity Error (qd_des - qd)", ylabel="Δqd [rad/s]",  fname_prefix="joint_vel_error",  name=name, outdir=OUTDIR)
    plot_joint_series(times, qd_arr,     title="Joint Velocity (qd)",                ylabel="qd [rad/s]",   fname_prefix="joint_velocity",   name=name, outdir=OUTDIR)

    save_joint_series_csv(times, q_err_arr, qd_err_arr, qd_arr, name=name, outdir=OUTDIR)

    print(f"[DONE] {name} reached={metrics['reached_ratio']:.3f}, mean_err={metrics['mean_err_m']:.4f} m")
    return metrics


#plotting
#####################################################################
def maybe_make_gif():
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[INFO] imageio not available; skip GIF")
        return
    frames = sorted((OUTDIR / "frames").glob("*.png"))[:300]
    if not frames:
        return
    imgs = [imageio.imread(str(p)) for p in frames]
    imageio.mimsave(OUTDIR / "preview.gif", imgs, duration=0.05)
    print("[SAVE] GIF ->", OUTDIR / "preview.gif")

def numeric_sort_key(p):
    parts = p.stem.split("_")
    # parts = ['lawnmower', '000', '45']
    # fallback if fewer parts exist
    nums = [int(x) for x in parts[1:] if x.isdigit()]
    return nums

def make_gifs_per_traj(duration=0.002):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[INFO] imageio not available; skip GIF")
        return

    frames = sorted((OUTDIR / "frames").glob("*.png"))  
    groups = {}
    for p in frames:
        name = p.stem.split("_", 1)[0]  
        groups.setdefault(name, []).append(p)

    for name, fps in groups.items():
        fps = sorted(fps, key=numeric_sort_key)
        if not fps:
            continue
        imgs = [imageio.imread(str(p)) for p in fps]  
        imageio.mimsave(OUTDIR / f"{name}.gif", imgs, fps=30)
        imageio.mimsave(OUTDIR / f"{name}.mp4", imgs, fps=30, quality=8, codec='libx264')
        print(f"[SAVE] {name}.gif  ({len(fps)} frames, full span)")


def plot_controller_diagnostics(
    times, dist_errs, vel_mag, vel_xyz, name="run", outdir=OUTDIR
):
    times = np.asarray(times, dtype=np.float32)
    dist_errs = np.asarray(dist_errs, dtype=np.float32)
    vel_mag = np.asarray(vel_mag, dtype=np.float32)
    vel_xyz = np.asarray(vel_xyz, dtype=np.float32)  # shape [T,3]

    # --- 保存パス ---
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) 距離誤差
    plt.figure(figsize=(7,3))
    plt.plot(times, dist_errs)
    plt.xlabel("time [s]"); plt.ylabel("||EE - target|| [m]")
    plt.title(f"Distance Error vs Time ({name})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir / f"dist_error_vs_time_{name}.png", dpi=150)
    plt.close()

    # 2) 速度 |v|
    plt.figure(figsize=(7,3))
    plt.plot(times, vel_mag)
    plt.xlabel("time [s]"); plt.ylabel("|v| [m/s]")
    plt.title(f"EE Speed vs Time ({name})")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(outdir / f"speed_vs_time_{name}.png", dpi=150)
    plt.close()

    # 3) 速度の各成分 vx, vy, vz
    plt.figure(figsize=(7,4))
    plt.plot(times, vel_xyz[:,0], label="vx")
    plt.plot(times, vel_xyz[:,1], label="vy")
    plt.plot(times, vel_xyz[:,2], label="vz")
    plt.xlabel("time [s]"); plt.ylabel("v component [m/s]")
    plt.title(f"EE Velocity Components ({name})")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(outdir / f"velocity_components_{name}.png", dpi=150)
    plt.close()


    # 5) CSV 保存（共有・再解析用）
    import csv
    csv_path = outdir / f"controller_timeseries_{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_sec","dist_err_m","speed_mps","vx","vy","vz","P_term","D_term"])
        for t, de, sp, (vx,vy,vz) in zip(times, dist_errs, vel_mag, vel_xyz):
            w.writerow([float(t), float(de), float(sp), float(vx), float(vy), float(vz)])
    print(f"[SAVE] controller timeseries CSV -> {csv_path}")




def plot_joint_pd_time_series(times, values_Tx7, title, ylabel, fname_prefix, name="run", outdir=OUTDIR):
    """
    values_Tx7: shape [T, 7] の時系列
    7関節を7行のサブプロットで縦積み表示し、PNG保存します。
    """
    times = np.asarray(times, dtype=np.float32)
    vals = np.asarray(values_Tx7, dtype=np.float32)  # [T,7]

    fig, axes = plt.subplots(7, 1, figsize=(9, 12), sharex=True)
    for j in range(7):
        ax = axes[j]
        if vals.shape[0] > 0:
            ax.plot(times, vals[:, j])
        ax.set_ylabel(f"{ylabel}\n[joint {j}]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{title} ({name})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = Path(outdir) / f"{fname_prefix}_{name}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[SAVE] {title} -> {outpath}")




def save_joint_pd_csv(times, p_Tx7, d_Tx7, total_Tx7, name="run", outdir=OUTDIR):
    """
    1ファイルに times, 各関節の p,d,total をまとめて保存します。
    列名: t_sec, p_j0..p_j6, d_j0..d_j6, tot_j0..tot_j6
    """
    import csv
    times = np.asarray(times, dtype=np.float32)
    p = np.asarray(p_Tx7, dtype=np.float32)
    d = np.asarray(d_Tx7, dtype=np.float32)
    tot = np.asarray(total_Tx7, dtype=np.float32)

    outpath = Path(outdir) / f"joint_pd_terms_{name}.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        header = ["t_sec"] + [f"p_j{j}" for j in range(7)] + [f"d_j{j}" for j in range(7)] + [f"tot_j{j}" for j in range(7)]
        w.writerow(header)
        T = len(times)
        for i in range(T):
            row = [float(times[i])]
            row += [float(p[i, j]) for j in range(7)]
            row += [float(d[i, j]) for j in range(7)]
            row += [float(tot[i, j]) for j in range(7)]
            w.writerow(row)
    print(f"[SAVE] joint PD terms CSV -> {outpath}")

def plot_joint_series(times, values_Tx7, title, ylabel, fname_prefix, name="run", outdir=OUTDIR):
    """
    values_Tx7: shape [T, 7]
    7関節を7行の縦積みプロットで保存します。
    """
    times = np.asarray(times, dtype=np.float32)
    vals = np.asarray(values_Tx7, dtype=np.float32)

    fig, axes = plt.subplots(7, 1, figsize=(9, 12), sharex=True)
    for j in range(7):
        ax = axes[j]
        if vals.shape[0] > 0:
            ax.plot(times, vals[:, j])
        ax.set_ylabel(f"{ylabel}\n[joint {j}]")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"{title} ({name})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outpath = Path(outdir) / f"{fname_prefix}_{name}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[SAVE] {title} -> {outpath}")


def save_joint_series_csv(times, q_err_Tx7, qd_err_Tx7, qd_Tx7, name="run", outdir=OUTDIR):
    """
    times と関節ごとの位置誤差, 速度誤差, 実速度を1ファイルに保存。
    列: t_sec, qerr_j0..6, qderr_j0..6, qd_j0..6
    """
    import csv
    times = np.asarray(times, dtype=np.float32)
    q_err = np.asarray(q_err_Tx7, dtype=np.float32)
    qd_err = np.asarray(qd_err_Tx7, dtype=np.float32)
    qd = np.asarray(qd_Tx7, dtype=np.float32)

    outpath = Path(outdir) / f"joint_errors_and_velocity_{name}.csv"
    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        header = (["t_sec"]
                  + [f"qerr_j{j}" for j in range(7)]
                  + [f"qderr_j{j}" for j in range(7)]
                  + [f"qd_j{j}" for j in range(7)])
        w.writerow(header)
        T = len(times)
        for i in range(T):
            row = [float(times[i])]
            row += [float(q_err[i, j]) for j in range(7)]
            row += [float(qd_err[i, j]) for j in range(7)]
            row += [float(qd[i, j]) for j in range(7)]
            w.writerow(row)
    print(f"[SAVE] joint errors & velocity CSV -> {outpath}")
#####################################################################





def main():
    env = FrankaSimEnv(
        model_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        image_size=IMAGE_SIZE,
        camera_name=CAMERA,
        success_thresh=SUCCESS_TOL,
        substeps=STEP_SUBSTEPS,
        normalizer=None,  
    )
    

    sid = env.physics.model.name2id('ee_target', 'site')
    site_body_id = env.physics.model.site_bodyid[sid]
    site_body_name = env.physics.model.id2name(site_body_id, 'body')
    print("ee_target is attached to body:", site_body_name)


    parent_bid = env.physics.model.body_parentid[site_body_id]  
    parent_name = 'ROOT' if parent_bid == -1 else env.physics.model.id2name(parent_bid, 'body')
    print("parent body:", parent_name)


    
    
    
    nv = env.physics.model.nv
    patch_franka_runtime(
        env,
        kp=KP,         
        kd=KD,                        
        dof_damping=DOF_DAMPING,   
        dof_armature=DOF_ARMATURE,
        ctrl_low=CTRL_LOW, 
        ctrl_high=CTRL_HIGH,   
        solver_iters=SOLVER_ITERS, 
        ls_iters=LS_ITERS,    
    )
    set_control_frequency_by_substeps(env, control_hz = Franka_FREQ)
    


    todo = [
        ("rectangle", NUM_PNTS, False),
        # ("lawnmower", NUM_PNTS, False),
        # ("lissajous", NUM_PNTS, False),
        # ("random", NUM_PNTS, False),
    ]
    all_metrics = []
    for kind, npts, shuf in todo:
        print(f'making {kind}....')
        traj = make_trajectory(kind=kind, n_points=npts, shuffle=shuf, seed=SEED)
        m = run_follow(env, traj, name=kind)
        all_metrics.append(m)

    with open(OUTDIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # maybe_make_gif()
    make_gifs_per_traj()
    print("\n✅ Trajectory evaluation complete. See:", OUTDIR)

if __name__ == "__main__":
    main()






