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

# ---- ログ設定（INFO以上を標準出力へ）----
logging.basicConfig(
    level=logging.INFO,  # 煩ければ INFO→WARNING に
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("traj_eval")


OUTDIR = Path("./robot_task_sim/traj_eval_out")
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "frames").mkdir(parents=True, exist_ok=True)

# === 評価パラメータ ===
IMAGE_SIZE = (128, 128)         # 保存フレーム解像度
CAMERA = "top_view"
SUCCESS_TOL = 0.1             # EE到達判定 (m)
STEP_SUBSTEPS = 50              
MAX_STEPS_PER_WAYPOINT = 300    # 各目標に対して許容する最大ステップ数
HOLD_STEPS_AT_TARGET = 10       # 収束後に少し保持して撮影
SAVE_EVERY = 1                # フレーム保存間隔
SEED = 0

NUM_PNTS = 200 #軌跡の分割ポイント数

# あなたの作業机の矩形領域（少しマージンを引いてIK失敗を避ける）
x_min, x_max = 0.315, 0.715
y_min, y_max = -0.2, 0.2
z_fixed = 0.10
MARGIN = 0.01

# ロボットパラメータ
Franka_FPS = 2.5 #2.5
KP=[None, None, None, None, None, None, None]#[4500, 4500, 3500, 3500, 2000, 2000, 2000] # 位置アクチュエータの比例ゲイン  (値を上げるほど追従が速くなるが、振動が起きやすい)
KD=[None, None, None, None, None, None, None]#[450, 450, 350, 350, 200, 200, 200]
DOF_DAMPING=None           # DOFダンピング（振動抑制）      (高いとブレーキがかかるよう動作する. 過剰だと応答が鈍くなる)
DOF_ARMATURE=None          # DOFアーマチュア（慣性付加）    (大きくすると応用が重く安定する)
CTRL_LOW=None              # アーム用ctrl下限（7要素）     (学習時や制御ポリシー設計と一致しているか確認)
CTRL_HIGH=None            # アーム用ctrl上限（7要素）     (学習時や制御ポリシー設計と一致しているか確認)
SOLVER_ITERS=None          # ソルバ反復                  (接触解決制度. 上げると接触安定性up, 速度down. デフォルトでは十分なことが多い)
LS_ITERS=None            # ラインサーチ反復             (大抵は solver_iters に比べ影響小. 接触剛性が高い場合のみ確認)

import collections.abc as cabc


def set_kp_per_joint(env, kp):
    """
    kp: 
      - float（全関節を同一値に）
      - 長さ=len(env.arm_actuator_ids) のシーケンス
        その中に None を含めると、その関節は現状値を維持します
    """
    
    m = env.physics.model
    arm_ids = list(env.arm_actuator_ids)
    n = len(arm_ids)


    if isinstance(kp, (int, float, np.floating)):
        kp_list = [float(kp)] * n
    else:
        try:
            kp_list = list(kp)
        except TypeError:
            raise TypeError("kp は float か、長さが関節数のシーケンスで指定してください")
        assert len(kp_list) == n, f"kp の長さは {n} 要素（関節数）にしてください"


    before = m.actuator_gainprm[arm_ids, 0].copy()
    print("kp before:", before)


    for aid, v in zip(arm_ids, kp_list):
        if v is None:
            continue
        m.actuator_gainprm[aid, 0] = float(v)

    env.physics.forward()
    after = m.actuator_gainprm[arm_ids, 0]
    print("kp after :", after)


def set_kd_per_joint(env, kd):
    """
    kd:
      - float（全関節を同一値に）
      - 長さ=len(env.arm_actuator_ids) のシーケンス
        その中に None を含めると、その関節は現状値を維持
    備考: MuJoCoの一般アクチュエータ(affine)では kd = -biasprm[2]
    """
    m = env.physics.model
    arm_ids = list(env.arm_actuator_ids)
    n = len(arm_ids)

    if isinstance(kd, (int, float, np.floating)):
        kd_list = [float(kd)] * n
    else:
        try:
            kd_list = list(kd)
        except TypeError:
            raise TypeError("kd は float か、長さが関節数のシーケンスで指定してください")
        assert len(kd_list) == n, f"kd の長さは {n} 要素（関節数）にしてください"

    # before（人が読みやすいように kd 値 = -biasprm[:,2] で表示）
    before = -m.actuator_biasprm[arm_ids, 2].copy()
    print("kd before:", before)

    for aid, v in zip(arm_ids, kd_list):
        if v is None:
            continue
        # kd = -biasprm[2] なので、biasprm[2] に -kd を入れる
        m.actuator_biasprm[aid, 2] = -float(v)

    env.physics.forward()
    after = -m.actuator_biasprm[arm_ids, 2]
    print("kd after :", after)



def patch_franka_runtime(env, kp=None, kd=None, dof_damping=None, dof_armature=None,
                         ctrl_low=None, ctrl_high=None, solver_iters=None, ls_iters=None):
    m = env.physics.model

    # --- kp（スカラー or 7次元ベクトル両対応）---
    if kp is not None:
        set_kp_per_joint(env, kp)
    if kd is not None:              
        set_kd_per_joint(env, kd) 

    # --- 以下はあなたの既存コードどおり ---
    if dof_damping is not None:
        arr = np.asarray(dof_damping, np.float32)
        assert arr.shape[0] == m.nv, "dof_damping の長さが nv と一致していません"
        m.dof_damping[:] = arr

    if dof_armature is not None:
        arr = np.asarray(dof_armature, np.float32)
        assert arr.shape[0] == m.nv, "dof_armature の長さが nv と一致していません"
        m.dof_armature[:] = arr

    if (ctrl_low is not None) or (ctrl_high is not None):
        lo = np.asarray(ctrl_low if ctrl_low is not None
                        else m.actuator_ctrlrange[env.arm_actuator_ids, 0], np.float32)
        hi = np.asarray(ctrl_high if ctrl_high is not None
                        else m.actuator_ctrlrange[env.arm_actuator_ids, 1], np.float32)
        assert lo.shape[0] == len(env.arm_actuator_ids)
        assert hi.shape[0] == len(env.arm_actuator_ids)
        m.actuator_ctrlrange[env.arm_actuator_ids, 0] = lo
        m.actuator_ctrlrange[env.arm_actuator_ids, 1] = hi
        env.ctrlrange = m.actuator_ctrlrange[env.arm_actuator_ids].copy()

    if solver_iters is not None:
        m.opt.iterations = int(solver_iters)
    if ls_iters is not None:
        m.opt.ls_iterations = int(ls_iters)

    env.physics.forward()


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
        # 長方形の外周をぐるっと回る
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
        # 往復スキャン（芝刈り軌跡）
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
    return traj

def ee(env):
    return env.get_ee_position().astype(np.float32)

def render_rgb(env, size=IMAGE_SIZE, camera=CAMERA):
    # 直接dm_controlレンダを使って [H,W,3] (0..1 float) で返す
    rgb = env.physics.render(height=size[0], width=size[1],
                             camera_id=env.camera_id if camera!="default" else -1)

    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255.0  # -> 0..1 float
    else:
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return rgb

def run_follow(env, traj_xyz, name="rectangle"):
    ee_traj = []
    tgt_traj = []
    reached = []
    steps_used = []
    total_frames = 0

    #フレームごと
    frame_ee_xy = []
    frame_tgt_xy = []

    log.info(f"[run] >>> start '{name}' | waypoints={len(traj_xyz)} tol={SUCCESS_TOL} substeps={STEP_SUBSTEPS}")
    t0 = time.time()

    # 初期化：最初の点近傍に寄せる（IKでワープ→安定化）
    env.reset()
    env.set_xyz(traj_xyz[0])
    for _ in range(50): env.physics.step()
    frame_ee_xy.append(env.get_ee_position()[:2].copy())
    frame_tgt_xy.append(traj_xyz[0][:2].copy())

    for i, target in enumerate(traj_xyz):
        log.info(f"[run] wp[{i+1}/{len(traj_xyz)}] target={np.round(target,4)}")
        # IKで目標関節角（q_des）を得る
        ik = env.calc_inverse_kinematic(target)
        q_des = ik.qpos[:7].copy()

        # 収束ループ
        step_cnt = 0
        while step_cnt < MAX_STEPS_PER_WAYPOINT:
            obs, reward, done, truncated, info = env.step(q_des)
            cur = ee(env)
            err = float(np.linalg.norm(cur - target))

            # if step_cnt % SAVE_EVERY == 0:
            #     rgb = render_rgb(env)
            #     plt.imsave(OUTDIR / "frames" / f"{name}_{i:03d}_{step_cnt:04d}.png", rgb)
            #     total_frames += 1
                
            #毎フレームのEE/targetをログ（XY）
            frame_ee_xy.append(cur[:2].copy())
            frame_tgt_xy.append(target[:2].copy())

            # 進捗ログ（25ステップ毎 or 初回）
            if step_cnt == 0 or (step_cnt % 25 == 0):
                log.info(f"[run]   step={step_cnt:4d}  err={err:.4f} m")

            if err < SUCCESS_TOL:
                # 少し保持してから次へ
                for _ in range(HOLD_STEPS_AT_TARGET):
                    env.step(q_des)
                reached.append((i, True))
                steps_used.append(step_cnt+HOLD_STEPS_AT_TARGET)
                break

            step_cnt += 1

        if step_cnt >= MAX_STEPS_PER_WAYPOINT:
            reached.append((i, False))
            steps_used.append(step_cnt)

        ee_traj.append(cur)
        tgt_traj.append(target)
        
    print('reached: ', reached)
    dt = time.time() - t0
    ee_traj = np.array(ee_traj, dtype=np.float32)
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
    # 可視化（XY投影）
    plt.figure(figsize=(5,5))
    plt.plot(tgt_traj[:,0], tgt_traj[:,1], linestyle="--", marker="o", markersize=10, label="target XY")
    plt.plot(ee_traj[:,0],  ee_traj[:,1],  linestyle="-",  marker=".", markersize=10, label="executed XY")

    
    plt.text(ee_traj[0,0], ee_traj[0,1], "S", color="orange", fontsize=12, fontweight="bold", ha="center", va="center")
    plt.text(ee_traj[-1,0], ee_traj[-1,1], "G", color="orange", fontsize=12, fontweight="bold", ha="center", va="center")

    plt.xlabel("X [m]"); plt.ylabel("Y [m]"); plt.title(f"Trajectory: {name}")
    plt.axis("equal"); plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTDIR / f"ee_traj_plot_{name}.png", dpi=150)
    plt.close()

    # エラー推移
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

    # 連続軌跡のXY図
    plt.figure(figsize=(5,5))
    plt.plot(frame_tgt_xy[:,0], frame_tgt_xy[:,1], 'g--', lw=1, marker="o",markersize=5, label='target XY (per frame)')
    plt.plot(frame_ee_xy[:,0],  frame_ee_xy[:,1],  'r-',  lw=1, marker="o",markersize=1,label='executed XY (per frame)')

    plt.text(frame_ee_xy[0,0], frame_ee_xy[0,1], "S", color="red", fontsize=12, fontweight="bold", ha="center", va="center")
    plt.text(frame_ee_xy[-1,0], frame_ee_xy[-1,1], "G", color="red", fontsize=12, fontweight="bold", ha="center", va="center")

    plt.xlabel("X [m]"); plt.ylabel("Y [m]")
    plt.title(f"Frame-wise trajectory: {name}")
    plt.axis("equal"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / f"ee_traj_plot_framewise_{name}.png", dpi=150)
    plt.close()
    print(f"[SAVE] framewise plot -> ee_traj_plot_framewise_{name}.png")
    

    print(f"[DONE] {name} reached={metrics['reached_ratio']:.3f}, mean_err={metrics['mean_err_m']:.4f} m")
    return metrics

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


def make_gifs_per_traj(duration=0.05):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[INFO] imageio not available; skip GIF")
        return

    frames = sorted((OUTDIR / "frames").glob("*.png"))  # 0埋めなので辞書順で時系列OK
    groups = {}
    for p in frames:
        name = p.stem.split("_", 1)[0]  # 先頭プレフィックス=軌跡名
        groups.setdefault(name, []).append(p)

    for name, fps in groups.items():
        fps = sorted(fps) 
        if not fps:
            continue
        imgs = [imageio.imread(str(p)) for p in fps]  
        imageio.mimsave(OUTDIR / f"{name}.gif", imgs, duration=duration)
        print(f"[SAVE] {name}.gif  ({len(fps)} frames, full span)")



def main():
    env = FrankaSimEnv(
        model_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        image_size=IMAGE_SIZE,
        camera_name=CAMERA,
        success_thresh=SUCCESS_TOL,
        substeps=STEP_SUBSTEPS,
        normalizer=None,  
    )
    
    
    
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
    set_control_frequency_by_substeps(env, control_hz = Franka_FPS)
    

    # 軌跡をいくつか評価
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
    # make_gifs_per_traj()
    print("\n✅ Trajectory evaluation complete. See:", OUTDIR)

if __name__ == "__main__":
    main()
