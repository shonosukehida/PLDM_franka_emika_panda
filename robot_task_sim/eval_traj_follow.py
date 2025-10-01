# evaluate_trajectory_following.py
from pathlib import Path
import numpy as np
import json, time
import matplotlib
matplotlib.use("Agg")  # 画面なしで保存
import matplotlib.pyplot as plt

# ==== あなたのプロジェクトの import に合わせて調整 ====
# 例: from pldm_envs.franka.envs import FrankaSimEnv
from pldm_envs.franka.envs import FrankaSimEnv
# ========================================================

OUTDIR = Path("./robot_task_sim/traj_eval_out")
OUTDIR.mkdir(parents=True, exist_ok=True)
(OUTDIR / "frames").mkdir(parents=True, exist_ok=True)

# === 評価パラメータ ===
IMAGE_SIZE = (128, 128)         # 保存フレーム解像度
CAMERA = "top_view"
SUCCESS_TOL = 0.01              # EE到達判定 (m)
STEP_SUBSTEPS = 50              # env.__init__(substeps=...) と同一が無難
MAX_STEPS_PER_WAYPOINT = 300    # 各目標に対して許容する最大ステップ数
HOLD_STEPS_AT_TARGET = 10       # 収束後に少し保持して撮影
SAVE_EVERY = 5                  # フレーム保存間隔
SEED = 0

# あなたの作業机の矩形領域（少しマージンを引いてIK失敗を避ける）
x_min, x_max = 0.315, 0.715
y_min, y_max = -0.2, 0.2
z_fixed = 0.10
MARGIN = 0.01

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

    t0 = time.time()

    # 初期化：最初の点近傍に寄せる（IKでワープ→安定化）
    env.reset()
    env.set_xyz(traj_xyz[0])
    for _ in range(50): env.physics.step()

    for i, target in enumerate(traj_xyz):
        # IKで目標関節角（q_des）を得る
        ik = env.calc_inverse_kinematic(target)
        q_des = ik.qpos[:7].copy()

        # 収束ループ
        step_cnt = 0
        while step_cnt < MAX_STEPS_PER_WAYPOINT:
            # この環境のstepは action を「望ましい qpos」として解釈してくれる（中でdqに変換）
            obs, reward, done, truncated, info = env.step(q_des)
            cur = ee(env)
            err = float(np.linalg.norm(cur - target))

            if step_cnt % SAVE_EVERY == 0:
                rgb = render_rgb(env)
                plt.imsave(OUTDIR / "frames" / f"{name}_{i:03d}_{step_cnt:04d}.png", rgb)
                total_frames += 1

            if err < SUCCESS_TOL:
                # 少し保持してから次へ
                for _ in range(HOLD_STEPS_AT_TARGET):
                    env.step(q_des)
                reached.append(True)
                steps_used.append(step_cnt+HOLD_STEPS_AT_TARGET)
                break

            step_cnt += 1

        if step_cnt >= MAX_STEPS_PER_WAYPOINT:
            reached.append(False)
            steps_used.append(step_cnt)

        ee_traj.append(cur)
        tgt_traj.append(target)

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

    # 可視化（XY投影）
    plt.figure(figsize=(5,5))
    plt.plot(tgt_traj[:,0], tgt_traj[:,1], linestyle="--", marker="o", markersize=2, label="target XY")
    plt.plot(ee_traj[:,0],  ee_traj[:,1],  linestyle="-",  marker=".", markersize=2, label="executed XY")
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

    # メトリクス保存
    with open(OUTDIR / f"metrics_{name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] {name} reached={metrics['reached_ratio']:.3f}, mean_err={metrics['mean_err_m']:.4f} m")
    return metrics

def maybe_make_gif():
    # 任意: imageio が入っていればGIF生成
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


def make_gifs_per_traj(max_frames=300, duration=0.05):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("[INFO] imageio not available; skip GIF")
        return
    frames = sorted((OUTDIR / "frames").glob("*.png"))
    groups = {}
    for p in frames:
        name = p.stem.split("_", 1)[0]  # 先頭の prefix が軌跡名
        groups.setdefault(name, []).append(p)
    print("[DBG] groups.keys()", groups.keys())
    for name, fps in groups.items():
        sel = fps[:max_frames]
        imgs = [imageio.imread(str(p)) for p in sel]
        imageio.mimsave(OUTDIR / f"{name}.gif", imgs, duration=duration)
        print(f"[SAVE] {name}.gif  ({len(sel)} frames)")


def main():
    # 環境生成
    env = FrankaSimEnv(
        model_path="mujoco_menagerie/franka_emika_panda/scene.xml",
        image_size=IMAGE_SIZE,
        camera_name=CAMERA,
        success_thresh=SUCCESS_TOL,
        substeps=STEP_SUBSTEPS,
        normalizer=None,   # ここは学習時に合わせてもOK
    )

    # 軌跡をいくつか評価
    todo = [
        ("rectangle", 80, False),
        ("lawnmower", 90, False),
        ("lissajous", 120, False),
        ("random", 80, False),
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
