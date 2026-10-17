import os, sys, csv, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import BallReachEnv, EVAL_DIST_LABELS, EVAL_ANGLE_LABELS
from core.constants import (
    TICKS_PER_DECISION, MAX_DECISIONS_STAGE0, EVAL_SAMPLES_PER_BIN,
)
from train.checkpoint import (
    load_ppo_model, ckpt_paths, discover_prefixes,
    evaluate_16bin_grid, count_to_base36_char,
)


def evaluate_model(ckpt_path, eval_seed=20261016, samples_per_bin=EVAL_SAMPLES_PER_BIN, verbose=False):
    if not os.path.exists(ckpt_path):
        print(f"[cli_tuner] 错误: 找不到权重文件 {ckpt_path}")
        return None

    env = BallReachEnv(mode="all")
    model, ckpt = load_ppo_model(ckpt_path, device="cpu")
    ckpt_mode = ckpt.get("mode", "all")

    print("========================================================================================")
    print(f"  CLI Tuner [PPO]: 评估 {os.path.basename(ckpt_path)} (ckpt_mode={ckpt_mode})")
    print(f"  ep={ckpt.get('episode', '?')}  note={ckpt.get('note', '')}")
    print("========================================================================================")

    t0 = time.time()
    res = evaluate_16bin_grid(
        env, model,
        max_decisions=MAX_DECISIONS_STAGE0,
        samples_per_bin=samples_per_bin,
        seed=eval_seed,
    )
    print(f"\n[16-Bin 分位评测 ({res['n_total']} 场景 = 16x{samples_per_bin}, 满分='zzzzzzzzzzzzzzzz', seed={eval_seed})]")
    print(f"  16-Bin 编码 (0-9a-z)        : [{res['grid_str']}]  (各格命中: {res['bin_touches']})")
    print(f"  总触球率 (Touch Rate)       : {res['touch_count']}/{res['n_total']} ({res['touch_rate']:.1%})")
    print(f"  命中平均用时 (Touch Time)   : {res['touch_sec']:.2f}s ({res['touch_ticks']:.1f}f / {res['touch_dec']:.2f}步)")
    print(f"  全样本含罚用时 (All Time)   : {res['all_sec']:.2f}s ({res['all_ticks']:.1f}f, 漏球按 {MAX_DECISIONS_STAGE0*4}f 计)")
    print(f"  触发跳跃/翻滚局数 (Jump/Flip): Jump={res['jump_eps']}/{res['n_total']} | Flip={res['flip_eps']}/{res['n_total']}")
    print(f"  平均总分 (Mean r_cum)       : {res['mean_rcum']:+.2f}")

    print("\n  [4 距离段 x 4 方位角 (F/R/L/B) 矩阵明细]:")
    header_angles = "  ".join(f"{a:>8}" for a in EVAL_ANGLE_LABELS)
    print(f"  {'Dist \\ Angle':<12}  {header_angles}")
    for d_idx, d_lab in enumerate(EVAL_DIST_LABELS):
        cells = []
        for a_idx in range(4):
            b_idx = d_idx * 4 + a_idx
            cnt = res["bin_touches"][b_idx]
            ch = count_to_base36_char(cnt)
            cells.append(f"{ch}({cnt:>2}/{samples_per_bin})")
        print(f"  {d_lab:<12}  " + "  ".join(f"{c:>8}" for c in cells))

    dt = time.time() - t0
    print(f"\nelapsed = {dt:.2f}s")
    return res


def benchmark_all_prefixes(prefixes=None, from_tag="best", eval_seed=20261016, samples_per_bin=EVAL_SAMPLES_PER_BIN):
    if not prefixes:
        prefixes = discover_prefixes()
    env = BallReachEnv(mode="all")

    rows = []
    for p in prefixes:
        latest_p, best_p = ckpt_paths(p)
        path = best_p if from_tag == "best" else latest_p
        if not os.path.exists(path):
            path = latest_p if from_tag == "best" else best_p
        if not os.path.exists(path):
            continue
        model, ckpt = load_ppo_model(path, device="cpu")
        res = evaluate_16bin_grid(
            env, model,
            max_decisions=MAX_DECISIONS_STAGE0,
            samples_per_bin=samples_per_bin,
            seed=eval_seed,
        )
        rows.append({
            "prefix": p,
            "path": os.path.basename(path),
            "ep": ckpt.get("episode", "?"),
            "grid_str": res["grid_str"],
            "tc": res["touch_count"],
            "tot": res["n_total"],
            "tr": res["touch_rate"],
            "touch_sec": res["touch_sec"],
            "touch_f": res["touch_ticks"],
            "all_sec": res["all_sec"],
            "all_f": res["all_ticks"],
            "flip": res["flip_eps"],
            "rcum": res["mean_rcum"],
        })

    if not rows:
        print("[cli_tuner] ./checkpoints/ 下未找到任何模型权重文件。")
        return []

    n_tot = 16 * samples_per_bin
    print("========================================================================================================================")
    print(f"  PPO 策略 16-Bin 基准排行榜 ({n_tot} 场景 = 16x{samples_per_bin}, 完美编码='zzzzzzzzzzzzzzzz', seed={eval_seed})")
    print("========================================================================================================================")
    rows_sorted = sorted(rows, key=lambda r: (r["all_f"], -r["tc"]))
    print(f"{'排序':<4} {'策略名称 (Prefix)':<18} {'16-Bin (0-9a-z)':<18} {'触球率':<13} {'全样本均时(▼)':<16} {'命中均时':<16} {'翻滚局数':<10} {'平均总分'}")
    print("-" * 120)
    for idx, r in enumerate(rows_sorted, 1):
        tc_str = f"{r['tc']}/{r['tot']}({r['tr']:.0%})"
        all_str = f"{r['all_sec']:.2f}s ({r['all_f']:.0f}f)"
        tch_str = f"{r['touch_sec']:.2f}s ({r['touch_f']:.0f}f)"
        fl_str = f"{r['flip']}/{r['tot']}"
        print(f"{idx:<6} {r['prefix']:<18} [{r['grid_str']}] {tc_str:<13} {all_str:<18} {tch_str:<16} {fl_str:<11} {r['rcum']:+.2f}")
    print("========================================================================================================================")
    return rows


def verify_and_compare_csv(csv_path, ckpt_path=None):
    import RocketSim as rs
    from vis.tuner.export import parse_spawn_state_from_row

    if not os.path.exists(csv_path):
        print(f"[cli_tuner] 找不到 CSV 文件: {csv_path}")
        return
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"[cli_tuner] CSV 为空: {csv_path}")
        return

    n_frames = len(rows)
    first = rows[0]
    last = rows[-1]
    r_cum = float(last.get("r_cum", 0.0))
    r_touch = float(last.get("r_touch", 0.0))
    final_dist = float(last.get("dist", 0.0))

    boost_frames = sum(1 for r in rows if int(float(r.get("boost", 0))) == 1)
    jump_frames  = [int(r["i"]) for r in rows if int(float(r.get("jump", 0))) == 1]
    flip_frames  = [int(r["i"]) for r in rows if int(float(r.get("is_flipping", 0))) == 1]

    print("==========================================================================")
    print(f"  CSV 轨迹深度诊断: {csv_path} ({n_frames} 帧 = {n_frames/120.0:.2f}s, src={first.get('source', '?')})")
    print(f"  终点 dist={final_dist:.1f} | r_touch={r_touch:+.2f} | r_cum={r_cum:+.2f}")
    print(f"  起跳帧={jump_frames[0] if jump_frames else '无'} | 首次翻滚帧={flip_frames[0] if flip_frames else '无'}")
    print(f"  喷气帧={boost_frames}/{n_frames} ({boost_frames/n_frames:.1%})")

    sd = parse_spawn_state_from_row(first)
    if sd is not None and "px" in first:
        env = BallReachEnv(mode=sd["mode"])
        env.reset(spawn_state=sd)
        max_car_err = 0.0
        max_ball_err = 0.0
        for r in rows:
            c = rs.CarControls()
            c.throttle  = float(r.get("thr", 0.0))
            c.steer     = float(r.get("str", 0.0))
            c.pitch     = float(r.get("ptc", 0.0))
            c.yaw       = float(r.get("yaw", 0.0))
            c.roll      = float(r.get("roll", 0.0))
            c.jump      = bool(int(float(r.get("jump", 0))))
            c.boost     = bool(int(float(r.get("boost", 0))))
            c.handbrake = bool(int(float(r.get("handbrake", 0))))
            env.car.set_controls(c)
            env.arena.step(1)
            snap = env.snapshot()
            cp_csv = (float(r["px"]), float(r["py"]), float(r["pz"]))
            bp_csv = (float(r["bx"]), float(r["by"]), float(r["bz"]))
            c_err = max(abs(a - b) for a, b in zip(snap["car_pos"], cp_csv))
            b_err = max(abs(a - b) for a, b in zip(snap["ball_pos"], bp_csv))
            max_car_err = max(max_car_err, c_err)
            max_ball_err = max(max_ball_err, b_err)
        print(f"  [Deterministic Replay Check] max_car_pos_err={max_car_err:.6f} uu | max_ball_pos_err={max_ball_err:.6f} uu")
    print("==========================================================================")


def main():
    p = argparse.ArgumentParser(description="CLI multi-checkpoint 16-bin evaluator & CSV diagnostics")
    p.add_argument("--prefix", type=str, nargs="*", default=None,
                   help="1 个或多个模型 prefix (不传则自动评测 ./checkpoints/ 下全部模型并输出排行榜)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="直接指定单个模型权重文件路径")
    p.add_argument("--from", dest="from_ckpt", default="best",
                   choices=["latest", "best"],
                   help="用 prefix 时读哪个权重 (default: best)")
    p.add_argument("--samples", type=int, default=EVAL_SAMPLES_PER_BIN,
                   help=f"每个分位格样本数 (default: {EVAL_SAMPLES_PER_BIN} -> 16x{EVAL_SAMPLES_PER_BIN}={16*EVAL_SAMPLES_PER_BIN} 场景)")
    p.add_argument("--seed", type=int, default=20261016,
                   help="评估随机种子 (default: 20261016)")
    p.add_argument("--verify", type=str, nargs="+", default=None,
                   help="校验并诊断指定的 1 个或多个 CSV 轨迹文件")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verify:
        for vp in args.verify:
            verify_and_compare_csv(vp, args.ckpt)
        return

    if args.ckpt is None and (not args.prefix or len(args.prefix) > 1):
        prefixes = args.prefix if args.prefix else discover_prefixes()
        if len(prefixes) > 1 or not args.prefix:
            benchmark_all_prefixes(
                prefixes=prefixes,
                from_tag=args.from_ckpt,
                eval_seed=args.seed,
                samples_per_bin=args.samples,
            )
            return

    ckpt_path = args.ckpt
    if ckpt_path is None:
        prefix = args.prefix[0] if args.prefix else discover_prefixes()[0]
        latest_p, best_p = ckpt_paths(prefix)
        ckpt_path = best_p if args.from_ckpt == "best" else latest_p
        if not os.path.exists(ckpt_path):
            alt_p = latest_p if args.from_ckpt == "best" else best_p
            if os.path.exists(alt_p):
                ckpt_path = alt_p

    evaluate_model(
        ckpt_path,
        eval_seed=args.seed,
        samples_per_bin=args.samples,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
