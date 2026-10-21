import os, sys, csv, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import BallReachEnv, STAGE_NAMES
from core.constants import (
    TICKS_PER_DECISION, MAX_DECISIONS, EVAL_SAMPLES_PER_STAGE, NUM_CURRICULUM_STAGES,
    NUM_CURRICULUM_STEPS, DIR5_LABELS, NUM_DIR5_BUCKETS,
)
from train.checkpoint import (
    load_ppo_model, ckpt_paths, discover_prefixes,
    evaluate_stage_grid, rate_to_digit_char,
)


def evaluate_dir5_buckets(env, model, samples_per_bucket=15, seed=20261016):
    import random
    from core.policy_net import multi_discrete_to_controls
    r = random.Random(seed)
    d_rates = []
    print(f"\n  [Dir5 五方向桶触球率诊断 (F/B/R/L/U, P1..P4)]:")
    for b_idx, b_lbl in enumerate(DIR5_LABELS):
        bits = "".join("1" if i == b_idx else "0" for i in range(NUM_DIR5_BUCKETS))
        env.set_spawn_dist(f"p[1111]u[{bits}]")
        touches = 0
        jumps = 0
        for _ in range(samples_per_bucket):
            random.seed(r.randint(1, 10**9))
            env.reset()
            obs = env.last_obs
            jumped = False
            ep_max_dec = getattr(env, "ep_max_decisions", MAX_DECISIONS)
            for _ in range(ep_max_dec):
                act, _, _ = model.act_step_fast(obs, deterministic=True)
                ctrl = multi_discrete_to_controls(act)
                if ctrl.jump:
                    jumped = True
                for _ in range(TICKS_PER_DECISION):
                    env.step_tick(ctrl)
                    if env._touched:
                        break
                if env._touched:
                    touches += 1
                    break
                obs, _, _ = env.get_obs()
            if jumped:
                jumps += 1
        rate = touches / max(1, samples_per_bucket)
        d_rates.append(rate)
        ch = rate_to_digit_char(rate)
        bar = "█" * int(round(rate * 10)) + "░" * (10 - int(round(rate * 10)))
        print(f"    Dir={b_lbl} : [{ch}] {bar}  {touches:>2}/{samples_per_bucket} ({rate:5.1%}) | Jump={jumps:>2}/{samples_per_bucket}")
    env.set_spawn_dist("")
    d_code = "".join(rate_to_digit_char(rt) for rt in d_rates)
    print(f"  --> 5-Dir 编码: U5[{d_code}]")
    return d_code


def evaluate_model(ckpt_path, eval_seed=20261016, samples_per_stage=EVAL_SAMPLES_PER_STAGE, verbose=False, eval_dir5=False, spawn_dist=None):
    if not os.path.exists(ckpt_path):
        print(f"[cli_tuner] 错误: 找不到权重文件 {ckpt_path}")
        return None

    env = BallReachEnv(mode="curriculum", stage_level=NUM_CURRICULUM_STAGES, spawn_dist=spawn_dist)
    model, ckpt = load_ppo_model(ckpt_path, device="cpu")
    ckpt_mode = ckpt.get("mode", "curriculum")
    ckpt_stage_lv = ckpt.get("stage_level", NUM_CURRICULUM_STAGES)
    ckpt_step_idx = int(ckpt.get("curriculum_step_idx", NUM_CURRICULUM_STEPS - 1))
    env.set_curriculum_state(ckpt_stage_lv, step_idx=ckpt_step_idx)

    print("========================================================================================")
    print(f"  CLI Tuner [Static Ball Master]: 评估 {os.path.basename(ckpt_path)} (mode={ckpt_mode}, phase_lv={ckpt_stage_lv}, step={ckpt_step_idx})")
    print(f"  ep={ckpt.get('episode', '?')}  note={ckpt.get('note', '')}")
    print("========================================================================================")

    t0 = time.time()
    res = evaluate_stage_grid(
        env, model,
        max_decisions=MAX_DECISIONS,
        samples_per_stage=samples_per_stage,
        active_stages=NUM_CURRICULUM_STAGES,
        seed=eval_seed,
    )
    print(f"\n[4-Phase 静态球课程矩阵评测 ({res['n_total']} 场景 = {NUM_CURRICULUM_STAGES}x{samples_per_stage}, 达标线='7777', seed={eval_seed})]")
    print(f"  4-Phase 编码 (0-9)          : [{res['grid_str']}]  (各阶命中: {res['bin_touches']})")
    print(f"  总触球率 (Touch Rate)       : {res['touch_count']}/{res['n_total']} ({res['touch_rate']:.1%})")
    print(f"  命中平均用时 (Touch Time)   : {res['touch_sec']:.2f}s ({res['touch_ticks']:.1f}f / {res['touch_dec']:.2f}步)")
    print(f"  全样本含罚用时 (All Time)   : {res['all_sec']:.2f}s ({res['all_ticks']:.1f}f)")
    print(f"  触发跳跃/翻滚局数 (Jump/Flip): Jump={res['jump_eps']}/{res['n_total']} | Flip={res['flip_eps']}/{res['n_total']}")
    print(f"  平均总分 (Mean r_cum)       : {res['mean_rcum']:+.2f}")

    print("\n  [4-Phase 课程明细]:")
    for s_idx, s_name in enumerate(STAGE_NAMES):
        cnt = res["bin_touches"][s_idx]
        rate = res["stage_rates"][s_idx]
        ch = rate_to_digit_char(rate)
        bar = "█" * int(round(rate * 10)) + "░" * (10 - int(round(rate * 10)))
        print(f"    {s_name:<30} : [{ch}] {bar}  {cnt:>2}/{samples_per_stage} ({rate:5.1%})")

    if eval_dir5:
        evaluate_dir5_buckets(env, model, samples_per_bucket=15, seed=eval_seed)

    dt = time.time() - t0
    print(f"\nelapsed = {dt:.2f}s")
    return res


def benchmark_all_prefixes(prefixes=None, from_tag="best", eval_seed=20261016, samples_per_stage=EVAL_SAMPLES_PER_STAGE):
    if not prefixes:
        prefixes = discover_prefixes()
    env = BallReachEnv(mode="curriculum", stage_level=NUM_CURRICULUM_STAGES)

    rows = []
    for p in prefixes:
        latest_p, best_p = ckpt_paths(p)
        path = best_p if from_tag == "best" else latest_p
        if not os.path.exists(path):
            path = latest_p if from_tag == "best" else best_p
        if not os.path.exists(path):
            continue
        model, ckpt = load_ppo_model(path, device="cpu")
        res = evaluate_stage_grid(
            env, model,
            max_decisions=MAX_DECISIONS,
            samples_per_stage=samples_per_stage,
            active_stages=NUM_CURRICULUM_STAGES,
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

    n_tot = NUM_CURRICULUM_STAGES * samples_per_stage
    print("========================================================================================================================")
    print(f"  PPO 静态球大王 4-Phase 课程矩阵排行榜 ({n_tot} 场景 = {NUM_CURRICULUM_STAGES}x{samples_per_stage}, 达标线='7777', seed={eval_seed})")
    print("========================================================================================================================")
    rows_sorted = sorted(rows, key=lambda r: (r["all_f"], -r["tc"]))
    print(f"{'排序':<4} {'策略名称 (Prefix)':<18} {'4-Phase (0-9)':<16} {'触球率':<13} {'全样本均时(▼)':<16} {'命中均时':<16} {'翻滚局数':<10} {'平均总分'}")
    print("-" * 120)
    for idx, r in enumerate(rows_sorted, 1):
        tc_str = f"{r['tc']}/{r['tot']}({r['tr']:.0%})"
        all_str = f"{r['all_sec']:.2f}s ({r['all_f']:.0f}f)"
        tch_str = f"{r['touch_sec']:.2f}s ({r['touch_f']:.0f}f)"
        fl_str = f"{r['flip']}/{r['tot']}"
        print(f"{idx:<6} {r['prefix']:<18} [{r['grid_str']}]           {tc_str:<13} {all_str:<18} {tch_str:<16} {fl_str:<11} {r['rcum']:+.2f}")
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
            env.step_tick(c)
            snap = env.snapshot()
            cp_csv = (float(r["px"]), float(r["py"]), float(r["pz"]))
            c_err = max(abs(a - b) for a, b in zip(snap["car_pos"], cp_csv))
            max_car_err = max(max_car_err, c_err)
        print(f"  [Deterministic Replay Check] max_car_pos_err={max_car_err:.6f} uu")
    print("==========================================================================")


def main():
    p = argparse.ArgumentParser(description="CLI multi-checkpoint 4-Phase Static Ball Master evaluator")
    p.add_argument("--prefix", type=str, nargs="*", default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--from", dest="from_ckpt", default="best", choices=["latest", "best"])
    p.add_argument("--dist", "--spawn-dist", dest="spawn_dist", type=str, default=None,
                   help="Custom scenario distribution e.g. 'p[1000]u[11111]'")
    p.add_argument("--dir5", "--aerial", dest="dir5", action="store_true", help="Run 5-direction bucket diagnostic table")
    p.add_argument("--samples", type=int, default=EVAL_SAMPLES_PER_STAGE)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--verify", type=str, nargs="+", default=None)
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
                samples_per_stage=args.samples,
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
        samples_per_stage=args.samples,
        verbose=args.verbose,
        eval_dir5=args.dir5,
        spawn_dist=args.spawn_dist,
    )


if __name__ == "__main__":
    main()
