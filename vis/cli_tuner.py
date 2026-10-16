import os, sys, csv, glob, argparse, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import BallReachEnv
from core.constants import (
    TICKS_PER_DECISION, MAX_DECISIONS_STAGE0,
)
from core.nn import multi_discrete_to_controls
from core.obs_v3 import build_obs_v3
from core.reward import RewardConfig, compute_step_reward
from train.checkpoint import (
    load_ppo_model, ckpt_paths, discover_prefixes,
)


def eval_ground_batch(env, model, batch, max_decisions=MAX_DECISIONS_STAGE0):
    dev = next(model.parameters()).device
    cfg = RewardConfig(max_decisions=max_decisions)
    touch_count = 0
    touch_ticks_list = []
    all_ticks_list = []
    rcum_list = []
    touch_rcum_list = []
    peak_cl_list = []
    jump_eps = 0
    flip_eps = 0

    max_ticks = max_decisions * TICKS_PER_DECISION

    for sd in batch:
        env.reset(spawn_state=sd)
        obs_curr = env.last_obs_v3
        prev_snap = env.snapshot()
        min_dist = prev_snap["dist"]
        peak_closing = 0.0
        closing_accum = 0.0
        ep_rcum = 0.0
        ep_ticks = 0
        touched_now = False
        ep_jumped = False
        ep_flipped = False

        for step in range(1, max_decisions + 1):
            if dev.type == "cpu" and hasattr(model, "act_step_fast"):
                act_np, _, _ = model.act_step_fast(obs_curr, deterministic=True)
            else:
                with torch.no_grad():
                    x_t = torch.tensor(obs_curr, dtype=torch.float32, device=dev)
                    act_t, _, _, _ = model.get_action_and_value(x_t, deterministic=True)
                    act_np = act_t.cpu().numpy()
            if int(act_np[3]) == 1:
                ep_jumped = True
            ctrl = multi_discrete_to_controls(act_np)
            ticks_run = 0
            wasted_ticks = 0.0
            boost_used_step = 0.0
            for _ in range(TICKS_PER_DECISION):
                wasted_ticks += float(env.step_with_waste_check(ctrl))
                boost_used_step += float(getattr(env, "last_boost_used", 0.0))
                ticks_run += 1
                if env._touched:
                    break
            ep_ticks += ticks_run
            snap = env.snapshot()
            if snap["is_flipping"]:
                ep_flipped = True
            touched_now = snap["touched"]
            timeout_now = (step == max_decisions) and not touched_now

            exact_step = ep_ticks / float(TICKS_PER_DECISION)
            r, detail = compute_step_reward(
                cfg,
                prev_dist=prev_snap["dist"], new_dist=snap["dist"],
                step=exact_step,
                touched=touched_now, timeout=timeout_now,
                air_time=snap["air_time"], align=snap["align"],
                prev_dist_h=prev_snap["dist_h"], new_dist_h=snap["dist_h"],
                closing=snap["closing"],
                peak_closing=peak_closing, closing_accum=closing_accum,
                ball_vel_to_goal=snap["ball_vel_to_goal"],
                ticks_elapsed=ticks_run,
                boost_wasted_ticks=wasted_ticks,
                boost_used=boost_used_step,
                min_dist=min_dist,
            )
            min_dist      = detail["min_dist"]
            peak_closing  = detail["peak_closing"]
            closing_accum = detail["closing_accum"]
            ep_rcum      += r
            prev_snap     = snap

            if touched_now:
                touch_count += 1
                touch_ticks_list.append(ep_ticks)
                touch_rcum_list.append(ep_rcum)
                break

            obs_curr, _, _ = build_obs_v3(
                env.car, env.arena,
                target_pos=env.target_pos,
                target_time=env.target_time,
            )

        if ep_jumped:
            jump_eps += 1
        if ep_flipped:
            flip_eps += 1
        all_ticks_list.append(ep_ticks if touched_now else max_ticks)
        rcum_list.append(ep_rcum)
        peak_cl_list.append(peak_closing)

    n_tot = len(batch)
    mean_touch_ticks = float(np.mean(touch_ticks_list)) if touch_ticks_list else float("inf")
    mean_all_ticks = float(np.mean(all_ticks_list)) if all_ticks_list else float("inf")
    return {
        "touch_count": touch_count,
        "n_total": n_tot,
        "touch_rate": touch_count / max(1, n_tot),
        "touch_ticks": mean_touch_ticks,
        "touch_dec": mean_touch_ticks / float(TICKS_PER_DECISION),
        "all_ticks": mean_all_ticks,
        "all_dec": mean_all_ticks / float(TICKS_PER_DECISION),
        "mean_rcum": float(np.mean(rcum_list)),
        "touch_rcum": float(np.mean(touch_rcum_list)) if touch_rcum_list else 0.0,
        "mean_peak_closing": float(np.mean(peak_cl_list)),
        "jump_eps": jump_eps,
        "flip_eps": flip_eps,
        "per_ticks": all_ticks_list,
    }


def evaluate_model(ckpt_path, eval_seed=20261016, n_ground_batches=1, verbose=False):
    if not os.path.exists(ckpt_path):
        print(f"[cli_tuner] 错误: 找不到权重文件 {ckpt_path}")
        return None

    env = BallReachEnv(mode="ground")
    model, ckpt = load_ppo_model(ckpt_path, device="cpu")
    ckpt_mode = ckpt.get("mode", "ground")
    if ckpt_mode == "stage0":
        ckpt_mode = "ground"

    print("========================================================================================")
    print(f"  CLI Tuner [PPO]: 评估 {os.path.basename(ckpt_path)} (ckpt_mode={ckpt_mode})")
    print(f"  ep={ckpt.get('episode', '?')}  note={ckpt.get('note', '')}")
    print("========================================================================================")

    t0 = time.time()
    rng = random.Random(eval_seed)
    batch = []
    for _ in range(max(1, n_ground_batches)):
        batch.extend(env.generate_stratified_stage0_batch(rng=rng))
    g_res = eval_ground_batch(env, model, batch)
    print(f"\n[Ground 分层随机基准 ({len(batch)} 场景, seed={eval_seed})]")
    print(f"  触球率 (Touch Rate)         : {g_res['touch_count']}/{g_res['n_total']} ({g_res['touch_rate']:.1%})")
    print(f"  命中平均帧数 (Touch Frames) : {g_res['touch_ticks']:.1f} 帧 ({g_res['touch_dec']:.2f} 步)")
    print(f"  全样本含罚帧数 (All Frames) : {g_res['all_ticks']:.1f} 帧 ({g_res['all_dec']:.2f} 步, 漏球按 {MAX_DECISIONS_STAGE0*4}f 计)")
    print(f"  平均峰值接近速度 (Closing)  : {g_res['mean_peak_closing']:.1f} uu/s")
    print(f"  触发跳跃/翻滚局数 (Jump/Flip): Jump={g_res['jump_eps']}/{g_res['n_total']} | Flip={g_res['flip_eps']}/{g_res['n_total']}")
    print(f"  平均总分 (Mean r_cum)       : {g_res['mean_rcum']:+.2f} (命中均分: {g_res['touch_rcum']:+.2f})")

    dt = time.time() - t0
    print(f"elapsed = {dt:.2f}s")
    return {"ground": g_res}


def benchmark_all_prefixes(prefixes=None, from_tag="best", eval_seed=20261016, n_ground_batches=1):
    if not prefixes:
        prefixes = discover_prefixes()
    env = BallReachEnv(mode="ground")
    rng = random.Random(eval_seed)
    batch = []
    for _ in range(max(1, n_ground_batches)):
        batch.extend(env.generate_stratified_stage0_batch(rng=rng))

    rows = []
    for p in prefixes:
        latest_p, best_p = ckpt_paths(p)
        path = best_p if from_tag == "best" else latest_p
        if not os.path.exists(path):
            path = latest_p if from_tag == "best" else best_p
        if not os.path.exists(path):
            continue
        model, ckpt = load_ppo_model(path, device="cpu")
        g_res = eval_ground_batch(env, model, batch)
        rows.append({
            "prefix": p,
            "path": os.path.basename(path),
            "ep": ckpt.get("episode", "?"),
            "g_tc": g_res["touch_count"],
            "g_tot": g_res["n_total"],
            "g_tr": g_res["touch_rate"],
            "g_touch_f": g_res["touch_ticks"],
            "g_touch_dec": g_res["touch_dec"],
            "g_all_f": g_res["all_ticks"],
            "g_all_dec": g_res["all_dec"],
            "g_peak": g_res["mean_peak_closing"],
            "g_flip": g_res["flip_eps"],
            "g_rcum": g_res["mean_rcum"],
        })

    if not rows:
        print("[cli_tuner] ./checkpoints/ 下未找到任何模型权重文件。")
        return []

    print("========================================================================================================================")
    print(f"  PPO 策略全场随机地面球基准排行榜 (Ground {len(batch)} 场景分层同题对决, seed={eval_seed})")
    print("========================================================================================================================")
    rows_g = sorted(rows, key=lambda r: (r["g_all_f"], -r["g_tc"]))
    print(f"{'排序':<4} {'策略名称 (Prefix)':<20} {'触球率':<12} {'全样本均帧(含罚▼)':<18} {'命中均帧(步数)':<18} {'峰值接近速度':<14} {'翻滚局数':<10} {'平均总分(r_cum)'}")
    print("-" * 120)
    for idx, r in enumerate(rows_g, 1):
        tc_str = f"{r['g_tc']}/{r['g_tot']}({r['g_tr']:.0%})"
        all_str = f"{r['g_all_f']:.1f}f ({r['g_all_dec']:.2f}s)"
        tch_str = f"{r['g_touch_f']:.1f}f ({r['g_touch_dec']:.2f}s)"
        pk_str = f"{r['g_peak']:.0f} uu/s"
        fl_str = f"{r['g_flip']}/{r['g_tot']}"
        print(f"{idx:<6} {r['prefix']:<20} {tc_str:<13} {all_str:<20} {tch_str:<20} {pk_str:<15} {fl_str:<11} {r['g_rcum']:+.2f}")
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
    peak_cl = float(last.get("peak_closing", 0.0))
    cl_acc  = float(last.get("closing_accum", 0.0))
    final_dist = float(last.get("dist", 0.0))

    boost_frames = sum(1 for r in rows if int(float(r.get("boost", 0))) == 1)
    jump_frames  = [int(r["i"]) for r in rows if int(float(r.get("jump", 0))) == 1]
    flip_frames  = [int(r["i"]) for r in rows if int(float(r.get("is_flipping", 0))) == 1]
    cl2000_frames = [int(r["i"]) for r in rows if float(r.get("closing", 0.0)) > 2000.0]

    print("==========================================================================")
    print(f"  CSV 轨迹深度诊断: {csv_path} ({n_frames} 帧 = {n_frames/120.0:.2f}s, src={first.get('source', '?')})")
    print(f"  终点 dist={final_dist:.1f} | peak_closing={peak_cl:.1f} | closing_accum={cl_acc:+.2f}")
    print(f"  r_touch={r_touch:+.2f} | r_cum={r_cum:+.2f}")
    print(f"  起跳帧={jump_frames[0] if jump_frames else '无'} | 首次翻滚帧={flip_frames[0] if flip_frames else '无'} | "
          f"首次超速(>2000)帧={cl2000_frames[0] if cl2000_frames else '无'} (共 {len(cl2000_frames)} 帧)")
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
    p = argparse.ArgumentParser(description="CLI multi-checkpoint Ground benchmark evaluator & CSV diagnostics")
    p.add_argument("--prefix", type=str, nargs="*", default=None,
                   help="1 个或多个模型 prefix (不传则自动评测 ./checkpoints/ 下全部模型并输出排行榜)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="直接指定单个模型权重文件路径")
    p.add_argument("--from", dest="from_ckpt", default="best",
                   choices=["latest", "best"],
                   help="用 prefix 时读哪个权重 (default: best)")
    p.add_argument("--batches", type=int, default=1,
                   help="Ground 分层评估批次倍数 (每批 24 场景, default: 1 -> 24 场景)")
    p.add_argument("--seed", type=int, default=20261016,
                   help="Ground 评估随机种子 (default: 20261016)")
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
                n_ground_batches=args.batches,
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
        n_ground_batches=args.batches,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
