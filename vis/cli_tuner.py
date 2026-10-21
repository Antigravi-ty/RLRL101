import os, sys, csv, argparse, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import KickoffEnv
from core.constants import (
    MAX_DECISIONS, NUM_KICKOFF_SLOTS, KICKOFF_SLOT_NAMES,
)
from train.checkpoint import (
    load_ppo_model, ckpt_paths, discover_prefixes,
    evaluate_5slot_kickoff,
)


def evaluate_model(ckpt_path, verbose=False):
    if not os.path.exists(ckpt_path):
        print(f"[cli_tuner] 错误: 找不到权重文件 {ckpt_path}")
        return None

    env = KickoffEnv()
    model, ckpt = load_ppo_model(ckpt_path, device="cpu")

    print("========================================================================================")
    print(f"  CLI Tuner [1v1 Fastest Kickoff]: 评估 {os.path.basename(ckpt_path)}")
    print(f"  ep={ckpt.get('episode', '?')}  note={ckpt.get('note', '')}")
    print("========================================================================================")

    t0 = time.time()
    res = evaluate_5slot_kickoff(env, model, max_decisions=MAX_DECISIONS)
    print(f"\n[5-Slot 标准开球位评测 (5 槽位 = DiagRight / DiagLeft / BackRight / BackLeft / Center, 悬挂缓冲=60f)]")
    print(f"  5-Slot 编码                 : [{res['grid_str']}]")
    print(f"  总触球率 (Touch Rate)       : {res['touch_count']}/{res['n_total']} ({res['touch_rate']:.1%})")
    print(f"  命中平均用时 (Touch Time)   : {res['touch_sec']:.2f}s ({res['touch_ticks']:.1f}f / {res['touch_dec']:.2f}步)")
    print(f"  全样本含罚用时 (All Time)   : {res['all_sec']:.2f}s ({res['all_ticks']:.1f}f)")
    print(f"  触发跳跃/翻滚槽位 (Jump/Flip): Jump={res['jump_eps']}/{res['n_total']} | Flip={res['flip_eps']}/{res['n_total']}")
    print(f"  平均总分 (Mean r_cum)       : {res['mean_rcum']:+.2f}")

    print("\n  [5-Slot 开球位明细]:")
    for s_idx, s_name in enumerate(KICKOFF_SLOT_NAMES):
        hit = res["slot_touches"][s_idx]
        f_cnt = res["slot_frames"][s_idx]
        sec = f_cnt / 120.0
        status = f"TOUCH @ {sec:4.2f}s ({f_cnt:3d}f)" if hit else f"TIMEOUT ({sec:4.2f}s / {f_cnt:3d}f)"
        print(f"    KO{s_idx} {s_name:<12} : {status}")

    dt = time.time() - t0
    print(f"\nelapsed = {dt:.2f}s")
    return res


def benchmark_all_prefixes(prefixes=None, from_tag="best"):
    if not prefixes:
        prefixes = discover_prefixes()
    env = KickoffEnv()

    rows = []
    for p in prefixes:
        latest_p, best_p = ckpt_paths(p)
        path = best_p if from_tag == "best" else latest_p
        if not os.path.exists(path):
            path = latest_p if from_tag == "best" else best_p
        if not os.path.exists(path):
            continue
        model, ckpt = load_ppo_model(path, device="cpu")
        res = evaluate_5slot_kickoff(env, model, max_decisions=MAX_DECISIONS)
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

    print("========================================================================================================================")
    print("  PPO 1v1 Fastest Kickoff 5-Slot 开球位排行榜 (60帧悬挂稳定 + 最快首触对局)")
    print("========================================================================================================================")
    rows_sorted = sorted(rows, key=lambda r: (-r["tc"], r["all_f"]))
    print(f"{'排序':<4} {'策略名称 (Prefix)':<18} {'5-Slot 编码':<24} {'触球率':<13} {'全样本均时(▼)':<16} {'命中均时':<16} {'翻滚槽数':<10} {'平均总分'}")
    print("-" * 120)
    for idx, r in enumerate(rows_sorted, 1):
        tc_str = f"{r['tc']}/{r['tot']}({r['tr']:.0%})"
        all_str = f"{r['all_sec']:.2f}s ({r['all_f']:.0f}f)"
        tch_str = f"{r['touch_sec']:.2f}s ({r['touch_f']:.0f}f)"
        fl_str = f"{r['flip']}/{r['tot']}"
        print(f"{idx:<6} {r['prefix']:<18} [{r['grid_str']}]   {tc_str:<13} {all_str:<18} {tch_str:<16} {fl_str:<11} {r['rcum']:+.2f}")
    print("========================================================================================================================")
    return rows


def verify_and_compare_csv(csv_paths):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    summaries = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"[cli_tuner] 找不到 CSV 文件: {csv_path}")
            continue
        with open(csv_path, "r") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            print(f"[cli_tuner] CSV 为空: {csv_path}")
            continue

        n_frames = len(rows)
        first = rows[0]
        last = rows[-1]
        r_cum = float(last.get("r_cum", 0.0))
        r_touch = float(last.get("r_touch", 0.0))
        final_dist = float(last.get("dist", 0.0))
        max_spd = max(float(r.get("car_speed", 0.0)) for r in rows)

        boost_frames = sum(1 for r in rows if int(float(r.get("boost", 0))) == 1)
        jump_frames  = [int(r["i"]) for r in rows if int(float(r.get("jump", 0))) == 1]
        flip_frames  = [int(r["i"]) for r in rows if int(float(r.get("is_flipping", 0))) == 1]
        ss_frames    = [int(r["i"]) for r in rows if int(float(r.get("is_supersonic", 0))) == 1]

        print("==========================================================================")
        print(f"  CSV 开球轨迹诊断: {csv_path}")
        print(f"  来源={first.get('source', '?')} | 开球位={first.get('spawn_id', '?')} | 总帧数={n_frames}f ({n_frames/120.0:.2f}s)")
        print(f"  终点 dist={final_dist:.1f} | 峰值速度={max_spd:.1f} | r_touch={r_touch:+.2f} | r_cum={r_cum:+.2f}")
        print(f"  起跳帧={jump_frames[0] if jump_frames else '无'} | 首次翻滚帧={flip_frames[0] if flip_frames else '无'} | 首次超音速帧={ss_frames[0] if ss_frames else '无'}")
        print(f"  喷气帧={boost_frames}/{n_frames} ({boost_frames/n_frames:.1%})")
        print("==========================================================================")
        summaries.append({
            "path": os.path.basename(csv_path),
            "src": first.get("source", "?"),
            "slot": first.get("spawn_id", "?"),
            "frames": n_frames,
            "sec": n_frames / 120.0,
            "max_spd": max_spd,
            "jump0": jump_frames[0] if jump_frames else -1,
            "flip0": flip_frames[0] if flip_frames else -1,
            "ss0": ss_frames[0] if ss_frames else -1,
            "r_cum": r_cum,
        })

    if len(summaries) >= 2:
        print("\n[Human vs. Model CSV 横向比对]:")
        for s in summaries:
            print(
                f"  • {s['path']:<36} src={s['src']:<6} slot={s['slot']:<14} "
                f"time={s['sec']:.2f}s ({s['frames']:3d}f) max_spd={s['max_spd']:6.1f} "
                f"jump={s['jump0']}f flip={s['flip0']}f ss={s['ss0']}f r_cum={s['r_cum']:+.2f}"
            )
    return summaries


def main():
    p = argparse.ArgumentParser(description="CLI 5-Slot Kickoff evaluator & Human vs Model CSV comparator")
    p.add_argument("--prefix", type=str, nargs="*", default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--from", dest="from_ckpt", default="best", choices=["latest", "best"])
    p.add_argument("--verify", type=str, nargs="+", default=None, help="Compare one or more exported CSV files")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.verify:
        verify_and_compare_csv(args.verify)
        return

    if args.ckpt is None and (not args.prefix or len(args.prefix) > 1):
        prefixes = args.prefix if args.prefix else discover_prefixes()
        if len(prefixes) > 1 or not args.prefix:
            benchmark_all_prefixes(
                prefixes=prefixes,
                from_tag=args.from_ckpt,
            )
            return

    ckpt_path = args.ckpt
    if ckpt_path is None:
        prefixes = discover_prefixes()
        prefix = args.prefix[0] if args.prefix else (prefixes[0] if prefixes else "kickoff_v1")
        latest_p, best_p = ckpt_paths(prefix)
        ckpt_path = best_p if args.from_ckpt == "best" else latest_p
        if not os.path.exists(ckpt_path):
            alt_p = latest_p if args.from_ckpt == "best" else best_p
            if os.path.exists(alt_p):
                ckpt_path = alt_p

    evaluate_model(ckpt_path, verbose=args.verbose)


if __name__ == "__main__":
    main()
