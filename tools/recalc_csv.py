#!/usr/bin/env python3
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.reward import RewardConfig, compute_step_reward


def recalc_file(path, cfg=None):
    if cfg is None:
        cfg = RewardConfig(max_decisions=180)
    with open(path, "r", newline="") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        return None

    old_rcum = float(rows[-1].get("r_cum", 0.0))
    init_dist = float(rows[0]["dist"])
    prev_dist = init_dist
    prev_spd = math.hypot(float(rows[0]["vx"]), float(rows[0]["vy"]), float(rows[0]["vz"]))
    prev_boost = float(rows[0].get("car_boost", 33.3))
    prev_flipping = bool(int(float(rows[0].get("is_flipping", 0))))

    totals = {
        "r_dist": 0.0,
        "r_time": 0.0,
        "r_boost": 0.0,
        "r_flip": 0.0,
        "r_touch": 0.0,
        "r_timeout": 0.0,
        "r_total": 0.0,
    }

    n_rows = len(rows)
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        dist = float(row["dist"])
        spd = math.hypot(float(row["vx"]), float(row["vy"]), float(row["vz"]))
        cur_boost = float(row.get("car_boost", prev_boost))
        boost_used = max(0.0, prev_boost - cur_boost)
        now_flipping = bool(int(float(row.get("is_flipping", 0))))
        flip_triggered = (not prev_flipping) and now_flipping
        touched = float(row.get("r_touch", 0.0)) > 0.0
        timeout = (float(row.get("r_timeout", 0.0)) < 0.0) or (idx == n_rows - 1 and not touched and n_rows >= cfg.max_decisions * 4)

        r_step, d = compute_step_reward(
            cfg,
            prev_dist=prev_dist,
            new_dist=dist,
            touched=touched,
            timeout=timeout,
            ticks_elapsed=1,
            boost_used=boost_used,
            flip_triggered=flip_triggered,
            flip_trigger_speed=prev_spd,
        )
        prev_dist = dist
        prev_spd = spd
        prev_boost = cur_boost
        prev_flipping = now_flipping

        for k in ("r_dist", "r_time", "r_boost", "r_flip", "r_touch", "r_timeout", "r_total"):
            totals[k] += d[k]

    return {
        "file": os.path.basename(path),
        "frames": n_rows,
        "old_rcum": old_rcum,
        **totals,
    }


def main():
    args = sys.argv[1:]
    if not args:
        args = sorted(glob.glob("data/*.csv"))
    files = []
    for a in args:
        expanded = sorted(glob.glob(a))
        files.extend(expanded if expanded else [a])

    cfg = RewardConfig(max_decisions=180)
    print(f"{'File':<28} | {'Frames':>6} | {'Old r_cum':>9} | {'New r_cum':>9} | {'r_dist':>7} | {'r_time':>7} | {'r_boost':>7} | {'r_flip':>8} | {'r_touch':>7} | {'r_timeout':>9}")
    print("-" * 122)
    for f in files:
        if not os.path.isfile(f):
            continue
        res = recalc_file(f, cfg)
        if not res:
            continue
        print(
            f"{res['file']:<28} | {res['frames']:>6d} | {res['old_rcum']:>+9.2f} | {res['r_total']:>+9.2f} | "
            f"{res['r_dist']:>+7.2f} | {res['r_time']:>+7.2f} | {res['r_boost']:>+7.2f} | {res['r_flip']:>+8.4f} | "
            f"{res['r_touch']:>+7.2f} | {res['r_timeout']:>+9.2f}"
        )


if __name__ == "__main__":
    main()
