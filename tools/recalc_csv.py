#!/usr/bin/env python3
"""tools/recalc_csv.py — 零依赖 (无需 PyTorch / RocketSim) CSV 奖励离线重算与对比工具。

直接读取 CSV 中已记录的物理状态序列 (dist, closing, vx, vy, vz, car_boost, on_ground, r_touch 等)，
使用最新的 RewardConfig 重新计算每一帧的:
  - r_dist (单局历史最近距离极值 min_dist watermark)
  - r_progress (峰值接近速度一次性突破奖)
  - r_closing_bonus
  - r_time
  - r_boost_waste (按 8.264 单帧满额加速度比例计算的浪费惩罚)
  - r_boost_used (净消耗氮气量接口，默认权重 0.0)
  - r_touch / r_timeout
并输出对比表格。

用法:
  python3 tools/recalc_csv.py data/*.csv
"""
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
    prev_dist_h = float(rows[0].get("dist_h", init_dist))
    min_dist = init_dist
    peak_closing = 0.0
    closing_accum = 0.0

    prev_spd = math.hypot(float(rows[0]["vx"]), float(rows[0]["vy"]), float(rows[0]["vz"]))
    prev_boost = float(rows[0]["car_boost"])
    if prev_boost <= 1.0 and any(float(r["car_boost"]) > 1.0 for r in rows):
        pass
    elif prev_boost <= 1.0:
        prev_boost *= 100.0

    totals = {
        "r_dist": 0.0,
        "r_progress": 0.0,
        "r_closing_bonus": 0.0,
        "r_time": 0.0,
        "r_boost_waste": 0.0,
        "r_boost_used": 0.0,
        "r_touch": 0.0,
        "r_timeout": 0.0,
        "r_total": 0.0,
        "wasted_units": 0.0,
        "boost_used": 0.0,
    }

    n_rows = len(rows)
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        dist = float(row["dist"])
        dist_h = float(row.get("dist_h", dist))
        closing = float(row.get("closing", 0.0))
        air_time = float(row.get("air_time", 0.0))
        align = float(row.get("align", 0.0))
        spd = math.hypot(float(row["vx"]), float(row["vy"]), float(row["vz"]))

        cur_b = float(row["car_boost"])
        if cur_b <= 1.0 and prev_boost <= 100.0 and float(rows[0]["car_boost"]) <= 1.0:
            cur_b *= 100.0

        boost_used = max(0.0, prev_boost - cur_b)

        # 若 CSV 已包含 boost_wasted 列则直接读取，否则由速度增量与 8.264 物理常数直接纯数学推算
        if "boost_wasted" in row:
            waste_ratio = float(row["boost_wasted"])
        else:
            waste_ratio = 0.0
            if float(row.get("boost", 0)) > 0.5 and boost_used > 1e-4:
                dv = max(0.0, spd - prev_spd)
                if spd >= 2291.736 and dv < 8.2639 - 0.05:
                    waste_ratio = max(0.0, min(1.0, 1.0 - dv / 8.2639))

        touched = float(row.get("r_touch", 0.0)) > 0.0
        timeout = (float(row.get("r_timeout", 0.0)) < 0.0) or (idx == n_rows - 1 and not touched and n_rows >= cfg.max_decisions * 4)
        step_equiv = (float(row["i"]) + 1.0) / 4.0

        r_step, d = compute_step_reward(
            cfg,
            prev_dist=prev_dist,
            new_dist=dist,
            step=step_equiv,
            touched=touched,
            timeout=timeout,
            air_time=air_time,
            align=align,
            prev_dist_h=prev_dist_h,
            new_dist_h=dist_h,
            closing=closing,
            peak_closing=peak_closing,
            closing_accum=closing_accum,
            ticks_elapsed=1,
            boost_wasted_ticks=waste_ratio,
            boost_used=boost_used,
            min_dist=min_dist,
        )
        min_dist = d["min_dist"]
        peak_closing = d["peak_closing"]
        closing_accum = d["closing_accum"]
        prev_dist = dist
        prev_dist_h = dist_h
        prev_spd = spd
        prev_boost = cur_b

        for k in ("r_dist", "r_progress", "r_closing_bonus", "r_time", "r_boost_waste", "r_boost_used", "r_touch", "r_timeout", "r_total"):
            totals[k] += d[k]
        totals["wasted_units"] += waste_ratio
        totals["boost_used"] += boost_used

    return {
        "file": os.path.basename(path),
        "frames": n_rows,
        "old_rcum": old_rcum,
        "min_dist": min_dist,
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
    print(f"{'File':<28} | {'Frames':>6} | {'Old r_cum':>9} | {'New r_cum':>9} | {'r_dist':>6} | {'r_prog':>6} | {'r_time':>6} | {'r_wast':>6} | {'r_touch':>7} | {'Wasted':>6} | {'BstUsed':>7}")
    print("-" * 126)
    for f in files:
        if not os.path.isfile(f):
            continue
        res = recalc_file(f, cfg)
        if not res:
            continue
        print(
            f"{res['file']:<28} | {res['frames']:>6d} | {res['old_rcum']:>+9.2f} | {res['r_total']:>+9.2f} | "
            f"{res['r_dist']:>+6.2f} | {res['r_progress']:>+6.2f} | {res['r_time']:>+6.2f} | "
            f"{res['r_boost_waste']:>+6.2f} | {res['r_touch']:>+7.2f} | {res['wasted_units']:>5.1f}f | {res['boost_used']:>6.1f}%"
        )


if __name__ == "__main__":
    main()
