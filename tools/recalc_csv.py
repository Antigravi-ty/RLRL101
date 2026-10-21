#!/usr/bin/env python3
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.reward import RewardConfig, compute_step_reward


def _row_to_snap(row, init_dist, min_dist):
    cx = float(row.get("px", row.get("car_x", 0.0)))
    cy = float(row.get("py", row.get("car_y", 0.0)))
    cz = float(row.get("pz", row.get("car_z", 17.0)))
    vx = float(row.get("vx", row.get("car_vx", 0.0)))
    vy = float(row.get("vy", row.get("car_vy", 0.0)))
    vz = float(row.get("vz", row.get("car_vz", 0.0)))
    bx = float(row.get("bx", row.get("ball_x", 0.0)))
    by = float(row.get("by", row.get("ball_y", 0.0)))
    bz = float(row.get("bz", row.get("ball_z", 93.0)))
    return {
        "car_pos": (cx, cy, cz),
        "car_vel": (vx, vy, vz),
        "car_speed": float(row.get("car_speed", (vx * vx + vy * vy + vz * vz) ** 0.5)),
        "ball_pos": (bx, by, bz),
        "up_z": float(row.get("up_z", 1.0)),
        "on_ground": bool(int(float(row.get("on_ground", 1)))),
        "is_flipping": bool(int(float(row.get("is_flipping", 0)))),
        "is_jumping": bool(int(float(row.get("fl_jumping", row.get("fl_is_jumping", 0))))),
        "has_djump": bool(int(float(row.get("fl_has_djump", 0)))),
        "air_time": float(row.get("air_time", 0.0)),
        "boost": float(row.get("car_boost", 33.0)),
        "dist": float(row.get("dist", init_dist)),
        "dist_h": float(row.get("dist_h", init_dist)),
        "init_dist": float(init_dist),
        "min_dist": float(min_dist),
        "align": float(row.get("align", 0.0)),
        "closing": float(row.get("closing", 0.0)),
    }


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
    min_dist = init_dist
    prev_boost = float(rows[0].get("car_boost", 100.0))
    prev_flipping = bool(int(float(rows[0].get("is_flipping", 0))))
    prev_djump = bool(int(float(rows[0].get("fl_has_djump", 0))))
    prev_snap = _row_to_snap(rows[0], init_dist, min_dist)

    totals = {
        "r_dist": 0.0,
        "r_slide": 0.0,
        "r_time": 0.0,
        "r_boost": 0.0,
        "r_jump": 0.0,
        "r_min_dist": 0.0,
        "r_touch": 0.0,
        "r_timeout": 0.0,
        "r_total": 0.0,
    }

    n_rows = len(rows)
    max_ticks = cfg.max_decisions * 4
    for idx, row in enumerate(rows):
        if idx == 0:
            continue
        dist = float(row["dist"])
        if dist < min_dist:
            min_dist = dist
        cur_boost = float(row.get("car_boost", prev_boost))
        boost_used = max(0.0, prev_boost - cur_boost)
        boost_pickups = 1 if (cur_boost - prev_boost) > 5.0 else 0
        jump_held = int(float(row.get("jump", 0)))
        on_ground = bool(int(float(row.get("on_ground", 1))))
        hb_held = int(float(row.get("handbrake", 0))) if on_ground else 0
        now_flipping = bool(int(float(row.get("is_flipping", 0))))
        now_djump = bool(int(float(row.get("fl_has_djump", 0))))
        flip_triggered = (not prev_flipping) and now_flipping
        djump_triggered = ((not prev_djump) and now_djump) and (not flip_triggered)
        touched = float(row.get("r_touch", 0.0)) > 0.0
        timeout = (float(row.get("r_timeout", 0.0)) < 0.0) or (idx == n_rows - 1 and not touched and n_rows >= max_ticks)
        new_snap = _row_to_snap(row, init_dist, min_dist)

        _, d = compute_step_reward(
            cfg,
            prev_dist=prev_dist,
            new_dist=dist,
            init_dist=init_dist,
            min_dist=min_dist,
            on_ground=on_ground,
            touched=touched,
            timeout=timeout,
            ticks_elapsed=1,
            boost_used=boost_used,
            boost_pickups=boost_pickups,
            jump_held_ticks=jump_held,
            handbrake_held_ticks=hb_held,
            djump_triggered=djump_triggered,
            flip_triggered=flip_triggered,
            prev_snap=prev_snap,
            new_snap=new_snap,
            ep_ticks=idx,
            max_decisions_ep=cfg.max_decisions,
        )
        prev_dist = dist
        prev_boost = cur_boost
        prev_flipping = now_flipping
        prev_djump = now_djump
        prev_snap = new_snap

        for k in totals:
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
    print(f"{'File':<28} | {'Frames':>6} | {'Old r_cum':>9} | {'New r_cum':>9} | {'r_dist':>7} | {'r_slide':>7} | {'r_time':>7} | {'r_boost':>7} | {'r_jump':>7} | {'r_min_d':>7} | {'r_touch':>7}")
    print("-" * 134)
    for f in files:
        if not os.path.isfile(f):
            continue
        res = recalc_file(f, cfg)
        if not res:
            continue
        print(
            f"{res['file']:<28} | {res['frames']:>6d} | {res['old_rcum']:>+9.2f} | {res['r_total']:>+9.2f} | "
            f"{res['r_dist']:>+7.2f} | {res['r_slide']:>+7.3f} | {res['r_time']:>+7.2f} | {res['r_boost']:>+7.2f} | "
            f"{res['r_jump']:>+7.3f} | {res['r_min_dist']:>+7.2f} | {res['r_touch']:>+7.2f}"
        )


if __name__ == "__main__":
    main()
