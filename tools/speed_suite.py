#!/usr/bin/env python3
"""tools/speed_suite.py —— 地面中心静态球大王 (Static Ball Master) 典型技能基准评测套件 (40 确定性场景)。

包含 5 组典型静态球场景 (每组 8 个场景，共 40 个场景)：
1. [G1:SpeedFlip (中远距地面静态靶极速冲刺/翻滚提速)] (8 场景)
   - 覆盖 800..4200 uu 地面静态球与低/高初始 Boost，检验 Speed-Flip 与拾取 Boost Pads 提速。
2. [G2:HalfFlip (正后方/背向静态靶 Half-Flip 与漂移掉头)] (8 场景)
   - 车头背对目标 (-Fwd 桶) + 静止/倒车初速/远离初速，检验 Half-Flip 与 Powerslide 掉头。
3. [G3:Powerslide (横向/斜向初速 + 侧向球 Powerslide 甩尾变向)] (8 场景)
   - 车头朝前但车辆横着/斜向滑行 (或向下砸地 cz<=75, vz<=0)，检验 Powerslide 与恢复抓地力。
4. [G4:UpPyramid (头顶/高空正上方 +Up 锥桶快速二段跳起飞 Fast Aerial)] (8 场景)
   - 球位于车辆 +Up 锥体 (Z=240..1750 uu)，检验地面快速起飞拦截。
5. [G5:LowAltJumpVar (离地生成 cz in [17, 75] + 跳跃/二段跳标志组合变体)] (8 场景)
   - 车辆微离地生成且速度水平或向下，覆盖 has_second_jump=True/False 与 is_jumping=True/False。
"""
import os
import sys
import math
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.env import BallReachEnv, euler_to_rot_vectors, get_adaptive_max_decisions
from core.spawner import dist_to_tier
from core.constants import TICKS_PER_DECISION, MAX_DECISIONS, NUM_CURRICULUM_STAGES
from core.policy_net import multi_discrete_to_controls
from train.checkpoint import load_ppo_model, ckpt_paths


def _make_scen(
    name, stage_idx, cx, cy, cz, yaw_d, cvx, cvy, cvz, bx, by, bz,
    boost=24.0, has_2jump=True, is_jumping=False,
):
    fwd, rgt, up = euler_to_rot_vectors(math.radians(yaw_d), 0.0, 0.0)
    dist_3d = math.hypot(bx - cx, by - cy, bz - cz)
    tier_idx = dist_to_tier(dist_3d)
    max_dec = get_adaptive_max_decisions(dist_3d, bz)
    return {
        "name": name,
        "mode": f"stage_{stage_idx + 1}",
        "stage_idx": stage_idx,
        "dist_tier": tier_idx,
        "max_decisions": max_dec,
        "is_hover_ball": (bz > 120.0),
        "car_on_ground": (cz <= 17.5),
        "is_jumping": bool(is_jumping),
        "has_second_jump": bool(has_2jump),
        "car_pos": (float(cx), float(cy), float(cz)),
        "car_vel": (float(cvx), float(cvy), float(cvz)),
        "car_ang": (0.0, 0.0, 0.0),
        "car_fwd": fwd, "car_rgt": rgt, "car_up": up,
        "car_boost": float(boost),
        "ball_pos": (float(bx), float(by), float(bz)),
        "ball_vel": (0.0, 0.0, 0.0),
        "ball_ang": (0.0, 0.0, 0.0),
    }


def build_speed_suite_scenarios():
    groups = {}

    # G1: Ground Speed-Flip & Boost Management (8 scenarios)
    g1_specs = [
        (0.0, -900.0, 90.0, 0.0, 0.0, 0.0, 28.0),
        (0.0, -1600.0, 90.0, 0.0, 0.0, 0.0, 20.0),
        (0.0, -2500.0, 90.0, 0.0, 0.0, 500.0, 32.0),
        (0.0, -3200.0, 90.0, 0.0, 0.0, 800.0, 75.0),
        (-1500.0, -1500.0, 45.0, 0.0, 1000.0, 1000.0, 18.0),
        (1200.0, -1800.0, 120.0, 600.0, -800.0, 1200.0, 12.0),
        (0.0, -2200.0, 90.0, 0.0, 0.0, 0.0, 0.0),
        (-1800.0, -2200.0, 50.0, 400.0, 1200.0, 1500.0, 0.0),
    ]
    groups["G1:SpeedFlip"] = [
        _make_scen(
            f"G1_{i+1}_D{int(math.hypot(bx-cx, by-cy))}_B{int(b)}",
            1 if math.hypot(bx - cx, by - cy) <= 1500 else 3,
            cx, cy, 17.0, yd,
            cspd * math.cos(math.radians(yd)), cspd * math.sin(math.radians(yd)), 0.0,
            bx, by, 93.15, boost=b, has_2jump=True,
        )
        for i, (cx, cy, yd, cspd, bx, by, b) in enumerate(g1_specs)
    ]

    # G2: Rear / Backwards & Half-Flip Turnaround (8 scenarios)
    g2_specs = [
        (450.0, 180.0, 0.0, 25.0, 93.15),
        (800.0, 180.0, 0.0, 30.0, 93.15),
        (1250.0, 180.0, 0.0, 65.0, 93.15),
        (1800.0, 170.0, 0.0, 20.0, 93.15),
        (950.0, 180.0, -400.0, 28.0, 93.15),
        (1400.0, 180.0, -650.0, 15.0, 93.15),
        (1200.0, 165.0, 500.0, 32.0, 93.15),
        (1100.0, 180.0, 0.0, 80.0, 420.0),
    ]
    g2 = []
    for i, (dist, rel_d, fspd, b, bz) in enumerate(g2_specs):
        ang = math.radians(90.0 + rel_d)
        bx, by = dist * math.cos(ang), dist * math.sin(ang)
        g2.append(_make_scen(
            f"G2_{i+1}_Rear{int(dist)}",
            1, 0.0, 0.0, 17.0, 90.0,
            0.0, fspd, 0.0,
            bx, by, bz, boost=b, has_2jump=True,
        ))
    groups["G2:HalfFlip"] = g2

    # G3: Powerslide & Sideways Velocity Recovery (8 scenarios: car facing +Y, sliding sideways/diagonally)
    g3_specs = [
        # (cx, cy, cz, yd, cvx, cvy, cvz, bx, by, bz, boost)
        (0.0, -500.0, 17.0, 90.0, 900.0, 0.0, 0.0, -700.0, -200.0, 93.15, 22.0),
        (0.0, -500.0, 17.0, 90.0, -1100.0, 200.0, 0.0, 800.0, 100.0, 93.15, 18.0),
        (0.0, 0.0, 17.0, 90.0, 1200.0, -400.0, 0.0, 0.0, 950.0, 93.15, 28.0),
        (0.0, 0.0, 35.0, 90.0, -950.0, 0.0, -250.0, -900.0, 600.0, 93.15, 25.0),
        (0.0, 0.0, 55.0, 0.0, 0.0, 1300.0, -400.0, 1100.0, 0.0, 93.15, 30.0),
        (-500.0, -500.0, 17.0, 45.0, 800.0, -800.0, 0.0, 400.0, 400.0, 93.15, 15.0),
        (500.0, -500.0, 45.0, 135.0, 900.0, 900.0, -300.0, -500.0, 500.0, 93.15, 60.0),
        (0.0, -800.0, 17.0, 90.0, 1400.0, 400.0, 0.0, -1100.0, 200.0, 220.0, 45.0),
    ]
    groups["G3:Powerslide"] = [
        _make_scen(
            f"G3_{i+1}_Slide",
            2, cx, cy, cz, yd, cvx, cvy, cvz, bx, by, bz, boost=b, has_2jump=True,
        )
        for i, (cx, cy, cz, yd, cvx, cvy, cvz, bx, by, bz, b) in enumerate(g3_specs)
    ]

    # G4: +Up Pyramid Aerial Takeoff from Ground (8 scenarios)
    g4_specs = [
        (0.0, -150.0, 90.0, 0.0, 0.0, 0.0, 320.0, 28.0),
        (0.0, -250.0, 90.0, 0.0, 0.0, 0.0, 520.0, 32.0),
        (0.0, -350.0, 90.0, 300.0, 0.0, 0.0, 740.0, 55.0),
        (-300.0, -300.0, 45.0, 0.0, 0.0, 0.0, 960.0, 75.0),
        (0.0, -500.0, 90.0, 400.0, 0.0, 0.0, 1180.0, 85.0),
        (400.0, -400.0, 135.0, 0.0, 0.0, 0.0, 1380.0, 95.0),
        (0.0, -650.0, 90.0, 500.0, 0.0, 0.0, 1580.0, 100.0),
        (-500.0, -500.0, 45.0, 300.0, 0.0, 0.0, 1780.0, 100.0),
    ]
    groups["G4:UpPyramid"] = [
        _make_scen(
            f"G4_{i+1}_UpZ{int(bz)}",
            1, cx, cy, 17.0, yd,
            cspd * math.cos(math.radians(yd)), cspd * math.sin(math.radians(yd)), 0.0,
            bx, by, bz, boost=b, has_2jump=True,
        )
        for i, (cx, cy, yd, cspd, bx, by, bz, b) in enumerate(g4_specs)
    ]

    # G5: Low-Altitude Spawn (cz in [25, 75], vz <= 0) + Jump/Double-Jump Flag Variations
    g5_specs = [
        # (cx, cy, cz, yd, cvx, cvy, cvz, bx, by, bz, boost, has_2j, is_jmp)
        (0.0, -600.0, 45.0, 90.0, 0.0, 600.0, -100.0, 0.0, 200.0, 260.0, 30.0, True, True),
        (0.0, -700.0, 65.0, 90.0, 0.0, 750.0, -250.0, 0.0, 300.0, 380.0, 45.0, True, False),
        (0.0, -650.0, 55.0, 90.0, 400.0, 500.0, -150.0, 0.0, 250.0, 180.0, 25.0, False, False),
        (-500.0, -500.0, 72.0, 45.0, 500.0, 500.0, -350.0, 300.0, 300.0, 93.15, 20.0, True, True),
        (500.0, -500.0, 72.0, 135.0, -500.0, 500.0, -400.0, -400.0, 400.0, 93.15, 18.0, False, False),
        (0.0, -900.0, 60.0, 90.0, -600.0, 400.0, -200.0, 500.0, 100.0, 93.15, 28.0, True, False),
        (0.0, -1100.0, 70.0, 90.0, 0.0, 900.0, -500.0, 0.0, 400.0, 520.0, 65.0, True, True),
        (0.0, -1200.0, 50.0, 90.0, 700.0, 300.0, -100.0, -600.0, 300.0, 310.0, 50.0, False, False),
    ]
    groups["G5:LowAltJump"] = [
        _make_scen(
            f"G5_{i+1}_CZ{int(cz)}_{'2J' if h2j else 'No2J'}",
            2, cx, cy, cz, yd, cvx, cvy, cvz, bx, by, bz,
            boost=b, has_2jump=h2j, is_jumping=ijmp,
        )
        for i, (cx, cy, cz, yd, cvx, cvy, cvz, bx, by, bz, b, h2j, ijmp) in enumerate(g5_specs)
    ]

    return groups


def evaluate_suite(model, env=None, max_decisions=MAX_DECISIONS):
    if env is None:
        env = BallReachEnv(mode="curriculum", stage_level=NUM_CURRICULUM_STAGES)
    groups = build_speed_suite_scenarios()
    summary = {}
    all_touches = 0
    all_total = 0
    all_frames = []
    all_touch_frames = []

    for g_name, scens in groups.items():
        g_tc = 0
        g_frames = []
        g_t_frames = []
        g_jump_frames = []
        for sd in scens:
            env.reset(spawn_state=sd)
            obs = env.last_obs
            first_jump_f = None
            ticks = 0
            ep_max_dec = sd.get("max_decisions", max_decisions)
            for _ in range(1, ep_max_dec + 1):
                act, _, _ = model.act_step_fast(obs, deterministic=True)
                ctrl = multi_discrete_to_controls(act)
                env.last_action = act
                for _ in range(TICKS_PER_DECISION):
                    if ctrl.jump and first_jump_f is None:
                        first_jump_f = ticks
                    env.step_tick(ctrl)
                    ticks += 1
                    if env._touched:
                        break
                if env._touched:
                    break
                obs, _, _ = env.get_obs()
            if env._touched:
                g_tc += 1
                all_touches += 1
                g_t_frames.append(ticks)
                all_touch_frames.append(ticks)
            else:
                ticks = ep_max_dec * TICKS_PER_DECISION
            g_frames.append(ticks)
            all_frames.append(ticks)
            all_total += 1
            if first_jump_f is not None:
                g_jump_frames.append(first_jump_f)

        summary[g_name] = {
            "touches": g_tc,
            "total": len(scens),
            "rate": g_tc / len(scens),
            "all_f": float(np.mean(g_frames)),
            "touch_f": float(np.mean(g_t_frames)) if g_t_frames else 720.0,
            "jump_f": float(np.mean(g_jump_frames)) if g_jump_frames else 720.0,
        }

    summary["TOTAL"] = {
        "touches": all_touches,
        "total": all_total,
        "rate": all_touches / max(1, all_total),
        "all_f": float(np.mean(all_frames)),
        "touch_f": float(np.mean(all_touch_frames)) if all_touch_frames else 720.0,
    }
    return summary


def print_suite_report(title, summary):
    tot = summary["TOTAL"]
    print("==========================================================================================")
    print(f"  Static Ball Master Benchmark Suite: {title}")
    print(f"  Overall: {tot['touches']}/{tot['total']} ({tot['rate']:.1%}) | All-Frames: {tot['all_f']:.1f}f ({tot['all_f']/120:.2f}s) | Touch-Frames: {tot['touch_f']:.1f}f ({tot['touch_f']/120:.2f}s)")
    print("------------------------------------------------------------------------------------------")
    for k, v in summary.items():
        if k == "TOTAL":
            continue
        print(
            f"  {k:<16} : {v['touches']:>2}/{v['total']} ({v['rate']:5.1%}) | "
            f"All={v['all_f']:5.1f}f ({v['all_f']/120:.2f}s) | "
            f"Touch={v['touch_f']:5.1f}f | FirstJump={v['jump_f']:5.1f}f"
        )
    print("==========================================================================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="*", default=["checkpoints/static_v0_best.pt"])
    args = ap.parse_args()
    env = BallReachEnv(mode="curriculum", stage_level=NUM_CURRICULUM_STAGES)
    for path in args.ckpt:
        if not os.path.exists(path):
            lp, bp = ckpt_paths(path)
            path = bp if os.path.exists(bp) else lp
        if not os.path.exists(path):
            print(f"Not found: {path}")
            continue
        model, _ = load_ppo_model(path, device="cpu")
        s = evaluate_suite(model, env)
        print_suite_report(os.path.basename(path), s)


if __name__ == "__main__":
    main()
