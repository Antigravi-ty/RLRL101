"""core/reward.py —— Freestyle Air-Dribble Showman 奖励系统：
中间空中盒子区 (Air-Box) 连击路程 × 触球倍数望远镜差分奖励 + 托球高度/速度匹配塑形 + 起球 (Pop/Catch) 引导 + 严禁进球重罚。

核心设计哲学:
1. 中间空中盒子区 (Central Air-Box) 连击路程 × 触球次数倍数大奖 (`r_combo` / `r_touch`):
   - 当球离开全部 6 个场地平面进入中间盒子区 (`bz >= 220`, `surface_clearance >= 180`) 后开始计分；
   - 每次空中触球 ($k = 1, 2, \\dots, K$) 记录球的三维坐标 $\\vec{P}_k$；
   - 累计空中连击折线总路程 $D_{\\text{chain}} = \\sum_{i=2}^{K} \\|\\vec{P}_i - \\vec{P}_{i-1}\\|_2$（按段平均高度加权：越高分越高）；
   - 整套动作总分为 $S_K = \\text{scale} \\times M_K \\times D_{\\text{weighted}}(K)$，其中触球倍数 $M_K \\approx K - 1$：
     * 开一脚大脚接住 ($K=2$) 仅拿 $1\\times$ 路程分；
     * 连续 5 次空中触球 ($K=5$) 则拿 $(d_{1\\to 2} + d_{2\\to 3} + d_{3\\to 4} + d_{4\\to 5}) \\times 4$ 倍路程高分！
   - 采用望远镜逐步差分 (Telescopic Step Delta) $\\Delta S_k = S_k - S_{k-1}$ 在每次空中触球瞬间即时发放，
     既保证整套连击落袋总分完全等于 $(d_{1\\to 2} + \\dots) \\times (K-1)$，又彻底消除 PPO 为提早领分而故意砸地的折现偏差。
2. 空中盒子区持续滞空与托球跟随塑形 (`r_air_box`):
   - 球在 Air-Box 内时，每 tick 按球高度 `(bz - 220) / 1824` 发放持续滞空加分；
   - 当车身处于球下方/近身跟随且速度矢量与球一致时，额外发放空中托球跟随奖励。
3. 自由起球 (Pop / Wall-to-Air / Catch) 与追球势能 (`r_dist`, `r_jump`):
   - 保留欧几里得距离势能差分 `(prev_dist - new_dist)` 引导快速追球；
   - 当球在地面/墙面附近被车顶起获得向上速度增量 ($\\Delta v_{z,\\text{ball}} > 0$) 或成功送入 Air-Box 时发放起球奖励。
4. 严禁进球重罚 (`r_goal` / `r_timeout`):
   - 一旦将球打入任意一侧球门 (`|ball_y| > 5120`)，立即触发 `-150.0` 超级大扣分并终止回合；
   - 当球靠近球门危险区 (`|by| > 4450, |bx| < 1150, bz < 750`) 且朝球门内运动时给予危险排斥惩罚，引导绕场来回盘带。
"""
import math
from dataclasses import dataclass
from .constants import (
    MAX_DECISIONS, TICKS_PER_DECISION,
    AIR_BOX_MIN_Z, CEILING_Z,
    GOAL_DANGER_Y, GOAL_DANGER_X, GOAL_DANGER_Z,
)
from .curriculum import min_boost_for_ball_z


@dataclass
class RewardConfig:
    distance_delta_scale:           float = 0.006     # Euclidean approach potential (6.0 pts per 1000 uu closed)
    touch_surface_dist:             float = 150.0     # Contact center-to-center distance
    time_penalty_per_tick:          float = 0.005     # Mild baseline tick cost (-0.6/s)
    boost_penalty_per_unit:         float = 0.0015    # Mild cost for non-productive ground boost waste
    boost_effective_bonus_per_unit: float = 0.010     # Positive bonus for aerial climb / dribble boost control
    boost_pickup_bonus:             float = 0.20      # Bonus per boost pad picked up in pads mode
    aerial_deficit_pickup_scale:    float = 0.035     # Bonus per unit of aerial boost deficit bridged
    jump_hold_penalty_per_tick:     float = 0.0005    # Mild penalty for holding jump in dead states
    djump_trigger_penalty:          float = 0.0       # Legacy compatibility field
    first_jump_hold_credit_tick:    float = 0.012     # Positive credit for valid first-jump takeoff
    jump_release_bridge_credit:     float = 0.18      # Positive credit on releasing jump in mid-air
    flip_impulse_bonus:             float = 0.55      # Bonus for productive flip / aerial dodge touch
    djump_takeoff_bonus:            float = 0.85      # Bonus for double-jump aerial launch toward high ball
    empty_jump_land_penalty:        float = 0.05      # Mild clawback on empty ground hop
    wrong_dodge_penalty:            float = 0.15      # Penalty for dodging away from ball
    air_recovery_scale:             float = 0.25      # Mid-air attitude control potential scale
    clean_land_bonus:               float = 0.15      # Bonus for clean upright landing
    halfflip_complete_bonus:        float = 0.35      # Bonus for half-flip recovery
    powerslide_bonus_per_tick:      float = 0.0030    # Bonus for powerslide turn alignment
    # --- Air-Dribble Showman Core Rewards ---
    combo_score_scale:              float = 1.00      # Scale on telescopic (Chained Distance * Touch Multiplier) delta
    air_touch_base_bonus:           float = 6.00      # Immediate bonus per valid Air-Box touch (scaled by height & chain)
    air_box_entry_bonus:            float = 4.50      # Bonus when popping/carrying the ball into the central Air-Box
    pop_upward_scale:               float = 2.20      # Bonus for imparting upward Z velocity to ground/wall ball
    air_height_tick_bonus:          float = 0.018     # Dense per-tick reward while ball stays high in Air-Box (+2.16/s at mid-high)
    air_carry_tick_bonus:           float = 0.028     # Dense per-tick reward when car shadows close under airborne ball (+3.36/s)
    goal_penalty:                   float = 150.0     # Massive penalty for scoring a goal (strictly forbidden!)
    goal_danger_tick_penalty:       float = 0.045     # Per-tick repulsion when driving ball toward either goal mouth
    # --- Legacy / Tuner Compatibility Fields ---
    touch_bonus:                    float = 15.0      # Base first-touch bonus
    touch_time_bonus_max:           float = 10.0      # Compatibility field
    min_dist_ratio_bonus:           float = 8.0       # Terminal approach ratio bonus when no air touch occurred
    timeout_penalty:                float = 0.0       # No timeout penalty in endless freestyle mode!
    max_decisions:                  int   = MAX_DECISIONS


def compute_min_dist_ratio_reward(
    cfg: RewardConfig,
    init_dist: float,
    min_dist: float,
    touched: bool = False,
) -> float:
    if touched:
        return float(cfg.min_dist_ratio_bonus)
    d0 = float(init_dist)
    if d0 <= 1e-4:
        return 0.0
    d_min = max(0.0, min(d0, float(min_dist)))
    ratio = max(0.0, min(1.0, (d0 - d_min) / d0))
    return float(cfg.min_dist_ratio_bonus) * ratio


def compute_step_reward(
    cfg: RewardConfig,
    prev_dist: float = 0.0,
    new_dist: float = 0.0,
    init_dist: float = None,
    min_dist: float = None,
    on_ground: bool = True,
    boost_used: float = 0.0,
    boost_pickups: int = 0,
    jump_held_ticks: int = 0,
    handbrake_held_ticks: int = 0,
    djump_triggered: bool = False,
    flip_triggered: bool = False,
    touched: bool = False,
    timeout: bool = False,
    ticks_elapsed: int = 1,
    prev_snap=None,
    new_snap=None,
    ep_ticks: int = 0,
    max_decisions_ep: int = None,
    **kwargs,
):
    elapsed_ticks = float(max(1, int(ticks_elapsed)))
    if new_snap is not None:
        on_ground = bool(new_snap.get("on_ground", on_ground))
        if init_dist is None:
            init_dist = new_snap.get("init_dist")
        if min_dist is None:
            min_dist = new_snap.get("min_dist")

    eff_prev_dist = float(prev_dist)
    eff_new_dist = float(new_dist)

    # Extract kinematic and Air-Box combo state from snapshots
    if prev_snap is not None and new_snap is not None:
        prev_on_ground = bool(prev_snap.get("on_ground", True))
        car_pos = new_snap.get("car_pos", (0.0, 0.0, 17.0))
        car_vel = new_snap.get("car_vel", (0.0, 0.0, 0.0))
        prev_vel = prev_snap.get("car_vel", (0.0, 0.0, 0.0))
        ball_pos = new_snap.get("ball_pos", (0.0, 0.0, 93.0))
        prev_ball_pos = prev_snap.get("ball_pos", ball_pos)
        ball_vel = new_snap.get("ball_vel", (0.0, 0.0, 0.0))
        prev_ball_vel = prev_snap.get("ball_vel", (0.0, 0.0, 0.0))

        cx, cy, cz = float(car_pos[0]), float(car_pos[1]), float(car_pos[2])
        vx, vy, vz = float(car_vel[0]), float(car_vel[1]), float(car_vel[2])
        prev_vz = float(prev_vel[2])
        bx, by, bz = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
        prev_bz = float(prev_ball_pos[2])
        bvx, bvy, bvz = float(ball_vel[0]), float(ball_vel[1]), float(ball_vel[2])
        prev_bvz = float(prev_ball_vel[2])

        dist_h = float(new_snap.get("dist_h", eff_new_dist))
        prev_align = float(prev_snap.get("align", 0.0))
        new_align = float(new_snap.get("align", 0.0))
        prev_closing = float(prev_snap.get("closing", 0.0))
        new_closing = float(new_snap.get("closing", 0.0))
        car_speed = float(new_snap.get("car_speed", math.hypot(vx, vy, vz)))
        prev_boost = float(prev_snap.get("boost", 100.0))
        cur_boost = float(new_snap.get("boost", prev_boost))
        up_z = float(new_snap.get("up_z", 1.0))

        in_air_box = bool(new_snap.get("in_air_box", False))
        prev_in_air_box = bool(prev_snap.get("in_air_box", False))
        combo_touches = int(new_snap.get("combo_touches", 0))
        combo_delta = max(
            0.0,
            float(new_snap.get("cum_combo_score", 0.0)) - float(prev_snap.get("cum_combo_score", 0.0)),
        )
        air_touches_delta = max(
            0,
            int(new_snap.get("total_air_touches", 0)) - int(prev_snap.get("total_air_touches", 0)),
        )
        touches_delta = max(
            0,
            int(new_snap.get("total_touches", 0)) - int(prev_snap.get("total_touches", 0)),
        )
        goal_scored = bool(new_snap.get("goal_scored", False))
    else:
        prev_on_ground = on_ground
        cx, cy, cz = 0.0, 0.0, (17.0 if on_ground else 250.0)
        vx, vy, vz = 0.0, 0.0, 0.0
        prev_vz = 0.0
        bx, by, bz = 0.0, 0.0, 93.0
        prev_bz = 93.0
        bvx, bvy, bvz = 0.0, 0.0, 0.0
        prev_bvz = 0.0
        dist_h = eff_new_dist
        prev_align = 0.5
        new_align = 0.5
        prev_closing = 0.0
        new_closing = (eff_prev_dist - eff_new_dist) * (120.0 / elapsed_ticks)
        car_speed = abs(new_closing)
        prev_boost = 100.0
        cur_boost = max(0.0, 100.0 - float(boost_used))
        up_z = 1.0
        in_air_box = False
        prev_in_air_box = False
        combo_touches = 1 if touched else 0
        combo_delta = 0.0
        air_touches_delta = 1 if (touched and not on_ground) else 0
        touches_delta = 1 if touched else 0
        goal_scored = False

    # 1. Approach Potential (`r_dist`) + Dense Air-Box Carry Shaping
    r_dist = cfg.distance_delta_scale * (eff_prev_dist - eff_new_dist)

    r_air_box = 0.0
    if in_air_box:
        # (a) Ball altitude reward inside the central Air-Box ("越高越好")
        height_norm = max(0.0, min(1.0, (bz - AIR_BOX_MIN_Z) / max(1.0, CEILING_Z - AIR_BOX_MIN_Z)))
        r_air_box += cfg.air_height_tick_bonus * (0.35 + 1.15 * height_norm) * elapsed_ticks

        # (b) Car shadowing / carrying underneath the airborne ball
        if (not on_ground) and eff_new_dist < 460.0 and cz <= bz + 55.0:
            prox = max(0.0, 1.0 - eff_new_dist / 460.0)
            rel_speed = math.hypot(vx - bvx, vy - bvy, vz - bvz)
            vel_match = max(0.0, 1.0 - rel_speed / 1100.0)
            # Extra credit when ball is rising or held aloft (`bvz >= -320`)
            loft_factor = 1.25 if bvz >= -180.0 else (0.85 if bvz >= -420.0 else 0.35)
            r_air_box += cfg.air_carry_tick_bonus * prox * (0.40 + 0.60 * vel_match) * loft_factor * elapsed_ticks

    r_dist += r_air_box

    # 2. Baseline time cost (`r_time`)
    r_time = -cfg.time_penalty_per_tick * elapsed_ticks

    # 3. Boost usage & pickup (`r_boost`, `r_pickup`)
    req_aerial_boost = min_boost_for_ball_z(bz)
    boost_gained = max(12.0 * float(max(0, int(boost_pickups))), max(0.0, cur_boost - prev_boost))
    if boost_pickups > 0 or boost_gained > 1.0:
        deficit_before = max(0.0, req_aerial_boost - prev_boost)
        deficit_bridged = min(boost_gained, deficit_before)
        r_pickup = cfg.aerial_deficit_pickup_scale * deficit_bridged + cfg.boost_pickup_bonus * 0.5
    else:
        r_pickup = 0.0

    used_b = max(0.0, float(boost_used))
    if used_b > 0.0:
        is_air_control = (not on_ground) and (
            (bz > cz - 40.0 and eff_new_dist < 650.0)
            or (bz > cz + 60.0 and vz > 20.0 and new_closing > 60.0)
        )
        is_ground_approach = on_ground and (car_speed < 2180.0) and (new_align > 0.35) and (new_closing > 200.0)
        if is_air_control:
            r_boost_use = 1.25 * cfg.boost_effective_bonus_per_unit * used_b
        elif is_ground_approach:
            r_boost_use = 0.75 * cfg.boost_effective_bonus_per_unit * used_b
        else:
            r_boost_use = -cfg.boost_penalty_per_unit * used_b
    else:
        r_boost_use = 0.0
    r_boost = r_boost_use + r_pickup

    # 4. Takeoff / Pop / Aerial Attitude Control (`r_jump`)
    r_jump = 0.0
    j_ticks = max(0, int(jump_held_ticks))
    if (not prev_on_ground) and (not on_ground) and eff_new_dist < 600.0:
        # Reward keeping nose pointed toward the ball while airborne
        if new_align > 0.55 and up_z > -0.20:
            r_jump += 0.004 * new_align * elapsed_ticks
    if djump_triggered and (bz > cz + 50.0) and (eff_new_dist < 900.0):
        r_jump += cfg.djump_takeoff_bonus * max(0.4, min(1.2, (bz - 160.0) / 300.0))
    elif flip_triggered and (new_closing > 100.0 or touches_delta > 0):
        r_jump += cfg.flip_impulse_bonus * (1.2 if not on_ground else 0.8)
    elif j_ticks > 0 and on_ground and vz <= 5.0:
        r_jump -= cfg.jump_hold_penalty_per_tick * float(j_ticks)

    # 5. Powerslide turn alignment (`r_slide`)
    hb_ticks = max(0, int(handbrake_held_ticks))
    if hb_ticks > 0 and on_ground:
        delta_align = new_align - prev_align
        if prev_align < 0.72 and delta_align > 0.002:
            r_slide = cfg.powerslide_bonus_per_tick * float(hb_ticks) + 0.18 * delta_align
        else:
            r_slide = -0.0005 * float(hb_ticks)
    else:
        r_slide = 0.0

    # 6. Core Showman Combo Reward:
    #    (a) Telescopic Chained-Distance * Touch-Multiplier Score Delta (`combo_delta`)
    #    (b) Per-Air-Box-Touch Immediate Height-Scaled Bonus
    #    (c) Ground/Wall Pop Upward Impulse & Air-Box Entry Bonus
    r_combo = cfg.combo_score_scale * combo_delta

    if air_touches_delta > 0:
        height_factor = 0.60 + 1.40 * max(0.0, min(1.0, (bz - AIR_BOX_MIN_Z) / max(1.0, CEILING_Z - AIR_BOX_MIN_Z)))
        chain_factor = 1.0 + 0.25 * float(min(10, max(0, combo_touches - 1)))
        r_combo += cfg.air_touch_base_bonus * height_factor * chain_factor * float(air_touches_delta)
    elif touches_delta > 0:
        # Ground/wall setup touch: reward popping the ball upward toward the Air-Box
        delta_bvz = max(0.0, bvz - prev_bvz)
        upward_pop = max(0.0, min(1.5, delta_bvz / 420.0))
        r_combo += 1.25 + cfg.pop_upward_scale * upward_pop

    # Bonus when launching the ball into the central Air-Box
    if (not prev_in_air_box) and in_air_box and (touches_delta > 0 or eff_new_dist < 420.0) and bvz > 40.0:
        r_combo += cfg.air_box_entry_bonus

    r_touch = r_combo

    # 7. Strict Anti-Goal Penalty & Goal-Mouth Repulsion (`r_goal` mapped into `r_timeout` slot for HUD/CSV)
    r_goal = 0.0
    if goal_scored:
        r_goal -= cfg.goal_penalty
    elif abs(by) > GOAL_DANGER_Y and abs(bx) < GOAL_DANGER_X and bz < GOAL_DANGER_Z:
        # Ball is inside the goal danger zone; penalize if moving toward the goal line
        toward_goal_vel = (bvy if by > 0.0 else -bvy)
        if toward_goal_vel > 60.0:
            danger_scale = min(2.5, toward_goal_vel / 500.0)
            r_goal -= cfg.goal_danger_tick_penalty * danger_scale * elapsed_ticks

    r_timeout = r_goal - (cfg.timeout_penalty if timeout else 0.0)

    # 8. Terminal minimum-distance shaping only if the episode ended without any touch
    total_ep_touches = int(new_snap.get("total_touches", 1 if touched else 0)) if new_snap is not None else (1 if touched else 0)
    if timeout and total_ep_touches == 0 and init_dist is not None and min_dist is not None:
        r_min_dist = compute_min_dist_ratio_reward(cfg, init_dist, min_dist, touched=False)
    else:
        r_min_dist = 0.0

    r_total = r_dist + r_time + r_boost + r_jump + r_slide + r_touch + r_timeout + r_min_dist

    detail = {
        "r_dist":     r_dist,
        "r_air_box":  r_air_box,
        "r_slide":    r_slide,
        "r_align":    r_slide,
        "r_time":     r_time,
        "r_boost":    r_boost,
        "r_pickup":   r_pickup,
        "r_jump":     r_jump,
        "r_touch":    r_touch,
        "r_combo":    r_combo,
        "r_goal":     r_goal,
        "r_timeout":  r_timeout,
        "r_min_dist": r_min_dist,
        "r_total":    r_total,
    }
    return r_total, detail
