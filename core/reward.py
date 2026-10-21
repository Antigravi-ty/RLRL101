"""core/reward.py —— 静态球大王：欧几里得距离势能守恒 + 零和托管跳跃翻滚链 (Zero-Sum Escrowed Jump-Flip Chain) + 氮气赤字感知奖励。

核心设计哲学:
1. 路径无关的欧几里得直线距离势能 (`r_dist`):
   - 逐步势能差分: `(prev_dist - new_dist) * distance_delta_scale`
   - 触球瞬间精准补齐碰撞残余距离 `(new_dist - touch_surface_dist)`，保证任意路径触球的距离总积分严格守恒。
2. 时间惩罚 (`r_time`) + 触球剩余时间稀疏大奖 (`r_touch`):
   - 每帧时间惩罚 `-time_penalty_per_tick`，触球瞬间额外获得 `touch_bonus + touch_time_bonus_max * rem_ratio`。
3. 零和托管跳跃-松手-翻滚/二段跳动作链 (`r_jump` — 防兔子跳刷分 + 全流程正向引导):
   - 阶段 A (起跳按住 `is_jumping=True, vz>0`): 发放微弱起跳托管正激励并记入 `_jump_escrow`；
   - 阶段 B (空中到达安全高度 `cz >= 28` 松开 Jump 键准备二段操作): 发放松手衔接托管奖励并记入 `_jump_escrow`；
   - 阶段 C1 (触发朝向目标的有效翻滚 Front-Flip / Speed-Flip / Back-Flip / Half-Flip): 全额赎回 `_jump_escrow` 并发放高额瞬时闭合冲量大奖 (`+1.15 ~ +1.65`)；
   - 阶段 C2 (在有效起跳窗口内触发针对高空球的 Double-Jump): 全额赎回 `_jump_escrow` 并发放二段跳拔高大奖；
   - 阶段 D (防刷分追缴 Clawback): 若车辆落地时未完成有效翻滚或二段跳 (单纯原地/路径上单跳刷分)，落地瞬间全额扣除 `_jump_escrow` 并追加空跳惩罚 `-empty_jump_land_penalty`，使单纯 Single-Jump 刷分严格负收益！
   - 阶段 E (空中姿态恢复势能差分 + 干净落地 / Half-Flip 180° 回正大奖): 引导 Flip-Cancel、Speed-Flip 与 Half-Flip 落地回正。
4. 氮气赤字与路线价值感知 (`r_boost`, `r_pickup`):
   - 结合 `min_boost_for_ball_z(bz)` 计算高空球氮气赤字：当 `boost < req_boost` 时，沿途拾取氮气垫提供高额赤字消除奖励；
   - 区分有效氮气加速/爬升 (正激励) 与超音速/背向无效喷气 (负惩罚)。
5. 上下文感知漂移转向 (`r_slide`):
   - 当车头未对准球 (`align < 0.72`) 且正在快速转向对准球 (`delta_align > 0`) 时给予漂移急转正奖励；直线行驶乱按漂移微弱扣分。
"""
import math
from dataclasses import dataclass
from .constants import MAX_DECISIONS, TICKS_PER_DECISION
from .curriculum import min_boost_for_ball_z


@dataclass
class RewardConfig:
    distance_delta_scale:           float = 0.010     # 10.0 pts per 1000 uu Euclidean distance closed
    touch_surface_dist:             float = 150.0     # Contact center-to-center distance for exact path conservation
    time_penalty_per_tick:          float = 0.045     # Time penalty per tick (-5.4/s) to drive maximum speed
    boost_penalty_per_unit:         float = 0.003     # Base cost for non-productive boost usage
    boost_effective_bonus_per_unit: float = 0.015     # Net positive bonus per unit of effective boost acceleration/climb
    boost_pickup_bonus:             float = 0.20      # Base bonus per boost pad picked up (scaled by boost gained & deficit)
    aerial_deficit_pickup_scale:    float = 0.040     # Extra bonus per unit of required aerial boost deficit bridged
    jump_hold_penalty_per_tick:     float = 0.0008    # Penalty for holding jump in dead/useless states
    djump_trigger_penalty:          float = 0.0       # Legacy compatibility field
    first_jump_hold_credit_tick:    float = 0.014     # Escrowed positive credit per tick during valid 1st-jump hold
    jump_release_bridge_credit:     float = 0.24      # Escrowed positive credit on releasing Jump in mid-air (cz >= 28)
    flip_impulse_bonus:             float = 1.30      # Base bonus on triggering a productive directional flip toward ball
    djump_takeoff_bonus:            float = 1.25      # Base bonus on triggering a valid double-jump for an elevated ball
    empty_jump_land_penalty:        float = 0.08      # Net penalty after clawing back escrow when landing from an empty hop
    wrong_dodge_penalty:            float = 0.30      # Penalty for flipping away from ball or double-jumping a ground ball
    air_recovery_scale:             float = 0.35      # Potential-based shaping scale for mid-air flip-cancel / upright align
    clean_land_bonus:               float = 0.28      # Bonus for landing upright with velocity aligned after a valid dodge
    halfflip_complete_bonus:        float = 0.65      # Extra bonus for completing a backward-to-forward Half-Flip turnaround
    powerslide_bonus_per_tick:      float = 0.0035    # Bonus per tick when powersliding to turn toward the ball
    touch_bonus:                    float = 70.0      # Base sparse touch score
    touch_time_bonus_max:           float = 50.0      # Early-arrival time bonus (up to +50.0)
    min_dist_ratio_bonus:           float = 20.0      # Terminal score based on (init_dist - min_dist) / init_dist
    timeout_penalty:                float = 12.0      # Terminal timeout penalty
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


def _air_recovery_potential(snap: dict, flip_start_align: float = 1.0) -> float:
    up_z = float(snap.get("up_z", 1.0))
    align = float(snap.get("align", 0.0))
    if flip_start_align < -0.25:
        # Half-Flip / Back-Flip turnaround potential:
        # Uncouples `align` from `max(0, up_z)` so Flip-Canceling (pushing W while inverted `up_z < 0`
        # to lock `align -> +1.0`) receives strong positive potential reward, followed by Air-Roll upright recovery!
        turn_progress = 0.5 * (align + 1.0)
        roll_upright = 0.5 * (up_z + 1.0) if align > 0.10 else 0.25 * max(0.0, up_z)
        return 0.65 * turn_progress + 0.35 * roll_upright
    u_pos = max(0.0, up_z)
    return 0.50 * u_pos + 0.50 * u_pos * max(0.0, align)


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

    # 1. Pure Euclidean straight-line distance potential (exact path conservation upon touch!)
    if touched:
        eff_new_dist = min(float(init_dist), cfg.touch_surface_dist) if init_dist is not None else cfg.touch_surface_dist
    else:
        eff_new_dist = float(new_dist)
    eff_prev_dist = float(prev_dist)
    r_dist = cfg.distance_delta_scale * (eff_prev_dist - eff_new_dist)

    # 2. Dense time penalty
    r_time = -cfg.time_penalty_per_tick * elapsed_ticks

    # Extract kinematic context from snapshots when available
    if prev_snap is not None and new_snap is not None:
        prev_on_ground = bool(prev_snap.get("on_ground", True))
        prev_jumping = bool(prev_snap.get("is_jumping", False))
        new_jumping = bool(new_snap.get("is_jumping", False))
        new_flipping = bool(new_snap.get("is_flipping", False))
        new_has_djump = bool(new_snap.get("has_djump", False))

        car_pos = new_snap.get("car_pos", (0.0, 0.0, 17.0))
        car_vel = new_snap.get("car_vel", (0.0, 0.0, 0.0))
        prev_vel = prev_snap.get("car_vel", (0.0, 0.0, 0.0))
        ball_pos = new_snap.get("ball_pos", (0.0, 0.0, 93.0))

        cz = float(car_pos[2])
        vz = float(car_vel[2])
        prev_vz = float(prev_vel[2])
        bz = float(ball_pos[2])
        dist_h = float(new_snap.get("dist_h", eff_new_dist))
        prev_align = float(prev_snap.get("align", 0.0))
        new_align = float(new_snap.get("align", 0.0))
        prev_closing = float(prev_snap.get("closing", 0.0))
        new_closing = float(new_snap.get("closing", 0.0))
        car_speed = float(new_snap.get("car_speed", math.hypot(car_vel[0], car_vel[1], car_vel[2])))
        prev_boost = float(prev_snap.get("boost", 0.0))
        cur_boost = float(new_snap.get("boost", prev_boost))
        up_z = float(new_snap.get("up_z", 1.0))
        is_diag_flip = bool(new_snap.get("last_flip_is_diagonal", False))
        thr_input = float(new_snap.get("last_thr_input", 1.0))

        # Carry jump-chain escrow state across snapshots
        jump_escrow = float(prev_snap.get("_jump_escrow", 0.0))
        air_valid_dodge = bool(prev_snap.get("_air_valid_dodge", False))
        air_release_awarded = bool(prev_snap.get("_air_release_awarded", False))
        flip_start_align = float(prev_snap.get("_flip_start_align", prev_align))
        prev_jump_held = bool(prev_snap.get("_jump_held", False))
    else:
        prev_on_ground = on_ground
        prev_jumping = False
        new_jumping = False
        new_flipping = flip_triggered
        new_has_djump = djump_triggered
        cz = 17.0 if on_ground else 50.0
        vz = 0.0
        prev_vz = 0.0
        bz = 93.0
        dist_h = eff_new_dist
        prev_align = 0.5
        new_align = 0.5
        prev_closing = 0.0
        new_closing = (eff_prev_dist - eff_new_dist) * (120.0 / elapsed_ticks)
        car_speed = abs(new_closing)
        prev_boost = 33.0
        cur_boost = max(0.0, 33.0 - float(boost_used))
        up_z = 1.0
        is_diag_flip = False
        thr_input = 1.0
        jump_escrow = 0.0
        air_valid_dodge = False
        air_release_awarded = False
        flip_start_align = prev_align
        prev_jump_held = False

    # Reset air state when resting on ground (not actively jumping)
    if prev_on_ground and on_ground and (not new_jumping):
        jump_escrow = 0.0
        air_valid_dodge = False
        air_release_awarded = False
        flip_start_align = new_align
    elif prev_on_ground and ((not on_ground) or (new_jumping and not prev_jumping)):
        flip_start_align = prev_align

    # 3. Context-Aware Boost Pickup & Boost Usage Reward (`r_pickup`, `r_boost`)
    req_aerial_boost = min_boost_for_ball_z(bz)
    boost_gained = max(12.0 * float(max(0, int(boost_pickups))), max(0.0, cur_boost - prev_boost))
    if boost_pickups > 0 or boost_gained > 1.0:
        # (a) Aerial boost deficit relief when ball is high and car lacks minimum climb boost
        deficit_before = max(0.0, req_aerial_boost - prev_boost)
        deficit_bridged = min(boost_gained, deficit_before)
        r_deficit = cfg.aerial_deficit_pickup_scale * deficit_bridged

        # (b) Ground transit boost value when low on boost and far from target
        dist_factor = max(0.25, min(1.50, eff_new_dist / 2000.0))
        low_boost_factor = max(0.20, min(1.30, (70.0 - prev_boost) / 55.0))
        r_transit_pad = cfg.boost_pickup_bonus * (0.60 + 0.022 * boost_gained * dist_factor * low_boost_factor)
        r_pickup = r_deficit + r_transit_pad
    else:
        r_pickup = 0.0

    used_b = max(0.0, float(boost_used))
    closing_gain = new_closing - prev_closing
    if used_b > 0.0:
        is_climbing_to_high_ball = (not on_ground) and (bz > cz + 75.0) and (vz > 40.0) and (new_closing > 120.0)
        is_accelerating_to_ball = (car_speed < 2195.0) and (new_align > 0.30) and (closing_gain > -25.0 or new_closing > 900.0)
        is_halfflip_boosting = (
            (not on_ground)
            and air_valid_dodge
            and (flip_start_align < -0.20)
            and (new_align > prev_align - 0.05 or new_align > 0.10)
        )
        is_wasting_supersonic = on_ground and (car_speed >= 2210.0) and (closing_gain <= 5.0)
        is_boosting_wrong_way = on_ground and (new_align < -0.15)
        if is_climbing_to_high_ball or is_halfflip_boosting:
            r_boost_use = 1.45 * cfg.boost_effective_bonus_per_unit * used_b
        elif is_accelerating_to_ball:
            # Extra sprint multiplier above 1050 uu/s to break Phase-3 speed-cap hesitation
            sprint_mult = 1.40 if (car_speed >= 1050.0 and new_align > 0.65 and dist_h > 420.0) else 1.0
            r_boost_use = cfg.boost_effective_bonus_per_unit * sprint_mult * used_b
        elif is_wasting_supersonic or is_boosting_wrong_way:
            r_boost_use = -2.5 * cfg.boost_penalty_per_unit * used_b
        else:
            r_boost_use = -cfg.boost_penalty_per_unit * used_b
    else:
        r_boost_use = 0.0

    # Penalize ground braking (pressing S) or unboosted coasting on open straightaways when aligned with target
    open_straightaway = (
        on_ground
        and (new_align > 0.78)
        and (dist_h > max(460.0, 1.45 * max(0.0, bz - cz)))
        and (car_speed < 2150.0)
    )
    if open_straightaway and prev_snap is not None and new_snap is not None:
        if thr_input < -0.2 and new_closing > 300.0:
            r_boost_use -= 0.012 * elapsed_ticks
        elif used_b == 0.0 and cur_boost > 15.0 and closing_gain < -8.0 and car_speed < 1950.0:
            r_boost_use -= 0.005 * elapsed_ticks

    r_boost = r_boost_use + r_pickup

    # 4. Zero-Sum Escrowed Jump -> Release -> Flip / Double-Jump Chain (`r_jump`)
    r_jump = 0.0
    j_ticks = max(0, int(jump_held_ticks))
    takeoff_window_h = max(900.0, 0.90 * max(0.0, new_closing) + 2.2 * max(0.0, bz - cz))
    wants_jump_for_height = (bz > 165.0) and (dist_h <= takeoff_window_h) and (cur_boost >= 0.45 * req_aerial_boost)
    wants_fwd_flip = (
        (new_align > 0.48)
        and (car_speed < 2150.0)
        and (dist_h > 520.0)
        and (bz <= 245.0 or dist_h > takeoff_window_h + 180.0)
    )
    wants_back_flip = (
        (new_align < -0.35)
        and (car_speed < 1900.0)
        and (dist_h > 360.0)
        and (bz <= 280.0 or dist_h > takeoff_window_h)
    )
    wants_jump_for_speed = wants_fwd_flip or wants_back_flip

    # Stage A: First-Jump Liftoff & Hold (escrowed positive credit; holding ~6-8 ticks clears nose-scrape height for flips)
    if new_jumping and (vz > 20.0) and j_ticks > 0 and (not air_valid_dodge):
        if wants_jump_for_height or wants_jump_for_speed:
            takeoff_mult = 1.35 if (wants_fwd_flip and car_speed > 450.0) else 1.0
            credit = cfg.first_jump_hold_credit_tick * float(j_ticks) * takeoff_mult
            if jump_escrow < 0.24:
                add_c = min(credit, 0.24 - jump_escrow)
                r_jump += add_c
                jump_escrow += add_c
        else:
            r_jump -= cfg.jump_hold_penalty_per_tick * float(j_ticks)
    elif j_ticks > 0 and not (flip_triggered or djump_triggered):
        r_jump -= cfg.jump_hold_penalty_per_tick * float(j_ticks)

    # Stage B: Mid-Air Jump Release Bridge (escrowed positive credit at safe dodge height cz >= 28, higher credit at cz >= 38)
    released_in_air = (
        (not on_ground)
        and (not air_valid_dodge)
        and (not air_release_awarded)
        and (not new_flipping)
        and (not new_has_djump)
        and (j_ticks == 0)
        and (prev_jumping or prev_jump_held)
        and (cz >= 28.0)
        and (vz > 35.0)
    )
    if released_in_air and (wants_jump_for_height or wants_jump_for_speed):
        height_clear_mult = 0.80 + 0.45 * min(1.0, max(0.0, (cz - 28.0) / 14.0))
        rel_credit = cfg.jump_release_bridge_credit * height_clear_mult
        r_jump += rel_credit
        jump_escrow += rel_credit
        air_release_awarded = True

    # Stage C1: Directional Flip / Speed-Flip / Back-Flip / Half-Flip Trigger
    if flip_triggered:
        raw_delta_c = 0.0
        if new_snap is not None:
            raw_delta_c = float(new_snap.get("last_dodge_delta_closing", 0.0))
        delta_closing = raw_delta_c if abs(raw_delta_c) > 1e-4 else (new_closing - prev_closing)

        # Productive flip toward ball (or maintaining high closing speed while dodging into ball)
        if delta_closing > 55.0 or (new_closing > 1150.0 and delta_closing > -25.0) or touched:
            jump_escrow = 0.0
            air_valid_dodge = True
            impulse_scale = max(0.45, min(1.35, max(delta_closing, 200.0) / 400.0))
            # Speed-flip / boosted flip synergy across medium-to-high takeoff speeds (car_speed > 550)
            if (used_b > 0.0 or cur_boost > 5.0) and car_speed > 550.0 and new_align > 0.45:
                speedflip_mult = 1.65 if is_diag_flip else 1.40
            elif is_diag_flip and new_align > 0.45:
                speedflip_mult = 1.30
            else:
                speedflip_mult = 1.0
            r_jump += cfg.flip_impulse_bonus * impulse_scale * speedflip_mult
            # Back-flip / Half-flip initiation bonus when car was facing away from ball
            if flip_start_align < -0.25 and delta_closing > 80.0:
                r_jump += 0.55
        else:
            # Counter-productive flip away from the ball: forfeit escrow + penalty
            if jump_escrow > 0.0:
                r_jump -= jump_escrow
                jump_escrow = 0.0
            r_jump -= cfg.wrong_dodge_penalty

    # Stage C2: Pure Vertical Double-Jump Trigger (for elevated balls in takeoff window)
    elif djump_triggered:
        raw_delta_vz = 0.0
        if new_snap is not None:
            raw_delta_vz = float(new_snap.get("last_dodge_delta_vz", 0.0))
        delta_vz = raw_delta_vz if abs(raw_delta_vz) > 1e-4 else (vz - prev_vz)

        valid_high_takeoff = (
            (bz > 195.0)
            and (cz < bz - 25.0)
            and (dist_h <= takeoff_window_h)
            and (new_closing > 80.0 or dist_h < 450.0)
            and (delta_vz > 100.0 or vz > 220.0)
            and (cur_boost >= 0.45 * req_aerial_boost)
        )
        if valid_high_takeoff or touched:
            jump_escrow = 0.0
            air_valid_dodge = True
            height_scale = max(0.45, min(1.30, (bz - 165.0) / 230.0))
            r_jump += cfg.djump_takeoff_bonus * height_scale
        else:
            if jump_escrow > 0.0:
                r_jump -= jump_escrow
                jump_escrow = 0.0
            r_jump -= cfg.wrong_dodge_penalty

    # Redeem escrow if single-jump directly touches an elevated ball
    if touched and (not on_ground) and (bz > 160.0):
        jump_escrow = 0.0
        air_valid_dodge = True

    # Stage D & E: Mid-Air Recovery Potential + Speed-Flip Cancel Stability + Landing Clawback + Clean-Landing & Half-Flip Bonus
    if (not prev_on_ground) and (not on_ground) and air_valid_dodge and prev_snap is not None and new_snap is not None:
        phi_prev = _air_recovery_potential(prev_snap, flip_start_align=flip_start_align)
        phi_new = _air_recovery_potential(new_snap, flip_start_align=flip_start_align)
        # During active flip torque (new_flipping=True), do not penalize mandatory pitch rotation; only reward positive cancel/turn progress!
        delta_phi = max(0.0, phi_new - phi_prev) if new_flipping else (phi_new - phi_prev)
        r_jump += cfg.air_recovery_scale * delta_phi
        # Speed-Flip Cancel stability bonus: maintaining upright nose alignment while boosting through a flip
        if new_flipping and flip_start_align >= 0.45 and up_z > 0.35 and new_align > 0.72 and used_b > 0.0:
            r_jump += 0.006 * elapsed_ticks

    just_landed = ((not prev_on_ground) and on_ground) or (prev_jumping and on_ground and (not new_jumping))
    if just_landed or (timeout and not on_ground):
        if (not air_valid_dodge) and (not touched):
            # Anti-Farming Clawback: empty single-jump hop loses all escrowed credit + hop penalty!
            r_jump -= (jump_escrow + cfg.empty_jump_land_penalty)
        elif just_landed and air_valid_dodge:
            # Clean upright landing bonus after flip / double-jump
            if up_z > 0.82 and (new_closing > 450.0 or new_align > 0.65):
                r_jump += cfg.clean_land_bonus
                # Half-Flip 180° turnaround completion bonus (started facing backward, landed facing ball)
                if flip_start_align < -0.25 and new_align > 0.50:
                    r_jump += cfg.halfflip_complete_bonus
        jump_escrow = 0.0
        air_valid_dodge = False
        air_release_awarded = False

    # Persist jump-chain state into new_snap for the next step
    if new_snap is not None:
        new_snap["_jump_escrow"] = float(jump_escrow)
        new_snap["_air_valid_dodge"] = bool(air_valid_dodge)
        new_snap["_air_release_awarded"] = bool(air_release_awarded)
        new_snap["_flip_start_align"] = float(flip_start_align)
        new_snap["_jump_held"] = bool(j_ticks > 0)

    # 5. Context-Aware Ground Powerslide Turn Reward (`r_slide`)
    hb_ticks = max(0, int(handbrake_held_ticks))
    if hb_ticks > 0 and on_ground:
        delta_align = new_align - prev_align
        if prev_align < 0.72 and delta_align > 0.002:
            r_slide = cfg.powerslide_bonus_per_tick * float(hb_ticks) + 0.22 * delta_align
        elif new_align > 0.90 and delta_align <= 0.0:
            r_slide = -0.0010 * float(hb_ticks)
        else:
            r_slide = 0.0005 * float(hb_ticks)
    else:
        r_slide = 0.0

    # 6. Terminal sparse rewards: early-touch bonus + timeout penalty + historical min_dist ratio bonus
    eff_max_dec = int(max_decisions_ep) if (max_decisions_ep is not None and max_decisions_ep > 0) else int(cfg.max_decisions)
    if touched:
        max_ticks = float(max(1, eff_max_dec * TICKS_PER_DECISION))
        rem_ratio = max(0.0, min(1.0, (max_ticks - float(ep_ticks)) / max_ticks)) if ep_ticks > 0 else 0.5
        r_touch = cfg.touch_bonus + cfg.touch_time_bonus_max * rem_ratio
    else:
        r_touch = 0.0

    r_timeout = -cfg.timeout_penalty if timeout else 0.0

    is_terminal = bool(touched or timeout)
    if is_terminal and init_dist is not None and min_dist is not None:
        r_min_dist = compute_min_dist_ratio_reward(cfg, init_dist, min_dist, touched=touched)
    elif is_terminal and touched:
        r_min_dist = float(cfg.min_dist_ratio_bonus)
    else:
        r_min_dist = 0.0

    r_total = r_dist + r_time + r_boost + r_jump + r_slide + r_touch + r_timeout + r_min_dist

    detail = {
        "r_dist":     r_dist,
        "r_slide":    r_slide,
        "r_align":    r_slide,      # Alias for plot/HUD compatibility
        "r_time":     r_time,
        "r_boost":    r_boost,
        "r_pickup":   r_pickup,
        "r_jump":     r_jump,
        "r_touch":    r_touch,
        "r_timeout":  r_timeout,
        "r_min_dist": r_min_dist,
        "r_total":    r_total,
    }
    return r_total, detail


