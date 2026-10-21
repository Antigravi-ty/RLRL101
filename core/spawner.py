"""core/spawner.py —— 静态球大王生成器 (车优先地面生成 + 5向半方盒球方位 + 4向水平/下压初速 + 75uu高度与跳跃标志变化 + 随机初始输入)。"""
import math
import random
import numpy as np
import RocketSim as rs

from .constants import (
    EVAL_SAMPLES_PER_BIN, NUM_CURRICULUM_STAGES, EVAL_SAMPLES_PER_STAGE,
    NUM_DIR5, NUM_VEL_DIR4, POOL_FRONTIER, POOL_MEMORY, POOL_RATIOS,
)
from .curriculum import (
    SAFE_SPAWN_X, SAFE_SPAWN_Y, SAFE_SPAWN_Z_MAX, SAFE_DJUMP_BALL_Z_MAX,
    SAFE_BALL_Z_MIN, SAFE_GND_CAR_Z, SAFE_MAX_CAR_Z,
    clamp_safe_arena_xy, is_safe_ball_pos, is_safe_car_pos,
    get_curriculum_step_spec, classify_local_dir5, sample_local_dir5_vector,
    sample_car_velocity_3d, dist_to_tier,
    sample_initial_boost, sample_random_prev_action, euler_to_rot_vectors,
)


def apply_car_jump_state(
    cs,
    on_ground: bool,
    has_second_jump: bool = True,
    is_jumping: bool = False,
    air_time: float = 0.20,
    init_prev_act=None,
):
    """Configures RocketSim CarState jump / double-jump flags and last_controls cleanly."""
    cs.is_on_ground = bool(on_ground)
    cs.is_flipping = False
    cs.has_flipped = False

    lc = rs.CarControls()
    if init_prev_act is not None and len(init_prev_act) >= 6:
        lc.throttle  = float(init_prev_act[0])
        lc.pitch     = float(-init_prev_act[0])
        lc.steer     = float(init_prev_act[1])
        lc.yaw       = float(init_prev_act[1])
        lc.roll      = float(init_prev_act[2])
        lc.jump      = bool(float(init_prev_act[3]) > 0.5) if is_jumping else False
        lc.boost     = bool(float(init_prev_act[4]) > 0.5)
        lc.handbrake = bool(float(init_prev_act[5]) > 0.5) if on_ground else False
    else:
        lc.jump = bool(is_jumping)
    cs.last_controls = lc

    if on_ground:
        cs.is_jumping = False
        cs.has_jumped = False
        cs.has_double_jumped = False
        cs.air_time = 0.0
        cs.air_time_since_jump = 0.0
        cs.jump_time = 0.0
    elif is_jumping:
        cs.is_jumping = True
        cs.has_jumped = True
        cs.has_double_jumped = False
        jt = float(min(0.18, max(0.02, air_time)))
        cs.jump_time = jt
        cs.air_time = jt
        cs.air_time_since_jump = 0.0
    elif has_second_jump:
        cs.is_jumping = False
        cs.has_jumped = True
        cs.has_double_jumped = False
        cs.air_time = float(min(1.0, max(0.08, air_time)))
        cs.air_time_since_jump = float(min(0.85, max(0.08, air_time)))
        cs.jump_time = 0.20
    else:
        cs.is_jumping = False
        cs.has_jumped = True
        cs.has_double_jumped = True
        cs.air_time = float(max(0.25, air_time))
        cs.air_time_since_jump = 1.50
        cs.jump_time = 0.20


class SpawnerMixin:
    def _next_dir5_from_deck(self, rng=random) -> int:
        deck = getattr(self, "_dir5_deck", None)
        if not deck:
            deck = list(range(NUM_DIR5))
            rng.shuffle(deck)
            self._dir5_deck = deck
        return int(deck.pop())

    def _next_vel_dir4_from_deck(self, rng=random) -> int:
        deck = getattr(self, "_vel_dir4_deck", None)
        if not deck:
            deck = list(range(NUM_VEL_DIR4))
            rng.shuffle(deck)
            self._vel_dir4_deck = deck
        return int(deck.pop())

    def _sample_car_first_state(
        self,
        d_lo: float,
        d_hi: float,
        ball_deg: float,
        vel_unlocked: bool,
        vel_deg: float,
        v_lo: float,
        v_hi: float,
        cz_max: float,
        dir_idx: int,
        boost_headroom: float = 47.0,
        boost_override: bool = True,
        bz_max: float = SAFE_DJUMP_BALL_Z_MAX,
        rng=random,
    ):
        """Car-First Spawner:
        1. Spawn car on ground (cz=17.0) or in authentic jump-chain bridge / low-altitude states (cz in [22, 75]).
        2. Given local direction bucket `dir_idx in 0..4` (F, B, R, L, U) and angle bias `ball_deg in [5°, 45°]`,
           sample a corner-safe ball position `ball_pos` at distance `d in [d_lo, d_hi]` with height capped at `bz_max`
           (480 uu Double-Jump envelope in Phases 1-4; 1900 uu Full-Arena in Phase 5).
        3. Provide realistic upward `vz > 0` for mid-jump (`is_jumping=True`) and ready-to-dodge (`has_second_jump=True`)
           bridge states so the policy seamlessly chains Ground -> Single Jump -> Release -> Flip / Double Jump.
        """
        d_bucket = int(dir_idx) % NUM_DIR5
        j_mode = -1
        bridge_spawn = False

        # 1. Car height & jump/double-jump state variation (including 25% Jump-Chain & Rolling-Takeoff Bridge scaffolding)
        if (not vel_unlocked) or (cz_max <= SAFE_GND_CAR_Z + 1.0):
            if (float(d_hi) >= 420.0 or d_bucket == 4) and rng.random() < 0.25:
                bridge_spawn = True
                j_mode = rng.choice((0, 1, 3)) if d_bucket != 4 else rng.choice((0, 1))
                if j_mode == 0:
                    on_ground = False
                    cz = rng.uniform(26.0, 46.0)
                    is_jumping = True
                    has_second_jump = True
                    air_time = rng.uniform(0.04, 0.14)
                elif j_mode == 1:
                    on_ground = False
                    cz = rng.uniform(36.0, 66.0)
                    is_jumping = False
                    has_second_jump = True
                    air_time = rng.uniform(0.12, 0.34)
                else:
                    # j_mode == 3: Ground rolling takeoff bridge (car on ground with forward/backward speed ready to 1st-jump)
                    on_ground = True
                    cz = SAFE_GND_CAR_Z
                    is_jumping = False
                    has_second_jump = True
                    air_time = 0.0
            else:
                cz = SAFE_GND_CAR_Z
                on_ground = True
                is_jumping = False
                has_second_jump = True
                air_time = 0.0
        else:
            if rng.random() < 0.48:
                cz = SAFE_GND_CAR_Z
                on_ground = True
                is_jumping = False
                has_second_jump = True
                air_time = 0.0
            else:
                on_ground = False
                j_mode = rng.choice((0, 1, 1, 2))  # Emphasize ready-to-dodge (j_mode=1) & mid-jump (j_mode=0)
                if j_mode == 0:
                    cz = rng.uniform(24.0, min(50.0, float(cz_max)))
                    is_jumping = True
                    has_second_jump = True
                    air_time = rng.uniform(0.03, 0.16)
                elif j_mode == 1:
                    cz = rng.uniform(36.0, min(SAFE_MAX_CAR_Z, max(50.0, float(cz_max))))
                    is_jumping = False
                    has_second_jump = True
                    air_time = rng.uniform(0.12, 0.50)
                else:
                    cz = rng.uniform(22.0, min(SAFE_MAX_CAR_Z, float(cz_max)))
                    is_jumping = False
                    has_second_jump = False
                    air_time = rng.uniform(0.25, 1.10)

        # 2. Sample local ball direction vector in dir_idx (F, B, R, L, U)
        u_local = sample_local_dir5_vector(d_bucket, ball_deg, rng=rng)
        ux_l, uy_l, uz_l = u_local
        local_az = math.atan2(uy_l, ux_l)
        z_cap = float(np.clip(bz_max, 220.0, SAFE_SPAWN_Z_MAX - 20.0))

        if d_bucket == 4:
            # +Up bucket: cap radial chord length so ball stays inside z_cap (480 uu in P1-P4, 1880 uu in P5)
            # while strictly preserving uz_l >= max(|ux_l|, |uy_l|) so classify_local_dir5 == 4
            d_ceil = max(165.0, (z_cap - cz) / max(uz_l, 0.58))
            d_hi_eff = max(165.0, min(float(d_hi), d_ceil))
            d_lo_eff = min(float(d_lo), max(160.0, d_hi_eff - 140.0))
        else:
            d_lo_eff = float(d_lo)
            d_hi_eff = float(d_hi)

        found = False
        cx = cy = bx = by = bz = yaw = 0.0

        # Car-First rejection loop: sample car (cx, cy) inside safe octagon, then place ball along feasible ray
        for _ in range(100):
            cx_c = rng.uniform(-SAFE_SPAWN_X + 80.0, SAFE_SPAWN_X - 80.0)
            cy_c = rng.uniform(-SAFE_SPAWN_Y + 80.0, SAFE_SPAWN_Y - 80.0)
            if not is_safe_car_pos(cx_c, cy_c, cz, on_ground=on_ground):
                continue

            dist_c = rng.uniform(d_lo_eff, max(d_lo_eff + 1.0, d_hi_eff))
            if d_bucket == 4:
                bz_c = float(np.clip(cz + dist_c * uz_l, max(SAFE_BALL_Z_MIN + 50.0, cz + 105.0), z_cap))
            else:
                # Lateral bucket (F/B/R/L): keep dz <= 0.52 * dist_c so classify_local_dir5 is guaranteed to be d_bucket
                max_lat_z = min(z_cap, cz + 0.52 * dist_c)
                if uz_l <= 0.0 or max_lat_z <= SAFE_BALL_Z_MIN + 15.0:
                    bz_c = SAFE_BALL_Z_MIN
                else:
                    djump_hi = min(SAFE_DJUMP_BALL_Z_MAX, max_lat_z)
                    low_hi = min(145.0, max_lat_z)
                    r_z = rng.random()
                    if z_cap <= SAFE_DJUMP_BALL_Z_MAX + 1.0:
                        # Phases 1-4 lateral (F/B/R/L): 75% ground/low-roller (so supersonic Boost, Speed-Flip & Half-Flip
                        # hit the ball cleanly without flying underneath it), 25% elevated Double-Jump balls
                        if r_z < 0.50:
                            bz_c = SAFE_BALL_Z_MIN
                        elif r_z < 0.75:
                            bz_c = float(rng.uniform(SAFE_BALL_Z_MIN, max(SAFE_BALL_Z_MIN, low_hi)))
                        else:
                            bz_c = float(rng.uniform(min(145.0, djump_hi), max(SAFE_BALL_Z_MIN, djump_hi)))
                    else:
                        # Phase 5 full-arena lateral: 45% ground/low, 30% Double-Jump band, 25% high aerial
                        if r_z < 0.45:
                            bz_c = float(rng.uniform(SAFE_BALL_Z_MIN, max(SAFE_BALL_Z_MIN, low_hi)))
                        elif r_z < 0.75:
                            bz_c = float(rng.uniform(min(145.0, djump_hi), max(SAFE_BALL_Z_MIN, djump_hi)))
                        else:
                            bz_c = float(rng.uniform(djump_hi, max_lat_z))

            dz = bz_c - cz
            if abs(dz) >= dist_c - 20.0:
                continue
            dh = math.sqrt(max(1.0, dist_c * dist_c - dz * dz))

            # Try purely uniform yaw first, then fallback to inward-pointing feasible yaws from the same (cx_c, cy_c)
            inward_world_az = math.atan2(-cy_c, -cx_c)
            candidate_yaws = [
                rng.uniform(-math.pi, math.pi),
                rng.uniform(-math.pi, math.pi),
                inward_world_az - local_az + rng.uniform(-0.6, 0.6),
                inward_world_az - local_az + rng.uniform(-0.25, 0.25),
            ]
            for yaw_c in candidate_yaws:
                yaw_norm = ((yaw_c + math.pi) % (2.0 * math.pi)) - math.pi
                world_az = yaw_norm + local_az
                bx_c = cx_c + dh * math.cos(world_az)
                by_c = cy_c + dh * math.sin(world_az)
                if is_safe_ball_pos(bx_c, by_c, bz_c):
                    cx, cy = cx_c, cy_c
                    bx, by, bz = bx_c, by_c, bz_c
                    yaw = yaw_norm
                    found = True
                    break
            if found:
                break

        if not found:
            # Guaranteed valid diagonal fallback inside safe octagon (enforcing 45-degree corner bevels)
            dist_c = float(np.clip(0.5 * (d_lo_eff + d_hi_eff), 160.0, 6000.0))
            if d_bucket == 4:
                cx, cy = 0.0, 0.0
                bz = float(np.clip(cz + dist_c * uz_l, 180.0, z_cap))
                dh = math.sqrt(max(1.0, dist_c * dist_c - (bz - cz) ** 2))
                yaw = rng.uniform(-math.pi, math.pi)
                bx, by = clamp_safe_arena_xy(cx + dh * math.cos(yaw + local_az), cy + dh * math.sin(yaw + local_az), margin=65.0)
            else:
                bz = SAFE_BALL_Z_MIN
                dh = math.sqrt(max(1.0, dist_c * dist_c - (bz - cz) ** 2))
                cx, cy = clamp_safe_arena_xy(0.0, -min(4200.0, 0.5 * dh), margin=45.0)
                bx, by = clamp_safe_arena_xy(0.0, cy + dh, margin=65.0)
                world_az = math.atan2(by - cy, bx - cx)
                yaw = ((world_az - local_az + math.pi) % (2.0 * math.pi)) - math.pi

        fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)

        # 3. Sample initial car velocity & consistent previous action for jump-chain states
        init_prev_act = sample_random_prev_action(r=rng)
        if vel_unlocked and v_hi > 1e-3:
            if (not on_ground) and (j_mode in (0, 1)) and (d_bucket in (0, 1)) and (rng.random() < 0.65):
                # Authentic Speed-Flip (+Fwd) / Back-Flip (-Fwd) mid-air takeoff bridge
                v_dir4 = d_bucket
                takeoff_v_lo = 450.0 if d_bucket == 0 else 250.0
                takeoff_v_hi = min(float(v_hi), 1450.0 if d_bucket == 0 else 950.0)
                cvx, cvy, _ = sample_car_velocity_3d(
                    vel_dir4_idx=v_dir4,
                    vel_deg=min(18.0, float(vel_deg)),
                    v_lo=min(takeoff_v_lo, takeoff_v_hi),
                    v_hi=takeoff_v_hi,
                    fwd=fwd,
                    rgt=rgt,
                    rng=rng,
                    on_ground=True,
                )
                cvz = 0.0
                init_prev_act[0] = 1.0 if d_bucket == 0 else -1.0
                init_prev_act[4] = 1.0 if d_bucket == 0 else 0.0
            else:
                v_dir4 = self._next_vel_dir4_from_deck(rng=rng)
                cvx, cvy, cvz = sample_car_velocity_3d(
                    vel_dir4_idx=v_dir4,
                    vel_deg=vel_deg,
                    v_lo=v_lo,
                    v_hi=v_hi,
                    fwd=fwd,
                    rgt=rgt,
                    rng=rng,
                    on_ground=on_ground,
                )
            if not on_ground:
                if j_mode == 0:
                    # Mid-first-jump hold: rising upward with Jump button held
                    cvz = float(rng.uniform(210.0, 315.0))
                    init_prev_act[3] = 1.0
                elif j_mode == 1 and rng.random() < 0.85:
                    # Released first jump near apex: rising upward with Jump button released (ready to Flip/DJump on step 0!)
                    cvz = float(rng.uniform(110.0, 275.0))
                    init_prev_act[3] = 0.0
        elif bridge_spawn:
            # Phase 1-2 Jump-Chain & Rolling-Takeoff Bridge state: give natural horizontal speed along car axis
            spd_h = rng.uniform(420.0, 1180.0) if d_bucket != 4 else rng.uniform(0.0, 350.0)
            if d_bucket == 1:  # -Fwd (ball behind car -> moving backward ready for Back-Flip / Half-Flip!)
                cvx, cvy = -spd_h * fwd[0], -spd_h * fwd[1]
                init_prev_act[0] = -1.0
            else:
                cvx, cvy = spd_h * fwd[0], spd_h * fwd[1]
                init_prev_act[0] = 1.0
            if on_ground:
                cvz = 0.0
                init_prev_act[3] = 0.0
            elif j_mode == 0:
                cvz = float(rng.uniform(220.0, 315.0))
                init_prev_act[3] = 1.0
            else:
                cvz = float(rng.uniform(125.0, 265.0))
                init_prev_act[3] = 0.0
        else:
            cvx = cvy = cvz = 0.0

        # Verify actual local direction bucket
        rel_x, rel_y, rel_z = bx - cx, by - cy, bz - cz
        rel_fwd = rel_x * fwd[0] + rel_y * fwd[1] + rel_z * fwd[2]
        rel_rgt = rel_x * rgt[0] + rel_y * rgt[1] + rel_z * rgt[2]
        rel_up  = rel_x * up[0]  + rel_y * up[1]  + rel_z * up[2]
        actual_dir5 = classify_local_dir5(rel_fwd, rel_rgt, rel_up)

        return {
            "ball_pos": (float(bx), float(by), float(bz)),
            "car_pos": (float(cx), float(cy), float(cz)),
            "car_vel": (float(cvx), float(cvy), float(cvz)),
            "car_ang": (0.0, 0.0, 0.0),
            "car_fwd": fwd,
            "car_rgt": rgt,
            "car_up": up,
            "car_boost": sample_initial_boost(
                r=rng,
                bz=bz,
                cz=cz,
                on_ground=on_ground,
                has_second_jump=has_second_jump,
                boost_headroom=boost_headroom,
                boost_override=boost_override,
            ),
            "on_ground": bool(on_ground),
            "is_jumping": bool(is_jumping),
            "has_second_jump": bool(has_second_jump),
            "air_time": float(air_time),
            "init_prev_act": tuple(float(x) for x in init_prev_act),
            "dir5_idx": int(actual_dir5),
        }

    def _apply_custom_spawn_dist(self, default_step_idx: int, rng=random):
        """Parses `self.spawn_dist_cfg.raw` (set via Tuner [U] dialog, Shift+1..7 hotkeys, or --dist CLI)
        and returns a custom `cfg_dict` when active, including dedicated Speed-Flip, Half-Flip, and Double-Jump drills.
        """
        import re
        raw = getattr(getattr(self, "spawn_dist_cfg", None), "raw", "") or ""
        s = raw.strip().lower()
        if not s or s in ("auto", "off", "none", "default"):
            return None

        # 1. Dedicated Speed-Flip drill: +Fwd diagonal-front (10°..24°), mid-far (1700..3800 uu), low ball, 24..45 boost
        if s in ("sf", "speedflip", "mode=speedflip", "speed_flip"):
            cfg_dict = self._sample_car_first_state(
                d_lo=1700.0, d_hi=3800.0, ball_deg=24.0,
                vel_unlocked=True, vel_deg=12.0, v_lo=0.0, v_hi=750.0,
                cz_max=SAFE_GND_CAR_Z, dir_idx=0,
                boost_headroom=0.0, boost_override=False, bz_max=175.0, rng=rng,
            )
            cx, cy, _ = cfg_dict["car_pos"]
            bx, by, _ = cfg_dict["ball_pos"]
            fwd = cfg_dict["car_fwd"]
            spd0 = float(rng.uniform(0.0, 650.0))
            cfg_dict["car_pos"] = (float(cx), float(cy), float(SAFE_GND_CAR_Z))
            cfg_dict["ball_pos"] = (float(bx), float(by), float(rng.uniform(SAFE_BALL_Z_MIN, 155.0)))
            cfg_dict["car_vel"] = (float(spd0 * fwd[0]), float(spd0 * fwd[1]), 0.0)
            cfg_dict["on_ground"] = True
            cfg_dict["is_jumping"] = False
            cfg_dict["has_second_jump"] = True
            cfg_dict["air_time"] = 0.0
            cfg_dict["car_boost"] = float(rng.uniform(24.0, 45.0))
            cfg_dict["init_prev_act"] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return cfg_dict, 28

        # 2. Dedicated Half-Flip 180° turnaround drill: -Fwd strictly behind car (5°..16°), mid distance (1000..2600 uu)
        if s in ("hf", "halfflip", "mode=halfflip", "half_flip"):
            cfg_dict = self._sample_car_first_state(
                d_lo=1000.0, d_hi=2600.0, ball_deg=16.0,
                vel_unlocked=True, vel_deg=10.0, v_lo=0.0, v_hi=700.0,
                cz_max=SAFE_GND_CAR_Z, dir_idx=1,
                boost_headroom=0.0, boost_override=False, bz_max=165.0, rng=rng,
            )
            cx, cy, _ = cfg_dict["car_pos"]
            bx, by, _ = cfg_dict["ball_pos"]
            fwd = cfg_dict["car_fwd"]
            # 75% rolling backward (-250..-700 uu/s ready for immediate Back-Flip), 25% moving forward (+150..+500 uu/s)
            if rng.random() < 0.75:
                v_signed = -float(rng.uniform(250.0, 700.0))
                init_thr = -1.0
            else:
                v_signed = float(rng.uniform(0.0, 450.0))
                init_thr = 1.0
            cfg_dict["car_pos"] = (float(cx), float(cy), float(SAFE_GND_CAR_Z))
            cfg_dict["ball_pos"] = (float(bx), float(by), float(rng.uniform(SAFE_BALL_Z_MIN, 145.0)))
            cfg_dict["car_vel"] = (float(v_signed * fwd[0]), float(v_signed * fwd[1]), 0.0)
            cfg_dict["on_ground"] = True
            cfg_dict["is_jumping"] = False
            cfg_dict["has_second_jump"] = True
            cfg_dict["air_time"] = 0.0
            cfg_dict["car_boost"] = float(rng.uniform(25.0, 55.0))
            cfg_dict["init_prev_act"] = (init_thr, 0.0, 0.0, 0.0, 0.0, 0.0)
            return cfg_dict, 20

        # 3. Dedicated Double-Jump elevated ball drill: bz in [260, 475] uu
        if s in ("dj", "djump", "mode=djump", "doublejump"):
            d_idx = rng.choice((0, 0, 2, 3, 4))
            cfg_dict = self._sample_car_first_state(
                d_lo=480.0, d_hi=1650.0, ball_deg=35.0,
                vel_unlocked=True, vel_deg=25.0, v_lo=200.0, v_hi=1100.0,
                cz_max=SAFE_MAX_CAR_Z, dir_idx=d_idx,
                boost_headroom=20.0, boost_override=False, bz_max=SAFE_DJUMP_BALL_Z_MAX, rng=rng,
            )
            bx, by, bz = cfg_dict["ball_pos"]
            if bz < 260.0:
                cfg_dict["ball_pos"] = (float(bx), float(by), float(rng.uniform(265.0, 465.0)))
            return cfg_dict, 25

        # 4. General p[...] u[...] d[...] step=N distribution
        from .curriculum import PHASE_FIRST_STEP_IDX, NUM_CURRICULUM_STEPS
        from .constants import DIST_TIERS
        step_idx = default_step_idx
        m_step = re.search(r"step\s*=\s*(\d+)", s)
        if m_step:
            step_idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(m_step.group(1))))

        matches = dict(re.findall(r"([spud])\[([^\]]*)\]", s))
        p_str = matches.get("p") or matches.get("s")
        if p_str and p_str.isdigit() and any(int(c) > 0 for c in p_str):
            weights = [int(c) for c in p_str]
            p_choice = rng.choices(range(1, len(weights) + 1), weights=weights, k=1)[0]
            step_idx = PHASE_FIRST_STEP_IDX.get(p_choice, step_idx)

        spec = get_curriculum_step_spec(step_idx)
        u_str = matches.get("u")
        if u_str and len(u_str) == 5 and u_str.isdigit() and any(int(c) > 0 for c in u_str):
            u_weights = [int(c) for c in u_str]
            dir_idx = rng.choices(range(5), weights=u_weights, k=1)[0]
        else:
            dir_idx = self._next_dir5_from_deck(rng=rng)

        d_lo, d_hi = spec.d_lo, spec.d_hi
        d_str = matches.get("d")
        if d_str and len(d_str) == 3 and d_str.isdigit() and any(int(c) > 0 for c in d_str):
            d_weights = [int(c) for c in d_str]
            t_idx = rng.choices(range(3), weights=d_weights, k=1)[0]
            d_lo, d_hi = DIST_TIERS[t_idx]

        cfg_dict = self._sample_car_first_state(
            d_lo=d_lo, d_hi=d_hi, ball_deg=spec.ball_deg,
            vel_unlocked=spec.vel_unlocked, vel_deg=spec.vel_deg,
            v_lo=spec.v_lo, v_hi=spec.v_hi, cz_max=spec.cz_max,
            dir_idx=dir_idx, boost_headroom=spec.boost_headroom,
            boost_override=spec.boost_override, bz_max=spec.bz_max, rng=rng,
        )
        return cfg_dict, step_idx

    def _randomize_curriculum(self, step_idx: int = None, force_pool: int = None, rng=random):
        """Samples an episode according to the 5-Phase Curriculum and 70% Frontier / 30% Memory pool split."""
        if step_idx is None:
            step_idx = int(getattr(self.curriculum_state, "step_idx", 0))

        custom_res = self._apply_custom_spawn_dist(step_idx, rng=rng)
        if custom_res is not None:
            cfg_dict, custom_step_idx = custom_res
            spec = get_curriculum_step_spec(custom_step_idx)
            pool_id = POOL_FRONTIER
        else:
            spec = get_curriculum_step_spec(step_idx)
            if force_pool is not None:
                pool_id = int(force_pool)
            else:
                if (not spec.has_memory_pool) or (rng.random() < POOL_RATIOS[0]):
                    pool_id = POOL_FRONTIER
                else:
                    pool_id = POOL_MEMORY

            dir_idx = self._next_dir5_from_deck(rng=rng)

            if pool_id == POOL_MEMORY and spec.has_memory_pool:
                cfg_dict = self._sample_car_first_state(
                    d_lo=spec.mem_d_lo,
                    d_hi=spec.mem_d_hi,
                    ball_deg=spec.mem_ball_deg,
                    vel_unlocked=spec.mem_vel_unlocked,
                    vel_deg=spec.mem_vel_deg,
                    v_lo=spec.mem_v_lo,
                    v_hi=spec.mem_v_hi,
                    cz_max=spec.mem_cz_max,
                    dir_idx=dir_idx,
                    boost_headroom=spec.mem_boost_headroom,
                    boost_override=spec.mem_boost_override,
                    bz_max=spec.mem_bz_max,
                    rng=rng,
                )
            else:
                pool_id = POOL_FRONTIER
                cfg_dict = self._sample_car_first_state(
                    d_lo=spec.d_lo,
                    d_hi=spec.d_hi,
                    ball_deg=spec.ball_deg,
                    vel_unlocked=spec.vel_unlocked,
                    vel_deg=spec.vel_deg,
                    v_lo=spec.v_lo,
                    v_hi=spec.v_hi,
                    cz_max=spec.cz_max,
                    dir_idx=dir_idx,
                    boost_headroom=spec.boost_headroom,
                    boost_override=spec.boost_override,
                    bz_max=spec.bz_max,
                    rng=rng,
                )

        self.current_step_idx = spec.step_idx
        self.current_stage_idx = int(cfg_dict["dir5_idx"])
        self.current_pool_id = int(pool_id)
        self.current_dir5_idx = int(cfg_dict["dir5_idx"])
        self.current_dir6_idx = self.current_dir5_idx
        self.is_high_band_sample = (pool_id == POOL_FRONTIER)
        self.has_second_jump = bool(cfg_dict["has_second_jump"])
        self.is_jumping_init = bool(cfg_dict["is_jumping"])
        self.init_air_time = float(cfg_dict["air_time"])
        self.init_prev_act = cfg_dict["init_prev_act"]
        self.is_hover_ball = True
        self.hover_ball_pos = cfg_dict["ball_pos"]

        bx, by, bz = cfg_dict["ball_pos"]
        cx, cy, cz = cfg_dict["car_pos"]
        dist = math.hypot(bx - cx, by - cy, bz - cz)
        self.current_dist_tier = dist_to_tier(dist)

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(0.0, 0.0, 0.0)
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, cz)
        cs.vel = rs.Vec(*cfg_dict["car_vel"])
        cs.ang_vel = rs.Vec(*cfg_dict["car_ang"])
        cs.rot_mat.forward = rs.Vec(*cfg_dict["car_fwd"])
        cs.rot_mat.right   = rs.Vec(*cfg_dict["car_rgt"])
        cs.rot_mat.up      = rs.Vec(*cfg_dict["car_up"])
        cs.boost = float(cfg_dict["car_boost"])
        apply_car_jump_state(
            cs,
            on_ground=cfg_dict["on_ground"],
            has_second_jump=self.has_second_jump,
            is_jumping=self.is_jumping_init,
            air_time=self.init_air_time,
            init_prev_act=self.init_prev_act,
        )
        self.car.set_state(cs)

    def generate_5dir_eval_grid(
        self,
        samples_per_stage=EVAL_SAMPLES_PER_STAGE,
        max_stage=NUM_CURRICULUM_STAGES,
        rng=None,
    ):
        """Generates deterministic evaluation batches across the 5 local directions (F, B, R, L, U)
        for the current curriculum step.
        """
        r = rng if rng is not None else random.Random(20261019)
        step_idx = int(getattr(self.curriculum_state, "step_idx", 0))
        spec = get_curriculum_step_spec(step_idx)

        grid_batches = []
        per_dir = max(4, int(samples_per_stage) // 2)
        for d_idx in range(NUM_DIR5):
            dir_samples = []
            for _ in range(per_dir):
                cfg_dict = self._sample_car_first_state(
                    d_lo=spec.d_lo,
                    d_hi=spec.d_hi,
                    ball_deg=spec.ball_deg,
                    vel_unlocked=spec.vel_unlocked,
                    vel_deg=spec.vel_deg,
                    v_lo=spec.v_lo,
                    v_hi=spec.v_hi,
                    cz_max=spec.cz_max,
                    dir_idx=d_idx,
                    boost_headroom=spec.boost_headroom,
                    boost_override=spec.boost_override,
                    bz_max=spec.bz_max,
                    rng=r,
                )
                sd = {
                    "mode": f"dir_{d_idx}",
                    "stage_idx": d_idx,
                    "step_idx": step_idx,
                    "pool_id": POOL_FRONTIER,
                    "dir5_idx": d_idx,
                    "dir6_idx": d_idx,
                    "bin_id": d_idx,
                    "is_hover_ball": True,
                    "on_ground": cfg_dict["on_ground"],
                    "is_jumping": cfg_dict["is_jumping"],
                    "has_second_jump": cfg_dict["has_second_jump"],
                    "air_time": cfg_dict["air_time"],
                    "init_prev_act": cfg_dict["init_prev_act"],
                    "car_pos": cfg_dict["car_pos"],
                    "car_vel": cfg_dict["car_vel"],
                    "car_ang": cfg_dict["car_ang"],
                    "car_fwd": cfg_dict["car_fwd"],
                    "car_rgt": cfg_dict["car_rgt"],
                    "car_up":  cfg_dict["car_up"],
                    "car_boost": cfg_dict["car_boost"],
                    "ball_pos": cfg_dict["ball_pos"],
                    "ball_vel": (0.0, 0.0, 0.0),
                    "ball_ang": (0.0, 0.0, 0.0),
                }
                sd["spawn_id"] = self._compute_spawn_id(sd)
                dir_samples.append(sd)
            grid_batches.append(dir_samples)
        return grid_batches

    generate_6stage_eval_grid = generate_5dir_eval_grid
    generate_8stage_eval_grid = generate_5dir_eval_grid
    generate_16bin_eval_grid = generate_5dir_eval_grid
