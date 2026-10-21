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
    SAFE_SPAWN_X, SAFE_SPAWN_Y, SAFE_SPAWN_Z_MAX,
    SAFE_BALL_Z_MIN, SAFE_GND_CAR_Z, SAFE_MAX_CAR_Z,
    is_safe_ball_pos, is_safe_car_pos,
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
        lc.jump = False
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
        rng=random,
    ):
        """Car-First Spawner:
        1. Spawn car on ground (cz=17.0) or low height (cz in [17, 75] when vel_unlocked) with random horizontal yaw (pitch=0, roll=0).
        2. Given local direction bucket `dir_idx in 0..4` (F, B, R, L, U) and angle bias `ball_deg in [5°, 45°]`,
           sample a feasible ball position `ball_pos` at distance `d in [d_lo, d_hi]`.
        3. When `vel_unlocked`, sample initial velocity in 4 horizontal directions (±vel_deg) and downward pitch [-90°, 0°],
           plus jump & double-jump flag variations.
        4. Always use normal boost pads (`pads_enabled=True`), 70% < 33 initial boost, and random initial input (`init_prev_act`).
        """
        # 1. Car height & jump/double-jump state variation
        if (not vel_unlocked) or (cz_max <= SAFE_GND_CAR_Z + 1.0):
            cz = SAFE_GND_CAR_Z
            on_ground = True
            is_jumping = False
            has_second_jump = True
            air_time = 0.0
        else:
            if rng.random() < 0.35:
                cz = SAFE_GND_CAR_Z
                on_ground = True
                is_jumping = False
                has_second_jump = True
                air_time = 0.0
            else:
                cz = rng.uniform(22.0, min(SAFE_MAX_CAR_Z, float(cz_max)))
                on_ground = False
                j_mode = rng.choice((0, 1, 2))
                if j_mode == 0:
                    is_jumping = True
                    has_second_jump = True
                    air_time = rng.uniform(0.02, 0.18)
                elif j_mode == 1:
                    is_jumping = False
                    has_second_jump = True
                    air_time = rng.uniform(0.08, 0.80)
                else:
                    is_jumping = False
                    has_second_jump = False
                    air_time = rng.uniform(0.25, 1.10)

        # 2. Sample local ball direction vector in dir_idx (F, B, R, L, U)
        d_bucket = int(dir_idx) % NUM_DIR5
        u_local = sample_local_dir5_vector(d_bucket, ball_deg, rng=rng)
        ux_l, uy_l, uz_l = u_local
        local_az = math.atan2(uy_l, ux_l)

        if d_bucket == 4:
            # +Up bucket: cap radial chord length so ball never exceeds ceiling SAFE_SPAWN_Z_MAX
            d_ceil = (SAFE_SPAWN_Z_MAX - 25.0 - cz) / max(uz_l, 0.1)
            d_hi_eff = min(float(d_hi), d_ceil)
            d_lo_eff = min(float(d_lo), max(160.0, d_hi_eff - 150.0))
        else:
            d_lo_eff = float(d_lo)
            d_hi_eff = float(d_hi)

        found = False
        cx = cy = bx = by = bz = yaw = 0.0

        # Car-First rejection loop: sample car (cx, cy) and yaw first, then place ball along feasible ray
        for _ in range(100):
            cx_c = rng.uniform(-SAFE_SPAWN_X + 80.0, SAFE_SPAWN_X - 80.0)
            cy_c = rng.uniform(-SAFE_SPAWN_Y + 80.0, SAFE_SPAWN_Y - 80.0)
            if not is_safe_car_pos(cx_c, cy_c, cz, on_ground=on_ground):
                continue

            dist_c = rng.uniform(d_lo_eff, max(d_lo_eff + 1.0, d_hi_eff))
            if d_bucket == 4:
                bz_c = float(np.clip(cz + dist_c * uz_l, max(SAFE_BALL_Z_MIN, cz + 120.0), SAFE_SPAWN_Z_MAX - 20.0))
            else:
                # Lateral bucket (F/B/R/L): clamp bz >= 93.15 (ground ball when uz_l <= 0) and keep rel_up < horizontal
                max_lat_z = min(SAFE_SPAWN_Z_MAX - 20.0, cz + 0.68 * dist_c)
                bz_c = float(np.clip(cz + dist_c * uz_l, SAFE_BALL_Z_MIN, max(SAFE_BALL_Z_MIN, max_lat_z)))

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
            # Guaranteed valid diagonal fallback inside safe arena
            dist_c = float(np.clip(0.5 * (d_lo_eff + d_hi_eff), 160.0, 6000.0))
            if d_bucket == 4:
                cx, cy = 0.0, 0.0
                bz = float(np.clip(cz + dist_c * uz_l, 220.0, SAFE_SPAWN_Z_MAX - 30.0))
                dh = math.sqrt(max(1.0, dist_c * dist_c - (bz - cz) ** 2))
                yaw = rng.uniform(-math.pi, math.pi)
                bx = cx + dh * math.cos(yaw + local_az)
                by = cy + dh * math.sin(yaw + local_az)
            else:
                bz = SAFE_BALL_Z_MIN
                dh = math.sqrt(max(1.0, dist_c * dist_c - (bz - cz) ** 2))
                cx, cy = 0.0, -min(4200.0, 0.5 * dh)
                bx, by = 0.0, cx + dh
                world_az = math.atan2(by - cy, bx - cx)
                yaw = ((world_az - local_az + math.pi) % (2.0 * math.pi)) - math.pi

        fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)

        # 3. Sample initial car velocity (horizontal 4-dir ±vel_deg, vertical [-90°, 0°])
        if vel_unlocked and v_hi > 1e-3:
            v_dir4 = self._next_vel_dir4_from_deck(rng=rng)
            cvx, cvy, cvz = sample_car_velocity_3d(
                vel_dir4_idx=v_dir4,
                vel_deg=vel_deg,
                v_lo=v_lo,
                v_hi=v_hi,
                fwd=fwd,
                rgt=rgt,
                rng=rng,
            )
        else:
            cvx = cvy = cvz = 0.0

        # Verify actual local direction bucket
        rel_x, rel_y, rel_z = bx - cx, by - cy, bz - cz
        rel_fwd = rel_x * fwd[0] + rel_y * fwd[1] + rel_z * fwd[2]
        rel_rgt = rel_x * rgt[0] + rel_y * rgt[1] + rel_z * rgt[2]
        rel_up  = rel_x * up[0]  + rel_y * up[1]  + rel_z * up[2]
        actual_dir5 = classify_local_dir5(rel_fwd, rel_rgt, rel_up)

        init_prev_act = sample_random_prev_action(r=rng)

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

    def _randomize_curriculum(self, step_idx: int = None, force_pool: int = None, rng=random):
        """Samples an episode according to the 5-Phase Curriculum and 70% Frontier / 30% Memory pool split."""
        if step_idx is None:
            step_idx = int(getattr(self.curriculum_state, "step_idx", 0))
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
