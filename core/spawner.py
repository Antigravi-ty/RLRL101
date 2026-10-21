"""core/spawner.py —— Freestyle Air-Dribble 五类动态场景生成器 (Carry 空中带球 / Catch 动态接球 / Bounce 弹跳挑球 / Wall 边墙起飞 / Ground 地面自由)。"""
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
        cs.air_time = float(min(1.1, max(0.08, air_time)))
        cs.air_time_since_jump = float(min(0.95, max(0.08, air_time)))
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

    def _sample_freestyle_scenario(
        self,
        scenario_idx: int,
        spec,
        rng=random,
    ) -> dict:
        """Generates one of the 5 Air-Dribble Freestyle scenarios:
          0 (Carry) : Ball is in the central Air-Box moving upward/across; car is airborne right underneath it.
          1 (Catch) : Dynamic ball lobbed into the Air-Box; car must fly up, catch it, and air dribble.
          2 (Bounce): Bouncing ball in open field; car pops it into the Air-Box and follows.
          3 (Wall)  : Ball rolling toward/up the curved side wall; car pushes/launches it into the Air-Box.
          4 (Ground): Grounded or gently rolling ball; car autonomously sets up the pop/wall-launch.
        """
        s_bucket = int(scenario_idx) % NUM_DIR5
        bs_max = float(getattr(spec, "ball_speed_max", 1200.0))
        bz_cap = float(np.clip(spec.bz_max, 450.0, SAFE_SPAWN_Z_MAX - 50.0))

        if s_bucket == 0:
            # ---- Scenario 0: CARRY (Mid-Air Dribble Carry in Central Air-Box) ----
            bx = rng.uniform(-2100.0, 2100.0)
            by = rng.uniform(-2500.0, 2500.0)
            bz = rng.uniform(380.0, min(1150.0, bz_cap))
            # Choose flight direction roughly toward arena interior (avoiding goal mouths)
            inward_yaw = math.atan2(-by, -bx) + rng.uniform(-0.85, 0.85)
            # Bias away from direct +Y/-Y goal axis when near center X
            if abs(math.sin(inward_yaw)) > 0.88 and abs(bx) < 900.0:
                inward_yaw += math.copysign(0.55, bx if abs(bx) > 10.0 else rng.choice((-1.0, 1.0)))

            b_hspd = rng.uniform(220.0, min(720.0, bs_max * 0.65))
            bvx = b_hspd * math.cos(inward_yaw)
            bvy = b_hspd * math.sin(inward_yaw)
            bvz = rng.uniform(90.0, 380.0)

            # Place car closely underneath and slightly behind the ball
            back_dist = rng.uniform(95.0, 210.0)
            dz_under = rng.uniform(115.0, 200.0)
            cx, cy = clamp_safe_arena_xy(
                bx - back_dist * math.cos(inward_yaw) + rng.uniform(-35.0, 35.0),
                by - back_dist * math.sin(inward_yaw) + rng.uniform(-35.0, 35.0),
                margin=120.0,
            )
            cz = float(max(65.0, bz - dz_under))

            # Point car nose toward the ball (upward pitch)
            dx, dy, dz = bx - cx, by - cy, bz - cz
            dist_3d = max(1.0, math.hypot(dx, dy, dz))
            car_yaw = math.atan2(dy, dx) + rng.uniform(-0.12, 0.12)
            car_pitch = math.asin(np.clip(dz / dist_3d, 0.25, 0.92)) + rng.uniform(-0.10, 0.10)
            car_roll = rng.uniform(-0.20, 0.20)
            fwd, rgt, up = euler_to_rot_vectors(car_yaw, car_pitch, car_roll)

            # Match ball velocity plus gentle closing speed so the car immediately catches the ball
            close_extra = rng.uniform(80.0, 240.0)
            cvx = bvx + close_extra * (dx / dist_3d)
            cvy = bvy + close_extra * (dy / dist_3d)
            cvz = bvz + close_extra * (dz / dist_3d) + rng.uniform(40.0, 140.0)

            on_ground = False
            is_jumping = False
            has_second_jump = bool(rng.random() < 0.70)
            air_time = rng.uniform(0.18, 0.65)
            init_prev_act = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        elif s_bucket == 1:
            # ---- Scenario 1: CATCH (Dynamic Lobbed Ball -> Fly Up & Catch) ----
            cx = rng.uniform(-2400.0, 2400.0)
            cy = rng.uniform(-2800.0, 2800.0)
            on_ground = bool(rng.random() < 0.60)
            cz = SAFE_GND_CAR_Z if on_ground else rng.uniform(45.0, 240.0)

            car_yaw = math.atan2(-cy, -cx) + rng.uniform(-0.9, 0.9)
            dist_h = rng.uniform(max(240.0, spec.d_lo), min(1350.0, spec.d_hi))
            ball_az = car_yaw + rng.uniform(-math.radians(spec.ball_deg * 0.6), math.radians(spec.ball_deg * 0.6))
            bx, by = clamp_safe_arena_xy(
                cx + dist_h * math.cos(ball_az),
                cy + dist_h * math.sin(ball_az),
                margin=220.0,
            )
            bz = rng.uniform(260.0, min(880.0, bz_cap))

            # Ball is rising or floating across in front of the car
            lob_yaw = car_yaw + rng.uniform(-0.75, 0.75)
            lob_hspd = rng.uniform(120.0, min(680.0, bs_max * 0.6))
            bvx = lob_hspd * math.cos(lob_yaw)
            bvy = lob_hspd * math.sin(lob_yaw)
            bvz = rng.uniform(220.0, 720.0)

            car_pitch = 0.0 if on_ground else rng.uniform(0.20, 0.75)
            fwd, rgt, up = euler_to_rot_vectors(car_yaw, car_pitch, 0.0)
            c_spd = rng.uniform(350.0, min(1250.0, spec.v_hi))
            cvx = c_spd * fwd[0]
            cvy = c_spd * fwd[1]
            cvz = 0.0 if on_ground else rng.uniform(140.0, 380.0)
            is_jumping = False
            has_second_jump = True
            air_time = 0.0 if on_ground else rng.uniform(0.10, 0.40)
            init_prev_act = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        elif s_bucket == 2:
            # ---- Scenario 2: BOUNCE (Bouncing / Popping Ball in Open Field) ----
            cx = rng.uniform(-2500.0, 2500.0)
            cy = rng.uniform(-3000.0, 3000.0)
            cz = SAFE_GND_CAR_Z
            on_ground = True

            car_yaw = math.atan2(-cy, -cx) + rng.uniform(-0.8, 0.8)
            dist_h = rng.uniform(max(220.0, spec.d_lo), min(1450.0, spec.d_hi))
            ball_az = car_yaw + rng.uniform(-0.35, 0.35)
            bx, by = clamp_safe_arena_xy(
                cx + dist_h * math.cos(ball_az),
                cy + dist_h * math.sin(ball_az),
                margin=220.0,
            )
            bz = rng.uniform(125.0, 340.0)

            # Ball moving gently along or slightly toward the car's line with vertical bounce
            b_yaw = car_yaw + rng.uniform(-0.6, 0.6)
            b_hspd = rng.uniform(100.0, min(650.0, bs_max * 0.55))
            bvx = b_hspd * math.cos(b_yaw)
            bvy = b_hspd * math.sin(b_yaw)
            bvz = rng.uniform(-220.0, 480.0)

            fwd, rgt, up = euler_to_rot_vectors(car_yaw, 0.0, 0.0)
            c_spd = rng.uniform(350.0, min(1350.0, spec.v_hi))
            cvx, cvy, cvz = c_spd * fwd[0], c_spd * fwd[1], 0.0
            is_jumping = False
            has_second_jump = True
            air_time = 0.0
            init_prev_act = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        elif s_bucket == 3:
            # ---- Scenario 3: WALL (Side-Wall Roll & Launch Setup) ----
            side_sign = rng.choice((-1.0, 1.0))
            bx = side_sign * rng.uniform(2100.0, 3450.0)
            by = rng.uniform(-2600.0, 2600.0)
            bz = SAFE_BALL_Z_MIN

            # Ball rolling toward the side wall at a slight angle
            wall_yaw = (0.0 if side_sign > 0 else math.pi) + rng.uniform(-0.42, 0.42)
            b_hspd = rng.uniform(480.0, min(1050.0, bs_max * 0.8))
            bvx = b_hspd * math.cos(wall_yaw)
            bvy = b_hspd * math.sin(wall_yaw)
            bvz = 0.0

            trail_d = rng.uniform(300.0, min(950.0, spec.d_hi))
            cx, cy = clamp_safe_arena_xy(
                bx - trail_d * math.cos(wall_yaw),
                by - trail_d * math.sin(wall_yaw),
                margin=100.0,
            )
            cz = SAFE_GND_CAR_Z
            on_ground = True
            car_yaw = wall_yaw + rng.uniform(-0.15, 0.15)
            fwd, rgt, up = euler_to_rot_vectors(car_yaw, 0.0, 0.0)
            c_spd = b_hspd + rng.uniform(120.0, 450.0)
            cvx, cvy, cvz = c_spd * fwd[0], c_spd * fwd[1], 0.0
            is_jumping = False
            has_second_jump = True
            air_time = 0.0
            init_prev_act = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        else:
            # ---- Scenario 4: GROUND (Free Ground / Autonomous Setup) ----
            cx = rng.uniform(-2600.0, 2600.0)
            cy = rng.uniform(-3200.0, 3200.0)
            cz = SAFE_GND_CAR_Z
            on_ground = True
            car_yaw = math.atan2(-cy, -cx) + rng.uniform(-math.pi * 0.5, math.pi * 0.5)
            dist_h = rng.uniform(spec.d_lo, spec.d_hi)
            ball_az = car_yaw + rng.uniform(-math.radians(spec.ball_deg), math.radians(spec.ball_deg))
            bx, by = clamp_safe_arena_xy(
                cx + dist_h * math.cos(ball_az),
                cy + dist_h * math.sin(ball_az),
                margin=180.0,
            )
            bz = SAFE_BALL_Z_MIN if rng.random() < 0.65 else rng.uniform(SAFE_BALL_Z_MIN, 220.0)
            b_spd = rng.uniform(0.0, min(600.0, bs_max * 0.5))
            b_yaw = rng.uniform(-math.pi, math.pi)
            bvx = b_spd * math.cos(b_yaw)
            bvy = b_spd * math.sin(b_yaw)
            bvz = 0.0 if bz <= 96.0 else rng.uniform(-150.0, 300.0)

            fwd, rgt, up = euler_to_rot_vectors(car_yaw, 0.0, 0.0)
            c_spd = rng.uniform(0.0, min(1200.0, spec.v_hi))
            cvx, cvy, cvz = c_spd * fwd[0], c_spd * fwd[1], 0.0
            is_jumping = False
            has_second_jump = True
            air_time = 0.0
            init_prev_act = tuple(float(x) for x in sample_random_prev_action(r=rng))

        return {
            "ball_pos": (float(bx), float(by), float(bz)),
            "ball_vel": (float(bvx), float(bvy), float(bvz)),
            "ball_ang": (0.0, 0.0, 0.0),
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
                boost_headroom=spec.boost_headroom,
                boost_override=spec.boost_override,
            ),
            "on_ground": bool(on_ground),
            "is_jumping": bool(is_jumping),
            "has_second_jump": bool(has_second_jump),
            "air_time": float(air_time),
            "init_prev_act": tuple(float(x) for x in init_prev_act),
            "dir5_idx": int(s_bucket),
            "is_hover_ball": False,
        }

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
        bz_max: float = 1650.0,
        rng=random,
    ):
        """Compatibility wrapper around `_sample_freestyle_scenario`."""
        spec = get_curriculum_step_spec(getattr(self, "current_step_idx", 0))
        return self._sample_freestyle_scenario(dir_idx, spec, rng=rng)

    def _apply_custom_spawn_dist(self, default_step_idx: int, rng=random):
        """Parses `self.spawn_dist_cfg.raw` (set via Tuner [U] dialog, Shift+1..7 hotkeys, or --dist CLI)."""
        import re
        raw = getattr(getattr(self, "spawn_dist_cfg", None), "raw", "") or ""
        s = raw.strip().lower()
        if not s or s in ("auto", "off", "none", "default"):
            return None

        spec = get_curriculum_step_spec(default_step_idx)

        # Dedicated Air-Dribble scenario presets
        preset_map = {
            "carry": 0, "mode=carry": 0,
            "catch": 1, "mode=catch": 1,
            "bounce": 2, "mode=bounce": 2, "pop": 2, "mode=pop": 2,
            "wall": 3, "mode=wall": 3,
            "ground": 4, "mode=ground": 4,
            "sf": 0, "speedflip": 0, "mode=speedflip": 0, "speed_flip": 0,
            "hf": 1, "halfflip": 1, "mode=halfflip": 1, "half_flip": 1,
            "dj": 2, "djump": 2, "mode=djump": 2, "doublejump": 2,
        }
        if s in preset_map:
            b_idx = preset_map[s]
            cfg_dict = self._sample_freestyle_scenario(b_idx, spec, rng=rng)
            cfg_dict["dir5_idx"] = b_idx
            return cfg_dict, default_step_idx

        from .curriculum import PHASE_FIRST_STEP_IDX, NUM_CURRICULUM_STEPS
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

        cfg_dict = self._sample_freestyle_scenario(dir_idx, spec, rng=rng)
        cfg_dict["dir5_idx"] = int(dir_idx)
        return cfg_dict, step_idx

    def _randomize_curriculum(self, step_idx: int = None, force_pool: int = None, rng=random):
        """Samples a dynamic Air-Dribble episode according to the 4-Phase Curriculum and 5 Scenario Buckets."""
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
            use_spec = get_curriculum_step_spec(max(0, step_idx - 2)) if (pool_id == POOL_MEMORY and step_idx >= 2) else spec
            cfg_dict = self._sample_freestyle_scenario(dir_idx, use_spec, rng=rng)

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
        self.is_hover_ball = bool(cfg_dict.get("is_hover_ball", False))
        self.hover_ball_pos = cfg_dict["ball_pos"] if self.is_hover_ball else None

        bx, by, bz = cfg_dict["ball_pos"]
        bvx, bvy, bvz = cfg_dict.get("ball_vel", (0.0, 0.0, 0.0))
        bax, bay, baz = cfg_dict.get("ball_ang", (0.0, 0.0, 0.0))
        cx, cy, cz = cfg_dict["car_pos"]
        dist = math.hypot(bx - cx, by - cy, bz - cz)
        self.current_dist_tier = dist_to_tier(dist)

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(bvx, bvy, bvz)
        bs.ang_vel = rs.Vec(bax, bay, baz)
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
        """Generates deterministic evaluation batches across all 5 Freestyle scenarios (Carry, Catch, Bounce, Wall, Ground)."""
        r = rng if rng is not None else random.Random(20261021)
        step_idx = int(getattr(self.curriculum_state, "step_idx", 0))
        spec = get_curriculum_step_spec(step_idx)

        grid_batches = []
        per_dir = max(2, int(samples_per_stage))
        for d_idx in range(NUM_DIR5):
            dir_samples = []
            for _ in range(per_dir):
                cfg_dict = self._sample_freestyle_scenario(d_idx, spec, rng=r)
                sd = {
                    "mode": f"dir_{d_idx}",
                    "stage_idx": d_idx,
                    "step_idx": step_idx,
                    "pool_id": POOL_FRONTIER,
                    "dir5_idx": d_idx,
                    "dir6_idx": d_idx,
                    "bin_id": d_idx,
                    "is_hover_ball": False,
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
                    "ball_vel": cfg_dict["ball_vel"],
                    "ball_ang": cfg_dict["ball_ang"],
                }
                sd["spawn_id"] = self._compute_spawn_id(sd)
                dir_samples.append(sd)
            grid_batches.append(dir_samples)
        return grid_batches

    generate_6stage_eval_grid = generate_5dir_eval_grid
    generate_8stage_eval_grid = generate_5dir_eval_grid
    generate_16bin_eval_grid = generate_5dir_eval_grid
