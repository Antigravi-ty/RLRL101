import os
import json
import math
import random
import hashlib
from dataclasses import dataclass, asdict
import numpy as np
import RocketSim as rs
from .constants import (
    LOW_BOOST_PROB, LOW_BOOST_MAX, EVAL_SAMPLES_PER_BIN,
    NUM_CURRICULUM_STAGES, EVAL_SAMPLES_PER_STAGE, AERIAL_HEIGHT_BOOST_REF,
)
from .obs_v3 import build_obs_v3, OPP_GOAL_POS


FOCUS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "focus_region.json",
)


@dataclass
class FocusRegionConfig:
    enabled: bool = False
    prob: float = 0.50
    dist_min: float = 150.0
    dist_max: float = 900.0
    angle_deg_min: float = 135.0
    angle_deg_max: float = 180.0
    both_sides: bool = True
    ball_z_min: float = 93.15
    ball_z_max: float = 93.15
    car_z_min: float = 17.0
    car_z_max: float = 17.0
    car_speed_min: float = 0.0
    car_speed_max: float = 0.0
    ball_speed_min: float = 0.0
    ball_speed_max: float = 0.0


@dataclass
class AerialCurriculumState:
    max_z: float = 400.0
    phase: int = 0
    boost_margin: int = 30
    high_band_touches: int = 0
    high_band_total: int = 0

    def code_str(self) -> str:
        hb_rate = (self.high_band_touches / self.high_band_total) if self.high_band_total > 0 else 0.0
        if self.phase == 0:
            return f"Z0400_DJ({self.high_band_touches}/{self.high_band_total}:{hb_rate:.0%})"
        elif self.phase == 1:
            z_lo = int(round(self.max_z - 100.0))
            z_hi = int(round(self.max_z))
            return f"Z{z_hi:04d}[{z_lo}-{z_hi}:{self.high_band_touches}/{self.high_band_total}={hb_rate:.0%}]_B100"
        elif self.phase == 2:
            return f"Z2000_CSV+{self.boost_margin:02d}({self.high_band_touches}/{self.high_band_total}={hb_rate:.0%})"
        else:
            return "Z2000_DONE"


STAGE_NAMES = (
    "S1:Gnd-BallStat-CarStat",
    "S2:Gnd-BallDyn-CarStat",
    "S3:Gnd-BallStat-CarMove",
    "S4:Gnd-BallDyn-CarMove",
    "S5:AirHover-CarStat",
    "S6:AirHover-CarMove",
    "S7:AirDyn-CarStat",
    "S8:AirDyn-CarMove",
)

STAGE_SHORT_LABELS = (
    "1:GSS", "2:GDS", "3:GSM", "4:GDM",
    "5:ASS", "6:ASM", "7:ADS", "8:ADM",
)


def interpolate_csv_boost_for_height(z: float) -> float:
    pts = AERIAL_HEIGHT_BOOST_REF
    if z <= pts[0][0]:
        return float(pts[0][1]) * max(0.0, (z - 93.15) / max(1.0, pts[0][0] - 93.15))
    for i in range(len(pts) - 1):
        z0, b0 = pts[i]
        z1, b1 = pts[i + 1]
        if z0 <= z <= z1:
            t = (z - z0) / max(1e-6, z1 - z0)
            return float(b0 + t * (b1 - b0))
    return float(pts[-1][1])


def compute_aerial_spawn_boost(bz: float, aerial_state: AerialCurriculumState = None, r=random) -> float:
    if bz <= 400.0:
        return sample_initial_boost(r=r)
    if aerial_state is None or aerial_state.phase <= 1:
        return 100.0
    csv_needed = interpolate_csv_boost_for_height(bz)
    return float(np.clip(csv_needed + float(aerial_state.boost_margin), 0.0, 100.0))


def load_focus_config(path=FOCUS_CONFIG_PATH) -> FocusRegionConfig:
    if not os.path.isfile(path):
        return FocusRegionConfig()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in FocusRegionConfig.__dataclass_fields__}
        return FocusRegionConfig(**valid)
    except Exception:
        return FocusRegionConfig()


def save_focus_config(cfg: FocusRegionConfig, path=FOCUS_CONFIG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def sample_initial_boost(r=random, low_prob=LOW_BOOST_PROB, low_max=LOW_BOOST_MAX) -> float:
    if r.random() < low_prob:
        return float(r.uniform(0.0, low_max))
    return float(r.uniform(low_max, 100.0))


def euler_to_rot_vectors(yaw: float, pitch: float = 0.0, roll: float = 0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    fwd = (cp * cy, cp * sy, sp)
    rgt = (sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp)
    up  = (-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp)
    return fwd, rgt, up


class BallReachEnv:
    def __init__(self, mode="curriculum", focus_cfg: FocusRegionConfig = None, stage_level: int = 1):
        self.mode = "ground" if mode == "stage0" else ("target" if mode == "stage1" else (mode or "curriculum"))
        self.stage_level = max(1, min(NUM_CURRICULUM_STAGES, int(stage_level)))
        self.current_stage_idx = 0
        self.aerial_state = AerialCurriculumState()
        self.is_hover_ball = False
        self.hover_ball_pos = None
        self.is_high_band_sample = False

        self.focus_cfg = focus_cfg if focus_cfg is not None else load_focus_config()
        self._focus_mtime = os.path.getmtime(FOCUS_CONFIG_PATH) if os.path.isfile(FOCUS_CONFIG_PATH) else None

        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        self.pads = self.arena.get_boost_pads()
        self.num_pads = len(self.pads)

        self._touched = False
        self._boost_pickups_tick = 0

        def _on_touch(arena, car, data):
            self._touched = True

        def _on_boost_pickup(arena, car, boost_pad, data):
            self._boost_pickups_tick += 1

        self.arena.set_ball_touch_callback(_on_touch)
        self.arena.set_boost_pickup_callback(_on_boost_pickup)

        self.last_spawn_state = None
        self.spawn_id = "0000"
        self.target_pos = None
        self.target_time = None
        self.last_boost_used = 0.0
        self.last_boost_pickups = 0
        self.last_flip_triggered = False
        self.last_flip_trigger_speed = 0.0

    def set_curriculum_state(self, stage_level: int, max_z: float = 400.0, phase: int = 0, boost_margin: int = 30):
        self.stage_level = max(1, min(NUM_CURRICULUM_STAGES, int(stage_level)))
        self.aerial_state.max_z = float(max_z)
        self.aerial_state.phase = int(phase)
        self.aerial_state.boost_margin = int(boost_margin)

    def maybe_reload_focus_config(self):
        if not os.path.isfile(FOCUS_CONFIG_PATH):
            return False
        mt = os.path.getmtime(FOCUS_CONFIG_PATH)
        if self._focus_mtime is None or mt > self._focus_mtime:
            self._focus_mtime = mt
            self.focus_cfg = load_focus_config()
            return True
        return False

    def _compute_spawn_id(self, sd):
        key_str = (
            f"{sd.get('mode','a')}_{sd.get('stage_idx',0)}_"
            f"{sd['car_pos'][0]:.1f}_{sd['car_pos'][1]:.1f}_{sd['car_pos'][2]:.1f}_"
            f"{sd['car_fwd'][0]:.3f}_{sd['car_fwd'][1]:.3f}_"
            f"{sd['ball_pos'][0]:.1f}_{sd['ball_pos'][1]:.1f}_{sd['ball_pos'][2]:.1f}_"
            f"{sd['ball_vel'][0]:.1f}_{sd['ball_vel'][1]:.1f}_{sd['ball_vel'][2]:.1f}"
        )
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()[:4]

    def _capture_spawn_state(self, use_mode):
        cs = self.car.get_state()
        bs = self.arena.ball.get_state()
        sd = {
            "mode": str(use_mode),
            "stage_idx": int(self.current_stage_idx),
            "is_hover_ball": bool(self.is_hover_ball),
            "is_high_band_sample": bool(self.is_high_band_sample),
            "car_pos": (float(cs.pos.x), float(cs.pos.y), float(cs.pos.z)),
            "car_vel": (float(cs.vel.x), float(cs.vel.y), float(cs.vel.z)),
            "car_ang": (float(cs.ang_vel.x), float(cs.ang_vel.y), float(cs.ang_vel.z)),
            "car_fwd": (float(cs.rot_mat.forward.x), float(cs.rot_mat.forward.y), float(cs.rot_mat.forward.z)),
            "car_rgt": (float(cs.rot_mat.right.x),   float(cs.rot_mat.right.y),   float(cs.rot_mat.right.z)),
            "car_up":  (float(cs.rot_mat.up.x),      float(cs.rot_mat.up.y),      float(cs.rot_mat.up.z)),
            "car_boost": float(cs.boost),
            "ball_pos": (float(bs.pos.x), float(bs.pos.y), float(bs.pos.z)),
            "ball_vel": (float(bs.vel.x), float(bs.vel.y), float(bs.vel.z)),
            "ball_ang": (float(bs.ang_vel.x), float(bs.ang_vel.y), float(bs.ang_vel.z)),
            "target_pos": self.target_pos,
            "target_time": self.target_time,
        }
        sd["spawn_id"] = self._compute_spawn_id(sd)
        return sd

    def _reset_boost_pads(self):
        for p in self.pads:
            ps = p.get_state()
            ps.is_active = True
            ps.cooldown = 0.0
            p.set_state(ps)

    def _apply_spawn_state(self, sd):
        self.arena.remove_car(self.car)
        self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        self._touched = False
        self._boost_pickups_tick = 0
        self._reset_boost_pads()

        self.current_stage_idx = int(sd.get("stage_idx", 0))
        self.is_hover_ball = bool(sd.get("is_hover_ball", sd.get("mode") == "static_aerial" or self.current_stage_idx in (4, 5)))
        self.is_high_band_sample = bool(sd.get("is_high_band_sample", False))
        self.hover_ball_pos = tuple(sd["ball_pos"]) if self.is_hover_ball else None

        cs = self.car.get_state()
        cs.pos = rs.Vec(*sd["car_pos"])
        cs.vel = rs.Vec(*sd["car_vel"])
        cs.ang_vel = rs.Vec(*sd["car_ang"])
        cs.rot_mat.forward = rs.Vec(*sd["car_fwd"])
        cs.rot_mat.right   = rs.Vec(*sd["car_rgt"])
        cs.rot_mat.up      = rs.Vec(*sd["car_up"])
        cs.boost = float(sd["car_boost"])
        cs.is_on_ground = (float(sd["car_pos"][2]) <= 20.0)
        self.car.set_state(cs)
        self.car.set_controls(rs.CarControls())

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(*sd["ball_pos"])
        bs.vel = rs.Vec(*sd["ball_vel"])
        bs.ang_vel = rs.Vec(*sd["ball_ang"])
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        self.target_pos = sd.get("target_pos")
        self.target_time = sd.get("target_time")
        self.spawn_id = sd.get("spawn_id") or self._compute_spawn_id(sd)

    def _lock_hover_ball_if_needed(self):
        if self.is_hover_ball and (not self._touched) and self.hover_ball_pos is not None:
            bs = self.arena.ball.get_state()
            bs.pos = rs.Vec(*self.hover_ball_pos)
            bs.vel = rs.Vec(0.0, 0.0, 0.0)
            bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            self.arena.ball.set_state(bs)

    def step_tick(self, ctrl):
        cs_before = self.car.get_state()
        prev_spd = float(math.sqrt(cs_before.vel.x**2 + cs_before.vel.y**2 + cs_before.vel.z**2))
        prev_boost = float(cs_before.boost)
        was_flipping = bool(cs_before.is_flipping)

        self._boost_pickups_tick = 0
        self._lock_hover_ball_if_needed()

        self.car.set_controls(ctrl)
        self.arena.step(1)

        self._lock_hover_ball_if_needed()

        cs_after = self.car.get_state()
        cur_boost = float(cs_after.boost)
        now_flipping = bool(cs_after.is_flipping)

        # Boost consumption only when car actually uses boost (ignore pad pickup increases)
        boost_used = max(0.0, prev_boost - cur_boost) if ctrl.boost else 0.0
        flip_triggered = (not was_flipping) and now_flipping

        self.last_boost_used = boost_used
        self.last_boost_pickups = self._boost_pickups_tick
        self.last_flip_triggered = flip_triggered
        self.last_flip_trigger_speed = prev_spd if flip_triggered else 0.0
        return boost_used, flip_triggered, self.last_flip_trigger_speed

    def apply_initial_backward_prefix(self, duration_ticks=30, jump_last_prob=0.5, rng=random):
        steer_choice = rng.choice((0, 1, 2))
        do_jump_last = (rng.random() < jump_last_prob)
        for t in range(duration_ticks):
            c = rs.CarControls()
            c.throttle = -1.0
            c.pitch = 1.0
            c.steer = float(steer_choice - 1)
            c.yaw = float(steer_choice - 1)
            c.jump = bool(do_jump_last and (t == duration_ticks - 1))
            self._lock_hover_ball_if_needed()
            self.car.set_controls(c)
            self.arena.step(1)
            self._lock_hover_ball_if_needed()
            if self._touched:
                break
        self._boost_pickups_tick = 0
        obs_v3, obs_v2, parts = build_obs_v3(
            self.car, self.arena,
            target_pos=self.target_pos, target_time=self.target_time,
        )
        self.last_obs_v3 = obs_v3
        self.last_parts = parts

    def reset(self, mode=None, replay_last=False, spawn_state=None, stage_override=None, sticky_init_prob=0.0):
        self.maybe_reload_focus_config()
        if spawn_state is not None:
            self.last_spawn_state = dict(spawn_state)
            if "spawn_id" not in self.last_spawn_state:
                self.last_spawn_state["spawn_id"] = self._compute_spawn_id(self.last_spawn_state)
            self._apply_spawn_state(self.last_spawn_state)
        elif replay_last and self.last_spawn_state is not None:
            self._apply_spawn_state(self.last_spawn_state)
        else:
            use_mode = mode or self.mode
            if use_mode == "stage0":
                use_mode = "ground"
            elif use_mode == "stage1":
                use_mode = "target"

            self.arena.remove_car(self.car)
            self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
            self._touched = False
            self._boost_pickups_tick = 0
            self._reset_boost_pads()

            self.target_pos = None
            self.target_time = None
            self.is_hover_ball = False
            self.hover_ball_pos = None
            self.is_high_band_sample = False

            if self.focus_cfg.enabled and random.random() < self.focus_cfg.prob:
                self._randomize_focus_region(self.focus_cfg)
                use_mode = "focus"
            elif stage_override is not None:
                self._randomize_curriculum_stage(int(stage_override))
                use_mode = f"stage_{int(stage_override)+1}"
            elif use_mode in ("curriculum", "auto"):
                chosen_stage = random.randint(0, self.stage_level - 1)
                self._randomize_curriculum_stage(chosen_stage)
                use_mode = f"stage_{chosen_stage+1}"
            elif use_mode == "ground":
                chosen_stage = random.randint(0, min(self.stage_level, 4) - 1)
                self._randomize_curriculum_stage(chosen_stage)
            elif use_mode == "static_aerial":
                chosen_stage = random.choice((4, 5))
                self._randomize_curriculum_stage(chosen_stage)
            elif use_mode == "aerial":
                chosen_stage = random.choice((6, 7))
                self._randomize_curriculum_stage(chosen_stage)
            elif use_mode == "target":
                self._randomize_target()
            elif use_mode in ("all", "mixed"):
                chosen_stage = random.randint(0, NUM_CURRICULUM_STAGES - 1)
                self._randomize_curriculum_stage(chosen_stage)
            else:
                chosen_stage = random.randint(0, self.stage_level - 1)
                self._randomize_curriculum_stage(chosen_stage)

            self.last_spawn_state = self._capture_spawn_state(use_mode)
            self._apply_spawn_state(self.last_spawn_state)

            if sticky_init_prob > 0.0 and random.random() < sticky_init_prob:
                self.apply_initial_backward_prefix(duration_ticks=30, jump_last_prob=0.5)

        obs_v3, obs_v2, parts = build_obs_v3(
            self.car, self.arena,
            target_pos=self.target_pos, target_time=self.target_time,
        )
        self.last_obs_v3 = obs_v3
        self.last_parts = parts
        dist = float(parts["D.ball_rel_b"][3])
        return obs_v2, dist

    def _randomize_curriculum_stage(self, stage_idx: int, rng=random):
        stage_idx = max(0, min(NUM_CURRICULUM_STAGES - 1, int(stage_idx)))
        self.current_stage_idx = stage_idx

        is_aerial = (stage_idx >= 4)
        if not is_aerial:
            # Ground stages 0..3:
            # 0: ball static, car static
            # 1: ball dynamic, car static
            # 2: ball static, car moving
            # 3: ball dynamic, car moving
            ball_dyn = stage_idx in (1, 3)
            car_move = stage_idx in (2, 3)
            self.is_hover_ball = False
            self.hover_ball_pos = None
            self.is_high_band_sample = False

            bx = rng.uniform(-2600.0, 2600.0)
            by = rng.uniform(-3000.0, 3000.0)
            bz = 93.15
            if ball_dyn:
                bspd = rng.uniform(120.0, 550.0)
                bdir = rng.uniform(-math.pi, math.pi)
                bvx, bvy, bvz = bspd * math.cos(bdir), bspd * math.sin(bdir), 0.0
            else:
                bvx, bvy, bvz = 0.0, 0.0, 0.0

            bs = self.arena.ball.get_state()
            bs.pos = rs.Vec(bx, by, bz)
            bs.vel = rs.Vec(bvx, bvy, bvz)
            bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            self.arena.ball.set_state(bs)

            # Balanced distance & angle sampling (ensures rear/close-range coverage for backward/half-flip)
            if rng.random() < 0.30:
                dist_h = rng.uniform(180.0, 950.0)
            else:
                dist_h = rng.uniform(600.0, 3800.0)

            bearing = rng.uniform(-math.pi, math.pi)
            cx = float(np.clip(bx - dist_h * math.cos(bearing), -3200.0, 3200.0))
            cy = float(np.clip(by - dist_h * math.sin(bearing), -4200.0, 3200.0))
            yaw = rng.uniform(-math.pi, math.pi)
            fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)

            cs = self.car.get_state()
            cs.pos = rs.Vec(cx, cy, 17.0)
            if car_move:
                cspd = rng.uniform(200.0, 1600.0)
                cdir = yaw + (0.0 if rng.random() < 0.7 else math.pi) + rng.uniform(-0.3, 0.3)
                cs.vel = rs.Vec(cspd * math.cos(cdir), cspd * math.sin(cdir), 0.0)
            else:
                cs.vel = rs.Vec(0.0, 0.0, 0.0)
            cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            cs.rot_mat.forward = rs.Vec(*fwd)
            cs.rot_mat.right   = rs.Vec(*rgt)
            cs.rot_mat.up      = rs.Vec(*up)
            cs.boost = sample_initial_boost(r=rng)
            cs.is_on_ground = True
            self.car.set_state(cs)
        else:
            # Aerial stages 4..7:
            # 4: hover ball (static aerial, !touched zero-G), car static
            # 5: hover ball (static aerial, !touched zero-G), car moving
            # 6: dynamic aerial ball, car static
            # 7: dynamic aerial ball, car moving
            ball_hover = stage_idx in (4, 5)
            car_move   = stage_idx in (5, 7)
            self.is_hover_ball = ball_hover

            bx = rng.uniform(-2000.0, 2000.0)
            by = rng.uniform(-2200.0, 2200.0)

            max_z = max(220.0, float(self.aerial_state.max_z))
            if ball_hover:
                if self.aerial_state.phase == 1 and max_z > 400.0 and rng.random() < 0.45:
                    # Ensure ample sampling in the top [max_z-100, max_z] examination band
                    bz = rng.uniform(max(400.0, max_z - 100.0), max_z)
                elif self.aerial_state.phase == 2 and rng.random() < 0.50:
                    # Ensure ample sampling of high balls (>500) during CSV +margin ladder
                    bz = rng.uniform(500.1, 2000.0)
                elif self.aerial_state.phase == 0 and rng.random() < 0.35:
                    # Smooth single-jump to double-jump bootstrap band (strictly requires jump)
                    bz = rng.uniform(145.0, 230.0)
                else:
                    bz = rng.uniform(145.0, max_z)
                self.is_high_band_sample = (
                    (self.aerial_state.phase == 0 and bz <= 400.0)
                    or (self.aerial_state.phase == 1 and (max_z - 100.0) <= bz <= max_z)
                    or (self.aerial_state.phase >= 2 and bz > 500.0)
                )
                bvx, bvy, bvz = 0.0, 0.0, 0.0
                self.hover_ball_pos = (float(bx), float(by), float(bz))
            else:
                bz = rng.uniform(220.0, min(1200.0, max(500.0, max_z)))
                self.is_high_band_sample = False
                bspd = rng.uniform(80.0, 420.0)
                bdir = rng.uniform(-math.pi, math.pi)
                bvx = bspd * math.cos(bdir)
                bvy = bspd * math.sin(bdir)
                bvz = rng.uniform(150.0, 520.0)
                self.hover_ball_pos = None

            bs = self.arena.ball.get_state()
            bs.pos = rs.Vec(bx, by, bz)
            bs.vel = rs.Vec(bvx, bvy, bvz)
            bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            self.arena.ball.set_state(bs)

            # Horizontal distance proportional to height so jump/aerial is physically reachable
            if bz <= 400.0:
                dist_h = rng.uniform(150.0, 1400.0)
            else:
                dist_h = rng.uniform(max(300.0, bz * 0.6), min(3200.0, bz * 1.8 + 800.0))

            bearing = rng.uniform(-math.pi, math.pi)
            cx = float(np.clip(bx - dist_h * math.cos(bearing), -3000.0, 3000.0))
            cy = float(np.clip(by - dist_h * math.sin(bearing), -3800.0, 2800.0))
            yaw = rng.uniform(-math.pi, math.pi)
            fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)

            cs = self.car.get_state()
            cs.pos = rs.Vec(cx, cy, 17.0)
            if car_move:
                cspd = rng.uniform(200.0, 1400.0)
                cdir = yaw + (0.0 if rng.random() < 0.75 else math.pi) + rng.uniform(-0.25, 0.25)
                cs.vel = rs.Vec(cspd * math.cos(cdir), cspd * math.sin(cdir), 0.0)
            else:
                cs.vel = rs.Vec(0.0, 0.0, 0.0)
            cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            cs.rot_mat.forward = rs.Vec(*fwd)
            cs.rot_mat.right   = rs.Vec(*rgt)
            cs.rot_mat.up      = rs.Vec(*up)
            cs.boost = compute_aerial_spawn_boost(bz, self.aerial_state, r=rng)
            cs.is_on_ground = True
            self.car.set_state(cs)

    def _randomize_focus_region(self, fc: FocusRegionConfig):
        for _ in range(50):
            cx = random.uniform(-2800.0, 2800.0)
            cy = random.uniform(-3600.0, 2800.0)
            cz = random.uniform(fc.car_z_min, fc.car_z_max)
            yaw = random.uniform(-math.pi, math.pi)
            rel_deg = random.uniform(fc.angle_deg_min, fc.angle_deg_max)
            if fc.both_sides and random.random() < 0.5:
                rel_deg = -rel_deg
            rel_rad = math.radians(rel_deg)
            bearing = yaw + rel_rad
            dist_h = random.uniform(fc.dist_min, fc.dist_max)
            bx = cx + dist_h * math.cos(bearing)
            by = cy + dist_h * math.sin(bearing)
            if -3500.0 <= bx <= 3500.0 and -4500.0 <= by <= 4500.0:
                break
        else:
            cx, cy, cz = 0.0, 0.0, max(17.0, fc.car_z_min)
            yaw = math.pi * 0.5
            rel_rad = math.radians(0.5 * (fc.angle_deg_min + fc.angle_deg_max))
            bearing = yaw + rel_rad
            dist_h = 0.5 * (fc.dist_min + fc.dist_max)
            bx = cx + dist_h * math.cos(bearing)
            by = cy + dist_h * math.sin(bearing)

        bz = random.uniform(fc.ball_z_min, fc.ball_z_max)
        bspd = random.uniform(fc.ball_speed_min, fc.ball_speed_max)
        bdir = random.uniform(-math.pi, math.pi)
        bvx = bspd * math.cos(bdir)
        bvy = bspd * math.sin(bdir)
        bvz = random.uniform(-200.0, 400.0) if bz > 120.0 and bspd > 0 else 0.0

        self.is_hover_ball = (bz > 120.0 and bspd == 0.0)
        self.hover_ball_pos = (float(bx), float(by), float(bz)) if self.is_hover_ball else None

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(bvx, bvy, bvz)
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        pitch = random.uniform(-0.35, 0.35) if cz > 35.0 else 0.0
        roll  = random.uniform(-0.35, 0.35) if cz > 35.0 else 0.0
        fwd, rgt, up = euler_to_rot_vectors(yaw, pitch, roll)

        cspd = random.uniform(fc.car_speed_min, fc.car_speed_max)
        cdir = yaw + random.uniform(-math.pi, math.pi)
        cvx = cspd * math.cos(cdir)
        cvy = cspd * math.sin(cdir)
        cvz = random.uniform(-150.0, 250.0) if cz > 35.0 and cspd > 0 else 0.0

        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, cz)
        cs.vel = rs.Vec(cvx, cvy, cvz)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(*fwd)
        cs.rot_mat.right   = rs.Vec(*rgt)
        cs.rot_mat.up      = rs.Vec(*up)
        cs.boost = compute_aerial_spawn_boost(bz, self.aerial_state)
        cs.is_on_ground = (cz <= 20.0)
        self.car.set_state(cs)

    def _randomize_target(self):
        self._randomize_curriculum_stage(0)
        bs = self.arena.ball.get_state()
        bx, by = float(bs.pos.x), float(bs.pos.y)
        for _ in range(30):
            tx = random.uniform(-3000.0, 3000.0)
            ty = random.uniform(-3000.0, 4800.0)
            d_b2t = math.hypot(tx - bx, ty - by)
            if 1000.0 <= d_b2t <= 4200.0:
                break
        else:
            tx, ty = 0.0, 5120.0
            d_b2t = max(1000.0, math.hypot(tx - bx, ty - by))

        tz = random.uniform(93.15, 450.0)
        desired_avg_spd = random.uniform(900.0, 2200.0)
        tt = float(np.clip(d_b2t / desired_avg_spd, 0.5, 2.8))
        self.target_pos = (float(tx), float(ty), float(tz))
        self.target_time = round(tt, 3)

    def generate_8stage_eval_grid(
        self,
        samples_per_stage=EVAL_SAMPLES_PER_STAGE,
        max_stage=NUM_CURRICULUM_STAGES,
        rng=None,
    ):
        r = rng if rng is not None else random.Random(20261016)
        angle_bins = [
            (-math.pi / 4.0, math.pi / 4.0),
            (math.pi / 4.0, 3.0 * math.pi / 4.0),
            (-3.0 * math.pi / 4.0, -math.pi / 4.0),
            (3.0 * math.pi / 4.0, 5.0 * math.pi / 4.0),
        ]
        dist_bins = [
            (200.0, 800.0),
            (800.0, 2000.0),
            (2000.0, 3600.0),
        ]

        grid_batches = []
        n_stages = max(1, min(NUM_CURRICULUM_STAGES, int(max_stage)))
        for s_idx in range(n_stages):
            stage_samples = []
            is_aerial = (s_idx >= 4)
            ball_dyn  = s_idx in (1, 3, 6, 7)
            car_move  = s_idx in (2, 3, 5, 7)
            is_hover  = s_idx in (4, 5)

            for k in range(samples_per_stage):
                d_lo, d_hi = dist_bins[k % len(dist_bins)]
                a_lo, a_hi = angle_bins[(k // len(dist_bins)) % len(angle_bins)]

                if is_aerial:
                    max_z = max(250.0, float(self.aerial_state.max_z))
                    bz = r.uniform(145.0, max_z) if is_hover else r.uniform(220.0, min(1100.0, max(500.0, max_z)))
                    if bz <= 400.0:
                        d_lo, d_hi = 200.0, 1300.0
                    else:
                        d_lo, d_hi = max(350.0, bz * 0.6), min(3000.0, bz * 1.7 + 700.0)
                else:
                    bz = 93.15

                for _ in range(50):
                    bx = r.uniform(-2200.0, 2200.0)
                    by = r.uniform(-2500.0, 2500.0)
                    dist = r.uniform(d_lo, d_hi)
                    bearing = r.uniform(-math.pi, math.pi)
                    cx = bx - dist * math.cos(bearing)
                    cy = by - dist * math.sin(bearing)
                    if -3000.0 <= cx <= 3000.0 and -3800.0 <= cy <= 3000.0:
                        break
                else:
                    bx, by = 0.0, 500.0
                    dist = 0.5 * (d_lo + d_hi)
                    bearing = math.pi * 0.5
                    cx, cy = 0.0, 500.0 - dist

                rel_ang = r.uniform(a_lo, a_hi)
                yaw = bearing - rel_ang
                fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)

                if ball_dyn:
                    bspd = r.uniform(120.0, 450.0)
                    bdir = r.uniform(-math.pi, math.pi)
                    bvx = bspd * math.cos(bdir)
                    bvy = bspd * math.sin(bdir)
                    bvz = r.uniform(150.0, 450.0) if is_aerial else 0.0
                else:
                    bvx, bvy, bvz = 0.0, 0.0, 0.0

                if car_move:
                    cspd = r.uniform(250.0, 1300.0)
                    cdir = yaw + (0.0 if (k % 3 != 0) else math.pi)
                    cvx, cvy, cvz = cspd * math.cos(cdir), cspd * math.sin(cdir), 0.0
                else:
                    cvx, cvy, cvz = 0.0, 0.0, 0.0

                if is_aerial:
                    boost_val = compute_aerial_spawn_boost(bz, self.aerial_state, r=r)
                else:
                    boost_val = float(r.uniform(0.0, LOW_BOOST_MAX)) if (k % 10 < 7) else float(r.uniform(LOW_BOOST_MAX, 100.0))

                sd = {
                    "mode": f"stage_{s_idx+1}",
                    "stage_idx": s_idx,
                    "bin_id": s_idx,
                    "is_hover_ball": is_hover,
                    "car_pos": (float(cx), float(cy), 17.0),
                    "car_vel": (float(cvx), float(cvy), float(cvz)),
                    "car_ang": (0.0, 0.0, 0.0),
                    "car_fwd": fwd,
                    "car_rgt": rgt,
                    "car_up":  up,
                    "car_boost": boost_val,
                    "ball_pos": (float(bx), float(by), float(bz)),
                    "ball_vel": (float(bvx), float(bvy), float(bvz)),
                    "ball_ang": (0.0, 0.0, 0.0),
                    "target_pos": None,
                    "target_time": None,
                }
                sd["spawn_id"] = self._compute_spawn_id(sd)
                stage_samples.append(sd)
            grid_batches.append(stage_samples)
        return grid_batches

    def generate_16bin_eval_grid(self, samples_per_bin=EVAL_SAMPLES_PER_BIN, rng=None):
        return self.generate_8stage_eval_grid(
            samples_per_stage=samples_per_bin,
            max_stage=NUM_CURRICULUM_STAGES,
            rng=rng,
        )

    def get_obs_v3(self, intent_override=None):
        obs_v3, obs_v2, parts = build_obs_v3(
            self.car, self.arena,
            intent_override=intent_override,
            target_pos=self.target_pos, target_time=self.target_time,
        )
        self.last_obs_v3 = obs_v3
        self.last_parts = parts
        return obs_v3, obs_v2, parts

    def snapshot(self):
        cs = self.car.get_state()
        bs = self.arena.ball.get_state()
        up = self.car.get_up_dir()
        fwd = self.car.get_forward_dir()
        right = self.car.get_right_dir()
        dx = float(bs.pos.x - cs.pos.x)
        dy = float(bs.pos.y - cs.pos.y)
        dz = float(bs.pos.z - cs.pos.z)
        rel = np.array([dx, dy, dz])
        rel_fwd = abs(float(dx * fwd.x + dy * fwd.y + dz * fwd.z))
        rel_rgt = abs(float(dx * right.x + dy * right.y + dz * right.z))
        rel_up  = abs(float(dx * up.x + dy * up.y + dz * up.z))
        vec_dist = (abs(dx), abs(dy), abs(dz))
        dist = float(np.linalg.norm(rel))
        dist_h = float(np.hypot(dx, dy))
        rel_vel = np.array([bs.vel.x - cs.vel.x, bs.vel.y - cs.vel.y, bs.vel.z - cs.vel.z])
        closing_speed = -float(np.dot(rel, rel_vel)) / max(dist, 1e-6)
        fwd_h = np.array([fwd.x, fwd.y])
        d_h = np.array([dx, dy])
        n = np.linalg.norm(d_h) + 1e-6
        align = float(np.dot(fwd_h, d_h / n))

        car_speed = float(math.sqrt(cs.vel.x**2 + cs.vel.y**2 + cs.vel.z**2))
        ball_pos_np = np.array([bs.pos.x, bs.pos.y, bs.pos.z], dtype=np.float32)
        ball_vel_np = np.array([bs.vel.x, bs.vel.y, bs.vel.z], dtype=np.float32)
        tgt_np = OPP_GOAL_POS if self.target_pos is None else np.asarray(self.target_pos, dtype=np.float32)
        b2g = tgt_np - ball_pos_np
        b2g_dir = b2g / (float(np.linalg.norm(b2g)) + 1e-6)
        ball_vel_to_goal = float(np.dot(ball_vel_np, b2g_dir))
        ball_speed = float(np.linalg.norm(ball_vel_np))

        return {
            "car_pos": (cs.pos.x, cs.pos.y, cs.pos.z),
            "car_vel": (cs.vel.x, cs.vel.y, cs.vel.z),
            "car_ang": (cs.ang_vel.x, cs.ang_vel.y, cs.ang_vel.z),
            "car_speed": car_speed,
            "is_supersonic": bool(cs.is_supersonic),
            "ball_pos": (bs.pos.x, bs.pos.y, bs.pos.z),
            "ball_vel": (bs.vel.x, bs.vel.y, bs.vel.z),
            "ball_speed": ball_speed,
            "ball_vel_to_goal": ball_vel_to_goal,
            "target_pos": self.target_pos,
            "target_time": self.target_time,
            "up_z": up.z,
            "fwd":   (fwd.x,   fwd.y,   fwd.z),
            "up":    (up.x,    up.y,    up.z),
            "right": (right.x, right.y, right.z),
            "on_ground": bool(cs.is_on_ground),
            "is_flipping": bool(cs.is_flipping),
            "is_jumping": bool(cs.is_jumping),
            "air_time": float(cs.air_time),
            "boost": float(cs.boost),
            "dist": dist,
            "vec_dist": vec_dist,
            "dist_h": dist_h,
            "align": align,
            "closing": closing_speed,
            "touched": self._touched,
            "stage_idx": self.current_stage_idx,
            "is_high_band_sample": self.is_high_band_sample,
        }


EVAL_DIST_LABELS = ["150..600", "600..1500", "1500..2800", "2800..4400"]
EVAL_ANGLE_LABELS = ["Front(F)", "Right(R)", "Left(L)", "Back(B)"]
