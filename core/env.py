import os
import json
import math
import random
import hashlib
from dataclasses import dataclass, asdict
import numpy as np
import RocketSim as rs
from .constants import LOW_BOOST_PROB, LOW_BOOST_MAX, EVAL_SAMPLES_PER_BIN
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
    def __init__(self, mode="all", focus_cfg: FocusRegionConfig = None):
        self.mode = "ground" if mode == "stage0" else ("target" if mode == "stage1" else (mode or "all"))
        self.focus_cfg = focus_cfg if focus_cfg is not None else load_focus_config()
        self._focus_mtime = os.path.getmtime(FOCUS_CONFIG_PATH) if os.path.isfile(FOCUS_CONFIG_PATH) else None

        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        self.pads = self.arena.get_boost_pads()
        self.num_pads = len(self.pads)

        self._touched = False
        def _on_touch(arena, car, data):
            self._touched = True
        self.arena.set_ball_touch_callback(_on_touch)

        self.last_spawn_state = None
        self.spawn_id = "0000"
        self.target_pos = None
        self.target_time = None
        self.last_boost_used = 0.0
        self.last_flip_triggered = False
        self.last_flip_trigger_speed = 0.0

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
            f"{sd.get('mode','a')}_{sd['car_pos'][0]:.1f}_{sd['car_pos'][1]:.1f}_{sd['car_pos'][2]:.1f}_"
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
        self._reset_boost_pads()

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

    def step_tick(self, ctrl):
        cs_before = self.car.get_state()
        prev_spd = float(math.sqrt(cs_before.vel.x**2 + cs_before.vel.y**2 + cs_before.vel.z**2))
        prev_boost = float(cs_before.boost)
        was_flipping = bool(cs_before.is_flipping)

        self.car.set_controls(ctrl)
        self.arena.step(1)

        cs_after = self.car.get_state()
        cur_boost = float(cs_after.boost)
        now_flipping = bool(cs_after.is_flipping)

        boost_used = max(0.0, prev_boost - cur_boost)
        flip_triggered = (not was_flipping) and now_flipping

        self.last_boost_used = boost_used
        self.last_flip_triggered = flip_triggered
        self.last_flip_trigger_speed = prev_spd if flip_triggered else 0.0
        return boost_used, flip_triggered, self.last_flip_trigger_speed

    def reset(self, mode=None, replay_last=False, spawn_state=None):
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
            self._reset_boost_pads()

            self.target_pos = None
            self.target_time = None

            if self.focus_cfg.enabled and random.random() < self.focus_cfg.prob:
                self._randomize_focus_region(self.focus_cfg)
                use_mode = "focus"
            elif use_mode in ("all", "mixed"):
                self._randomize_all_variations()
            elif use_mode == "ground":
                self._randomize_ground()
            elif use_mode == "target":
                self._randomize_target()
            elif use_mode == "static_aerial":
                self._randomize_static_aerial()
            elif use_mode == "aerial":
                self._randomize_aerial()
            else:
                self._randomize_all_variations()

            self.last_spawn_state = self._capture_spawn_state(use_mode)
            self._apply_spawn_state(self.last_spawn_state)

        obs_v3, obs_v2, parts = build_obs_v3(
            self.car, self.arena,
            target_pos=self.target_pos, target_time=self.target_time,
        )
        self.last_obs_v3 = obs_v3
        self.last_parts = parts
        dist = float(parts["D.ball_rel_b"][3])
        return obs_v2, dist

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
        cs.boost = sample_initial_boost()
        cs.is_on_ground = (cz <= 20.0)
        self.car.set_state(cs)

    def _randomize_all_variations(self):
        car_in_air = (random.random() < 0.25)
        ball_in_air = (random.random() < 0.35)
        ball_moving = (random.random() < 0.65)
        car_moving  = (random.random() < 0.55)

        bx = random.uniform(-3000.0, 3000.0)
        by = random.uniform(-3800.0, 3800.0)
        bz = random.uniform(180.0, 1100.0) if ball_in_air else 93.15

        if ball_moving:
            bspd_h = random.uniform(50.0, 1200.0)
            bdir = random.uniform(-math.pi, math.pi)
            bvx = bspd_h * math.cos(bdir)
            bvy = bspd_h * math.sin(bdir)
            bvz = random.uniform(-300.0, 600.0) if ball_in_air else 0.0
        else:
            bvx, bvy, bvz = 0.0, 0.0, (random.uniform(200.0, 500.0) if ball_in_air else 0.0)

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(bvx, bvy, bvz)
        bs.ang_vel = rs.Vec(random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0), random.uniform(-3.0, 3.0)) if ball_moving else rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        dist_h = random.uniform(120.0, 4200.0)
        bearing = random.uniform(-math.pi, math.pi)
        cx = float(np.clip(bx - dist_h * math.cos(bearing), -3400.0, 3400.0))
        cy = float(np.clip(by - dist_h * math.sin(bearing), -4400.0, 4400.0))
        cz = random.uniform(45.0, 900.0) if car_in_air else 17.0

        yaw = random.uniform(-math.pi, math.pi)
        pitch = random.uniform(-0.6, 0.6) if car_in_air else 0.0
        roll  = random.uniform(-0.6, 0.6) if car_in_air else 0.0
        fwd, rgt, up = euler_to_rot_vectors(yaw, pitch, roll)

        if car_moving:
            cspd = random.uniform(100.0, 2100.0)
            if random.random() < 0.7:
                cv_yaw = yaw + (0.0 if random.random() < 0.75 else math.pi) + random.uniform(-0.35, 0.35)
            else:
                cv_yaw = random.uniform(-math.pi, math.pi)
            cvx = cspd * math.cos(cv_yaw)
            cvy = cspd * math.sin(cv_yaw)
            cvz = random.uniform(-300.0, 400.0) if car_in_air else 0.0
        else:
            cvx, cvy, cvz = 0.0, 0.0, 0.0

        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, cz)
        cs.vel = rs.Vec(cvx, cvy, cvz)
        cs.ang_vel = rs.Vec(
            random.uniform(-2.0, 2.0) if car_in_air else 0.0,
            random.uniform(-2.0, 2.0) if car_in_air else 0.0,
            random.uniform(-2.0, 2.0) if car_moving else 0.0,
        )
        cs.rot_mat.forward = rs.Vec(*fwd)
        cs.rot_mat.right   = rs.Vec(*rgt)
        cs.rot_mat.up      = rs.Vec(*up)
        cs.boost = sample_initial_boost()
        cs.is_on_ground = not car_in_air
        self.car.set_state(cs)

    def _randomize_ground(self):
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2800.0, 2800.0)
        by = random.uniform(-3200.0, 3200.0)
        bs.pos = rs.Vec(bx, by, 93.15)
        if random.random() < 0.5:
            bs.vel = rs.Vec(0.0, 0.0, 0.0)
        else:
            bs.vel = rs.Vec(random.uniform(-400.0, 400.0), random.uniform(-400.0, 400.0), 0.0)
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        dist_h = random.uniform(120.0, 4200.0)
        bearing = random.uniform(-math.pi, math.pi)
        cx = float(np.clip(bx - dist_h * math.cos(bearing), -3200.0, 3200.0))
        cy = float(np.clip(by - dist_h * math.sin(bearing), -4200.0, 3200.0))

        yaw = random.uniform(-math.pi, math.pi)
        fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)

        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        if random.random() < 0.4:
            cspd = random.uniform(100.0, 1800.0)
            cdir = yaw + (0.0 if random.random() < 0.7 else math.pi)
            cs.vel = rs.Vec(cspd * math.cos(cdir), cspd * math.sin(cdir), 0.0)
        else:
            cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(*fwd)
        cs.rot_mat.right   = rs.Vec(*rgt)
        cs.rot_mat.up      = rs.Vec(*up)
        cs.boost = sample_initial_boost()
        cs.is_on_ground = True
        self.car.set_state(cs)

    def _randomize_target(self):
        self._randomize_ground()
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

    def _randomize_static_aerial(self):
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2000.0, 2000.0)
        by = random.uniform(-2000.0, 2500.0)
        bz = random.uniform(220.0, 750.0)
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(random.uniform(-80.0, 80.0), random.uniform(-80.0, 80.0), random.uniform(350.0, 650.0))
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        for _ in range(20):
            cx = random.uniform(-2500.0, 2500.0)
            cy = random.uniform(-3500.0, 2000.0)
            if 400.0 <= math.hypot(cx - bx, cy - by) <= 2800.0:
                break
        yaw = random.uniform(-math.pi, math.pi)
        fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)
        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(*fwd)
        cs.rot_mat.right   = rs.Vec(*rgt)
        cs.rot_mat.up      = rs.Vec(*up)
        cs.boost = sample_initial_boost()
        cs.is_on_ground = True
        self.car.set_state(cs)

    def _randomize_aerial(self):
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2000.0, 2000.0)
        by = random.uniform(-2000.0, 2500.0)
        bz = random.uniform(300.0, 1100.0)
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(random.uniform(-400.0, 400.0), random.uniform(-400.0, 400.0), random.uniform(100.0, 500.0))
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        cx = random.uniform(-2500.0, 2500.0)
        cy = random.uniform(-3800.0, 1000.0)
        yaw = random.uniform(-math.pi, math.pi)
        fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)
        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(*fwd)
        cs.rot_mat.right   = rs.Vec(*rgt)
        cs.rot_mat.up      = rs.Vec(*up)
        cs.boost = sample_initial_boost()
        cs.is_on_ground = True
        self.car.set_state(cs)

    def generate_16bin_eval_grid(self, samples_per_bin=EVAL_SAMPLES_PER_BIN, rng=None):
        r = rng if rng is not None else random.Random(20261016)
        dist_bins = [
            (150.0, 600.0),
            (600.0, 1500.0),
            (1500.0, 2800.0),
            (2800.0, 4400.0),
        ]
        angle_bins = [
            (-math.pi / 4.0, math.pi / 4.0),
            (math.pi / 4.0, 3.0 * math.pi / 4.0),
            (-3.0 * math.pi / 4.0, -math.pi / 4.0),
            (3.0 * math.pi / 4.0, 5.0 * math.pi / 4.0),
        ]

        grid_batches = []
        for b_idx, (d_lo, d_hi) in enumerate(dist_bins):
            for a_idx, (a_lo, a_hi) in enumerate(angle_bins):
                bin_id = b_idx * 4 + a_idx
                bin_samples = []
                for s_idx in range(samples_per_bin):
                    for _ in range(50):
                        bx = r.uniform(-2400.0, 2400.0)
                        by = r.uniform(-2800.0, 2800.0)
                        dist = r.uniform(d_lo, d_hi)
                        bearing = r.uniform(-math.pi, math.pi)
                        cx = bx - dist * math.cos(bearing)
                        cy = by - dist * math.sin(bearing)
                        if -3200.0 <= cx <= 3200.0 and -4200.0 <= cy <= 3200.0:
                            break
                    else:
                        bx, by = 0.0, 500.0
                        dist = 0.5 * (d_lo + d_hi)
                        bearing = math.pi * 0.5
                        cx, cy = 0.0, 500.0 - dist

                    rel_ang = r.uniform(a_lo, a_hi)
                    yaw = bearing - rel_ang
                    fwd, rgt, up = euler_to_rot_vectors(yaw, 0.0, 0.0)

                    bspd = 0.0 if (s_idx % 2 == 0) else r.uniform(100.0, 450.0)
                    bdir = r.uniform(-math.pi, math.pi)
                    bvx = bspd * math.cos(bdir)
                    bvy = bspd * math.sin(bdir)

                    boost_val = float(r.uniform(0.0, LOW_BOOST_MAX)) if (s_idx % 10 < 7) else float(r.uniform(LOW_BOOST_MAX, 100.0))

                    sd = {
                        "mode": "ground",
                        "bin_id": bin_id,
                        "car_pos": (float(cx), float(cy), 17.0),
                        "car_vel": (0.0, 0.0, 0.0),
                        "car_ang": (0.0, 0.0, 0.0),
                        "car_fwd": fwd,
                        "car_rgt": rgt,
                        "car_up":  up,
                        "car_boost": boost_val,
                        "ball_pos": (float(bx), float(by), 93.15),
                        "ball_vel": (float(bvx), float(bvy), 0.0),
                        "ball_ang": (0.0, 0.0, 0.0),
                        "target_pos": None,
                        "target_time": None,
                    }
                    sd["spawn_id"] = self._compute_spawn_id(sd)
                    bin_samples.append(sd)
                grid_batches.append(bin_samples)
        return grid_batches

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
        rel = np.array([bs.pos.x - cs.pos.x, bs.pos.y - cs.pos.y, bs.pos.z - cs.pos.z])
        dist = float(np.linalg.norm(rel))
        dist_h = float(np.hypot(rel[0], rel[1]))
        rel_vel = np.array([bs.vel.x - cs.vel.x, bs.vel.y - cs.vel.y, bs.vel.z - cs.vel.z])
        closing_speed = -float(np.dot(rel, rel_vel)) / max(dist, 1e-6)
        fwd_h = np.array([fwd.x, fwd.y])
        d_h = np.array([rel[0], rel[1]])
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
            "dist_h": dist_h,
            "align": align,
            "closing": closing_speed,
            "touched": self._touched,
        }


EVAL_DIST_LABELS = ["150..600", "600..1500", "1500..2800", "2800..4400"]
EVAL_ANGLE_LABELS = ["Front(F)", "Right(R)", "Left(L)", "Back(B)"]
