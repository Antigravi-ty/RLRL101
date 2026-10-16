import math
import random
import hashlib
import numpy as np
import RocketSim as rs
from .obs_v3 import build_obs_v3, OPP_GOAL_POS


class BallReachEnv:
    def __init__(self, mode="ground"):
        self.mode = "ground" if mode in ("stage0", None) else ("target" if mode == "stage1" else mode)
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
        self._shadow_arena = rs.Arena(rs.GameMode.SOCCAR)
        self._shadow_car = self._shadow_arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        self._ball_sim_arena = rs.Arena(rs.GameMode.SOCCAR)

    def _compute_spawn_id(self, sd):
        key_str = (
            f"{sd.get('mode','g')}_{sd['car_pos'][0]:.1f}_{sd['car_pos'][1]:.1f}_"
            f"{sd['car_fwd'][0]:.3f}_{sd['car_fwd'][1]:.3f}_"
            f"{sd['ball_pos'][0]:.1f}_{sd['ball_pos'][1]:.1f}_"
            f"{sd['ball_vel'][0]:.1f}_{sd['ball_vel'][1]:.1f}"
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
        while self._shadow_car.id < self.car.id:
            self._shadow_arena.remove_car(self._shadow_car)
            self._shadow_car = self._shadow_arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
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

    def step_with_waste_check(self, ctrl):
        cs_before = self.car.get_state()
        prev_boost = float(cs_before.boost)
        v_no_vec = None
        v_no_mag = None
        b_no_boost = prev_boost
        was_on_ground = bool(cs_before.is_on_ground)

        if ctrl.boost and prev_boost > 0.01:
            self.arena.clone_into(self._shadow_arena)
            c_no = rs.CarControls()
            c_no.throttle  = ctrl.throttle
            c_no.steer     = ctrl.steer
            c_no.pitch     = ctrl.pitch
            c_no.yaw       = ctrl.yaw
            c_no.roll      = ctrl.roll
            c_no.jump      = ctrl.jump
            c_no.boost     = False
            c_no.handbrake = ctrl.handbrake
            self._shadow_car.set_controls(c_no)
            self._shadow_arena.step(1)
            cf_s = self._shadow_car.get_state()
            v_no_vec = (float(cf_s.vel.x), float(cf_s.vel.y), float(cf_s.vel.z))
            v_no_mag = float(math.sqrt(v_no_vec[0]**2 + v_no_vec[1]**2 + v_no_vec[2]**2))
            b_no_boost = float(cf_s.boost)

        self.car.set_controls(ctrl)
        self.arena.step(1)

        cur_s = self.car.get_state()
        cur_boost = float(cur_s.boost)
        boost_used = max(0.0, prev_boost - cur_boost)

        waste_ratio = 0.0
        if v_no_vec is not None and cur_boost < b_no_boost - 1e-4:
            vx, vy, vz = float(cur_s.vel.x), float(cur_s.vel.y), float(cur_s.vel.z)
            v_with_mag = float(math.sqrt(vx*vx + vy*vy + vz*vz))
            if v_with_mag >= 2290.0:
                if was_on_ground:
                    dv = max(0.0, v_with_mag - v_no_mag)
                    full_dv = 8.2639
                else:
                    dv = max(0.0, v_with_mag - v_no_mag)
                    full_dv = 8.8611
                if dv < full_dv - 0.05:
                    waste_ratio = max(0.0, min(1.0, 1.0 - (dv / full_dv)))

        self.last_boost_used = boost_used
        self.last_waste_ratio = waste_ratio
        return waste_ratio

    def reset(self, mode=None, replay_last=False, spawn_state=None):
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
            elif use_mode == "mixed":
                use_mode = "ground" if random.random() < 0.5 else "aerial"

            self.arena.remove_car(self.car)
            self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
            self._touched = False
            self._reset_boost_pads()

            self.target_pos = None
            self.target_time = None
            if use_mode in ("ground", "stage0"):
                self._randomize_stage0()
            elif use_mode in ("target", "stage1"):
                self._randomize_stage1()
            elif use_mode == "static_aerial":
                self._randomize_static_aerial()
            elif use_mode == "aerial":
                self._randomize_aerial()

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

    def _randomize_stage0(self):
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2800.0, 2800.0)
        by = random.uniform(-3200.0, 3200.0)
        bs.pos = rs.Vec(bx, by, 93.15)
        bs.vel = rs.Vec(random.uniform(-300.0, 300.0), random.uniform(-300.0, 300.0), 0.0)
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        for _ in range(20):
            cx = random.uniform(-3200.0, 3200.0)
            cy = random.uniform(-4200.0, 3200.0)
            if math.hypot(cx - bx, cy - by) > 900.0:
                break

        yaw = random.uniform(-math.pi, math.pi)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)

        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(cos_y, sin_y, 0.0)
        cs.rot_mat.right   = rs.Vec(-sin_y, cos_y, 0.0)
        cs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        cs.boost = random.uniform(33.3, 100.0)
        cs.is_on_ground = True
        self.car.set_state(cs)

    def _randomize_stage1(self):
        self._randomize_stage0()
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

    def evaluate_shot_cpa(self, max_sim_ticks=360):
        tgt = OPP_GOAL_POS if self.target_pos is None else np.asarray(self.target_pos, dtype=np.float32)
        tx, ty, tz = float(tgt[0]), float(tgt[1]), float(tgt[2])

        bs = self.arena.ball.get_state()
        self._ball_sim_arena.ball.set_state(bs)

        d_prev = math.sqrt((bs.pos.x - tx)**2 + (bs.pos.y - ty)**2 + (bs.pos.z - tz)**2)
        min_d = d_prev
        min_tick = 0
        has_decreased = False

        step_ticks = 2
        for tick in range(step_ticks, int(max_sim_ticks) + 1, step_ticks):
            self._ball_sim_arena.step(step_ticks)
            b_now = self._ball_sim_arena.ball.get_state()
            d_now = math.sqrt((b_now.pos.x - tx)**2 + (b_now.pos.y - ty)**2 + (b_now.pos.z - tz)**2)
            if d_now < d_prev - 1e-3:
                has_decreased = True
                if d_now < min_d:
                    min_d = d_now
                    min_tick = tick
            elif d_now > d_prev + 1e-3:
                break
            d_prev = d_now

        if not has_decreased:
            return float("inf"), 0.0, float(self.target_time or 0.0)

        cpa_time = float(min_tick) / 120.0
        time_err = abs(cpa_time - float(self.target_time)) if self.target_time is not None else 0.0
        return float(min_d), cpa_time, float(time_err)

    def generate_stratified_stage0_batch(self, rng=None, include_target=None):
        r = rng if rng is not None else random
        if include_target is None:
            include_target = (self.mode in ("target", "stage1"))
        dist_bins = [(950.0, 1800.0), (1800.0, 3200.0), (3200.0, 4800.0)]
        angle_bins = [
            (0.0, math.pi / 6.0),
            (math.pi / 6.0, math.pi / 2.0),
            (math.pi / 2.0, 3.0 * math.pi / 4.0),
            (3.0 * math.pi / 4.0, math.pi),
        ]
        bvel_bins = [(0.0, 80.0), (200.0, 400.0)]

        batch = []
        for d_lo, d_hi in dist_bins:
            for a_lo, a_hi in angle_bins:
                for v_lo, v_hi in bvel_bins:
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

                    rel_ang = r.uniform(a_lo, a_hi) * (1.0 if r.random() < 0.5 else -1.0)
                    yaw = bearing + rel_ang
                    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

                    bspd = r.uniform(v_lo, v_hi)
                    bdir = r.uniform(-math.pi, math.pi)
                    bvx = bspd * math.cos(bdir)
                    bvy = bspd * math.sin(bdir)

                    t_pos = None
                    t_time = None
                    if include_target:
                        for _ in range(30):
                            tx = r.uniform(-3000.0, 3000.0)
                            ty = r.uniform(-3000.0, 4800.0)
                            d_b2t = math.hypot(tx - bx, ty - by)
                            if 1000.0 <= d_b2t <= 4200.0:
                                break
                        else:
                            tx, ty = 0.0, 5120.0
                            d_b2t = max(1000.0, math.hypot(tx - bx, ty - by))
                        tz = r.uniform(93.15, 450.0)
                        desired_avg_spd = r.uniform(900.0, 2200.0)
                        t_time = round(float(np.clip(d_b2t / desired_avg_spd, 0.5, 2.8)), 3)
                        t_pos = (float(tx), float(ty), float(tz))

                    sd = {
                        "mode": "target" if include_target else "ground",
                        "car_pos": (float(cx), float(cy), 17.0),
                        "car_vel": (0.0, 0.0, 0.0),
                        "car_ang": (0.0, 0.0, 0.0),
                        "car_fwd": (float(cos_y), float(sin_y), 0.0),
                        "car_rgt": (float(-sin_y), float(cos_y), 0.0),
                        "car_up":  (0.0, 0.0, 1.0),
                        "car_boost": float(r.uniform(35.0, 100.0)),
                        "ball_pos": (float(bx), float(by), 93.15),
                        "ball_vel": (float(bvx), float(bvy), 0.0),
                        "ball_ang": (0.0, 0.0, 0.0),
                        "target_pos": t_pos,
                        "target_time": t_time,
                    }
                    sd["spawn_id"] = self._compute_spawn_id(sd)
                    batch.append(sd)
        return batch

    def _randomize_static_aerial(self):
        bs = self.arena.ball.get_state()
        bx = random.uniform(-2000.0, 2000.0)
        by = random.uniform(-2000.0, 2500.0)
        bz = random.uniform(220.0, 750.0)
        bs.pos = rs.Vec(bx, by, bz)
        bs.vel = rs.Vec(random.uniform(-80.0, 80.0), random.uniform(-80.0, 80.0), random.uniform(350.0, 650.0))
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        for _ in range(20):
            cx = random.uniform(-2500.0, 2500.0)
            cy = random.uniform(-3500.0, 2000.0)
            if 900.0 <= math.hypot(cx - bx, cy - by) <= 2800.0:
                break
        yaw = random.uniform(-math.pi, math.pi)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(cos_y, sin_y, 0.0)
        cs.rot_mat.right   = rs.Vec(-sin_y, cos_y, 0.0)
        cs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        cs.boost = random.uniform(60.0, 100.0)
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
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        cx = random.uniform(-2500.0, 2500.0)
        cy = random.uniform(-3800.0, 1000.0)
        yaw = random.uniform(-math.pi, math.pi)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        cs = self.car.get_state()
        cs.pos = rs.Vec(cx, cy, 17.0)
        cs.vel = rs.Vec(0.0, 0.0, 0.0)
        cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        cs.rot_mat.forward = rs.Vec(cos_y, sin_y, 0.0)
        cs.rot_mat.right   = rs.Vec(-sin_y, cos_y, 0.0)
        cs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        cs.boost = random.uniform(60.0, 100.0)
        cs.is_on_ground = True
        self.car.set_state(cs)

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
