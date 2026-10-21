"""core/env.py —— BallReachEnv 静态球大王仿真环境 (4-Phase 车优先课程 + 地面 45f 悬挂稳定 + 全场正常 Boost Pads + 随机初始输入)。"""
import math
import random
import hashlib
import numpy as np
import RocketSim as rs

from .constants import (
    NUM_CURRICULUM_STAGES, MAX_DECISIONS, GROUND_SETTLE_TICKS,
    POOL_FRONTIER,
)
from .obs import build_obs, format_prev_action_vec
from .curriculum import (
    CurriculumState, STAGE_NAMES, STAGE_SHORT_LABELS,
    SAFE_SPAWN_X, SAFE_SPAWN_Y, SAFE_SPAWN_Z_MAX,
    NUM_CURRICULUM_STEPS, get_curriculum_step_spec, stage_level_to_step_idx,
    classify_local_dir5, dist_to_tier, get_adaptive_max_decisions,
    sample_initial_boost, sample_random_prev_action, euler_to_rot_vectors,
)
from .spawner import SpawnerMixin, apply_car_jump_state


class _SpawnDistCfg:
    def __init__(self, raw: str = ""):
        self.raw = (raw or "").strip()

    @property
    def is_active(self) -> bool:
        return bool(self.raw and self.raw.lower() not in ("auto", "off", "none", "default"))


class BallReachEnv(SpawnerMixin):
    def __init__(
        self,
        mode="curriculum",
        stage_level: int = 1,
        step_idx: int = None,
        spawn_dist: str = None,
        **kwargs,
    ):
        self.mode = mode or "curriculum"
        self.stage_level = max(1, min(NUM_CURRICULUM_STAGES, int(stage_level)))
        init_step = step_idx if step_idx is not None else stage_level_to_step_idx(self.stage_level)
        self.current_step_idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(init_step)))
        self.current_stage_idx = 0
        self.current_pool_id = POOL_FRONTIER
        self.current_dir5_idx = 0
        self.current_dir6_idx = 0
        self.current_dist_tier = 0
        self.spawn_dist_cfg = _SpawnDistCfg(spawn_dist or "")
        self.curriculum_state = CurriculumState(
            step_idx=self.current_step_idx,
            phase=get_curriculum_step_spec(self.current_step_idx).phase,
        )
        self.aerial_state = self.curriculum_state
        self.stage_rates = [0.0] * NUM_CURRICULUM_STAGES
        self.is_hover_ball = True
        self.hover_ball_pos = None
        self.is_high_band_sample = True
        self.has_second_jump = True
        self.is_jumping_init = False
        self.init_air_time = 0.0
        self.init_prev_act = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._dir5_deck = []
        self._vel_dir4_deck = []

        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.ball = self.arena.ball
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
        self.last_boost_used = 0.0
        self.last_boost_pickups = 0
        self.last_flip_triggered = False
        self.last_djump_triggered = False
        self.last_jump_held = False
        self.last_handbrake_held = False
        self.prev_action_vec = np.zeros(6, dtype=np.float32)
        self.init_dist = 1000.0
        self.min_dist = 1000.0
        self.ep_max_decisions = MAX_DECISIONS

    @property
    def curriculum_step_idx(self) -> int:
        return self.current_step_idx

    @property
    def current_dir_idx(self) -> int:
        return self.current_dir5_idx

    def set_spawn_dist(self, spec: str):
        self.spawn_dist_cfg = _SpawnDistCfg(spec or "")

    def sample_spawn(
        self,
        mode="curriculum",
        curriculum_step_idx: int = None,
        force_pool: int = None,
        force_dir_idx: int = None,
        rng=random,
    ):
        step_i = self.current_step_idx if curriculum_step_idx is None else int(curriculum_step_idx)
        spec = get_curriculum_step_spec(step_i)
        d_idx = self._next_dir5_from_deck(rng=rng) if force_dir_idx is None else int(force_dir_idx)
        pool_id = POOL_FRONTIER if force_pool is None else int(force_pool)
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
            rng=rng,
        )
        sd = {
            "mode": str(mode),
            "step_idx": int(spec.step_idx),
            "stage_idx": int(cfg_dict["dir5_idx"]),
            "pool_id": int(pool_id),
            "dir_idx": int(cfg_dict["dir5_idx"]),
            "dir5_idx": int(cfg_dict["dir5_idx"]),
            "dir6_idx": int(cfg_dict["dir5_idx"]),
            "is_hover_ball": True,
            "is_high_band_sample": (pool_id == POOL_FRONTIER),
            "on_ground": bool(cfg_dict["on_ground"]),
            "is_jumping": bool(cfg_dict["is_jumping"]),
            "has_second_jump": bool(cfg_dict["has_second_jump"]),
            "air_time": float(cfg_dict["air_time"]),
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
        return sd

    def set_curriculum_state(
        self,
        stage_level: int = 1,
        stage_rates=None,
        step_idx: int = None,
        **kwargs,
    ):
        if step_idx is not None:
            self.current_step_idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(step_idx)))
        else:
            self.current_step_idx = stage_level_to_step_idx(stage_level)
        spec = get_curriculum_step_spec(self.current_step_idx)
        self.stage_level = max(1, min(NUM_CURRICULUM_STAGES, spec.phase))
        self.curriculum_state.step_idx = self.current_step_idx
        self.curriculum_state.phase = spec.phase
        if stage_rates is not None:
            self.stage_rates = [float(x) for x in stage_rates[:NUM_CURRICULUM_STAGES]]

    def _compute_spawn_id(self, sd):
        key_str = (
            f"{sd.get('mode','a')}_{sd.get('step_idx',0)}_"
            f"{sd['car_pos'][0]:.1f}_{sd['car_pos'][1]:.1f}_{sd['car_pos'][2]:.1f}_"
            f"{sd['car_fwd'][0]:.3f}_{sd['car_fwd'][1]:.3f}_"
            f"{sd['ball_pos'][0]:.1f}_{sd['ball_pos'][1]:.1f}_{sd['ball_pos'][2]:.1f}"
        )
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()[:4]

    def _capture_spawn_state(self, use_mode):
        cs = self.car.get_state()
        bs = self.arena.ball.get_state()
        sd = {
            "mode": str(use_mode),
            "step_idx": int(self.current_step_idx),
            "stage_idx": int(self.current_stage_idx),
            "pool_id": int(self.current_pool_id),
            "dir5_idx": int(self.current_dir5_idx),
            "dir6_idx": int(self.current_dir5_idx),
            "dist_tier": int(self.current_dist_tier),
            "is_hover_ball": bool(self.is_hover_ball),
            "is_high_band_sample": bool(self.is_high_band_sample),
            "on_ground": bool(cs.is_on_ground),
            "is_jumping": bool(self.is_jumping_init),
            "has_second_jump": bool(self.has_second_jump),
            "air_time": float(self.init_air_time),
            "init_prev_act": tuple(float(x) for x in self.init_prev_act),
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
        }
        sd["spawn_id"] = self._compute_spawn_id(sd)
        return sd

    def _reset_boost_pads(self):
        """All 34 boost pads are always normal (active with 0 cooldown on reset)."""
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

        self.current_step_idx = int(sd.get("step_idx", self.current_step_idx))
        self.current_stage_idx = int(sd.get("stage_idx", 0))
        self.current_pool_id = int(sd.get("pool_id", POOL_FRONTIER))
        self.is_hover_ball = bool(sd.get("is_hover_ball", True))
        self.is_high_band_sample = bool(sd.get("is_high_band_sample", self.current_pool_id == POOL_FRONTIER))
        self.hover_ball_pos = tuple(sd["ball_pos"]) if self.is_hover_ball else None
        self.has_second_jump = bool(sd.get("has_second_jump", True))
        self.is_jumping_init = bool(sd.get("is_jumping", False))
        self.init_air_time = float(sd.get("air_time", 0.20))

        raw_prev = sd.get("init_prev_act")
        if raw_prev is not None and len(raw_prev) >= 6:
            self.init_prev_act = tuple(float(x) for x in raw_prev[:6])
        else:
            self.init_prev_act = tuple(float(x) for x in sample_random_prev_action())

        on_ground = bool(sd.get("on_ground", (float(sd["car_pos"][2]) <= 20.0 and float(sd["car_up"][2]) >= 0.95)))

        if on_ground and GROUND_SETTLE_TICKS > 0:
            # Place ball high out of reach while settling the ground car's raycast suspension for 45 ticks
            bs_tmp = self.arena.ball.get_state()
            bs_tmp.pos = rs.Vec(0.0, 0.0, 1950.0)
            bs_tmp.vel = rs.Vec(0.0, 0.0, 0.0)
            bs_tmp.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            self.arena.ball.set_state(bs_tmp)

            cs_settle = self.car.get_state()
            cs_settle.pos = rs.Vec(float(sd["car_pos"][0]), float(sd["car_pos"][1]), 17.0)
            cs_settle.vel = rs.Vec(0.0, 0.0, 0.0)
            cs_settle.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            cs_settle.rot_mat.forward = rs.Vec(*sd["car_fwd"])
            cs_settle.rot_mat.right   = rs.Vec(*sd["car_rgt"])
            cs_settle.rot_mat.up      = rs.Vec(*sd["car_up"])
            cs_settle.boost = float(sd["car_boost"])
            apply_car_jump_state(cs_settle, on_ground=True, has_second_jump=True)
            self.car.set_state(cs_settle)
            self.car.set_controls(rs.CarControls())
            self.arena.step(GROUND_SETTLE_TICKS)
            self._touched = False
            self._boost_pickups_tick = 0

        self._reset_boost_pads()

        cs = self.car.get_state()
        settled_z = float(cs.pos.z) if on_ground else float(sd["car_pos"][2])
        cs.pos = rs.Vec(float(sd["car_pos"][0]), float(sd["car_pos"][1]), settled_z)
        cs.vel = rs.Vec(*sd["car_vel"])
        cs.ang_vel = rs.Vec(*sd["car_ang"])
        cs.rot_mat.forward = rs.Vec(*sd["car_fwd"])
        cs.rot_mat.right   = rs.Vec(*sd["car_rgt"])
        cs.rot_mat.up      = rs.Vec(*sd["car_up"])
        cs.boost = float(sd["car_boost"])
        apply_car_jump_state(
            cs,
            on_ground=on_ground,
            has_second_jump=self.has_second_jump,
            is_jumping=self.is_jumping_init,
            air_time=self.init_air_time,
            init_prev_act=self.init_prev_act,
        )
        self.car.set_state(cs)

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(*sd["ball_pos"])
        bs.vel = rs.Vec(*sd.get("ball_vel", (0.0, 0.0, 0.0)))
        bs.ang_vel = rs.Vec(*sd.get("ball_ang", (0.0, 0.0, 0.0)))
        bs.rot_mat.forward = rs.Vec(1.0, 0.0, 0.0)
        bs.rot_mat.right   = rs.Vec(0.0, 1.0, 0.0)
        bs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
        self.arena.ball.set_state(bs)

        dx = float(sd["ball_pos"][0] - cs.pos.x)
        dy = float(sd["ball_pos"][1] - cs.pos.y)
        dz = float(sd["ball_pos"][2] - cs.pos.z)
        fwd = sd["car_fwd"]
        rgt = sd["car_rgt"]
        up  = sd["car_up"]
        if "dir5_idx" in sd:
            self.current_dir5_idx = int(sd["dir5_idx"])
        elif "dir6_idx" in sd:
            self.current_dir5_idx = int(sd["dir6_idx"])
        else:
            self.current_dir5_idx = classify_local_dir5(
                dx * fwd[0] + dy * fwd[1] + dz * fwd[2],
                dx * rgt[0] + dy * rgt[1] + dz * rgt[2],
                dx * up[0]  + dy * up[1]  + dz * up[2],
            )
        self.current_dir6_idx = self.current_dir5_idx

        self.spawn_id = sd.get("spawn_id") or self._compute_spawn_id(sd)
        self.prev_action_vec = format_prev_action_vec(self.init_prev_act)

        d0 = math.sqrt(dx * dx + dy * dy + dz * dz)
        self.init_dist = max(1.0, d0)
        self.min_dist = self.init_dist
        self.current_dist_tier = int(sd.get("dist_tier", dist_to_tier(d0)))
        c_spd = math.hypot(float(sd["car_vel"][0]), float(sd["car_vel"][1]), float(sd["car_vel"][2]))
        self.ep_max_decisions = get_adaptive_max_decisions(d0, float(sd["ball_pos"][2]), MAX_DECISIONS, car_spd=c_spd)

    def _lock_hover_ball_if_needed(self):
        if self.is_hover_ball and (not self._touched) and self.hover_ball_pos is not None:
            bs = self.arena.ball.get_state()
            bs.pos = rs.Vec(*self.hover_ball_pos)
            bs.vel = rs.Vec(0.0, 0.0, 0.0)
            bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            self.arena.ball.set_state(bs)

    def step_tick(self, ctrl):
        cs_before = self.car.get_state()
        prev_boost = float(cs_before.boost)
        was_flipping = bool(cs_before.is_flipping)
        was_djumped = bool(cs_before.has_double_jumped)
        was_on_ground = bool(cs_before.is_on_ground)

        # In air, handbrake (powerslide) has no physical effect — enforce clean 0
        if not was_on_ground:
            ctrl.handbrake = False

        self._boost_pickups_tick = 0
        self._lock_hover_ball_if_needed()

        self.car.set_controls(ctrl)
        self.arena.step(1)

        self._lock_hover_ball_if_needed()

        cs_after = self.car.get_state()
        cur_boost = float(cs_after.boost)

        bs_after = self.arena.ball.get_state()
        cur_dist = math.hypot(
            float(bs_after.pos.x - cs_after.pos.x),
            float(bs_after.pos.y - cs_after.pos.y),
            float(bs_after.pos.z - cs_after.pos.z),
        )
        if cur_dist < self.min_dist:
            self.min_dist = cur_dist

        now_flipping = bool(cs_after.is_flipping)
        now_djumped = bool(cs_after.has_double_jumped)

        self.prev_action_vec = np.array([
            float(ctrl.throttle if abs(ctrl.throttle) >= abs(ctrl.pitch) else -ctrl.pitch),
            float(ctrl.steer if abs(ctrl.steer) >= abs(ctrl.yaw) else ctrl.yaw),
            float(ctrl.roll),
            1.0 if ctrl.jump else 0.0,
            1.0 if ctrl.boost else 0.0,
            1.0 if ctrl.handbrake else 0.0,
        ], dtype=np.float32)

        boost_used = max(0.0, prev_boost - cur_boost) if ctrl.boost else 0.0
        flip_triggered = (not was_flipping) and now_flipping
        djump_triggered = (not was_djumped) and now_djumped
        jump_held = bool(ctrl.jump)
        handbrake_held = bool(ctrl.handbrake) and was_on_ground

        self.last_boost_used = boost_used
        self.last_boost_pickups = self._boost_pickups_tick
        self.last_flip_triggered = flip_triggered
        self.last_djump_triggered = djump_triggered
        self.last_jump_held = jump_held
        self.last_handbrake_held = handbrake_held
        return boost_used, flip_triggered, djump_triggered

    def reset(self, mode=None, replay_last=False, spawn_state=None, stage_override=None, **kwargs):
        if spawn_state is not None:
            self.last_spawn_state = dict(spawn_state)
            if "spawn_id" not in self.last_spawn_state:
                self.last_spawn_state["spawn_id"] = self._compute_spawn_id(self.last_spawn_state)
            self._apply_spawn_state(self.last_spawn_state)
        elif replay_last and self.last_spawn_state is not None:
            self._apply_spawn_state(self.last_spawn_state)
        else:
            use_mode = mode or self.mode
            self.arena.remove_car(self.car)
            self.car = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
            self._touched = False
            self._boost_pickups_tick = 0

            self.is_hover_ball = True
            self.hover_ball_pos = None
            self.is_high_band_sample = True

            if stage_override is not None:
                from .curriculum import PHASE_FIRST_STEP_IDX
                p_num = max(1, min(5, int(stage_override) + 1))
                t_step = PHASE_FIRST_STEP_IDX.get(p_num, 0)
                self._randomize_curriculum(step_idx=t_step)
                use_mode = f"phase_{p_num}"
            elif use_mode in ("all", "mixed"):
                self._randomize_curriculum(step_idx=NUM_CURRICULUM_STEPS - 1)
                use_mode = "all"
            else:
                self._randomize_curriculum(step_idx=self.current_step_idx)
                use_mode = f"step_{self.current_step_idx}"

            self.last_spawn_state = self._capture_spawn_state(use_mode)
            self._apply_spawn_state(self.last_spawn_state)

        obs_full, obs_panel, parts = build_obs(
            self.car, self.arena,
            prev_action_vec=self.prev_action_vec,
        )
        self.last_obs = obs_full
        self.last_obs_v3 = obs_full
        self.last_obs_panel = obs_panel
        self.last_parts = parts
        dist = float(parts["D.ball_rel_b"][6])
        return obs_panel, dist

    def get_obs(self):
        obs_full, obs_panel, parts = build_obs(
            self.car, self.arena,
            prev_action_vec=self.prev_action_vec,
        )
        self.last_obs = obs_full
        self.last_obs_v3 = obs_full
        self.last_obs_panel = obs_panel
        self.last_parts = parts
        return obs_full, obs_panel, parts

    def get_obs_v3(self):
        return self.get_obs()

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
        b_fwd = float(dx * fwd.x + dy * fwd.y + dz * fwd.z)
        b_rgt = float(dx * right.x + dy * right.y + dz * right.z)
        b_up  = float(dx * up.x + dy * up.y + dz * up.z)
        dist = float(np.linalg.norm(rel))
        if dist < self.min_dist:
            self.min_dist = dist
        inv_d = 1.0 / max(dist, 1e-6)
        dir_ball = (b_fwd * inv_d, b_rgt * inv_d, b_up * inv_d)
        dist_h = float(np.hypot(dx, dy))
        car_vel_np = np.array([cs.vel.x, cs.vel.y, cs.vel.z])
        closing_speed = float(np.dot(rel, car_vel_np)) * inv_d
        fwd_h = np.array([fwd.x, fwd.y])
        d_h = np.array([dx, dy])
        n = np.linalg.norm(d_h) + 1e-6
        align = float(np.dot(fwd_h, d_h / n))

        car_speed = float(math.sqrt(cs.vel.x**2 + cs.vel.y**2 + cs.vel.z**2))

        return {
            "car_pos": (cs.pos.x, cs.pos.y, cs.pos.z),
            "car_vel": (cs.vel.x, cs.vel.y, cs.vel.z),
            "car_ang": (cs.ang_vel.x, cs.ang_vel.y, cs.ang_vel.z),
            "car_speed": car_speed,
            "is_supersonic": bool(cs.is_supersonic),
            "ball_pos": (bs.pos.x, bs.pos.y, bs.pos.z),
            "ball_vel": (bs.vel.x, bs.vel.y, bs.vel.z),
            "up_z": up.z,
            "fwd":   (fwd.x,   fwd.y,   fwd.z),
            "up":    (up.x,    up.y,    up.z),
            "right": (right.x, right.y, right.z),
            "on_ground": bool(cs.is_on_ground),
            "is_flipping": bool(cs.is_flipping),
            "is_jumping": bool(cs.is_jumping),
            "has_djump": bool(cs.has_double_jumped),
            "air_time": float(cs.air_time),
            "boost": float(cs.boost),
            "dist": dist,
            "init_dist": self.init_dist,
            "min_dist": self.min_dist,
            "dir_ball": dir_ball,
            "dist_h": dist_h,
            "align": align,
            "closing": closing_speed,
            "touched": self._touched,
            "step_idx": self.current_step_idx,
            "stage_idx": self.current_stage_idx,
            "pool_id": self.current_pool_id,
            "dir5_idx": self.current_dir5_idx,
            "dir6_idx": self.current_dir5_idx,
            "dist_tier": self.current_dist_tier,
            "is_high_band_sample": self.is_high_band_sample,
            "ep_max_decisions": self.ep_max_decisions,
        }
