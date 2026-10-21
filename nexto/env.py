"""nexto/env.py —— 支持任意数量 (N vs M) 车辆的 RocketSim 标准开球对战环境 (NextoMatchEnv)。"""
import math
import random
import numpy as np
import RocketSim as rs

from core.constants import MAX_DECISIONS, NUM_CURRICULUM_STAGES, NUM_CURRICULUM_STEPS
from core.obs import build_obs, format_prev_action_vec
from core.spawner import apply_car_jump_state

# Rocket League 5 个标准开球位 (以蓝队 team=0 为基准，面朝 +Y 橙方球门) + 4 个超编后排扩展位 (支持单队最多 9 辆车)
BLUE_KICKOFF_SLOTS = (
    (-2048.0, -2560.0, 17.0, 0.25 * math.pi),  # 0: 右斜位 (Diag Right)
    ( 2048.0, -2560.0, 17.0, 0.75 * math.pi),  # 1: 左斜位 (Diag Left)
    ( -256.0, -3840.0, 17.0, 0.50 * math.pi),  # 2: 右中后 (Back Right)
    (  256.0, -3840.0, 17.0, 0.50 * math.pi),  # 3: 左中后 (Back Left)
    (    0.0, -4608.0, 17.0, 0.50 * math.pi),  # 4: 门前正中 (Goalie Center)
    (-2304.0, -4608.0, 17.0, 0.50 * math.pi),  # 5: 扩展后排右 1
    ( 2304.0, -4608.0, 17.0, 0.50 * math.pi),  # 6: 扩展后排左 1
    (-2688.0, -4608.0, 17.0, 0.50 * math.pi),  # 7: 扩展后排右 2
    ( 2688.0, -4608.0, 17.0, 0.50 * math.pi),  # 8: 扩展后排左 2
)


class _DummySpawnDistCfg:
    raw = "kickoff"
    is_active = False


class NextoMatchEnv:
    """支持任意蓝/橙队人数 (如 3v3, 6v2, 1v1 等) 的多车实时对抗环境。"""

    def __init__(self, n_blue: int = 3, n_orange: int = 3, seed: int = None):
        self.n_blue = max(0, int(n_blue))
        self.n_orange = max(0, int(n_orange))
        if self.n_blue + self.n_orange < 1:
            self.n_blue = 1
        self.n_cars = self.n_blue + self.n_orange
        self._rng = random.Random(seed)

        # Compatibility attributes for Tuner UI / HUD
        self.mode = f"nexto_{self.n_blue}v{self.n_orange}"
        self.stage_level = NUM_CURRICULUM_STAGES
        self.current_step_idx = NUM_CURRICULUM_STEPS - 1
        self.curriculum_step_idx = self.current_step_idx
        self.current_stage_idx = 0
        self.current_dir_idx = 0
        self.current_dir5_idx = 0
        self.current_dist_tier = 1
        self.ep_max_decisions = MAX_DECISIONS
        self.current_max_decisions = MAX_DECISIONS
        self.spawn_dist_cfg = _DummySpawnDistCfg()
        self.has_second_jump = True
        self.last_spawn_state = None
        self.target_pos = None
        self.focus_cfg = None

        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.ball = self.arena.ball
        self.pads = self.arena.get_boost_pads()
        self.num_pads = len(self.pads)

        self.cars = []
        self.car_teams = []
        self.car_labels = []
        self.car_id_to_idx = {}

        for i in range(self.n_blue):
            c = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
            idx = len(self.cars)
            self.cars.append(c)
            self.car_teams.append(0)
            self.car_labels.append(f"#{idx + 1} BLUE-{i + 1}")
            self.car_id_to_idx[c.id] = idx

        for i in range(self.n_orange):
            c = self.arena.add_car(rs.Team.ORANGE, rs.CarConfig.OCTANE)
            idx = len(self.cars)
            self.cars.append(c)
            self.car_teams.append(1)
            self.car_labels.append(f"#{idx + 1} ORANGE-{i + 1}")
            self.car_id_to_idx[c.id] = idx

        self.car_teams = np.array(self.car_teams, dtype=np.int32)
        self.focused_car_idx = 0

        # Scoreboard & telemetry state
        self.blue_score = 0
        self.orange_score = 0
        self.goal_scored_this_tick = False
        self.last_goal_team = None
        self.kickoff_Pause_ticks = 0

        self._touched_by_car = [False] * self.n_cars
        self._boost_pickups_by_car = [0] * self.n_cars
        self.prev_action_vecs = [np.zeros(6, dtype=np.float32) for _ in range(self.n_cars)]
        self.init_dists = [1000.0] * self.n_cars
        self.min_dists = [1000.0] * self.n_cars

        # Compatibility single-car telemetry for the currently focused car
        self._touched = False
        self.last_boost_used = 0.0
        self.last_boost_pickups = 0
        self.last_flip_triggered = False
        self.last_djump_triggered = False
        self.last_jump_held = False
        self.last_handbrake_held = False
        self.last_dodge_delta_closing = 0.0
        self.last_dodge_delta_vz = 0.0
        self.last_flip_is_diagonal = False
        self.last_thr_input = 0.0
        self.init_dist = 1000.0
        self.min_dist = 1000.0
        self.last_obs = None
        self.last_obs_panel = None
        self.last_parts = None

        def _on_touch(arena, car, data=None, *args, **kwargs):
            idx = self.car_id_to_idx.get(car.id)
            if idx is not None:
                self._touched_by_car[idx] = True
            self._touched = True

        def _on_boost_pickup(arena, car, boost_pad=None, data=None, *args, **kwargs):
            idx = self.car_id_to_idx.get(car.id)
            if idx is not None:
                self._boost_pickups_by_car[idx] += 1

        def _on_goal(arena, scoring_team, data=None, *args, **kwargs):
            self.goal_scored_this_tick = True
            self.last_goal_team = int(scoring_team)
            if int(scoring_team) == 0:
                self.blue_score += 1
            else:
                self.orange_score += 1

        self.arena.set_ball_touch_callback(_on_touch)
        self.arena.set_boost_pickup_callback(_on_boost_pickup)
        try:
            self.arena.set_goal_score_callback(_on_goal)
        except Exception:
            pass

        self.is_nexto_env = True
        self.reset()

    @property
    def score_blue(self) -> int:
        return self.blue_score

    @property
    def score_orange(self) -> int:
        return self.orange_score

    @property
    def car(self):
        """返回当前聚焦车辆，保持与单车组件 (如 HUD / Camera / CarStatePanel) 的零耦合兼容。"""
        return self.cars[self.focused_car_idx]

    def select_car(self, index_0_based: int) -> int:
        """选择 0..n_cars-1 的车辆；若超出当前车辆总数，自动落到最后一辆车。"""
        if self.n_cars <= 0:
            return 0
        self.focused_car_idx = max(0, min(int(index_0_based), self.n_cars - 1))
        self.get_obs(self.focused_car_idx)
        return self.focused_car_idx

    def cycle_car(self, delta: int) -> int:
        """切换到上一辆 (delta=-1, 键 9) 或下一辆 (delta=+1, 键 0) 车辆。"""
        if self.n_cars <= 0:
            return 0
        self.focused_car_idx = (self.focused_car_idx + int(delta)) % self.n_cars
        self.get_obs(self.focused_car_idx)
        return self.focused_car_idx

    def car_label(self, idx: int = None) -> str:
        target_idx = self.focused_car_idx if idx is None else int(idx)
        if 0 <= target_idx < len(self.car_labels):
            return self.car_labels[target_idx]
        return "#1"

    def focused_car_label(self) -> str:
        return self.car_label(self.focused_car_idx)

    def set_curriculum_state(self, *args, **kwargs):
        pass

    def _reset_boost_pads(self):
        for p in self.pads:
            ps = p.get_state()
            ps.is_active = True
            ps.cooldown = 0.0
            p.set_state(ps)

    def reset_random_kickoff(self):
        """重置球场为随机标准 Kickoff Spawn 阵型，支持任意蓝橙队人数。"""
        seed_val = self._rng.randint(0, 1_000_000)
        self.arena.reset_kickoff(seed_val)
        self._reset_boost_pads()

        # Reset ball to exact center kickoff
        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(0.0, 0.0, 93.15)
        bs.vel = rs.Vec(0.0, 0.0, 0.0)
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        # Permute the 5 standard kickoff slots + 4 overflow slots
        std_indices = [0, 1, 2, 3, 4]
        self._rng.shuffle(std_indices)
        overflow_indices = [5, 6, 7, 8]
        self._rng.shuffle(overflow_indices)
        slot_order = std_indices + overflow_indices

        blue_cars = [c for c, t in zip(self.cars, self.car_teams) if t == 0]
        orange_cars = [c for c, t in zip(self.cars, self.car_teams) if t == 1]

        blue_slots = [slot_order[i % len(slot_order)] for i in range(len(blue_cars))]
        orange_slots = [slot_order[i % len(slot_order)] for i in range(len(orange_cars))]
        self._rng.shuffle(blue_slots)
        self._rng.shuffle(orange_slots)

        def _place_team_cars(team_cars, team_slot_indices, is_orange: bool):
            sign = -1.0 if is_orange else 1.0
            yaw_offset = math.pi if is_orange else 0.0
            for idx_in_team, (c, slot_i) in enumerate(zip(team_cars, team_slot_indices)):
                sx, sy, sz, syaw = BLUE_KICKOFF_SLOTS[slot_i]
                # If > 9 cars on one team, stagger slightly along X
                extra_tier = idx_in_team // len(BLUE_KICKOFF_SLOTS)
                x_pos = sign * (sx + extra_tier * 320.0)
                y_pos = sign * sy
                yaw = syaw + yaw_offset
                fx, fy = math.cos(yaw), math.sin(yaw)
                # Left-handed Unreal/RocketSim basis: when facing +Y (fx=0, fy=1), right is (-1, 0, 0)
                rx, ry = -fy, fx

                cs = c.get_state()
                cs.pos = rs.Vec(float(x_pos), float(y_pos), float(sz))
                cs.vel = rs.Vec(0.0, 0.0, 0.0)
                cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
                cs.rot_mat.forward = rs.Vec(float(fx), float(fy), 0.0)
                cs.rot_mat.right   = rs.Vec(float(rx), float(ry), 0.0)
                cs.rot_mat.up      = rs.Vec(0.0, 0.0, 1.0)
                cs.boost = 33.333
                cs.is_demoed = False
                cs.demo_respawn_timer = 0.0
                apply_car_jump_state(cs, on_ground=True, has_second_jump=True)
                c.set_state(cs)
                c.set_controls(rs.CarControls())

        _place_team_cars(blue_cars, blue_slots, is_orange=False)
        _place_team_cars(orange_cars, orange_slots, is_orange=True)

        self.goal_scored_this_tick = False
        self._touched = False
        for i, c in enumerate(self.cars):
            self._touched_by_car[i] = False
            self._boost_pickups_by_car[i] = 0
            self.prev_action_vecs[i] = np.zeros(6, dtype=np.float32)
            cs = c.get_state()
            d0 = math.hypot(float(cs.pos.x), float(cs.pos.y), float(93.15 - cs.pos.z))
            self.init_dists[i] = max(1.0, d0)
            self.min_dists[i] = self.init_dists[i]

        self.init_dist = self.init_dists[self.focused_car_idx]
        self.min_dist = self.min_dists[self.focused_car_idx]

    def reset(self, mode=None, replay_last=False, spawn_state=None, **kwargs):
        self.reset_random_kickoff()
        _, obs_panel, parts = self.get_obs(self.focused_car_idx)
        dist = float(parts["D.ball_rel_b"][6])
        return obs_panel, dist

    def step_all_cars(self, controls_list: list):
        """对全场所有车辆执行 1 物理步 (1/120s)，返回每辆车的 (boost_used, flip_triggered, djump_triggered, pickups) 列表。"""
        pre_boost = []
        pre_flip = []
        pre_djump = []
        pre_gnd = []

        for i, c in enumerate(self.cars):
            cs = c.get_state()
            pre_boost.append(float(cs.boost))
            pre_flip.append(bool(cs.is_flipping))
            pre_djump.append(bool(cs.has_double_jumped))
            pre_gnd.append(bool(cs.is_on_ground))
            self._touched_by_car[i] = False
            self._boost_pickups_by_car[i] = 0
            ctrl = controls_list[i] if i < len(controls_list) else rs.CarControls()
            c.set_controls(ctrl)

        self.goal_scored_this_tick = False
        self._touched = False
        self.arena.step(1)

        bs = self.arena.ball.get_state()
        bx, by, bz = float(bs.pos.x), float(bs.pos.y), float(bs.pos.z)
        # Fallback goal check if goal callback wasn't triggered by Arena
        if not self.goal_scored_this_tick and abs(by) > 5120.0 + 90.0:
            self.goal_scored_this_tick = True
            if by > 0.0:
                self.blue_score += 1
                self.last_goal_team = 0
            else:
                self.orange_score += 1
                self.last_goal_team = 1

        step_results = []
        for i, c in enumerate(self.cars):
            ctrl = controls_list[i] if i < len(controls_list) else rs.CarControls()
            cs = c.get_state()
            cur_boost = float(cs.boost)
            now_flip = bool(cs.is_flipping)
            now_djump = bool(cs.has_double_jumped)

            b_used = max(0.0, pre_boost[i] - cur_boost) if ctrl.boost else 0.0
            f_trig = (not pre_flip[i]) and now_flip
            dj_trig = (not pre_djump[i]) and now_djump
            pickups = self._boost_pickups_by_car[i]

            d_now = math.hypot(bx - float(cs.pos.x), by - float(cs.pos.y), bz - float(cs.pos.z))
            if d_now < self.min_dists[i]:
                self.min_dists[i] = d_now

            self.prev_action_vecs[i] = format_prev_action_vec([
                float(ctrl.throttle if abs(ctrl.throttle) >= abs(ctrl.pitch) else -ctrl.pitch),
                float(ctrl.steer if abs(ctrl.steer) >= abs(ctrl.yaw) else ctrl.yaw),
                float(ctrl.roll),
                1.0 if ctrl.jump else 0.0,
                1.0 if ctrl.boost else 0.0,
                1.0 if ctrl.handbrake else 0.0,
            ])

            if i == self.focused_car_idx:
                self.last_boost_used = b_used
                self.last_boost_pickups = pickups
                self.last_flip_triggered = f_trig
                self.last_djump_triggered = dj_trig
                self.last_jump_held = bool(ctrl.jump)
                self.last_handbrake_held = bool(ctrl.handbrake) and pre_gnd[i]
                self.init_dist = self.init_dists[i]
                self.min_dist = self.min_dists[i]

            step_results.append((b_used, f_trig, dj_trig, pickups))

        return step_results

    def get_obs(self, car_idx: int = None):
        idx = self.focused_car_idx if car_idx is None else max(0, min(int(car_idx), self.n_cars - 1))
        obs_full, obs_panel, parts = build_obs(
            self.cars[idx],
            self.arena,
            prev_action_vec=self.prev_action_vecs[idx],
        )
        if idx == self.focused_car_idx:
            self.last_obs = obs_full
            self.last_obs_panel = obs_panel
            self.last_parts = parts
        return obs_full, obs_panel, parts

    def snapshot_car(self, idx: int):
        idx = max(0, min(int(idx), self.n_cars - 1))
        c = self.cars[idx]
        cs = c.get_state()
        bs = self.arena.ball.get_state()
        up = c.get_up_dir()
        fwd = c.get_forward_dir()
        right = c.get_right_dir()

        dx = float(bs.pos.x - cs.pos.x)
        dy = float(bs.pos.y - cs.pos.y)
        dz = float(bs.pos.z - cs.pos.z)
        dist = float(math.hypot(dx, dy, dz))
        if dist < self.min_dists[idx]:
            self.min_dists[idx] = dist
        inv_d = 1.0 / max(dist, 1e-6)
        b_fwd = float(dx * fwd.x + dy * fwd.y + dz * fwd.z)
        b_rgt = float(dx * right.x + dy * right.y + dz * right.z)
        b_up  = float(dx * up.x + dy * up.y + dz * up.z)
        dir_ball = (b_fwd * inv_d, b_rgt * inv_d, b_up * inv_d)
        dist_h = float(math.hypot(dx, dy))
        closing_speed = float(dx * cs.vel.x + dy * cs.vel.y + dz * cs.vel.z) * inv_d
        n_h = math.hypot(dx, dy) + 1e-6
        align = float((fwd.x * dx + fwd.y * dy) / n_h)
        car_speed = float(math.hypot(cs.vel.x, cs.vel.y, cs.vel.z))
        has_flip_or_jump = bool(cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump)

        return {
            "car_idx": idx,
            "team": int(self.car_teams[idx]),
            "label": self.car_labels[idx],
            "car_pos": (float(cs.pos.x), float(cs.pos.y), float(cs.pos.z)),
            "car_vel": (float(cs.vel.x), float(cs.vel.y), float(cs.vel.z)),
            "car_ang": (float(cs.ang_vel.x), float(cs.ang_vel.y), float(cs.ang_vel.z)),
            "car_speed": car_speed,
            "is_supersonic": bool(cs.is_supersonic),
            "is_demoed": bool(cs.is_demoed),
            "ball_pos": (float(bs.pos.x), float(bs.pos.y), float(bs.pos.z)),
            "ball_vel": (float(bs.vel.x), float(bs.vel.y), float(bs.vel.z)),
            "up_z": float(up.z),
            "fwd":   (float(fwd.x),   float(fwd.y),   float(fwd.z)),
            "up":    (float(up.x),    float(up.y),    float(up.z)),
            "right": (float(right.x), float(right.y), float(right.z)),
            "on_ground": bool(cs.is_on_ground),
            "is_flipping": bool(cs.is_flipping),
            "is_jumping": bool(cs.is_jumping),
            "has_djump": bool(cs.has_double_jumped),
            "has_flip_or_jump": has_flip_or_jump,
            "air_time": float(cs.air_time),
            "flip_time": float(cs.flip_time),
            "boost": float(cs.boost),
            "dist": dist,
            "init_dist": self.init_dists[idx],
            "min_dist": self.min_dists[idx],
            "dir_ball": dir_ball,
            "dist_h": dist_h,
            "align": align,
            "closing": closing_speed,
            "last_dodge_delta_closing": 0.0,
            "last_dodge_delta_vz": 0.0,
            "last_flip_is_diagonal": False,
            "last_thr_input": 1.0,
            "touched": bool(self._touched_by_car[idx]),
            "step_idx": self.current_step_idx,
            "stage_idx": 0,
            "pool_id": 0,
            "dir5_idx": 0,
            "dir6_idx": 0,
            "dist_tier": 1,
            "is_high_band_sample": True,
            "ep_max_decisions": MAX_DECISIONS,
        }

    def snapshot(self):
        return self.snapshot_car(self.focused_car_idx)

    def snapshot_all_cars(self):
        return [self.snapshot_car(i) for i in range(self.n_cars)]
