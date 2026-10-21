"""core/env.py —— KickoffEnv (BallReachEnv): RocketSim 原生开球重置 + 60 Frames 悬挂稳定缓冲 + 1v1 双车对抗环境。"""
import math
import random
import hashlib
import numpy as np
import RocketSim as rs

from .constants import (
    NUM_KICKOFF_SLOTS, KICKOFF_SLOT_NAMES, KICKOFF_SLOT_SHORT, KICKOFF_SLOT_SEEDS,
    MAX_DECISIONS, SETTLE_TICKS,
)
from .obs import build_obs, format_prev_action_vec

STAGE_NAMES = KICKOFF_SLOT_NAMES
STAGE_SHORT_LABELS = KICKOFF_SLOT_SHORT


class _SpawnDistCfg:
    def __init__(self, raw: str = ""):
        self.raw = (raw or "").strip()

    @property
    def is_active(self) -> bool:
        return bool(self.raw and self.raw.lower() not in ("auto", "off", "none", "default", "kickoff", "all"))


def parse_slot_from_spec(spec: str):
    """解析开球位锁定配置，例如 'slot=0', '0', 'dr', 'dl', 'br', 'bl', 'ct'。返回 slot_idx (0..4) 或 None(全随机/轮询)。"""
    s = (spec or "").strip().lower()
    if not s or s in ("auto", "off", "none", "default", "kickoff", "all"):
        return None
    if s.startswith("slot="):
        s = s.split("=", 1)[1].strip()
    elif s.startswith("step="):
        s = s.split("=", 1)[1].strip()
    short_map = {
        "dr": 0, "diagright": 0, "right_diag": 0,
        "dl": 1, "diagleft": 1, "left_diag": 1,
        "br": 2, "backright": 2, "right_back": 2,
        "bl": 3, "backleft": 3, "left_back": 3,
        "ct": 4, "center": 4, "goalie": 4,
    }
    if s in short_map:
        return short_map[s]
    if s.isdigit():
        v = int(s)
        if 0 <= v < NUM_KICKOFF_SLOTS:
            return v
        if 1 <= v <= NUM_KICKOFF_SLOTS:
            return v - 1
    return None


class KickoffEnv:
    """1v1 标准开球竞速环境 (Blue vs Orange):
    - 重置时调用 `arena.reset_kickoff(seed)` 生成对称开球位；
    - 强制先执行 `SETTLE_TICKS = 60` 帧 (0.50s) 零输入物理步以彻底稳定悬挂 (Suspension Settling)；
    - 第 61 帧起作为 `tick = 0` 允许模型或人类输入；
    - 任意一方率先触球 (First-Touch) 或满 6.0 秒 (720 帧) 即结束当前回合。
    """

    def __init__(
        self,
        mode: str = "kickoff",
        slot_idx: int = None,
        spawn_dist: str = None,
        seed: int = None,
        **kwargs,
    ):
        self.mode = mode or "kickoff"
        self._rng = random.Random(seed)
        self.spawn_dist_cfg = _SpawnDistCfg(spawn_dist or "")
        parsed_slot = parse_slot_from_spec(spawn_dist) if spawn_dist else None
        self.locked_slot_idx = parsed_slot if parsed_slot is not None else (
            int(slot_idx) % NUM_KICKOFF_SLOTS if slot_idx is not None else None
        )
        self.fixed_slot_idx = self.locked_slot_idx
        self.settle_ticks = SETTLE_TICKS
        self.current_slot_idx = self.locked_slot_idx if self.locked_slot_idx is not None else 0
        self.last_slot_idx = None
        self._slot_deck = []

        # Compatibility attributes for Tuner / CLI
        self.stage_level = NUM_KICKOFF_SLOTS
        self.current_step_idx = self.current_slot_idx
        self.current_stage_idx = self.current_slot_idx
        self.current_dir5_idx = self.current_slot_idx
        self.current_dist_tier = 2
        self.ep_max_decisions = MAX_DECISIONS
        self.current_max_decisions = MAX_DECISIONS
        self.has_second_jump = True

        self.arena = rs.Arena(rs.GameMode.SOCCAR)
        self.ball = self.arena.ball
        self.car_blue = self.arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)
        self.car_orange = self.arena.add_car(rs.Team.ORANGE, rs.CarConfig.OCTANE)
        self.cars = [self.car_blue, self.car_orange]
        self.car_teams = [0, 1]
        self.car_labels = ["#1 BLUE", "#2 ORANGE"]
        self.n_blue = 1
        self.n_orange = 1
        self.n_cars = 2
        self.focused_car_idx = 0
        self.car_id_to_idx = {self.car_blue.id: 0, self.car_orange.id: 1}

        self.pads = self.arena.get_boost_pads()
        self.num_pads = len(self.pads)

        self._touched = False
        self._touched_by_car = [False, False]
        self.winner_car_idx = None  # 0=Blue won, 1=Orange won, -1=Tie, None=Ongoing/Timeout
        self._boost_pickups_by_car = [0, 0]
        self._boost_pickups_tick = 0

        def _on_touch(arena=None, car=None, data=None, *args, **kwargs):
            if car is not None:
                idx = self.car_id_to_idx.get(car.id)
                if idx is not None:
                    self._touched_by_car[idx] = True
            self._touched = True

        def _on_boost_pickup(arena=None, car=None, boost_pad=None, data=None, *args, **kwargs):
            if car is not None:
                idx = self.car_id_to_idx.get(car.id)
                if idx is not None:
                    self._boost_pickups_by_car[idx] += 1
                    if idx == self.focused_car_idx:
                        self._boost_pickups_tick += 1

        self.arena.set_ball_touch_callback(_on_touch)
        self.arena.set_boost_pickup_callback(_on_boost_pickup)

        self.prev_action_vecs = [np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)]
        self.init_dists = [1000.0, 1000.0]
        self.min_dists = [1000.0, 1000.0]

        self.last_spawn_state = None
        self.spawn_id = "0000"
        self.last_boost_used = 0.0
        self.last_boost_pickups = 0
        self.last_flip_triggered = False
        self.last_djump_triggered = False
        self.last_jump_held = False
        self.last_handbrake_held = False
        self.last_thr_input = 0.0
        self.init_dist = 1000.0
        self.min_dist = 1000.0
        self.last_obs = None
        self.last_obs_v3 = None
        self.last_obs_panel = None
        self.last_parts = None
        self.last_obs_pair = (None, None)

    @property
    def car(self):
        return self.cars[self.focused_car_idx]

    @property
    def opp_car(self):
        return self.cars[1 - self.focused_car_idx]

    @property
    def prev_action_vec(self):
        return self.prev_action_vecs[self.focused_car_idx]

    @prev_action_vec.setter
    def prev_action_vec(self, val):
        self.prev_action_vecs[self.focused_car_idx] = format_prev_action_vec(val)

    @property
    def curriculum_step_idx(self) -> int:
        return self.current_slot_idx

    @property
    def current_dir_idx(self) -> int:
        return self.current_slot_idx

    def select_car(self, index_0_based: int) -> int:
        self.focused_car_idx = max(0, min(int(index_0_based), self.n_cars - 1))
        self.get_obs(self.focused_car_idx)
        return self.focused_car_idx

    def cycle_car(self, delta: int) -> int:
        self.focused_car_idx = (self.focused_car_idx + int(delta)) % self.n_cars
        self.get_obs(self.focused_car_idx)
        return self.focused_car_idx

    def car_label(self, idx: int = None) -> str:
        target_idx = self.focused_car_idx if idx is None else int(idx)
        if 0 <= target_idx < len(self.car_labels):
            return self.car_labels[target_idx]
        return "#1 BLUE"

    def set_kickoff_slot(self, slot_idx):
        if slot_idx is None:
            self.locked_slot_idx = None
            self.fixed_slot_idx = None
        else:
            self.locked_slot_idx = int(slot_idx) % NUM_KICKOFF_SLOTS
            self.fixed_slot_idx = self.locked_slot_idx
            self.current_slot_idx = self.locked_slot_idx

    def set_spawn_dist(self, spec: str):
        self.spawn_dist_cfg = _SpawnDistCfg(spec or "")
        self.locked_slot_idx = parse_slot_from_spec(spec)
        self.fixed_slot_idx = self.locked_slot_idx
        if self.locked_slot_idx is not None:
            self.current_slot_idx = self.locked_slot_idx

    def set_curriculum_state(self, stage_level: int = 1, step_idx: int = None, **kwargs):
        if step_idx is not None:
            self.current_slot_idx = int(step_idx) % NUM_KICKOFF_SLOTS
            self.current_step_idx = self.current_slot_idx
            self.current_stage_idx = self.current_slot_idx
            self.current_dir5_idx = self.current_slot_idx

    def _next_slot_idx(self, cycle_sequential: bool = False) -> int:
        if self.locked_slot_idx is not None:
            return int(self.locked_slot_idx)
        if cycle_sequential and self.last_slot_idx is not None:
            return (int(self.last_slot_idx) + 1) % NUM_KICKOFF_SLOTS
        if not self._slot_deck:
            self._slot_deck = list(range(NUM_KICKOFF_SLOTS))
            self._rng.shuffle(self._slot_deck)
        if self.last_slot_idx is not None and len(self._slot_deck) > 1 and self._slot_deck[-1] == self.last_slot_idx:
            self._slot_deck[-1], self._slot_deck[0] = self._slot_deck[0], self._slot_deck[-1]
        nxt = self._slot_deck.pop()
        if self.last_slot_idx is not None and nxt == self.last_slot_idx and NUM_KICKOFF_SLOTS > 1:
            nxt = (nxt + 1) % NUM_KICKOFF_SLOTS
        return nxt

    def _compute_spawn_id(self, sd):
        slot_i = sd.get("slot_idx", 0)
        key_str = (
            f"ko_slot{slot_i}_"
            f"{sd['car_pos'][0]:.1f}_{sd['car_pos'][1]:.1f}_{sd['car_pos'][2]:.2f}"
        )
        return f"S{slot_i}_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:4]

    def _reset_kickoff_and_settle(self, slot_idx: int):
        """执行 RocketSim 原生 reset_kickoff(seed) 并空跑 60 frames 稳定悬挂。"""
        slot_idx = int(slot_idx) % NUM_KICKOFF_SLOTS
        self.current_slot_idx = slot_idx
        self.last_slot_idx = slot_idx
        self.current_step_idx = slot_idx
        self.current_stage_idx = slot_idx
        self.current_dir5_idx = slot_idx

        zero_ctrl = rs.CarControls()
        self.car_blue.set_controls(zero_ctrl)
        self.car_orange.set_controls(zero_ctrl)

        seed_val = KICKOFF_SLOT_SEEDS[slot_idx]
        self.arena.reset_kickoff(seed_val)
        ko_b = self.car_blue.get_state()
        ko_o = self.car_orange.get_state()
        ko_bx, ko_by, ko_f_b, ko_r_b, ko_u_b = (
            float(ko_b.pos.x), float(ko_b.pos.y),
            ko_b.rot_mat.forward, ko_b.rot_mat.right, ko_b.rot_mat.up,
        )
        ko_ox, ko_oy, ko_f_o, ko_r_o, ko_u_o = (
            float(ko_o.pos.x), float(ko_o.pos.y),
            ko_o.rot_mat.forward, ko_o.rot_mat.right, ko_o.rot_mat.up,
        )

        # 60 Frames (0.50s) 悬挂稳定缓冲: 零输入让物理引擎自然稳定车辆悬挂
        if SETTLE_TICKS > 0:
            self.arena.step(SETTLE_TICKS)

        # 保持 60 帧稳定后的悬挂高度 z 与触地状态，消除上一局残余车轮转速带来的 0.07uu 水平微漂移
        for c_obj, kx, ky, kf, kr, ku in (
            (self.car_blue, ko_bx, ko_by, ko_f_b, ko_r_b, ko_u_b),
            (self.car_orange, ko_ox, ko_oy, ko_f_o, ko_r_o, ko_u_o),
        ):
            cs = c_obj.get_state()
            cs.pos = rs.Vec(kx, ky, float(cs.pos.z))
            cs.vel = rs.Vec(0.0, 0.0, 0.0)
            cs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
            cs.rot_mat.forward = kf
            cs.rot_mat.right = kr
            cs.rot_mat.up = ku
            c_obj.set_state(cs)

        # 确保所有 34 个氮气垫处于激活满状态，球严格在 (0, 0, 93.15) 静止
        for p in self.pads:
            ps = p.get_state()
            ps.is_active = True
            ps.cooldown = 0.0
            p.set_state(ps)

        bs = self.arena.ball.get_state()
        bs.pos = rs.Vec(0.0, 0.0, 93.15)
        bs.vel = rs.Vec(0.0, 0.0, 0.0)
        bs.ang_vel = rs.Vec(0.0, 0.0, 0.0)
        self.arena.ball.set_state(bs)

        self._touched = False
        self._touched_by_car = [False, False]
        self.winner_car_idx = None
        self._boost_pickups_by_car = [0, 0]
        self._boost_pickups_tick = 0
        self.prev_action_vecs = [np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)]

        bx, by, bz = float(bs.pos.x), float(bs.pos.y), float(bs.pos.z)
        for i, c in enumerate(self.cars):
            cs = c.get_state()
            d0 = math.hypot(bx - float(cs.pos.x), by - float(cs.pos.y), bz - float(cs.pos.z))
            self.init_dists[i] = max(1.0, d0)
            self.min_dists[i] = self.init_dists[i]

        self.init_dist = self.init_dists[self.focused_car_idx]
        self.min_dist = self.min_dists[self.focused_car_idx]
        self.ep_max_decisions = MAX_DECISIONS
        self.current_max_decisions = MAX_DECISIONS

        cs_b = self.car_blue.get_state()
        cs_o = self.car_orange.get_state()
        sd = {
            "mode": f"kickoff_slot_{slot_idx}",
            "slot_idx": slot_idx,
            "slot_name": KICKOFF_SLOT_NAMES[slot_idx],
            "step_idx": slot_idx,
            "stage_idx": slot_idx,
            "dir5_idx": slot_idx,
            "on_ground": bool(cs_b.is_on_ground),
            "has_second_jump": True,
            "car_pos": (float(cs_b.pos.x), float(cs_b.pos.y), float(cs_b.pos.z)),
            "car_vel": (float(cs_b.vel.x), float(cs_b.vel.y), float(cs_b.vel.z)),
            "car_ang": (float(cs_b.ang_vel.x), float(cs_b.ang_vel.y), float(cs_b.ang_vel.z)),
            "car_fwd": (float(cs_b.rot_mat.forward.x), float(cs_b.rot_mat.forward.y), float(cs_b.rot_mat.forward.z)),
            "car_rgt": (float(cs_b.rot_mat.right.x),   float(cs_b.rot_mat.right.y),   float(cs_b.rot_mat.right.z)),
            "car_up":  (float(cs_b.rot_mat.up.x),      float(cs_b.rot_mat.up.y),      float(cs_b.rot_mat.up.z)),
            "car_boost": float(cs_b.boost),
            "opp_pos": (float(cs_o.pos.x), float(cs_o.pos.y), float(cs_o.pos.z)),
            "ball_pos": (bx, by, bz),
            "ball_vel": (0.0, 0.0, 0.0),
            "ball_ang": (0.0, 0.0, 0.0),
        }
        sd["spawn_id"] = self._compute_spawn_id(sd)
        self.last_spawn_state = sd
        self.spawn_id = sd["spawn_id"]

    def reset(self, mode=None, replay_last=False, spawn_state=None, slot_idx=None, **kwargs):
        if spawn_state is not None and "slot_idx" in spawn_state:
            target_slot = int(spawn_state["slot_idx"]) % NUM_KICKOFF_SLOTS
        elif slot_idx is not None:
            target_slot = int(slot_idx) % NUM_KICKOFF_SLOTS
        elif replay_last and self.last_slot_idx is not None:
            target_slot = int(self.last_slot_idx)
        else:
            target_slot = self._next_slot_idx(cycle_sequential=(mode == "cycle"))

        self._reset_kickoff_and_settle(target_slot)

        obs_b, panel_b, parts_b = build_obs(
            self.car_blue, self.arena,
            prev_action_vec=self.prev_action_vecs[0],
            opp_car=self.car_orange,
            is_orange=False,
        )
        obs_o, panel_o, parts_o = build_obs(
            self.car_orange, self.arena,
            prev_action_vec=self.prev_action_vecs[1],
            opp_car=self.car_blue,
            is_orange=True,
        )
        self.last_obs_pair = (obs_b, obs_o)
        if self.focused_car_idx == 0:
            self.last_obs = obs_b
            self.last_obs_v3 = obs_b
            self.last_obs_panel = panel_b
            self.last_parts = parts_b
        else:
            self.last_obs = obs_o
            self.last_obs_v3 = obs_o
            self.last_obs_panel = panel_o
            self.last_parts = parts_o

        dist = float(self.last_parts["D.ball_rel_b"][6])
        return self.last_obs_panel, dist

    def _sanitize_ctrl(self, car_obj, ctrl):
        """应用物理动作清洗: 空中禁用手刹，翻滚且车头偏离球时禁用反向喷气。"""
        cs_before = car_obj.get_state()
        bs_before = self.arena.ball.get_state()
        was_on_ground = bool(cs_before.is_on_ground)
        was_flipping = bool(cs_before.is_flipping)

        if not was_on_ground:
            ctrl.handbrake = False
        if was_flipping and ctrl.boost:
            dx_f = float(bs_before.pos.x - cs_before.pos.x)
            dy_f = float(bs_before.pos.y - cs_before.pos.y)
            dz_f = float(bs_before.pos.z - cs_before.pos.z)
            dist_f = max(1e-6, math.hypot(dx_f, dy_f, dz_f))
            align_f = (
                dx_f * float(cs_before.rot_mat.forward.x)
                + dy_f * float(cs_before.rot_mat.forward.y)
                + dz_f * float(cs_before.rot_mat.forward.z)
            ) / dist_f
            if align_f < 0.25:
                ctrl.boost = False
        return cs_before, was_on_ground, was_flipping, bool(cs_before.has_double_jumped), float(cs_before.boost)

    def step_duel_tick(self, ctrl_blue, ctrl_orange):
        """1v1 双车同步推进 1 物理帧 (1/120s)。
        返回:
          (b_boost_used, b_flip, b_djump), (o_boost_used, o_flip, o_djump)
        """
        _, b_gnd, b_flip_pre, b_dj_pre, b_boost_pre = self._sanitize_ctrl(self.car_blue, ctrl_blue)
        _, o_gnd, o_flip_pre, o_dj_pre, o_boost_pre = self._sanitize_ctrl(self.car_orange, ctrl_orange)

        self._boost_pickups_by_car[0] = 0
        self._boost_pickups_by_car[1] = 0
        self._boost_pickups_tick = 0

        self.car_blue.set_controls(ctrl_blue)
        self.car_orange.set_controls(ctrl_orange)
        self.arena.step(1)

        bs_after = self.arena.ball.get_state()
        bx, by, bz = float(bs_after.pos.x), float(bs_after.pos.y), float(bs_after.pos.z)

        results = []
        for i, (c, ctrl, gnd_pre, flip_pre, dj_pre, boost_pre) in enumerate((
            (self.car_blue, ctrl_blue, b_gnd, b_flip_pre, b_dj_pre, b_boost_pre),
            (self.car_orange, ctrl_orange, o_gnd, o_flip_pre, o_dj_pre, o_boost_pre),
        )):
            cs_after = c.get_state()
            cur_boost = float(cs_after.boost)
            cur_dist = math.hypot(bx - float(cs_after.pos.x), by - float(cs_after.pos.y), bz - float(cs_after.pos.z))
            if cur_dist < self.min_dists[i]:
                self.min_dists[i] = cur_dist

            self.prev_action_vecs[i] = np.array([
                float(ctrl.throttle if abs(ctrl.throttle) >= abs(ctrl.pitch) else -ctrl.pitch),
                float(ctrl.steer if abs(ctrl.steer) >= abs(ctrl.yaw) else ctrl.yaw),
                float(ctrl.roll),
                1.0 if ctrl.jump else 0.0,
                1.0 if ctrl.boost else 0.0,
                1.0 if ctrl.handbrake else 0.0,
            ], dtype=np.float32)

            boost_used = max(0.0, boost_pre - cur_boost) if ctrl.boost else 0.0
            flip_trig = (not flip_pre) and bool(cs_after.is_flipping)
            djump_trig = (not dj_pre) and bool(cs_after.has_double_jumped)

            if i == self.focused_car_idx:
                self.last_thr_input = float(self.prev_action_vecs[i][0])
                self.last_boost_used = boost_used
                self.last_boost_pickups = self._boost_pickups_by_car[i]
                self.last_flip_triggered = flip_trig
                self.last_djump_triggered = djump_trig
                self.last_jump_held = bool(ctrl.jump)
                self.last_handbrake_held = bool(ctrl.handbrake) and gnd_pre
                self.init_dist = self.init_dists[i]
                self.min_dist = self.min_dists[i]

            results.append((boost_used, flip_trig, djump_trig))

        if self._touched and self.winner_car_idx is None:
            b_t, o_t = self._touched_by_car[0], self._touched_by_car[1]
            if b_t and o_t:
                self.winner_car_idx = -1  # Simultaneous touch tie
            elif b_t:
                self.winner_car_idx = 0   # Blue won first touch
            elif o_t:
                self.winner_car_idx = 1   # Orange won first touch

        return results[0], results[1]

    def step_tick(self, ctrl, opp_ctrl=None):
        """单车兼容接口 (用于人类手动录制或单车评测):
        默认控制聚焦车 (Blue)，若未提供 opp_ctrl 则对手保持静止空指令，不干扰人类触球录制。
        """
        if opp_ctrl is None:
            opp_ctrl = rs.CarControls()
        if self.focused_car_idx == 0:
            res_b, _ = self.step_duel_tick(ctrl, opp_ctrl)
            return res_b
        else:
            _, res_o = self.step_duel_tick(opp_ctrl, ctrl)
            return res_o

    def get_obs_pair(self):
        """返回 (obs_blue_42, obs_orange_42) 供 1v1 Self-Play 训练使用。"""
        obs_b, panel_b, parts_b = build_obs(
            self.car_blue, self.arena,
            prev_action_vec=self.prev_action_vecs[0],
            opp_car=self.car_orange,
            is_orange=False,
        )
        obs_o, panel_o, parts_o = build_obs(
            self.car_orange, self.arena,
            prev_action_vec=self.prev_action_vecs[1],
            opp_car=self.car_blue,
            is_orange=True,
        )
        self.last_obs_pair = (obs_b, obs_o)
        if self.focused_car_idx == 0:
            self.last_obs = obs_b
            self.last_obs_v3 = obs_b
            self.last_obs_panel = panel_b
            self.last_parts = parts_b
        else:
            self.last_obs = obs_o
            self.last_obs_v3 = obs_o
            self.last_obs_panel = panel_o
            self.last_parts = parts_o
        return obs_b, obs_o

    def get_obs(self, car_idx: int = None):
        idx = self.focused_car_idx if car_idx is None else max(0, min(int(car_idx), 1))
        car_obj = self.cars[idx]
        opp_obj = self.cars[1 - idx]
        obs_full, obs_panel, parts = build_obs(
            car_obj, self.arena,
            prev_action_vec=self.prev_action_vecs[idx],
            opp_car=opp_obj,
            is_orange=(idx == 1),
        )
        if idx == self.focused_car_idx:
            self.last_obs = obs_full
            self.last_obs_v3 = obs_full
            self.last_obs_panel = obs_panel
            self.last_parts = parts
        return obs_full, obs_panel, parts

    def get_obs_v3(self):
        return self.get_obs()

    def snapshot_car(self, idx: int = 0):
        idx = max(0, min(int(idx), 1))
        c = self.cars[idx]
        opp = self.cars[1 - idx]
        cs = c.get_state()
        ocs = opp.get_state()
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
        n_h = dist_h + 1e-6
        align = float((fwd.x * dx + fwd.y * dy) / n_h)
        car_speed = float(math.hypot(cs.vel.x, cs.vel.y, cs.vel.z))
        has_flip_or_jump = bool(cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump)

        opp_dist = float(math.hypot(float(bs.pos.x - ocs.pos.x), float(bs.pos.y - ocs.pos.y), float(bs.pos.z - ocs.pos.z)))
        ctrl = c.get_controls()
        boosting = 1.0 if (bool(ctrl.boost) and float(cs.boost) > 0.1) else 0.0

        return {
            "car_idx": idx,
            "team": idx,
            "label": self.car_labels[idx],
            "slot_idx": self.current_slot_idx,
            "slot_name": KICKOFF_SLOT_NAMES[self.current_slot_idx],
            "car_pos": (float(cs.pos.x), float(cs.pos.y), float(cs.pos.z)),
            "car_vel": (float(cs.vel.x), float(cs.vel.y), float(cs.vel.z)),
            "car_ang": (float(cs.ang_vel.x), float(cs.ang_vel.y), float(cs.ang_vel.z)),
            "car_speed": car_speed,
            "is_supersonic": bool(cs.is_supersonic),
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
            "djump_or_flip": bool(cs.has_double_jumped or cs.is_flipping),
            "has_flip_or_jump": has_flip_or_jump,
            "air_time": float(cs.air_time),
            "flip_time": float(cs.flip_time),
            "boost": float(cs.boost),
            "boosting": boosting,
            "dist": dist,
            "opp_dist": opp_dist,
            "init_dist": self.init_dists[idx],
            "min_dist": self.min_dists[idx],
            "dir_ball": dir_ball,
            "dist_h": dist_h,
            "align": align,
            "closing": closing_speed,
            "touched": bool(self._touched_by_car[idx]),
            "opp_touched": bool(self._touched_by_car[1 - idx]),
            "any_touched": bool(self._touched),
            "winner_car_idx": self.winner_car_idx,
            "step_idx": self.current_slot_idx,
            "stage_idx": self.current_slot_idx,
            "dir5_idx": self.current_slot_idx,
            "ep_max_decisions": self.ep_max_decisions,
        }

    def snapshot_pair(self):
        return self.snapshot_car(0), self.snapshot_car(1)

    def snapshot_all_cars(self):
        return [self.snapshot_car(0), self.snapshot_car(1)]

    def snapshot(self, car_idx: int = None):
        target_idx = self.focused_car_idx if car_idx is None else int(car_idx)
        return self.snapshot_car(target_idx)

    def generate_5slot_eval_grid(self, samples_per_stage: int = 1):
        """生成 5 个标准开球位的评测列表。由于每个开球位物理确定，默认每个位置测 1 次即可 (或按需重复)。"""
        n_rep = max(1, int(samples_per_stage))
        grid = []
        for s_idx in range(NUM_KICKOFF_SLOTS):
            stage_samples = [{"slot_idx": s_idx, "mode": f"kickoff_slot_{s_idx}"} for _ in range(n_rep)]
            grid.append(stage_samples)
        return grid

    generate_6stage_eval_grid = generate_5slot_eval_grid


# 兼容别名
BallReachEnv = KickoffEnv
