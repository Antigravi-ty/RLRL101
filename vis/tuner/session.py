import numpy as np
from core.constants import TICKS_PER_DECISION, MAX_DECISIONS, OBS_PANEL_DIM
from core.obs import OBS_GROUPS, GROUP_ORDER
from core.policy_net import multi_discrete_to_controls
from core.reward import compute_step_reward
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN,
    PHYSICS_HZ, SPEED_MIN, SPEED_MAX,
)
from .controls import build_car_controls, ctrl_to_dict


_OBS_LABELS = []
for _grp in GROUP_ORDER:
    _, _, _labels = OBS_GROUPS[_grp]
    for _l in _labels:
        _OBS_LABELS.append(f"{_grp}.{_l}")
assert len(_OBS_LABELS) == OBS_PANEL_DIM, len(_OBS_LABELS)


def _flatten_raw(parts):
    out = []
    for grp in GROUP_ORDER:
        out.extend(float(v) for v in parts[grp])
    return np.array(out, dtype=np.float64)


def snapshot_all_cars(env):
    """Capture state of both Blue (#1) and Orange (#2) cars for 3D rendering & playback."""
    out = []
    for idx, c in enumerate(env.cars):
        s_i = env.snapshot(car_idx=idx)
        cs_i = c.get_state()
        ctrl_i = c.get_controls()
        out.append({
            "car_idx": idx,
            "team": int(env.car_teams[idx]),
            "label": env.car_label(idx),
            "car_pos": s_i["car_pos"],
            "car_vel": s_i["car_vel"],
            "car_speed": s_i["car_speed"],
            "fwd": s_i["fwd"],
            "up": s_i["up"],
            "right": s_i["right"],
            "boost": s_i["boost"],
            "boosting": 1.0 if (bool(ctrl_i.boost) and s_i["boost"] > 0.1) else 0.0,
            "is_jumping": bool(cs_i.is_jumping),
            "djump_or_flip": bool(cs_i.has_double_jumped or cs_i.is_flipping),
            "dist": s_i["dist"],
        })
    return out


class Session:
    def __init__(self, env, cfg, opp_model=None, vs_model: bool = True):
        self.env = env
        self.cfg = cfg
        self.opp_model = opp_model
        self.vs_model = bool(vs_model)
        self.deterministic_opp = True
        self._opp_ticks_left = 0
        self._last_obs_o = None
        self.current_opp_action = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        self.current_opp_ctrl = multi_discrete_to_controls(self.current_opp_action)
        self.speed = 1.0
        self.plot_mode = 1
        self.max_buffer_frames = 720
        self.reset()

    def set_opp_model(self, model):
        self.opp_model = model

    def set_buffer_duration(self, max_frames: int):
        self.max_buffer_frames = max(10, int(max_frames))

    def reset(self, replay_last=False, spawn_state=None):
        self.env.reset(replay_last=replay_last, spawn_state=spawn_state)
        self.cfg.max_decisions = getattr(self.env, "ep_max_decisions", MAX_DECISIONS)
        self.state = STATE_IDLE
        self.frames = []
        self.playback_idx = 0
        self.playback_playing = False
        self.r_cum = 0.0
        self.prev_snap = None
        self._prev_boost = None
        self._last_time = None
        self._opp_ticks_left = 0
        _, self._last_obs_o = self.env.get_obs_pair()
        self.speed = getattr(self, "speed", 1.0)
        self.plot_mode = getattr(self, "plot_mode", 1)

        self._obs_min = None
        self._obs_max = None
        self._raw_min = None
        self._raw_max = None
        self._freerun_steps = 0

    def begin_recording(self):
        self.state = STATE_RECORDING
        self._last_time = None
        self._opp_ticks_left = 0
        _, self._last_obs_o = self.env.get_obs_pair()

    def advance_recording(self, keys, now):
        if self._last_time is None:
            self._last_time = now
            return None
        dt = now - self._last_time
        self._last_time = now

        target_hz = PHYSICS_HZ * self.speed
        n = max(1, min(int(round(dt * target_hz)), 30))
        for _ in range(n):
            if self.state != STATE_RECORDING:
                break
            reason = self._step_once(keys)
            if reason:
                return reason
        return None

    def _infer_opp_ctrl(self):
        if (not getattr(self, "vs_model", False)) or (getattr(self, "opp_model", None) is None):
            return None
        if getattr(self, "_opp_ticks_left", 0) <= 0:
            if self._last_obs_o is None:
                self._last_obs_o, _, _ = self.env.get_obs(car_idx=1)
            det = bool(getattr(self, "deterministic_opp", True))
            act_o, _, _ = self.opp_model.act_step_fast(self._last_obs_o, deterministic=det)
            self.current_opp_action = np.asarray(act_o, dtype=np.int64)
            self.current_opp_ctrl = multi_discrete_to_controls(self.current_opp_action)
            self._opp_ticks_left = TICKS_PER_DECISION
        self._opp_ticks_left -= 1
        return self.current_opp_ctrl

    def _step_once(self, keys):
        ctrl = build_car_controls(keys)
        pre_cs = self.env.car.get_state()
        if not pre_cs.is_on_ground:
            ctrl.handbrake = False
        input_dict = ctrl_to_dict(ctrl)

        opp_ctrl = self._infer_opp_ctrl()
        boost_used, flip_triggered, djump_triggered = self.env.step_tick(ctrl, opp_ctrl=opp_ctrl)
        snap = self.env.snapshot(car_idx=0)
        _, obs_o = self.env.get_obs_pair()
        self._last_obs_o = obs_o
        obs_full, obs_panel, obs_parts = self.env.get_obs(car_idx=0)
        all_cars = snapshot_all_cars(self.env)

        cs = self.env.car.get_state()
        flag_snap = {
            "on_ground":     bool(cs.is_on_ground),
            "is_jumping":    bool(cs.is_jumping),
            "is_flipping":   bool(cs.is_flipping),
            "is_supersonic": bool(cs.is_supersonic),
            "auto_flip":     bool(cs.is_auto_flipping),
            "has_flipped":   bool(cs.has_flipped),
            "has_djump":     bool(cs.has_double_jumped),
            "flip_or_jump":  bool(cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump),
            "flip_reset":    bool(cs.has_flip_reset),
            "world_cont":    bool(cs.has_world_contact),
            "air_time":      float(cs.air_time),
            "flip_time":     float(cs.flip_time),
            "jump_time":     float(cs.jump_time),
            "boost_time":    float(cs.boosting_time),
        }

        max_dec = getattr(self.env, "ep_max_decisions", getattr(self.cfg, "max_decisions", MAX_DECISIONS))
        rec_limit = max_dec * TICKS_PER_DECISION
        prev = self.prev_snap
        tb = bool(self.env._touched_by_car[0])
        to = bool(self.env._touched_by_car[1])
        any_touch = tb or to

        if prev is None:
            r_total = 0.0
            detail = {
                "r_dist": 0.0,
                "r_slide": 0.0,
                "r_time": 0.0,
                "r_boost": 0.0,
                "r_jump": 0.0,
                "r_touch": 0.0,
                "r_timeout": 0.0,
                "r_min_dist": 0.0,
                "r_total": 0.0,
            }
            timeout = False
        else:
            timeout = (len(self.frames) + 1) >= rec_limit and (not any_touch)
            r_total, detail = compute_step_reward(
                self.cfg,
                prev_dist=prev["dist"],
                new_dist=snap["dist"],
                init_dist=snap.get("init_dist", self.env.init_dist),
                min_dist=snap.get("min_dist", self.env.min_dist),
                prev_dir_ball=prev.get("dir_ball"),
                new_dir_ball=snap.get("dir_ball"),
                on_ground=snap["on_ground"],
                boost_used=boost_used,
                boost_pickups=self.env.last_boost_pickups,
                jump_held_ticks=1 if ctrl.jump else 0,
                handbrake_held_ticks=1 if self.env.last_handbrake_held else 0,
                djump_triggered=djump_triggered,
                flip_triggered=flip_triggered,
                touched=tb,
                opp_touched=to,
                timeout=timeout,
                ticks_elapsed=1,
                prev_snap=prev,
                new_snap=snap,
                ep_ticks=len(self.frames) + 1,
                max_decisions_ep=max_dec,
            )
            self.r_cum += r_total

        self.frames.append({
            "i": len(self.frames),
            "source": "human",
            "spawn_state": self.env.last_spawn_state,
            "input": input_dict,
            "car_pos": snap["car_pos"], "car_vel": snap["car_vel"],
            "car_speed": snap["car_speed"], "is_supersonic": snap["is_supersonic"],
            "ball_pos": snap["ball_pos"], "ball_vel": snap["ball_vel"],
            "up_z": snap["up_z"], "on_ground": snap["on_ground"],
            "is_flipping": snap["is_flipping"], "air_time": snap["air_time"],
            "boost": snap["boost"],
            "boost_used": float(boost_used),
            "dist": snap["dist"], "dist_h": snap["dist_h"], "align": snap["align"],
            "closing": snap["closing"],
            "fwd":   snap["fwd"],
            "up":    snap["up"],
            "right": snap["right"],
            "r_total": r_total, "r_cum": self.r_cum, "detail": detail,
            "obs_v2":    obs_panel,
            "obs_v3":    obs_full,
            "obs_parts": obs_parts,
            "flags":     flag_snap,
            "all_cars":  all_cars,
            "winner_car_idx": self.env.winner_car_idx,
        })
        self.prev_snap = snap

        if tb and to:
            return "tie_touch"
        if tb:
            return "touch"
        if to:
            return "touch_model_opp"
        if timeout:
            return "timeout"
        return None

    def finish_recording(self, reason):
        self.state = STATE_DONE
        self.playback_idx = max(0, len(self.frames) - 1)
        n_f = len(self.frames)
        sec = n_f / 120.0
        extra_str = ""
        if self.frames:
            bvel = self.frames[-1].get("ball_vel", (0.0, 0.0, 0.0))
            if reason in ("touch", "touch_blue"):
                extra_str = f" | 球纵向vy={bvel[1]:+.0f}"
            elif reason == "tie_touch":
                extra_str = " | 同时触球(平局)"
            elif reason == "touch_model_opp":
                extra_str = f" | 对手Model先触球(剩{self.frames[-1].get('dist', 0.0):.0f}uu)"
        return (
            f"完成 ({reason})  {n_f} 帧 ({sec:.2f}s){extra_str} / "
            f"累计 {self.r_cum:+.2f}  [J 回放 | R 重置当前 | N 新开球位 | H 切单人/对战 | Enter 导出CSV]"
        )

    def tick_playback(self):
        if not self.playback_playing or not self.frames:
            return False
        self.playback_idx = min(self.playback_idx + 1, len(self.frames) - 1)
        if self.playback_idx >= len(self.frames) - 1:
            self.playback_playing = False
            return True
        return False

    def seek(self, delta):
        if not self.frames:
            return
        self.playback_idx = max(0, min(self.playback_idx + delta,
                                       len(self.frames) - 1))

    def toggle_playback(self):
        if self.state == STATE_DONE:
            self.state = STATE_PLAYBACK
            self.playback_playing = True
            if self.playback_idx >= len(self.frames) - 1:
                self.playback_idx = 0
        else:
            self.playback_playing = not self.playback_playing

    def adjust_speed(self, delta):
        self.speed = max(SPEED_MIN, min(SPEED_MAX,
                                        round(self.speed + delta, 2)))
        self._last_time = None

    def begin_freerun(self):
        self.state = STATE_FREERUN
        self._last_time = None
        self._obs_min = None
        self._obs_max = None
        self._raw_min = None
        self._raw_max = None
        self._freerun_steps = 0

    def advance_freerun(self, keys, now):
        if self._last_time is None:
            self._last_time = now
            return
        dt = now - self._last_time
        self._last_time = now

        target_hz = PHYSICS_HZ * self.speed
        n = max(1, min(int(round(dt * target_hz)), 30))
        for _ in range(n):
            if self.state != STATE_FREERUN:
                break
            self._freerun_step(keys)

    def _freerun_step(self, keys):
        ctrl = build_car_controls(keys)
        self.env.step_tick(ctrl)
        _, obs_panel, parts = self.env.get_obs(car_idx=0)
        raw = _flatten_raw(parts)

        if self._obs_min is None:
            self._obs_min = obs_panel.copy()
            self._obs_max = obs_panel.copy()
            self._raw_min = raw.copy()
            self._raw_max = raw.copy()
        else:
            np.minimum(self._obs_min, obs_panel, out=self._obs_min)
            np.maximum(self._obs_max, obs_panel, out=self._obs_max)
            np.minimum(self._raw_min, raw, out=self._raw_min)
            np.maximum(self._raw_max, raw, out=self._raw_max)
        self._freerun_steps += 1

    def end_freerun(self):
        self.state = STATE_IDLE
        return (self._obs_min, self._obs_max,
                self._raw_min, self._raw_max,
                self._freerun_steps)

    def format_freerun_report(self):
        if self._obs_min is None:
            return "FREERUN: no data"
        lines = []
        lines.append(f"FREERUN steps={self._freerun_steps}")
        lines.append(f"{'idx':>3} {'label':<32} "
                     f"{'obs_min':>10} {'obs_max':>10} "
                     f"{'raw_min':>12} {'raw_max':>12}")
        for i, lab in enumerate(_OBS_LABELS):
            lines.append(
                f"{i:>3} {lab:<32} "
                f"{self._obs_min[i]:>10.4f} {self._obs_max[i]:>10.4f} "
                f"{self._raw_min[i]:>12.2f} {self._raw_max[i]:>12.2f}"
            )
        return "\n".join(lines)
