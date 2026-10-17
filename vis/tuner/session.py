import numpy as np
from core.constants import TOUCH_DIST, OBS_DIM, TICKS_PER_DECISION, MAX_DECISIONS
from core.obs_v2 import build_obs_v2, OBS_V2_GROUPS, GROUP_ORDER
from core.reward import compute_step_reward
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN,
    PHYSICS_HZ, RECORD_LIMIT, SPEED_MIN, SPEED_MAX,
)
from .controls import build_car_controls, ctrl_to_dict


_OBS_LABELS = []
for _grp in GROUP_ORDER:
    _, _, _labels = OBS_V2_GROUPS[_grp]
    for _l in _labels:
        _OBS_LABELS.append(f"{_grp}.{_l}")
assert len(_OBS_LABELS) == OBS_DIM, len(_OBS_LABELS)


def _flatten_raw(parts):
    out = []
    for grp in GROUP_ORDER:
        out.extend(float(v) for v in parts[grp])
    return np.array(out, dtype=np.float64)


class Session:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.speed = 1.0
        self.plot_mode = 1
        self.reset()

    def reset(self, replay_last=False, spawn_state=None):
        self.env.reset(replay_last=replay_last, spawn_state=spawn_state)
        from core.constants import MAX_DECISIONS_STAGE0
        self.cfg.max_decisions = MAX_DECISIONS_STAGE0
        self.state = STATE_IDLE
        self.frames = []
        self.playback_idx = 0
        self.playback_playing = False
        self.r_cum = 0.0
        self.prev_snap = None
        self._prev_boost = None
        self._last_time = None
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

    def _step_once(self, keys):
        ctrl = build_car_controls(keys)
        boost_used, flip_triggered, flip_trigger_speed = self.env.step_tick(ctrl)
        snap = self.env.snapshot()

        obs_v2, obs_parts = build_obs_v2(
            self.env.car, self.env.arena.ball.get_state())

        cs = self.env.car.get_state()
        flag_snap = {
            "on_ground":     bool(cs.is_on_ground),
            "is_jumping":    bool(cs.is_jumping),
            "is_flipping":   bool(cs.is_flipping),
            "is_supersonic": bool(cs.is_supersonic),
            "auto_flip":     bool(cs.is_auto_flipping),
            "has_flipped":   bool(cs.has_flipped),
            "has_djump":     bool(cs.has_double_jumped),
            "flip_or_jump":  bool(cs.has_flip_or_jump),
            "flip_reset":    bool(cs.has_flip_reset),
            "world_cont":    bool(cs.has_world_contact),
            "air_time":      float(cs.air_time),
            "flip_time":     float(cs.flip_time),
            "jump_time":     float(cs.jump_time),
            "boost_time":    float(cs.boosting_time),
        }

        max_dec = getattr(self.cfg, "max_decisions", MAX_DECISIONS)
        rec_limit = max_dec * TICKS_PER_DECISION
        prev = self.prev_snap
        if prev is None:
            r_total = 0.0
            detail = {
                "r_dist": 0.0,
                "r_time": 0.0,
                "r_boost": 0.0,
                "r_flip": 0.0,
                "r_touch": 0.0,
                "r_timeout": 0.0,
                "r_total": 0.0,
            }
            touched = False
            timeout = False
        else:
            touched = snap["touched"]
            timeout = (len(self.frames) + 1) >= rec_limit
            r_total, detail = compute_step_reward(
                self.cfg,
                prev_dist=prev["dist"],
                new_dist=snap["dist"],
                touched=touched,
                timeout=timeout,
                ticks_elapsed=1,
                boost_used=boost_used,
                flip_triggered=flip_triggered,
                flip_trigger_speed=flip_trigger_speed,
            )
            self.r_cum += r_total

        self.frames.append({
            "i": len(self.frames),
            "source": "human",
            "spawn_state": self.env.last_spawn_state,
            "input": ctrl_to_dict(ctrl),
            "car_pos": snap["car_pos"], "car_vel": snap["car_vel"],
            "car_speed": snap["car_speed"], "is_supersonic": snap["is_supersonic"],
            "ball_pos": snap["ball_pos"], "ball_vel": snap["ball_vel"],
            "up_z": snap["up_z"], "on_ground": snap["on_ground"],
            "is_flipping": snap["is_flipping"], "air_time": snap["air_time"],
            "boost": snap["boost"],
            "dist": snap["dist"], "dist_h": snap["dist_h"], "align": snap["align"],
            "closing": snap["closing"],
            "fwd":   snap["fwd"],
            "up":    snap["up"],
            "right": snap["right"],
            "r_total": r_total, "r_cum": self.r_cum, "detail": detail,
            "obs_v2":    obs_v2,
            "obs_parts": obs_parts,
            "flags":     flag_snap,
        })
        self.prev_snap = snap

        if touched or timeout:
            return "touch" if touched else "timeout"
        return None

    def finish_recording(self, reason):
        self.state = STATE_DONE
        self.playback_idx = max(0, len(self.frames) - 1)
        return (f"完成 ({reason})  {len(self.frames)} 帧 / "
                f"累计 {self.r_cum:+.2f}  [Space 回放 / R 重置]")

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
        self.env.car.set_controls(ctrl)
        self.env.arena.step(1)
        obs_v2, parts = build_obs_v2(
            self.env.car, self.env.arena.ball.get_state())
        raw = _flatten_raw(parts)

        if self._obs_min is None:
            self._obs_min = obs_v2.copy()
            self._obs_max = obs_v2.copy()
            self._raw_min = raw.copy()
            self._raw_max = raw.copy()
        else:
            np.minimum(self._obs_min, obs_v2, out=self._obs_min)
            np.maximum(self._obs_max, obs_v2, out=self._obs_max)
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
