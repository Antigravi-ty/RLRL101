import numpy as np
from core.constants import TICKS_PER_DECISION, MAX_DECISIONS, OBS_PANEL_DIM
from core.obs import OBS_GROUPS, GROUP_ORDER
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


class Session:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self.speed = 1.0
        self.plot_mode = 1
        self.reset()

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
        self.last_finish_reason = None
        self.takeover_info = None
        self.speed = getattr(self, "speed", 1.0)
        self.plot_mode = getattr(self, "plot_mode", 1)

        self._obs_min = None
        self._obs_max = None
        self._raw_min = None
        self._raw_max = None
        self._freerun_steps = 0

    def restore_from_frames(self, prefix_frames, start_recording=True, takeover_info=None):
        """Restores the exact RocketSim physics and session state up to prefix_frames[-1] for mid-trajectory human takeover."""
        import RocketSim as rs
        if not prefix_frames:
            self.reset(replay_last=True)
            return

        spawn_sd = prefix_frames[0].get("spawn_state") or self.env.last_spawn_state
        self.env.reset(replay_last=True, spawn_state=spawn_sd)
        for f in prefix_frames:
            inp = f.get("input") or {}
            ctrl = rs.CarControls()
            ctrl.throttle = float(inp.get("thr", 0.0))
            ctrl.steer = float(inp.get("str", 0.0))
            ctrl.pitch = float(inp.get("ptc", 0.0))
            ctrl.yaw = float(inp.get("yaw", 0.0))
            ctrl.roll = float(inp.get("roll", 0.0))
            ctrl.jump = bool(inp.get("jump", False))
            ctrl.boost = bool(inp.get("boost", False))
            ctrl.handbrake = bool(inp.get("handbrake", False))
            self.env.step_tick(ctrl)

        self.env.get_obs()
        self.cfg.max_decisions = getattr(self.env, "ep_max_decisions", MAX_DECISIONS)
        self.frames = [dict(f) for f in prefix_frames]
        for idx_i, fr_i in enumerate(self.frames):
            fr_i["i"] = idx_i
        self.playback_idx = len(self.frames) - 1
        self.playback_playing = False
        self.r_cum = float(self.frames[-1].get("r_cum", 0.0))
        self.prev_snap = self.env.snapshot()
        self._prev_boost = float(self.prev_snap["boost"])
        self._last_time = None
        self.last_finish_reason = None
        if takeover_info is not None:
            self.takeover_info = dict(takeover_info)
        self.state = STATE_RECORDING if start_recording else STATE_IDLE

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
        pre_cs = self.env.car.get_state()
        if not pre_cs.is_on_ground:
            ctrl.handbrake = False
        input_dict = ctrl_to_dict(ctrl)

        boost_used, flip_triggered, djump_triggered = self.env.step_tick(ctrl)
        snap = self.env.snapshot()
        obs_full, obs_panel, obs_parts = self.env.get_obs()

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
            touched = False
            timeout = False
        else:
            touched = snap["touched"]
            timeout = (len(self.frames) + 1) >= rec_limit
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
                touched=touched,
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
        })
        self.prev_snap = snap

        if touched or timeout:
            return "touch" if touched else "timeout"
        return None

    def format_takeover_short(self, reason=None):
        ti = getattr(self, "takeover_info", None)
        if not ti:
            return ""
        r_str = str(reason or getattr(self, "last_finish_reason", None) or "done")
        b_len = int(ti.get("baseline_len", len(self.frames)))
        b_reason = str(ti.get("baseline_reason", "touch"))
        b_src = str(ti.get("baseline_source", "MODEL"))
        cur_len = len(self.frames)
        diff_f = cur_len - b_len
        diff_s = diff_f / PHYSICS_HZ
        if r_str == "touch" and b_reason == "touch":
            if diff_f < 0:
                return f"🏆 比{b_src}快 {-diff_f}帧 ({-diff_s:.2f}s)! [{cur_len}f vs {b_len}f]"
            elif diff_f == 0:
                return f"🤝 与{b_src}速度持平! [{cur_len}f vs {b_len}f]"
            else:
                return f"🐢 比{b_src}慢 +{diff_f}帧 (+{diff_s:.2f}s) [{cur_len}f vs {b_len}f]"
        elif r_str == "touch" and b_reason != "touch":
            return f"🏆 接管成功触球 ({cur_len}f)! ({b_src}原轨 {b_reason})"
        elif r_str != "touch" and b_reason == "touch":
            return f"❌ 接管{r_str} ({cur_len}f) vs {b_src} {b_len}f触球"
        return f"对比{b_src}: {cur_len}f vs {b_len}f ({diff_f:+d}f)"

    def format_takeover_result(self, reason=None):
        ti = getattr(self, "takeover_info", None)
        r_str = str(reason or getattr(self, "last_finish_reason", None) or "done")
        if not ti:
            return (f"完成 ({r_str})  {len(self.frames)} 帧 / "
                    f"累计 {self.r_cum:+.2f}  [J 回放 / R 重置]")
        t_idx = int(ti.get("takeover_idx", 0))
        b_len = int(ti.get("baseline_len", len(self.frames)))
        b_src = str(ti.get("baseline_source", "MODEL"))
        cur_len = len(self.frames)
        seg_h = max(0, cur_len - 1 - t_idx)
        seg_b = max(0, b_len - 1 - t_idx)
        short = self.format_takeover_short(r_str)
        return (
            f"完成 ({r_str}) [中途接管 @f={t_idx}]  "
            f"总 {cur_len}帧({cur_len/PHYSICS_HZ:.2f}s) vs {b_src} {b_len}帧({b_len/PHYSICS_HZ:.2f}s) "
            f"(接管后 {seg_h}f vs {seg_b}f) | {short} | "
            f"累计 {self.r_cum:+.2f}  [Backspace 回到接管点 / ←→ 回溯 / M 切Model]"
        )

    def finish_recording(self, reason):
        self.state = STATE_DONE
        self.last_finish_reason = str(reason)
        self.playback_idx = max(0, len(self.frames) - 1)
        if getattr(self, "takeover_info", None):
            return self.format_takeover_result(reason)
        return (f"完成 ({reason})  {len(self.frames)} 帧 / "
                f"累计 {self.r_cum:+.2f}  [J 回放 / R 重置]")

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
        pre_cs = self.env.car.get_state()
        if not pre_cs.is_on_ground:
            ctrl.handbrake = False
        self.env.step_tick(ctrl)
        _, obs_panel, parts = self.env.get_obs()
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
