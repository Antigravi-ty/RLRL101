"""vis/tuner/nexto_session.py —— 面向 Nexto 多车实时对抗的解耦会话管理器 (带环形缓冲自动防溢出与多车视角切换)。"""
from collections import deque
import numpy as np

from core.constants import MAX_DECISIONS
from core.reward import compute_step_reward
from nexto.model import NextoAgent, action_8d_to_car_controls, action_8d_to_input_dict
from nexto.obs import NextoObsBuilder
from .session import Session
from .ui_constants import (
    STATE_RECORDING, STATE_DONE, STATE_PLAYBACK,
    PHYSICS_HZ, SPEED_MIN, SPEED_MAX,
)

NEXTO_TICK_SKIP = 8
DEFAULT_BUFFER_MAX_FRAMES = 600  # 5.0s rolling window at 120Hz, prevents memory overflow


class NextoSession(Session):
    """驱动全场所有 (n_blue + n_orange) 辆 Nexto 赛车的实时对战会话。"""

    is_nexto_session = True

    def __init__(self, env, cfg, model_path=None, max_buffer_frames=DEFAULT_BUFFER_MAX_FRAMES):
        self.env = env
        self.cfg = cfg
        self.speed = 1.0
        self.plot_mode = 1
        self.max_buffer_frames = max(60, int(max_buffer_frames))
        self.agent = NextoAgent(model_path=model_path, deterministic=True, stochastic_kickoffs=True)
        self.deterministic = True
        self.auto_kickoff_on_goal = True

        self.frames = deque(maxlen=self.max_buffer_frames)
        self.state = STATE_RECORDING
        self.playback_idx = 0
        self.playback_playing = False
        self._last_time = None
        self._step_seq = 0
        self._ticks_until_decision = 0
        self._goal_Pause_ticks = 0
        self._need_trail_clear = False

        self._init_match_buffers()
        self.reset(replay_last=False)

    def _init_match_buffers(self):
        n = self.env.n_cars
        self.prev_actions_8d = np.zeros((n, 8), dtype=np.float32)
        self.current_act_indices = np.zeros(n, dtype=np.int64)
        self.current_controls = [action_8d_to_car_controls(self.prev_actions_8d[i]) for i in range(n)]
        self.car_probs_90 = np.full((n, 90), 1.0 / 90.0, dtype=np.float32)
        self.car_attn_ent = np.zeros(n, dtype=np.float32)
        self.car_attn_max = np.zeros(n, dtype=np.float32)
        self.car_r_cum = [0.0] * n
        self.car_prev_snaps = [None] * n

        # HUD compatibility fields for the currently focused car
        self.current_ppo_action = [1, 1, 1, 0, 0, 0]
        self.last_ppo_probs = None
        self.last_ppo_value = 0.0
        self.last_pad_ent = 0.0
        self.last_pad_max = 0.0
        self.r_cum = 0.0
        self._prev_boost = None

    def reset(self, replay_last=False, spawn_state=None):
        """重置为随机标准 Kickoff Spawn 并立即继续实时对抗。"""
        self.env.reset_by_config()
        self.cfg.max_decisions = MAX_DECISIONS
        self.frames.clear()
        self.playback_idx = 0
        self.playback_playing = False
        self._last_time = None
        self._step_seq = 0
        self._ticks_until_decision = 0
        self._goal_Pause_ticks = 0
        self._need_trail_clear = True
        self._init_match_buffers()
        self.state = STATE_RECORDING

    def begin_recording(self, replay_last=False, spawn_state=None):
        self.reset(replay_last=replay_last, spawn_state=spawn_state)

    def pop_trail_clear_request(self) -> bool:
        if self._need_trail_clear:
            self._need_trail_clear = False
            return True
        return False

    def set_buffer_duration(self, max_frames: int):
        """动态调整环形缓冲区持续帧数 (如 600 -> 100)，自动截断超出窗口的旧帧并与下方图表同步。"""
        self.max_buffer_frames = max(10, int(max_frames))
        recent = list(self.frames)[-self.max_buffer_frames:]
        self.frames = deque(recent, maxlen=self.max_buffer_frames)
        if self.frames:
            self.playback_idx = max(0, min(self.playback_idx, len(self.frames) - 1))
        else:
            self.playback_idx = 0

    def select_car(self, index_0_based: int) -> int:
        new_idx = self.env.select_car(index_0_based)
        self._reproject_frames_to_car(new_idx)
        return new_idx

    def cycle_car(self, delta: int) -> int:
        new_idx = self.env.cycle_car(delta)
        self._reproject_frames_to_car(new_idx)
        return new_idx

    def _reproject_frames_to_car(self, car_idx: int):
        """当切换当前聚焦车辆 (键 1..8 / 9 / 0) 时，将环形缓冲区内的历史帧同步投影至该车。"""
        for fr in self.frames:
            all_cars = fr.get("all_cars")
            if all_cars and 0 <= car_idx < len(all_cars):
                self._apply_car_entry_to_frame(fr, all_cars[car_idx])
        if self.frames:
            last_fr = self.frames[-1]
            self.r_cum = last_fr["r_cum"]
            self.last_ppo_probs = last_fr.get("ppo_probs")
            self.last_pad_ent = last_fr.get("pad_ent", 0.0)
            self.last_pad_max = last_fr.get("pad_max", 0.0)
            self.current_ppo_action = last_fr["input"].get("ppo_channels", [1, 1, 1, 0, 0, 0])

    @staticmethod
    def _apply_car_entry_to_frame(fr: dict, entry: dict):
        fr["car_idx"] = entry["car_idx"]
        fr["car_label"] = entry["label"]
        fr["team"] = entry["team"]
        fr["role_id"] = entry.get("role_id", 0)
        fr["role_code"] = entry.get("role_code", "CHSR")
        fr["role_name"] = entry.get("role_name", "Chaser")
        fr["role_eta_s"] = entry.get("role_eta_s", 0.0)
        fr["role_potential"] = entry.get("role_potential", 0.0)
        fr["role_reason"] = entry.get("role_reason", "")
        fr["input"] = entry["input"]
        fr["car_pos"] = entry["car_pos"]
        fr["car_vel"] = entry["car_vel"]
        fr["car_speed"] = entry["car_speed"]
        fr["is_supersonic"] = entry["is_supersonic"]
        fr["up_z"] = entry["up_z"]
        fr["on_ground"] = entry["on_ground"]
        fr["is_flipping"] = entry["is_flipping"]
        fr["air_time"] = entry["air_time"]
        fr["boost"] = entry["boost"]
        fr["boost_used"] = entry["boost_used"]
        fr["dist"] = entry["dist"]
        fr["dist_h"] = entry["dist_h"]
        fr["align"] = entry["align"]
        fr["closing"] = entry["closing"]
        fr["fwd"] = entry["fwd"]
        fr["up"] = entry["up"]
        fr["right"] = entry["right"]
        fr["r_total"] = entry["r_total"]
        fr["r_cum"] = entry["r_cum"]
        fr["detail"] = entry["detail"]
        fr["obs_v2"] = entry["obs_v2"]
        fr["obs_v3"] = entry["obs_v3"]
        fr["obs_parts"] = entry["obs_parts"]
        fr["flags"] = entry["flags"]
        fr["ppo_value"] = 0.0
        fr["ppo_probs"] = entry["ppo_probs"]
        fr["pad_ent"] = entry["pad_ent"]
        fr["pad_max"] = entry["pad_max"]

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
            self._step_once({})
        return None

    def _step_once(self, keys=None):
        self.agent.deterministic = bool(self.deterministic)

        # Decision step every NEXTO_TICK_SKIP (8) ticks
        if self._ticks_until_decision <= 0:
            bs = self.env.arena.ball.get_state()
            is_kickoff = (abs(float(bs.pos.x)) < 1.0 and abs(float(bs.pos.y)) < 1.0 and np.hypot(bs.vel.x, bs.vel.y) < 5.0)
            q, kv, m = NextoObsBuilder.build_batch_obs(
                self.env.cars,
                self.env.car_teams,
                self.env.arena,
                self.env.pads,
                self.prev_actions_8d,
            )
            acts_8d, act_indices, probs_90, mean_attn = self.agent.act_batch(q, kv, m, is_kickoff=is_kickoff)
            self.prev_actions_8d = acts_8d
            self.current_act_indices = act_indices
            self.car_probs_90 = probs_90
            self.current_controls = [action_8d_to_car_controls(acts_8d[i]) for i in range(self.env.n_cars)]
            for i in range(self.env.n_cars):
                ent_i, max_i = self.agent.attention_stats(mean_attn[i])
                self.car_attn_ent[i] = ent_i
                self.car_attn_max[i] = max_i
            self._ticks_until_decision = NEXTO_TICK_SKIP

        step_results = self.env.step_all_cars(self.current_controls)
        self._ticks_until_decision -= 1
        self._step_seq += 1

        # If a goal was scored, automatically reset to a new random standard Kickoff Spawn (or pause if disabled)
        if self.env.goal_scored_this_tick:
            if self.auto_kickoff_on_goal:
                self.env.reset_by_config()
                self._ticks_until_decision = 0
                self._need_trail_clear = True
                self.prev_actions_8d.fill(0.0)
                self.car_prev_snaps = [None] * self.env.n_cars
            else:
                self.state = STATE_DONE
                self.playback_idx = max(0, len(self.frames) - 1)
                self.playback_playing = False

        bs = self.env.arena.ball.get_state()
        ball_pos = (float(bs.pos.x), float(bs.pos.y), float(bs.pos.z))
        ball_vel = (float(bs.vel.x), float(bs.vel.y), float(bs.vel.z))

        all_car_entries = []
        for i, car in enumerate(self.env.cars):
            b_used, f_trig, dj_trig, pickups = step_results[i]
            ctrl_i = self.current_controls[i]
            snap_i = self.env.snapshot_car(i)
            obs_full_i, obs_panel_i, obs_parts_i = self.env.get_obs(i)
            cs_i = car.get_state()

            flag_snap = {
                "on_ground":     bool(cs_i.is_on_ground),
                "is_jumping":    bool(cs_i.is_jumping),
                "is_flipping":   bool(cs_i.is_flipping),
                "is_supersonic": bool(cs_i.is_supersonic),
                "auto_flip":     bool(cs_i.is_auto_flipping),
                "has_flipped":   bool(cs_i.has_flipped),
                "has_djump":     bool(cs_i.has_double_jumped),
                "flip_or_jump":  bool(cs_i.has_flip_or_jump() if callable(cs_i.has_flip_or_jump) else cs_i.has_flip_or_jump),
                "flip_reset":    bool(cs_i.has_flip_reset),
                "world_cont":    bool(cs_i.has_world_contact),
                "air_time":      float(cs_i.air_time),
                "flip_time":     float(cs_i.flip_time),
                "jump_time":     float(cs_i.jump_time),
                "boost_time":    float(cs_i.boosting_time),
            }

            prev_i = self.car_prev_snaps[i]
            if prev_i is None:
                r_total = 0.0
                detail = {
                    "r_dist": 0.0, "r_slide": 0.0, "r_time": 0.0, "r_boost": 0.0,
                    "r_jump": 0.0, "r_touch": 0.0, "r_timeout": 0.0, "r_min_dist": 0.0, "r_total": 0.0,
                }
            else:
                r_total, detail = compute_step_reward(
                    self.cfg,
                    prev_dist=prev_i["dist"],
                    new_dist=snap_i["dist"],
                    init_dist=snap_i["init_dist"],
                    min_dist=snap_i["min_dist"],
                    prev_dir_ball=prev_i.get("dir_ball"),
                    new_dir_ball=snap_i.get("dir_ball"),
                    on_ground=snap_i["on_ground"],
                    boost_used=b_used,
                    boost_pickups=pickups,
                    jump_held_ticks=1 if ctrl_i.jump else 0,
                    handbrake_held_ticks=1 if (ctrl_i.handbrake and snap_i["on_ground"]) else 0,
                    djump_triggered=dj_trig,
                    flip_triggered=f_trig,
                    touched=snap_i["touched"],
                    timeout=False,
                    ticks_elapsed=1,
                    prev_snap=prev_i,
                    new_snap=snap_i,
                    ep_ticks=self._step_seq,
                    max_decisions_ep=MAX_DECISIONS,
                )
                self.car_r_cum[i] += r_total
            self.car_prev_snaps[i] = snap_i

            input_dict = action_8d_to_input_dict(self.prev_actions_8d[i], int(self.current_act_indices[i]))
            ch_probs = self.agent.marginal_channel_probs(self.car_probs_90[i])

            all_car_entries.append({
                "car_idx": i,
                "label": self.env.car_labels[i],
                "team": int(self.env.car_teams[i]),
                "role_id": snap_i.get("role_id", 0),
                "role_code": snap_i.get("role_code", "CHSR"),
                "role_name": snap_i.get("role_name", "Chaser"),
                "role_eta_s": snap_i.get("role_eta_s", 0.0),
                "role_potential": snap_i.get("role_potential", 0.0),
                "role_reason": snap_i.get("role_reason", ""),
                "input": input_dict,
                "car_pos": snap_i["car_pos"],
                "car_vel": snap_i["car_vel"],
                "car_speed": snap_i["car_speed"],
                "is_supersonic": snap_i["is_supersonic"],
                "up_z": snap_i["up_z"],
                "on_ground": snap_i["on_ground"],
                "is_jumping": snap_i["is_jumping"],
                "is_flipping": snap_i["is_flipping"],
                "djump_or_flip": bool(snap_i["has_djump"] or snap_i["is_flipping"]),
                "air_time": snap_i["air_time"],
                "boost": snap_i["boost"],
                "boost_used": float(b_used),
                "boosting": 1.0 if (b_used > 0.001 or (ctrl_i.boost and snap_i["boost"] > 0.1)) else 0.0,
                "dist": snap_i["dist"],
                "dist_h": snap_i["dist_h"],
                "align": snap_i["align"],
                "closing": snap_i["closing"],
                "fwd": snap_i["fwd"],
                "up": snap_i["up"],
                "right": snap_i["right"],
                "r_total": r_total,
                "r_cum": self.car_r_cum[i],
                "detail": detail,
                "obs_v2": obs_panel_i,
                "obs_v3": obs_full_i,
                "obs_parts": obs_parts_i,
                "flags": flag_snap,
                "ppo_probs": ch_probs,
                "pad_ent": float(self.car_attn_ent[i]),
                "pad_max": float(self.car_attn_max[i]),
            })

        f_idx = self.env.focused_car_idx
        focused_entry = all_car_entries[f_idx]
        frame = {
            "i": self._step_seq,
            "source": "nexto",
            "spawn_state": None,
            "ball_pos": ball_pos,
            "ball_vel": ball_vel,
            "blue_score": self.env.blue_score,
            "orange_score": self.env.orange_score,
            "all_cars": all_car_entries,
        }
        self._apply_car_entry_to_frame(frame, focused_entry)
        self.frames.append(frame)

        self.r_cum = focused_entry["r_cum"]
        self.last_ppo_probs = focused_entry["ppo_probs"]
        self.last_ppo_value = 0.0
        self.last_pad_ent = focused_entry["pad_ent"]
        self.last_pad_max = focused_entry["pad_max"]
        self.current_ppo_action = focused_entry["input"]["ppo_channels"]
        return None

    def toggle_playback(self):
        if self.state == STATE_RECORDING:
            self.state = STATE_DONE
            self.playback_idx = max(0, len(self.frames) - 1)
            self.playback_playing = False
        elif self.state == STATE_DONE:
            self.state = STATE_PLAYBACK
            self.playback_playing = True
            if self.playback_idx >= len(self.frames) - 1:
                self.playback_idx = 0
        elif self.state == STATE_PLAYBACK:
            self.state = STATE_RECORDING
            self.playback_playing = False
            self._last_time = None
