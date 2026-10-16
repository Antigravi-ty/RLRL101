import os
import numpy as np
import torch

from core.constants import (
    OBS_V3_DIM, TICKS_PER_DECISION, MAX_DECISIONS, MAX_DECISIONS_STAGE0,
)
from core.nn import multi_discrete_to_controls, multi_discrete_to_key_str
from core.obs_v3 import build_obs_v3
from core.reward import compute_step_reward
from train.checkpoint import load_ppo_model
from .session import Session
from .ui_constants import STATE_RECORDING, RECORD_LIMIT


class AutoSession(Session):
    def __init__(self, env, cfg, ckpt_path):
        super().__init__(env, cfg)
        self.ckpt_path = ckpt_path
        self.model = None
        self.ckpt_meta = {}
        self.current_ppo_action = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        self.current_ppo_ctrl = multi_discrete_to_controls(self.current_ppo_action)
        self._ppo_ticks_left = 0
        self.last_ppo_probs = None
        self.last_ppo_value = 0.0
        self.reload()

    def set_ckpt(self, new_path):
        self.ckpt_path = new_path
        self.reload()

    def reload(self):
        self.model, self.ckpt_meta = load_ppo_model(self.ckpt_path, device="cpu")
        ckpt_mode = self.ckpt_meta.get("mode", "ground")
        if ckpt_mode == "stage0":
            ckpt_mode = "ground"
        self.env.mode = ckpt_mode
        self.cfg.max_decisions = MAX_DECISIONS_STAGE0

    def begin_recording(self, replay_last=False, spawn_state=None):
        self.state = STATE_RECORDING
        self._last_time = None
        self._ppo_ticks_left = 0
        self.frames = []
        self.r_cum = 0.0
        self.prev_snap = None
        self.env.reset(mode=self.env.mode, replay_last=replay_last, spawn_state=spawn_state)
        self._last_obs_v3 = self.env.last_obs_v3

    def _step_once(self, keys):
        if self._ppo_ticks_left <= 0:
            with torch.no_grad():
                x_t = torch.tensor(self._last_obs_v3, dtype=torch.float32)
                act_t, _, _, _ = self.model.get_action_and_value(x_t, deterministic=True)
                self.current_ppo_action = act_t.cpu().numpy()
                self.last_ppo_probs, self.last_ppo_value = self.model.get_channel_probs(x_t)
            self.current_ppo_ctrl = multi_discrete_to_controls(self.current_ppo_action)
            self._ppo_ticks_left = TICKS_PER_DECISION

        pre_step_speed = self.env.step_tick(self.current_ppo_ctrl)
        self._ppo_ticks_left -= 1
        input_dict = self._ppo_action_to_input_dict(self.current_ppo_action, self.current_ppo_ctrl)

        snap = self.env.snapshot()
        obs_v3, obs_v2, obs_parts = build_obs_v3(
            self.env.car, self.env.arena,
            target_pos=self.env.target_pos,
            target_time=self.env.target_time,
        )
        self._last_obs_v3 = obs_v3

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
                car_speed=pre_step_speed,
                touched=touched,
                timeout=timeout,
                ticks_elapsed=1,
            )
            self.r_cum += r_total

        self.frames.append({
            "i": len(self.frames),
            "source": "model",
            "spawn_state": self.env.last_spawn_state,
            "input": input_dict,
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
            "obs_v3":    obs_v3,
            "obs_parts": obs_parts,
            "flags":     flag_snap,
            "ppo_value": self.last_ppo_value,
            "ppo_probs": self.last_ppo_probs,
        })
        self.prev_snap = snap

        if touched or timeout:
            return "touch" if touched else "timeout"
        return None

    def _ppo_action_to_input_dict(self, act_arr, ctrl):
        return {
            "thr": float(ctrl.throttle),
            "str": float(ctrl.steer),
            "ptc": float(ctrl.pitch),
            "yaw": float(ctrl.yaw),
            "roll": float(ctrl.roll),
            "jump": bool(ctrl.jump),
            "boost": bool(ctrl.boost),
            "handbrake": bool(ctrl.handbrake),
            "ppo_channels": [int(x) for x in act_arr],
            "keys_str": multi_discrete_to_key_str(act_arr),
        }
