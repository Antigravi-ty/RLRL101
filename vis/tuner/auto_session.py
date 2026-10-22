import numpy as np
try:
    import torch
except ImportError:
    torch = None

from core.constants import (
    TICKS_PER_DECISION, MAX_DECISIONS, NUM_CURRICULUM_STAGES, NUM_CURRICULUM_STEPS,
)
from core.policy_net import multi_discrete_to_controls, multi_discrete_to_key_str
from core.reward import compute_step_reward
from train.checkpoint import load_ppo_model
from .session import Session
from .ui_constants import STATE_RECORDING


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
        self.last_pad_ent = 0.0
        self.last_pad_max = 0.0
        self.deterministic = True
        self.reload()

    def set_ckpt(self, new_path):
        self.ckpt_path = new_path
        self.reload()

    def reload(self):
        self.model, self.ckpt_meta = load_ppo_model(self.ckpt_path, device="cpu")
        ckpt_mode = self.ckpt_meta.get("mode", "curriculum")
        self.env.mode = ckpt_mode
        slv = int(self.ckpt_meta.get("stage_level", NUM_CURRICULUM_STAGES))
        c_step = int(self.ckpt_meta.get(
            "curriculum_step_idx",
            self.ckpt_meta.get("step_idx", NUM_CURRICULUM_STEPS - 1),
        ))
        self.env.set_curriculum_state(slv, step_idx=c_step)
        self.cfg.max_decisions = getattr(self.env, "ep_max_decisions", MAX_DECISIONS)

    def begin_recording(self, replay_last=False, spawn_state=None):
        self.state = STATE_RECORDING
        self._last_time = None
        self._ppo_ticks_left = 0
        self.frames = []
        self.r_cum = 0.0
        self.prev_snap = None
        self.last_finish_reason = None
        self.takeover_info = None
        self.env.reset(mode=self.env.mode, replay_last=replay_last, spawn_state=spawn_state)
        self.cfg.max_decisions = getattr(self.env, "ep_max_decisions", MAX_DECISIONS)
        self._last_obs = self.env.last_obs

    def instant_simulate(self, replay_last=False, spawn_state=None) -> str:
        """Instantly simulates the entire episode to completion (touch or timeout) without waiting for real-time ticks."""
        self.begin_recording(replay_last=replay_last, spawn_state=spawn_state)
        return self.complete_remaining()

    def complete_remaining(self) -> str:
        """Completes remaining ticks of the current episode so the full model baseline is available for scrubbing/takeover."""
        max_dec = getattr(self.env, "ep_max_decisions", getattr(self.cfg, "max_decisions", MAX_DECISIONS))
        max_ticks = max_dec * TICKS_PER_DECISION + 4
        rem = max(1, max_ticks - len(self.frames))
        reason = "timeout"
        for _ in range(rem):
            r = self._step_once({})
            if r:
                reason = r
                break
        return self.finish_recording(reason)

    def _step_once(self, keys):
        if self._ppo_ticks_left <= 0:
            if torch is not None and getattr(self.model, "_np_state", None) is None:
                with torch.no_grad():
                    x_t = torch.tensor(self._last_obs, dtype=torch.float32)
                    act_t, _, _, _ = self.model.get_action_and_value(x_t, deterministic=self.deterministic)
                    self.current_ppo_action = act_t.cpu().numpy()
                    self.last_ppo_probs, self.last_ppo_value = self.model.get_channel_probs(x_t)
                    if hasattr(self.model, "get_attention_stats"):
                        ent_val, max_val, _ = self.model.get_attention_stats()
                        self.last_pad_ent = float(ent_val)
                        self.last_pad_max = float(max_val)
            else:
                act_np, _, _ = self.model.act_step_fast(self._last_obs, deterministic=self.deterministic)
                self.current_ppo_action = np.asarray(act_np, dtype=np.int64)
                self.last_ppo_probs, self.last_ppo_value = self.model.get_channel_probs(self._last_obs)
                if hasattr(self.model, "get_attention_stats"):
                    ent_val, max_val, _ = self.model.get_attention_stats()
                    self.last_pad_ent = float(ent_val)
                    self.last_pad_max = float(max_val)
            self.current_ppo_ctrl = multi_discrete_to_controls(self.current_ppo_action)
            self._ppo_ticks_left = TICKS_PER_DECISION

        boost_used, flip_triggered, djump_triggered = self.env.step_tick(self.current_ppo_ctrl)
        self._ppo_ticks_left -= 1
        input_dict = self._ppo_action_to_input_dict(self.current_ppo_action, self.current_ppo_ctrl)

        snap = self.env.snapshot()
        obs_full, obs_panel, obs_parts = self.env.get_obs()
        self._last_obs = obs_full

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
                jump_held_ticks=1 if self.current_ppo_ctrl.jump else 0,
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
            "source": "model",
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
            "ppo_value": self.last_ppo_value,
            "ppo_probs": self.last_ppo_probs,
            "pad_ent":   self.last_pad_ent,
            "pad_max":   self.last_pad_max,
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

    def _freerun_step(self, keys):
        """Advances the PPO model in unlimited FREERUN mode without appending frames (zero memory growth)."""
        if self._ppo_ticks_left <= 0:
            obs_curr = getattr(self, "_last_obs", self.env.last_obs)
            act_np, _, _ = self.model.act_step_fast(obs_curr, deterministic=self.deterministic)
            self.current_ppo_action = np.asarray(act_np, dtype=np.int64)
            self.current_ppo_ctrl = multi_discrete_to_controls(self.current_ppo_action)
            self._ppo_ticks_left = TICKS_PER_DECISION
        self.env.step_tick(self.current_ppo_ctrl)
        self._ppo_ticks_left -= 1
        obs_full, _, _ = self.env.get_obs()
        self._last_obs = obs_full
        self._freerun_steps += 1
        if self.env._touched:
            self.env.reset(mode=self.env.mode, replay_last=False)
            self._last_obs = self.env.last_obs
            self._ppo_ticks_left = 0

