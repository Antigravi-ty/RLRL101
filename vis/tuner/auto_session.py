import os
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from core.constants import TICKS_PER_DECISION, MAX_DECISIONS
from core.policy_net import ActorCritic, multi_discrete_to_controls, multi_discrete_to_key_str
from core.reward import compute_step_reward
from train.checkpoint import load_ppo_model
from .session import Session, snapshot_all_cars
from .ui_constants import STATE_RECORDING


class AutoSession(Session):
    def __init__(self, env, cfg, ckpt_path=None):
        super().__init__(env, cfg)
        self.ckpt_path = ckpt_path
        self.model = None
        self.ckpt_meta = {}
        self.current_ppo_action = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        self.current_ppo_ctrl = multi_discrete_to_controls(self.current_ppo_action)
        self.current_opp_action = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
        self.current_opp_ctrl = multi_discrete_to_controls(self.current_opp_action)
        self._ppo_ticks_left = 0
        self.last_ppo_probs = None
        self.last_ppo_value = 0.0
        self.last_pad_ent = 0.0
        self.last_pad_max = 0.0
        self.deterministic = True
        self._last_obs_b = None
        self._last_obs_o = None
        self.reload()

    def set_ckpt(self, new_path):
        self.ckpt_path = new_path
        self.reload()

    def reload(self):
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            self.model, self.ckpt_meta = load_ppo_model(self.ckpt_path, device="cpu")
        else:
            self.model = ActorCritic()
            self.ckpt_meta = {}
        self.cfg.max_decisions = getattr(self.env, "ep_max_decisions", MAX_DECISIONS)

    def select_car(self, slot_idx: int) -> int:
        idx = self.env.select_car(slot_idx)
        return idx

    def cycle_car(self, delta: int) -> int:
        idx = self.env.cycle_car(delta)
        return idx

    def begin_recording(self, replay_last=False, spawn_state=None):
        self.state = STATE_RECORDING
        self._last_time = None
        self._ppo_ticks_left = 0
        self.frames = []
        self.r_cum = 0.0
        self.prev_snap = None
        self.env.reset(replay_last=replay_last, spawn_state=spawn_state)
        self.cfg.max_decisions = getattr(self.env, "ep_max_decisions", MAX_DECISIONS)
        self._last_obs_b, _, _ = self.env.get_obs(car_idx=0)
        self._last_obs_o, _, _ = self.env.get_obs(car_idx=1)

    def instant_simulate(self, replay_last=False, spawn_state=None) -> str:
        """Instantly simulates the 1v1 kickoff episode to completion (touch or timeout)."""
        self.begin_recording(replay_last=replay_last, spawn_state=spawn_state)
        max_dec = getattr(self.env, "ep_max_decisions", getattr(self.cfg, "max_decisions", MAX_DECISIONS))
        max_ticks = max_dec * TICKS_PER_DECISION + 4
        reason = "timeout"
        for _ in range(max_ticks):
            r = self._step_once({})
            if r:
                reason = r
                break
        return self.finish_recording(reason)

    def _infer_action(self, obs_vec, record_probs=False):
        if self.model is None:
            self.model = ActorCritic()
        det = getattr(self, "deterministic", True)
        if torch is not None and getattr(self.model, "_np_state", None) is None:
            with torch.no_grad():
                x_t = torch.tensor(obs_vec, dtype=torch.float32)
                act_t, _, _, _ = self.model.get_action_and_value(x_t, deterministic=det)
                act_np = act_t.cpu().numpy().astype(np.int64)
                if record_probs:
                    self.last_ppo_probs, self.last_ppo_value = self.model.get_channel_probs(x_t)
        else:
            act_np, _, _ = self.model.act_step_fast(obs_vec, deterministic=det)
            act_np = np.asarray(act_np, dtype=np.int64)
            if record_probs:
                self.last_ppo_probs, self.last_ppo_value = self.model.get_channel_probs(obs_vec)
        return act_np

    def _step_once(self, keys=None):
        if getattr(self, "_last_obs_b", None) is None:
            self._last_obs_b, _, _ = self.env.get_obs(car_idx=0)
        if getattr(self, "_last_obs_o", None) is None:
            self._last_obs_o, _, _ = self.env.get_obs(car_idx=1)

        f_idx = int(getattr(self.env, "focused_car_idx", 0))
        if getattr(self, "_ppo_ticks_left", 0) <= 0:
            act_b = self._infer_action(self._last_obs_b, record_probs=(f_idx == 0))
            act_o = self._infer_action(self._last_obs_o, record_probs=(f_idx == 1))
            self.current_blue_action = act_b
            self.current_ppo_action = act_b if f_idx == 0 else act_o
            self.current_ppo_ctrl = multi_discrete_to_controls(act_b)
            self.current_opp_action = act_o
            self.current_opp_ctrl = multi_discrete_to_controls(act_o)
            self._ppo_ticks_left = TICKS_PER_DECISION

        res_b, res_o = self.env.step_duel_tick(self.current_ppo_ctrl, self.current_opp_ctrl)
        self._ppo_ticks_left -= 1

        focused_ctrl = self.current_ppo_ctrl if f_idx == 0 else self.current_opp_ctrl
        input_dict = self._ppo_action_to_input_dict(self.current_ppo_action, focused_ctrl)
        act_b_cur = getattr(self, "current_blue_action", self.current_ppo_action)
        ctrl_b_dict = self._ppo_action_to_input_dict(act_b_cur, self.env.car_blue.get_controls())
        ctrl_o_dict = self._ppo_action_to_input_dict(self.current_opp_action, self.env.car_orange.get_controls())

        snap = self.env.snapshot(car_idx=f_idx)
        obs_b, panel_b, parts_b = self.env.get_obs(car_idx=0)
        obs_o, panel_o, parts_o = self.env.get_obs(car_idx=1)
        self._last_obs_b = obs_b
        self._last_obs_o = obs_o

        obs_full = obs_b if f_idx == 0 else obs_o
        obs_panel = panel_b if f_idx == 0 else panel_o
        obs_parts = parts_b if f_idx == 0 else parts_o
        all_cars = snapshot_all_cars(self.env)

        cs = self.env.cars[f_idx].get_state()
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

        boost_used, flip_triggered, djump_triggered = res_b if f_idx == 0 else res_o
        tb = bool(self.env._touched_by_car[0])
        to = bool(self.env._touched_by_car[1])
        my_touch = tb if f_idx == 0 else to
        opp_touch = to if f_idx == 0 else tb

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
            timeout = False
        else:
            timeout = (len(self.frames) + 1) >= rec_limit and not (tb or to)
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
                boost_pickups=self.env._boost_pickups_by_car[f_idx],
                jump_held_ticks=1 if focused_ctrl.jump else 0,
                handbrake_held_ticks=1 if (focused_ctrl.handbrake and snap["on_ground"]) else 0,
                djump_triggered=djump_triggered,
                flip_triggered=flip_triggered,
                touched=my_touch,
                opp_touched=opp_touch,
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
            "car_idx": f_idx,
            "spawn_state": self.env.last_spawn_state,
            "input": input_dict,
            "dual_ctrls": (ctrl_b_dict, ctrl_o_dict),
            "_ppo_ticks_left": int(self._ppo_ticks_left),
            "_opp_ticks_left": int(self._ppo_ticks_left),
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
            "ppo_value": getattr(self, "last_ppo_value", 0.0),
            "ppo_probs": getattr(self, "last_ppo_probs", None),
            "pad_ent":   0.0,
            "pad_max":   0.0,
        })
        self.prev_snap = snap

        if tb and to:
            return "tie_touch"
        if tb:
            return "touch_blue"
        if to:
            return "touch_orange"
        if timeout:
            return "timeout"
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
