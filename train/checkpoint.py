import os
import copy
import random
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from core.constants import (
    OBS_V3_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS,
    TICKS_PER_DECISION, MAX_DECISIONS_STAGE0,
)
from core.nn import ActorCritic, multi_discrete_to_controls
from core.obs_v3 import build_obs_v3

CKPT_DIR    = "./checkpoints"
CKPT_LATEST = os.path.join(CKPT_DIR, "ground_v0_latest.pt")
CKPT_BEST   = os.path.join(CKPT_DIR, "ground_v0_best.pt")


def ensure_dir():
    os.makedirs(CKPT_DIR, exist_ok=True)


def normalize_prefix(prefix=None, default_mode="ground"):
    if not prefix:
        m = "ground" if default_mode in ("stage0", "ground", None, "") else (
            "target" if default_mode in ("stage1", "target") else str(default_mode)
        )
        return f"{m}_v0"
    p = str(prefix).strip().strip("_")
    return p or "ground_v0"


def ckpt_name(prefix=None, tag="latest", default_mode="ground"):
    norm = normalize_prefix(prefix, default_mode=default_mode)
    return f"{norm}_{tag}.pt"


def ckpt_paths(prefix=None, default_mode="ground"):
    ensure_dir()
    norm = normalize_prefix(prefix, default_mode=default_mode)
    latest_path = os.path.join(CKPT_DIR, f"{norm}_latest.pt")
    best_path   = os.path.join(CKPT_DIR, f"{norm}_best.pt")
    return latest_path, best_path


def discover_prefixes():
    if not os.path.isdir(CKPT_DIR):
        return ["ground_v0"]
    found = set()
    for name in os.listdir(CKPT_DIR):
        for suffix in ("_latest.pt", "_best.pt"):
            if name.endswith(suffix) and len(name) > len(suffix):
                mid = name[:-len(suffix)]
                found.add(mid)
                break
    return sorted(found) or ["ground_v0"]


def save_ckpt(path, model, episode, mode="ground", note="", extra=None):
    ensure_dir()
    payload = {
        "arch":        "ppo",
        "mode":        mode,
        "state_dict":  model.state_dict(),
        "obs_dim":     getattr(model, "obs_dim", OBS_V3_DIM),
        "hidden_dim":  getattr(model, "hidden_dim", PPO_HIDDEN_DIM),
        "action_dims": getattr(model, "action_dims", PPO_ACTION_DIMS),
        "episode":     episode,
        "note":        note,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_ppo_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    obs_dim = ckpt.get("obs_dim", OBS_V3_DIM)
    hidden_dim = ckpt.get("hidden_dim", PPO_HIDDEN_DIM)
    action_dims = ckpt.get("action_dims", PPO_ACTION_DIMS)
    model = ActorCritic(obs_dim=obs_dim, hidden=hidden_dim, action_dims=action_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_ckpt(path, net, strict_dims=True):
    ckpt = torch.load(path, map_location="cpu")
    if strict_dims and ckpt.get("obs_dim", OBS_V3_DIM) != getattr(net, "obs_dim", OBS_V3_DIM):
        raise ValueError(f"PPO obs_dim mismatch: {ckpt.get('obs_dim')} / {net.obs_dim}")
    net.load_state_dict(ckpt["state_dict"])
    return ckpt


def evaluate_scenario_batch(env, model, spawn_batch, max_decisions=MAX_DECISIONS_STAGE0):
    dev = next(model.parameters()).device
    touch_count = 0
    touch_steps = []
    all_steps = []

    for sd in spawn_batch:
        env.reset(spawn_state=sd)
        obs_curr = env.last_obs_v3
        touched_step = float(max_decisions)
        ep_ticks = 0

        for step in range(1, max_decisions + 1):
            if dev.type == "cpu" and hasattr(model, "act_step_fast"):
                act_np, _, _ = model.act_step_fast(obs_curr, deterministic=True)
            else:
                with torch.no_grad():
                    x_t = torch.tensor(obs_curr, dtype=torch.float32, device=dev)
                    act_t, _, _, _ = model.get_action_and_value(x_t, deterministic=True)
                    act_np = act_t.cpu().numpy()
            ctrl = multi_discrete_to_controls(act_np)
            for _ in range(TICKS_PER_DECISION):
                env.car.set_controls(ctrl)
                env.arena.step(1)
                ep_ticks += 1
                if env._touched:
                    break

            if env._touched:
                touch_count += 1
                exact_step = ep_ticks / float(TICKS_PER_DECISION)
                touch_steps.append(exact_step)
                touched_step = exact_step
                break

            obs_curr, _, _ = build_obs_v3(
                env.car, env.arena,
                target_pos=env.target_pos,
                target_time=env.target_time,
            )

        all_steps.append(touched_step)

    mean_step = float(np.mean(touch_steps)) if touch_steps else float("inf")
    return touch_count, len(spawn_batch), mean_step, all_steps


class BestTracker:
    def __init__(self, save_path=None, mode="ground", eval_seed=20261016):
        self.save_path = save_path or CKPT_BEST
        self.mode = mode
        self._eval_rng = random.Random(eval_seed)

        self.best_mask = 0
        self.best_avg_frames = float("inf")
        self.best_all_avg = float("inf")
        self.best_ep = 0
        self.best_metrics = None
        self.best_model = None
        self.last_duel_info = ""

    def init_baseline(self, model, env):
        self.best_model = copy.deepcopy(model)
        self.best_model.eval()
        batch = env.generate_stratified_stage0_batch(rng=self._eval_rng)
        tc, n_tot, avg_step, all_steps = evaluate_scenario_batch(env, model, batch)
        all_avg = float(np.mean(all_steps))
        self.best_mask = tc
        self.best_avg_frames = avg_step
        self.best_all_avg = all_avg
        self.best_metrics = [(tc / max(1, n_tot), avg_step)]
        self.last_duel_info = (
            f"touch={tc}/{n_tot}({tc/max(1,n_tot):.0%}) "
            f"avg_step={avg_step:.1f}(all={all_avg:.1f})"
        )
        return tc, self.best_metrics, avg_step

    def maybe_save(self, model, episode, env):
        batch = env.generate_stratified_stage0_batch(rng=self._eval_rng)
        c_tc, n_tot, c_avg, c_all_steps = evaluate_scenario_batch(env, model, batch)
        c_all_avg = float(np.mean(c_all_steps))

        saved = False
        if self.best_model is None:
            self.best_model = copy.deepcopy(model)
            self.best_model.eval()
            self.best_mask = c_tc
            self.best_avg_frames = c_avg
            self.best_all_avg = c_all_avg
            self.best_ep = episode
            save_ckpt(
                self.save_path, model, episode, mode=self.mode,
                note=f"init_best touch={c_tc}/{n_tot} avg={c_avg:.2f} all_avg={c_all_avg:.2f}",
                extra={
                    "touch_count": c_tc, "n_total": n_tot,
                    "avg_frames": c_avg, "all_avg_frames": c_all_avg,
                    "touch_rate": c_tc / max(1, n_tot),
                },
            )
            saved = True
            self.last_duel_info = (
                f"touch={c_tc}/{n_tot}({c_tc/max(1,n_tot):.0%}) "
                f"step={c_avg:.1f}(all={c_all_avg:.1f}) [INIT_BEST]"
            )
        else:
            b_tc, _, b_avg, b_all_steps = evaluate_scenario_batch(env, self.best_model, batch)
            b_all_avg = float(np.mean(b_all_steps))
            max_dec = MAX_DECISIONS_STAGE0
            common_diffs = [
                c_s - b_s for c_s, b_s in zip(c_all_steps, b_all_steps)
                if c_s < max_dec - 1e-6 and b_s < max_dec - 1e-6
            ]
            common_deg = float(np.mean(common_diffs)) if common_diffs else 0.0
            if b_tc >= int(0.75 * n_tot):
                is_win = (
                    (c_tc >= b_tc)
                    and (c_all_avg < b_all_avg - 1e-6)
                    and (common_deg <= 0.5)
                )
            else:
                is_win = (c_tc > b_tc) or (c_tc == b_tc and c_all_avg < b_all_avg - 1e-6)

            if is_win:
                self.best_model = copy.deepcopy(model)
                self.best_model.eval()
                self.best_mask = c_tc
                self.best_avg_frames = c_avg
                self.best_all_avg = c_all_avg
                self.best_ep = episode
                save_ckpt(
                    self.save_path, model, episode, mode=self.mode,
                    note=f"duel_win touch={c_tc}/{n_tot}(vs {b_tc}) avg={c_avg:.2f}(vs {b_avg:.2f}) all={c_all_avg:.2f}(vs {b_all_avg:.2f})",
                    extra={
                        "touch_count": c_tc, "n_total": n_tot,
                        "avg_frames": c_avg, "all_avg_frames": c_all_avg,
                        "touch_rate": c_tc / max(1, n_tot),
                    },
                )
                saved = True
            self.last_duel_info = (
                f"duel[{self.mode}] {c_tc}/{n_tot}({c_avg:.1f}s/all:{c_all_avg:.1f}s) "
                f"vs best {b_tc}/{n_tot}({b_avg:.1f}s/all:{b_all_avg:.1f}s)"
            )
        metrics = [(c_tc / max(1, n_tot), c_avg)]
        return saved, c_tc, metrics, c_avg
