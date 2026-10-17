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
    EVAL_SAMPLES_PER_BIN,
)
from core.nn import ActorCritic, multi_discrete_to_controls, adapt_state_dict_input_dim
from core.obs_v3 import build_obs_v3
from core.reward import RewardConfig, compute_step_reward

CKPT_DIR    = "./checkpoints"
CKPT_LATEST = os.path.join(CKPT_DIR, "ground_v0_latest.pt")
CKPT_BEST   = os.path.join(CKPT_DIR, "ground_v0_best.pt")

BASE36_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


def count_to_base36_char(c: int) -> str:
    idx = max(0, min(35, int(c)))
    return BASE36_CHARS[idx]


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
    hidden_dim = ckpt.get("hidden_dim", PPO_HIDDEN_DIM)
    action_dims = ckpt.get("action_dims", PPO_ACTION_DIMS)
    model = ActorCritic(obs_dim=OBS_V3_DIM, hidden=hidden_dim, action_dims=action_dims).to(device)
    sd = adapt_state_dict_input_dim(ckpt["state_dict"], model)
    model.load_state_dict(sd)
    model.eval()
    return model, ckpt


def load_ckpt(path, net, strict_dims=False):
    ckpt = torch.load(path, map_location="cpu")
    if strict_dims and ckpt.get("obs_dim", OBS_V3_DIM) != getattr(net, "obs_dim", OBS_V3_DIM):
        raise ValueError(f"PPO obs_dim mismatch: {ckpt.get('obs_dim')} / {net.obs_dim}")
    sd = adapt_state_dict_input_dim(ckpt["state_dict"], net)
    net.load_state_dict(sd)
    return ckpt


def evaluate_16bin_grid(
    env,
    model,
    grid_batches=None,
    max_decisions=MAX_DECISIONS_STAGE0,
    samples_per_bin=EVAL_SAMPLES_PER_BIN,
    seed=20261016,
    rng=None,
):
    if grid_batches is None:
        r = rng if rng is not None else random.Random(seed)
        grid_batches = env.generate_16bin_eval_grid(samples_per_bin=samples_per_bin, rng=r)
    dev = next(model.parameters()).device
    cfg = RewardConfig(max_decisions=max_decisions)
    bin_counts = []
    bin_chars = []
    touch_steps = []
    all_steps = []
    rcum_list = []
    jump_eps = 0
    flip_eps = 0

    for bin_samples in grid_batches:
        b_tc = 0
        for sd in bin_samples:
            env.reset(spawn_state=sd)
            obs_curr = env.last_obs_v3
            prev_dist = env.snapshot()["dist"]
            touched_step = float(max_decisions)
            ep_ticks = 0
            ep_rcum = 0.0
            ep_jumped = False
            ep_flipped = False

            for step in range(1, max_decisions + 1):
                if dev.type == "cpu" and hasattr(model, "act_step_fast"):
                    act_np, _, _ = model.act_step_fast(obs_curr, deterministic=True)
                else:
                    with torch.no_grad():
                        x_t = torch.tensor(obs_curr, dtype=torch.float32, device=dev)
                        act_t, _, _, _ = model.get_action_and_value(x_t, deterministic=True)
                        act_np = act_t.cpu().numpy()
                if int(act_np[3]) == 1:
                    ep_jumped = True
                ctrl = multi_discrete_to_controls(act_np)
                ticks_run = 0
                boost_used_step = 0.0
                flip_trig_step = False
                flip_spd_step = 0.0
                for _ in range(TICKS_PER_DECISION):
                    b_u, f_trig, f_spd = env.step_tick(ctrl)
                    ticks_run += 1
                    ep_ticks += 1
                    boost_used_step += b_u
                    if f_trig and not flip_trig_step:
                        flip_trig_step = True
                        flip_spd_step = f_spd
                        ep_flipped = True
                    if env._touched:
                        break

                cs_now = env.car.get_state()
                bs_now = env.arena.ball.get_state()
                new_dist = float(np.hypot(
                    np.hypot(bs_now.pos.x - cs_now.pos.x, bs_now.pos.y - cs_now.pos.y),
                    bs_now.pos.z - cs_now.pos.z
                ))
                touched_now = env._touched
                timeout_now = (step == max_decisions) and not touched_now
                r_step, _ = compute_step_reward(
                    cfg,
                    prev_dist=prev_dist,
                    new_dist=new_dist,
                    touched=touched_now,
                    timeout=timeout_now,
                    ticks_elapsed=ticks_run,
                    boost_used=boost_used_step,
                    flip_triggered=flip_trig_step,
                    flip_trigger_speed=flip_spd_step,
                )
                ep_rcum += r_step
                prev_dist = new_dist

                if touched_now:
                    b_tc += 1
                    exact_step = ep_ticks / float(TICKS_PER_DECISION)
                    touch_steps.append(exact_step)
                    touched_step = exact_step
                    break

                obs_curr, _, _ = build_obs_v3(
                    env.car, env.arena,
                    target_pos=env.target_pos,
                    target_time=env.target_time,
                )

            if ep_jumped:
                jump_eps += 1
            if ep_flipped:
                flip_eps += 1
            all_steps.append(touched_step)
            rcum_list.append(ep_rcum)
        bin_counts.append(b_tc)
        bin_chars.append(count_to_base36_char(b_tc))

    grid_code = "".join(bin_chars)
    total_touch = sum(bin_counts)
    n_total = sum(len(b) for b in grid_batches)
    mean_touch_step = float(np.mean(touch_steps)) if touch_steps else float(max_decisions)
    mean_all_step = float(np.mean(all_steps)) if all_steps else float(max_decisions)
    touch_ticks = mean_touch_step * float(TICKS_PER_DECISION)
    all_ticks = mean_all_step * float(TICKS_PER_DECISION)
    return {
        "grid_code": grid_code,
        "grid_str": grid_code,
        "bin_counts": bin_counts,
        "bin_touches": bin_counts,
        "touch_count": total_touch,
        "n_total": n_total,
        "touch_rate": total_touch / max(1, n_total),
        "mean_touch_step": mean_touch_step,
        "mean_all_step": mean_all_step,
        "touch_dec": mean_touch_step,
        "all_dec": mean_all_step,
        "touch_ticks": touch_ticks,
        "all_ticks": all_ticks,
        "touch_sec": touch_ticks / 120.0,
        "all_sec": all_ticks / 120.0,
        "jump_eps": jump_eps,
        "flip_eps": flip_eps,
        "mean_rcum": float(np.mean(rcum_list)) if rcum_list else 0.0,
        "all_steps": all_steps,
    }


class BestTracker:
    def __init__(self, save_path=None, mode="all", eval_seed=20261016):
        self.save_path = save_path or CKPT_BEST
        self.mode = mode
        self._eval_rng = random.Random(eval_seed)

    def init_baseline(self, model, env):
        self.best_model = copy.deepcopy(model)
        self.best_model.eval()
        grid = env.generate_16bin_eval_grid(samples_per_bin=EVAL_SAMPLES_PER_BIN, rng=self._eval_rng)
        res = evaluate_16bin_grid(env, model, grid_batches=grid)
        self.best_mask = res["touch_count"]
        self.best_avg_frames = res["mean_touch_step"]
        self.best_all_avg = res["mean_all_step"]
        self.best_code = res["grid_code"]
        self.last_duel_info = (
            f"[{res['grid_code']}] {res['touch_count']}/{res['n_total']} "
            f"{res['touch_sec']:.2f}s(all:{res['all_sec']:.2f}s)"
        )
        return res

    def maybe_save(self, model, episode, env):
        grid = env.generate_16bin_eval_grid(samples_per_bin=EVAL_SAMPLES_PER_BIN, rng=self._eval_rng)
        c_res = evaluate_16bin_grid(env, model, grid_batches=grid)
        c_tc = c_res["touch_count"]
        n_tot = c_res["n_total"]
        c_avg = c_res["mean_touch_step"]
        c_all_avg = c_res["mean_all_step"]
        c_code = c_res["grid_code"]

        saved = False
        if getattr(self, "best_model", None) is None:
            self.best_model = copy.deepcopy(model)
            self.best_model.eval()
            self.best_mask = c_tc
            self.best_avg_frames = c_avg
            self.best_all_avg = c_all_avg
            self.best_code = c_code
            self.best_ep = episode
            save_ckpt(
                self.save_path, model, episode, mode=self.mode,
                note=f"init_best [{c_code}] {c_tc}/{n_tot} all_avg={c_all_avg:.2f}",
                extra={
                    "grid_code": c_code,
                    "touch_count": c_tc, "n_total": n_tot,
                    "avg_frames": c_avg, "all_avg_frames": c_all_avg,
                    "touch_rate": c_tc / max(1, n_tot),
                },
            )
            saved = True
            self.last_duel_info = f"[{c_code}] {c_tc}/{n_tot} {c_res['touch_sec']:.2f}s(all:{c_res['all_sec']:.2f}s) [INIT]"
        else:
            b_res = evaluate_16bin_grid(env, self.best_model, grid_batches=grid)
            b_tc = b_res["touch_count"]
            b_all_avg = b_res["mean_all_step"]
            b_code = b_res["grid_code"]

            is_win = (c_tc > b_tc) or (c_tc == b_tc and c_all_avg < b_all_avg - 1e-6)
            if is_win:
                self.best_model = copy.deepcopy(model)
                self.best_model.eval()
                self.best_mask = c_tc
                self.best_avg_frames = c_avg
                self.best_all_avg = c_all_avg
                self.best_code = c_code
                self.best_ep = episode
                save_ckpt(
                    self.save_path, model, episode, mode=self.mode,
                    note=f"win [{c_code}] {c_tc}/{n_tot}(vs {b_tc}) all={c_all_avg:.2f}(vs {b_all_avg:.2f})",
                    extra={
                        "grid_code": c_code,
                        "touch_count": c_tc, "n_total": n_tot,
                        "avg_frames": c_avg, "all_avg_frames": c_all_avg,
                        "touch_rate": c_tc / max(1, n_tot),
                    },
                )
                saved = True
            self.last_duel_info = (
                f"[{c_code}] {c_tc}/{n_tot}({c_res['all_sec']:.2f}s) "
                f"vs [{b_code}] {b_tc}/{n_tot}({b_res['all_sec']:.2f}s)"
            )
        return saved, c_tc, c_code, c_avg
