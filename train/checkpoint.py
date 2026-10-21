"""train/checkpoint.py —— 1v1 最快开球权重保存、纯 NumPy/PyTorch 双兼容加载与 5 标准开球位极速评测 (BestTracker)。"""
import os
import copy
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from core.constants import (
    OBS_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS,
    TICKS_PER_DECISION, MAX_DECISIONS, NUM_KICKOFF_SLOTS, KICKOFF_SLOT_SHORT,
)
from core.policy_net import ActorCritic, multi_discrete_to_controls
from core.reward import RewardConfig, compute_step_reward

CKPT_DIR    = "./checkpoints"
CKPT_LATEST = os.path.join(CKPT_DIR, "ko_v1_latest.pt")
CKPT_BEST   = os.path.join(CKPT_DIR, "ko_v1_best.pt")


def rate_to_digit_char(rate: float) -> str:
    d = max(0, min(9, int(float(rate) * 10.0 + 1e-6)))
    return str(d)


def ensure_dir():
    os.makedirs(CKPT_DIR, exist_ok=True)


def normalize_prefix(prefix=None, default_mode="ko"):
    if not prefix:
        return "ko_v1"
    p = str(prefix).strip().strip("_")
    return p or "ko_v1"


def ckpt_name(prefix=None, tag="latest", default_mode="ko"):
    norm = normalize_prefix(prefix, default_mode=default_mode)
    return f"{norm}_{tag}.pt"


def ckpt_paths(prefix=None, default_mode="ko"):
    ensure_dir()
    norm = normalize_prefix(prefix, default_mode=default_mode)
    latest_path = os.path.join(CKPT_DIR, f"{norm}_latest.pt")
    best_path   = os.path.join(CKPT_DIR, f"{norm}_best.pt")
    return latest_path, best_path


def discover_prefixes():
    if not os.path.isdir(CKPT_DIR):
        return ["ko_v1"]
    found = set()
    for name in os.listdir(CKPT_DIR):
        for suffix in ("_latest.pt", "_best.pt"):
            if name.endswith(suffix) and len(name) > len(suffix):
                mid = name[:-len(suffix)]
                found.add(mid)
                break
    return sorted(found) or ["ko_v1"]


def save_ckpt(path, model, episode, mode="kickoff", note="", extra=None):
    ensure_dir()
    payload = {
        "arch":        "ppo_kickoff_mlp",
        "mode":        mode,
        "state_dict":  model.state_dict(),
        "obs_dim":     getattr(model, "obs_dim", OBS_DIM),
        "hidden_dim":  getattr(model, "hidden_dim", PPO_HIDDEN_DIM),
        "action_dims": getattr(model, "action_dims", PPO_ACTION_DIMS),
        "episode":     episode,
        "note":        note,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _load_ckpt_file_raw(path, map_location="cpu"):
    if torch is not None:
        return torch.load(path, map_location=map_location, weights_only=False)
    import zipfile
    import pickle
    from collections import OrderedDict
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        pkl_name = next((n for n in names if n.endswith("/data.pkl") or n == "data.pkl"), None)
        if pkl_name is None:
            raise RuntimeError(f"Invalid .pt archive: {path}")
        storages = {}
        for name in names:
            if "/data/" in name and not name.endswith("/"):
                key = name.split("/data/")[1]
                storages[key] = z.read(name)

        class _NumpyPtUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        arr = np.frombuffer(storage, dtype=np.float32)
                        if len(size) == 0:
                            return arr[storage_offset]
                        total = int(np.prod(size))
                        return arr[storage_offset:storage_offset + total].reshape(size).copy()
                    return _rebuild
                if module == "torch" and name in ("FloatStorage", "LongStorage", "HalfStorage", "DoubleStorage"):
                    return name
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                return super().find_class(module, name)

            def persistent_load(self, pid):
                key = str(pid[2])
                return storages[key]

        return _NumpyPtUnpickler(z.open(pkl_name)).load()


def load_ppo_model(path, device="cpu"):
    ckpt = _load_ckpt_file_raw(path, map_location=device)
    obs_dim = ckpt.get("obs_dim", OBS_DIM)
    hidden_dim = ckpt.get("hidden_dim", PPO_HIDDEN_DIM)
    action_dims = ckpt.get("action_dims", PPO_ACTION_DIMS)
    model = ActorCritic(obs_dim=obs_dim, hidden=hidden_dim, action_dims=action_dims).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def load_ckpt(path, net, strict_dims=True):
    ckpt = _load_ckpt_file_raw(path, map_location="cpu")
    if strict_dims and (
        ckpt.get("obs_dim", OBS_DIM) != getattr(net, "obs_dim", OBS_DIM)
        or ckpt.get("hidden_dim", PPO_HIDDEN_DIM) != getattr(net, "hidden_dim", PPO_HIDDEN_DIM)
    ):
        raise ValueError(
            f"PPO arch mismatch: ckpt=({ckpt.get('obs_dim')},{ckpt.get('hidden_dim')}) "
            f"!= net=({net.obs_dim},{net.hidden_dim})"
        )
    net.load_state_dict(ckpt["state_dict"])
    return ckpt


def evaluate_5slot_kickoff(
    env,
    model,
    max_decisions=MAX_DECISIONS,
    **kwargs,
):
    """在全部 5 个标准开球位 (0:DR, 1:DL, 2:BR, 3:BL, 4:CT) 上进行确定性 (argmax) 极速评测。"""
    dev_type = "cpu"
    if torch is not None and hasattr(model, "parameters"):
        try:
            dev = next(model.parameters()).device
            dev_type = dev.type
        except Exception:
            dev_type = "cpu"

    cfg = RewardConfig(max_decisions=max_decisions)
    max_ticks = max_decisions * TICKS_PER_DECISION

    bin_counts = []
    stage_rates = []
    slot_frames = []
    slot_secs = []
    slot_speeds = []
    touch_steps = []
    all_steps = []
    rcum_list = []
    jump_eps = 0
    flip_eps = 0

    for slot_idx in range(NUM_KICKOFF_SLOTS):
        env.reset(slot_idx=slot_idx)
        obs_curr = env.last_obs
        prev_snap = env.snapshot()
        prev_dist = prev_snap["dist"]

        ep_ticks = 0
        ep_rcum = 0.0
        ep_jumped = False
        ep_flipped = False
        touched_now = False
        touch_spd = 0.0

        for step in range(1, max_decisions + 1):
            if dev_type == "cpu" and hasattr(model, "act_step_fast"):
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
            for _ in range(TICKS_PER_DECISION):
                _, f_trig, _ = env.step_tick(ctrl)
                ticks_run += 1
                ep_ticks += 1
                if f_trig:
                    ep_flipped = True
                if env._touched_by_car[0]:
                    break

            snap_now = env.snapshot()
            new_dist = snap_now["dist"]
            touched_now = snap_now["touched"]
            timeout_now = (step == max_decisions) and not touched_now

            r_step, _ = compute_step_reward(
                cfg,
                prev_dist=prev_dist,
                new_dist=new_dist,
                init_dist=snap_now["init_dist"],
                touched=touched_now,
                opp_touched=False,
                timeout=timeout_now,
                ticks_elapsed=ticks_run,
                ep_ticks=ep_ticks,
                max_decisions_ep=max_decisions,
            )
            ep_rcum += r_step
            prev_dist = new_dist

            if touched_now:
                touch_spd = snap_now["car_speed"]
                break

            obs_curr, _, _ = env.get_obs()

        if ep_jumped:
            jump_eps += 1
        if ep_flipped:
            flip_eps += 1

        exact_step = ep_ticks / float(TICKS_PER_DECISION) if touched_now else float(max_decisions)
        frames_val = ep_ticks if touched_now else max_ticks

        bin_counts.append(1 if touched_now else 0)
        stage_rates.append(1.0 if touched_now else 0.0)
        slot_frames.append(frames_val)
        slot_secs.append(frames_val / 120.0)
        slot_speeds.append(touch_spd)
        if touched_now:
            touch_steps.append(exact_step)
        all_steps.append(exact_step)
        rcum_list.append(ep_rcum)

    grid_code = "".join("1" if c > 0 else "0" for c in bin_counts)
    total_touch = sum(bin_counts)
    n_total = NUM_KICKOFF_SLOTS
    mean_touch_step = float(np.mean(touch_steps)) if touch_steps else float(max_decisions)
    mean_all_step = float(np.mean(all_steps)) if all_steps else float(max_decisions)
    touch_ticks = mean_touch_step * float(TICKS_PER_DECISION)
    all_ticks = float(np.mean(slot_frames))
    min_ticks = float(np.min(slot_frames))

    slot_summary = " ".join(
        f"{KICKOFF_SLOT_SHORT[i]}:{slot_frames[i]}f({slot_secs[i]:.2f}s)"
        for i in range(NUM_KICKOFF_SLOTS)
    )

    return {
        "grid_code": grid_code,
        "grid_str": grid_code,
        "slot_summary": slot_summary,
        "slot_touches": bin_counts,
        "slot_frames": slot_frames,
        "slot_secs": slot_secs,
        "slot_speeds": slot_speeds,
        "bin_counts": bin_counts,
        "bin_touches": bin_counts,
        "stage_rates": stage_rates,
        "touch_count": total_touch,
        "n_total": n_total,
        "touch_rate": total_touch / max(1, n_total),
        "mean_touch_step": mean_touch_step,
        "mean_all_step": mean_all_step,
        "touch_dec": mean_touch_step,
        "all_dec": mean_all_step,
        "touch_ticks": touch_ticks,
        "all_ticks": all_ticks,
        "min_ticks": min_ticks,
        "touch_sec": touch_ticks / 120.0,
        "all_sec": all_ticks / 120.0,
        "min_sec": min_ticks / 120.0,
        "jump_eps": jump_eps,
        "flip_eps": flip_eps,
        "mean_rcum": float(np.mean(rcum_list)) if rcum_list else 0.0,
        "all_steps": all_steps,
    }


evaluate_stage_grid = evaluate_5slot_kickoff
evaluate_6stage_grid = evaluate_5slot_kickoff


class BestTracker:
    """追踪并保存 5 个标准开球位总用时最短 (Fastest) 的最优模型权重。"""
    def __init__(self, save_path=None, mode="kickoff", load_existing=False, **kwargs):
        self.save_path = save_path or CKPT_BEST
        self.mode = mode
        self.best_model = None
        self.best_touch_count = 0
        self.best_all_ticks = float("inf")
        self.best_slot_frames = [720] * NUM_KICKOFF_SLOTS
        self.best_ep = 0
        self.last_duel_info = ""

        if load_existing and self.save_path and os.path.exists(self.save_path):
            try:
                m, ck = load_ppo_model(self.save_path, device="cpu")
                self.best_model = m
                self.best_touch_count = int(ck.get("touch_count", 0))
                self.best_all_ticks = float(ck.get("all_avg_ticks", float("inf")))
                self.best_ep = int(ck.get("episode", 0))
            except Exception:
                pass

    def pop_completed_result(self):
        return None

    def wait_and_close(self):
        pass

    def maybe_save(self, model, episode, env, extra_meta=None, **kwargs):
        c_res = evaluate_5slot_kickoff(env, model)
        c_tc = c_res["touch_count"]
        n_tot = c_res["n_total"]
        c_ticks = c_res["all_ticks"]
        c_sec = c_res["all_sec"]

        meta = {
            "grid_code": c_res["grid_code"],
            "touch_count": c_tc,
            "n_total": n_tot,
            "all_avg_ticks": c_ticks,
            "all_avg_sec": c_sec,
            "min_ticks": c_res["min_ticks"],
            "slot_frames": c_res["slot_frames"],
            "touch_rate": c_tc / max(1, n_tot),
        }
        if extra_meta:
            meta.update(extra_meta)

        is_win = (
            self.best_model is None
            or (c_tc > self.best_touch_count)
            or (c_tc == self.best_touch_count and c_ticks < self.best_all_ticks - 1e-4)
        )

        prev_ticks = self.best_all_ticks
        prev_tc = self.best_touch_count

        if is_win:
            self.best_model = copy.deepcopy(model).to("cpu")
            self.best_model.eval()
            self.best_touch_count = c_tc
            self.best_all_ticks = c_ticks
            self.best_slot_frames = list(c_res["slot_frames"])
            self.best_ep = episode
            save_ckpt(
                self.save_path, model, episode, mode=self.mode,
                note=f"5slot {c_tc}/{n_tot} avg={c_ticks:.1f}f({c_sec:.2f}s) [{c_res['slot_summary']}]",
                extra=meta,
            )

        if prev_ticks == float("inf"):
            self.last_duel_info = (
                f"5Slot={c_tc}/{n_tot} avg={c_ticks:.1f}f({c_sec:.2f}s) [{c_res['slot_summary']}] *INIT_BEST*"
            )
        else:
            self.last_duel_info = (
                f"5Slot={c_tc}/{n_tot} avg={c_ticks:.1f}f({c_sec:.2f}s) "
                f"vs best={prev_tc}/{n_tot}({prev_ticks:.1f}f) [{c_res['slot_summary']}]"
                + (" *NEW FASTEST*" if is_win else "")
            )
        return is_win, c_tc, c_res["grid_code"], c_ticks, c_res
