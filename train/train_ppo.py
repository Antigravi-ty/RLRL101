import os, sys, time, random, math, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from collections import deque

from core.constants import (
    OBS_V3_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS,
    TICKS_PER_DECISION, MAX_DECISIONS_STAGE0,
    POST_TOUCH_STEPS,
    STICKY_EXPLORE_PROB, STICKY_DURATION_TICKS,
    LOGIT_CLAMP_RANGE, NUM_CURRICULUM_STAGES,
)
from core.nn import ActorCritic, multi_discrete_to_controls
from core.obs_v3 import build_obs_v3
from core.env import (
    BallReachEnv, FocusRegionConfig, AerialCurriculumState,
    save_focus_config, load_focus_config, STAGE_SHORT_LABELS,
)
from core.reward import RewardConfig, compute_step_reward, compute_vector_dist
from train.ppo import RolloutBuffer, ppo_update, get_device, compute_gae_1d
from train.checkpoint import (
    ensure_dir, save_ckpt, load_ckpt, BestTracker, ckpt_paths,
    format_8stage_code, check_stage_gate,
)

GAMMA            = 0.995
GAE_LAMBDA       = 0.95
ROLLOUT_STEPS    = int(os.environ.get("ROLLOUT_STEPS", "2048"))
PPO_EPOCHS       = int(os.environ.get("PPO_EPOCHS", "4"))
MINIBATCH_SIZE   = int(os.environ.get("MINIBATCH_SIZE", "256"))
CLIP_COEF        = 0.2
ENT_COEF         = 0.005
VF_COEF          = 0.5
SYM_COEF         = float(os.environ.get("SYM_COEF", "0.02"))
MAX_GRAD_NORM    = 0.5
REWARD_SCALE     = 0.05


def cosine_lr_cyclic(ep, period, lr_max, lr_min):
    t = ((ep - 1) % max(period, 1)) / max(period, 1)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))


def _relax_cpu_affinity():
    if hasattr(os, "sched_setaffinity"):
        try:
            n_cpus = os.cpu_count() or 4
            os.sched_setaffinity(0, set(range(n_cpus)))
        except Exception:
            pass


def _worker_loop(conn, mode, seed, logit_clamp):
    _relax_cpu_affinity()
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = BallReachEnv(mode=mode, stage_level=1)
    cfg = RewardConfig(max_decisions=MAX_DECISIONS_STAGE0)
    local_model = ActorCritic(
        obs_dim=OBS_V3_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=logit_clamp,
    ).to("cpu")
    local_model.eval()

    def _reset_ep(sticky_prob):
        env.reset(sticky_init_prob=sticky_prob)
        return env.last_obs_v3, env.snapshot()

    obs_v3, prev_snap = _reset_ep(0.0)
    step = 0
    ep_ticks = 0
    ep_reward = 0.0

    while True:
        cmd, data = conn.recv()
        if cmd == "close":
            conn.close()
            break
        elif cmd == "sync_curriculum":
            stage_lv, max_z, phase, boost_margin = data
            env.set_curriculum_state(stage_lv, max_z=max_z, phase=phase, boost_margin=boost_margin)
        elif cmd == "collect_chunk":
            state_dict, chunk_steps, sticky_prob = data
            if state_dict is not None:
                local_model.load_state_dict(state_dict)

            c_obs  = np.zeros((chunk_steps, OBS_V3_DIM), dtype=np.float32)
            c_act  = np.zeros((chunk_steps, len(PPO_ACTION_DIMS)), dtype=np.int64)
            c_logp = np.zeros(chunk_steps, dtype=np.float32)
            c_rew  = np.zeros(chunk_steps, dtype=np.float32)
            c_done = np.zeros(chunk_steps, dtype=np.float32)
            c_val  = np.zeros(chunk_steps, dtype=np.float32)
            infos  = []
            total_ticks = 0

            for t in range(chunk_steps):
                act_np, logp_val, val_scalar = local_model.act_step_fast(obs_v3, deterministic=False)
                ctrl = multi_discrete_to_controls(act_np)

                ticks_run = 0
                boost_used_step = 0.0
                boost_pickups_step = 0
                flip_triggered_step = False
                flip_speed_step = 0.0

                for _ in range(TICKS_PER_DECISION):
                    b_u, f_trig, f_spd = env.step_tick(ctrl)
                    boost_used_step += b_u
                    boost_pickups_step += env.last_boost_pickups
                    if f_trig and not flip_triggered_step:
                        flip_triggered_step = True
                        flip_speed_step = f_spd
                    ticks_run += 1
                    if env._touched:
                        break

                step += 1
                ep_ticks += ticks_run
                total_ticks += ticks_run
                snap = env.snapshot()
                touched_now = snap["touched"]
                timeout_now = (step >= MAX_DECISIONS_STAGE0) and not touched_now
                done = touched_now or timeout_now

                r, _ = compute_step_reward(
                    cfg,
                    prev_dist=prev_snap["dist"],
                    new_dist=snap["dist"],
                    prev_vec_dist=prev_snap["vec_dist"],
                    new_vec_dist=snap["vec_dist"],
                    boost_used=boost_used_step,
                    boost_pickups=boost_pickups_step,
                    flip_triggered=flip_triggered_step,
                    flip_trigger_speed=flip_speed_step,
                    touched=touched_now,
                    timeout=timeout_now,
                    ticks_elapsed=ticks_run,
                )
                ep_reward += r
                exact_step = ep_ticks / float(TICKS_PER_DECISION)

                c_obs[t]  = obs_v3
                c_act[t]  = act_np
                c_logp[t] = logp_val
                c_rew[t]  = r
                c_done[t] = float(done)
                c_val[t]  = val_scalar

                if done:
                    infos.append({
                        "touched": touched_now,
                        "step": exact_step,
                        "ep_reward": ep_reward,
                        "car_speed": snap["car_speed"],
                        "stage_idx": env.current_stage_idx,
                        "is_high_band_sample": env.is_high_band_sample,
                    })
                    obs_v3, prev_snap = _reset_ep(sticky_prob)
                    step = 0
                    ep_ticks = 0
                    ep_reward = 0.0
                else:
                    obs_v3, _, _ = build_obs_v3(
                        env.car, env.arena,
                        target_pos=env.target_pos, target_time=env.target_time,
                    )
                    prev_snap = snap

            if c_done[-1] > 0.5:
                last_val = 0.0
            else:
                _, _, last_val = local_model.act_step_fast(obs_v3, deterministic=False)

            conn.send((c_obs, c_act, c_logp, c_rew, c_done, c_val, float(last_val), total_ticks, infos))


class MultiCoreVecEnv:
    def __init__(self, n_workers, mode="curriculum", base_seed=42, logit_clamp=LOGIT_CLAMP_RANGE):
        import multiprocessing as mp
        ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
        self.n_workers = n_workers
        self.conns = []
        self.procs = []
        for i in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker_loop,
                args=(child_conn, mode, base_seed + i * 1009, logit_clamp),
                daemon=True,
            )
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)

    def sync_curriculum(self, stage_level: int, max_z: float, phase: int, boost_margin: int):
        payload = (int(stage_level), float(max_z), int(phase), int(boost_margin))
        for c in self.conns:
            c.send(("sync_curriculum", payload))

    def collect_rollout(self, model, total_steps: int, sticky_prob: float):
        chunk_steps = max(1, (total_steps + self.n_workers - 1) // self.n_workers)
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        for c in self.conns:
            c.send(("collect_chunk", (cpu_sd, chunk_steps, sticky_prob)))
        results = [c.recv() for c in self.conns]
        return results

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1.0)


class CurriculumTracker:
    def __init__(
        self,
        init_stage_level=1,
        unlock_threshold=0.70,
        window_size=200,
        min_samples_per_stage=40,
        aerial_state: AerialCurriculumState = None,
    ):
        self.stage_level = max(1, min(NUM_CURRICULUM_STAGES, int(init_stage_level)))
        self.unlock_threshold = float(unlock_threshold)
        self.min_samples_per_stage = int(min_samples_per_stage)
        self.stage_windows = [deque(maxlen=window_size) for _ in range(NUM_CURRICULUM_STAGES)]
        self.aerial_state = aerial_state if aerial_state is not None else AerialCurriculumState()
        self.high_band_window = deque(maxlen=100)

    def record_episode(self, stage_idx: int, touched: bool, is_high_band: bool = False):
        if 0 <= stage_idx < NUM_CURRICULUM_STAGES:
            self.stage_windows[stage_idx].append(1 if touched else 0)
        if stage_idx in (4, 5) and is_high_band:
            self.high_band_window.append(1 if touched else 0)
            self.aerial_state.high_band_touches = sum(self.high_band_window)
            self.aerial_state.high_band_total = len(self.high_band_window)

    def get_stage_rates(self):
        rates = []
        for i in range(NUM_CURRICULUM_STAGES):
            w = self.stage_windows[i]
            rates.append((sum(w) / len(w)) if len(w) > 0 else 0.0)
        return rates

    def get_8stage_code(self, unlocked_count=None) -> str:
        u = self.stage_level if unlocked_count is None else unlocked_count
        return format_8stage_code(self.get_stage_rates(), unlocked_stages=u)

    def check_eval_gate(self, min_touch=0.70, warmup_eps=60, total_eps=0) -> bool:
        if total_eps < warmup_eps:
            return False
        rates = self.get_stage_rates()
        return check_stage_gate(rates, self.stage_level, min_touch=min_touch)

    def maybe_advance(self, auto_unlock=True):
        if not auto_unlock:
            return None
        rates = self.get_stage_rates()

        # Check if all active required stages have enough samples and >= threshold
        req_count = 6 if self.stage_level == 7 else self.stage_level
        for i in range(req_count):
            if len(self.stage_windows[i]) < self.min_samples_per_stage:
                return None
            if rates[i] < self.unlock_threshold - 1e-6:
                return None

        # Special handling for Static Aerial sub-curriculum (Stages 5 & 6, indices 4 & 5)
        if self.stage_level in (5, 6):
            ast = self.aerial_state
            if len(self.high_band_window) >= 100:
                hb_rate = sum(self.high_band_window) / len(self.high_band_window)
                if hb_rate >= self.unlock_threshold - 1e-6:
                    if ast.phase == 0:
                        ast.phase = 1
                        ast.max_z = 500.0
                        self.high_band_window.clear()
                        ast.high_band_touches = 0
                        ast.high_band_total = 0
                        return f"[aerial-ladder] Z<=400 mastered ({hb_rate:.0%}) -> Phase 1: Z=500, Boost=100"
                    elif ast.phase == 1:
                        if ast.max_z < 2000.0 - 1e-3:
                            ast.max_z = min(2000.0, ast.max_z + 100.0)
                            self.high_band_window.clear()
                            ast.high_band_touches = 0
                            ast.high_band_total = 0
                            return f"[aerial-ladder] High band >=70% -> Expanded max_z to {ast.max_z:.0f} (Boost=100)"
                        else:
                            ast.phase = 2
                            ast.boost_margin = 30
                            self.high_band_window.clear()
                            ast.high_band_touches = 0
                            ast.high_band_total = 0
                            return f"[aerial-ladder] Z=2000 mastered -> Phase 2: CSV Boost +30 margin"
                    elif ast.phase == 2:
                        if ast.boost_margin > 0:
                            ast.boost_margin = max(0, ast.boost_margin - 12)
                            self.high_band_window.clear()
                            ast.high_band_touches = 0
                            ast.high_band_total = 0
                            return f"[aerial-ladder] CSV Boost margin reduced by 12 -> +{ast.boost_margin:02d}"
                        else:
                            ast.phase = 3
                            self.high_band_window.clear()
                            ast.high_band_touches = 0
                            ast.high_band_total = 0
                            return "[aerial-ladder] CSV +00 mastered -> Static Aerial Ladder DONE!"
            # Gate stage 5->6 on clearing at least Z<=400 (phase>=1), and stage 6->7 (Dynamic Aerial) on completing the full ladder (phase>=3)
            if self.stage_level == 5 and ast.phase < 1:
                return None
            if self.stage_level == 6 and ast.phase < 3:
                return None

        if self.stage_level < NUM_CURRICULUM_STAGES:
            self.stage_level += 1
            return (f"[curriculum] Stage {self.stage_level - 1} mastered "
                    f"([{self.get_8stage_code(self.stage_level - 1)}]) -> Unlocked Stage {self.stage_level}/8 "
                    f"({STAGE_SHORT_LABELS[self.stage_level - 1]})!")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="PPO Multi-Discrete Keyboard Training (8-Stage Curriculum)")
    p.add_argument("--prefix", default=None)
    p.add_argument("--from", "--from-ckpt", dest="from_ckpt", default="latest",
                   choices=["latest", "best", "none"])
    p.add_argument("--save-prefix", "--save_prefix", dest="save_prefix", default=None)
    p.add_argument("--mode", default=os.environ.get("TRAIN_MODE", "curriculum"),
                   choices=["curriculum", "auto", "ground", "target", "static_aerial", "aerial", "all", "mixed", "stage0", "stage1"])
    p.add_argument("--stage-level", type=int, default=None,
                   help="Initial unlocked curriculum stages 1..8 (default: 1 for curriculum/ground, 8 for all)")
    p.add_argument("--no-auto-unlock", action="store_true",
                   help="Disable automatic unlocking of next stage at 70%% touch rate")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")),
                   help="Number of parallel CPU env workers (default: 1, set 2 or 4 for multi-core)")
    p.add_argument("--sticky-prob", type=float, default=STICKY_EXPLORE_PROB,
                   help="Episode-initial 30-frame backward exploration probability (default: 0.02)")
    p.add_argument("--logit-clamp", type=float, default=LOGIT_CLAMP_RANGE,
                   help="Soft logit clamp range during training sampling to prevent head collapse (default: 4.5, 0=off)")
    p.add_argument("--eval-min-touch", type=float, default=0.70,
                   help="Minimum touch rate across required stages to trigger Best evaluation (default: 0.70)")
    p.add_argument("--eval-warmup-eps", type=int, default=60,
                   help="Warmup episodes before evaluating (default: 60)")
    p.add_argument("--sync-eval", action="store_true",
                   help="Run evaluation synchronously instead of background thread")
    p.add_argument("--focus-prob", type=float, default=None)
    p.add_argument("--focus-angle", type=str, default=None)
    p.add_argument("--focus-dist", type=str, default=None)
    p.add_argument("--lr-max", "--lr_max", dest="lr_max", type=float, default=2.5e-4)
    p.add_argument("--lr-min", "--lr_min", dest="lr_min", type=float, default=2e-5)
    p.add_argument("--lr-period", type=int, default=50000)
    p.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    p.add_argument("--ent-coef", type=float, default=ENT_COEF)
    p.add_argument("--sym-coef", type=float, default=SYM_COEF)
    p.add_argument("--vf-warmup", type=int, default=1)
    p.add_argument("--reset-jump", default="no", choices=["auto", "yes", "no"])
    p.add_argument("--episodes", type=int, default=int(os.environ.get("TRAIN_EPISODES", "0")))
    p.add_argument("--eval-every", type=int, default=int(os.environ.get("CKPT_EVAL_EVERY", "200")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    return p.parse_args()


def main():
    args = parse_args()
    _relax_cpu_affinity()
    ensure_dir()

    if args.mode == "stage0":
        args.mode = "ground"
    elif args.mode == "stage1":
        args.mode = "target"

    if args.focus_prob is not None or args.focus_angle is not None or args.focus_dist is not None:
        fc = load_focus_config()
        if args.focus_prob is not None:
            fc.prob = float(args.focus_prob)
            fc.enabled = fc.prob > 0.0
        else:
            fc.enabled = True
        if args.focus_angle:
            lo, hi = [float(x) for x in args.focus_angle.split(",")]
            fc.angle_deg_min, fc.angle_deg_max = lo, hi
        if args.focus_dist:
            lo, hi = [float(x) for x in args.focus_dist.split(",")]
            fc.dist_min, fc.dist_max = lo, hi
        save_focus_config(fc)

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1" if args.workers > 1 else "2")))
    device = get_device()

    read_prefix = args.prefix if args.prefix is not None else f"{args.mode}_v0"
    save_prefix = args.save_prefix if args.save_prefix is not None else read_prefix
    latest_save_path, best_save_path = ckpt_paths(save_prefix, default_mode=args.mode)
    read_latest_path, read_best_path = ckpt_paths(read_prefix, default_mode=args.mode)

    if args.stage_level is not None:
        init_stage_lv = max(1, min(NUM_CURRICULUM_STAGES, int(args.stage_level)))
    elif args.mode in ("all", "mixed"):
        init_stage_lv = 8
    elif args.mode == "static_aerial":
        init_stage_lv = 5
    elif args.mode == "aerial":
        init_stage_lv = 7
    else:
        init_stage_lv = 1

    env = BallReachEnv(mode=args.mode, stage_level=init_stage_lv)
    arena = env.arena

    model = ActorCritic(
        obs_dim=OBS_V3_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=args.logit_clamp,
    ).to(device)

    src = None
    if args.from_ckpt == "best":
        src = read_best_path
    elif args.from_ckpt == "latest":
        src = read_latest_path

    loaded_ok = False
    start_episode = 0
    if src and os.path.exists(src):
        try:
            ckpt = load_ckpt(src, model, strict_dims=False)
            ep_info = ckpt.get("episode", 0)
            if isinstance(ep_info, int) and ep_info > 0:
                start_episode = ep_info
            if args.stage_level is None and "stage_level" in ckpt:
                init_stage_lv = max(1, min(NUM_CURRICULUM_STAGES, int(ckpt["stage_level"])))
                env.stage_level = init_stage_lv
            if "aerial_max_z" in ckpt:
                env.aerial_state.max_z = float(ckpt["aerial_max_z"])
                env.aerial_state.phase = int(ckpt.get("aerial_phase", 0))
                env.aerial_state.boost_margin = int(ckpt.get("aerial_boost_margin", 30))
            loaded_ok = True
            print(f"[ppo] loaded {src} (ep={ep_info}, stage_level={init_stage_lv}) on device={device}")
            if args.reset_jump != "no":
                with torch.no_grad():
                    b_diff = float((model.actor[-1].bias[9] - model.actor[-1].bias[10]).item())
                if args.reset_jump == "yes" or b_diff > 1.8:
                    model.revive_jump_head(off_bias=0.0, weight_std=0.01)
                    print(f"[ppo] revived ch3 (Jump) head (prev bias_diff={b_diff:.2f} -> 0.0)")
        except Exception as e:
            print(f"[ppo] ckpt incompatible ({e}), starting fresh PPO weights")
    else:
        print(f"[ppo] starting fresh PPO weights on device={device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max, eps=1e-5)
    buffer = RolloutBuffer(args.rollout_steps, OBS_V3_DIM, action_channels=len(PPO_ACTION_DIMS))

    max_ep_decisions = MAX_DECISIONS_STAGE0
    cfg = RewardConfig(max_decisions=max_ep_decisions)

    curr_tracker = CurriculumTracker(
        init_stage_level=init_stage_lv,
        unlock_threshold=args.eval_min_touch,
        aerial_state=env.aerial_state,
    )

    tracker = BestTracker(
        save_path=best_save_path,
        mode=args.mode,
    )

    touch_history = deque(maxlen=500)
    time_history  = deque(maxlen=500)
    spd_history   = deque(maxlen=500)
    rew_history   = deque(maxlen=200)

    fps_t0 = time.time(); frames_count = 0
    done_touch = done_timeout = 0
    last_ppo_stats = {}
    ppo_updates_count = 0

    ep_limit_str = "INF (Ctrl+C to stop)" if args.episodes <= 0 else str(args.episodes)
    lr_cycle = args.lr_period

    print(f"[ppo] mode={args.mode} stage={curr_tracker.stage_level}/8 workers={args.workers} "
          f"sticky_init={args.sticky_prob:.0%} clamp=±{args.logit_clamp:.1f} "
          f"max_dec={max_ep_decisions} ({max_ep_decisions*TICKS_PER_DECISION} ticks) "
          f"| episodes={ep_limit_str} rollout={args.rollout_steps} ent={args.ent_coef:.4f} sym={args.sym_coef:.2f} "
          f"lr={args.lr_max:.2e}->{args.lr_min:.2e} | eval_gate={args.eval_min_touch:.0%} "
          f"| save: latest={latest_save_path}, best={best_save_path}")

    last_episode = start_episode
    vec_env = (
        MultiCoreVecEnv(args.workers, mode=args.mode, base_seed=seed, logit_clamp=args.logit_clamp)
        if args.workers > 1 else None
    )
    if vec_env is not None:
        vec_env.sync_curriculum(
            curr_tracker.stage_level,
            env.aerial_state.max_z,
            env.aerial_state.phase,
            env.aerial_state.boost_margin,
        )

    def _build_ckpt_extra():
        return {
            "stage_level": curr_tracker.stage_level,
            "grid_code": curr_tracker.get_8stage_code(),
            "aerial_max_z": env.aerial_state.max_z,
            "aerial_phase": env.aerial_state.phase,
            "aerial_boost_margin": env.aerial_state.boost_margin,
        }

    def _handle_episode_end(ep_num, touched, step_taken, ep_rew, car_spd, stage_idx, is_high_band, lr_now):
        nonlocal done_touch, done_timeout, fps_t0, frames_count
        touch_history.append(1 if touched else 0)
        rew_history.append(ep_rew)
        curr_tracker.record_episode(stage_idx, touched, is_high_band=is_high_band)
        if touched:
            time_history.append(step_taken)
            spd_history.append(car_spd)
            done_touch += 1
        else:
            done_timeout += 1

        # Poll completed async eval result if any
        completed_eval = tracker.pop_completed_result()
        if completed_eval is not None:
            _, _, _, _, async_msg = completed_eval
            print(async_msg)

        # Check stage advancement
        auto_unlock = (not args.no_auto_unlock) and (args.mode in ("curriculum", "auto"))
        adv_msg = curr_tracker.maybe_advance(auto_unlock=auto_unlock)
        if adv_msg:
            env.stage_level = curr_tracker.stage_level
            if vec_env is not None:
                vec_env.sync_curriculum(
                    curr_tracker.stage_level,
                    env.aerial_state.max_z,
                    env.aerial_state.phase,
                    env.aerial_state.boost_margin,
                )
            print(adv_msg)

        PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50" if (ep_num - start_episode) <= 500 else "200"))
        eval_line = ""
        if ep_num % args.eval_every == 0:
            gate_ok = curr_tracker.check_eval_gate(
                min_touch=args.eval_min_touch,
                warmup_eps=args.eval_warmup_eps,
                total_eps=(ep_num - start_episode),
            )
            if gate_ok:
                if args.sync_eval:
                    saved, _, _, _ = tracker.maybe_save(
                        model, ep_num, env,
                        active_stages=curr_tracker.stage_level,
                        extra_meta=_build_ckpt_extra(),
                    )
                    eval_line = f" | {tracker.last_duel_info}" + (" *BEST*" if saved else "")
                else:
                    started = tracker.trigger_async_eval(
                        model, ep_num,
                        active_stages=curr_tracker.stage_level,
                        aerial_state=env.aerial_state,
                        extra_meta=_build_ckpt_extra(),
                    )
                    eval_line = f" | eval-async(S{curr_tracker.stage_level})" if started else " | eval-busy"
            else:
                eval_line = f" | eval-skip([{curr_tracker.get_8stage_code()}]<{args.eval_min_touch:.0%})"

        if ep_num % PRINT_EVERY == 0:
            tr = sum(touch_history) / max(len(touch_history), 1)
            mean_t = (sum(time_history) / max(len(time_history), 1)) if time_history else 0.0
            mean_spd = (sum(spd_history) / max(len(spd_history), 1)) if spd_history else 0.0
            mean_r = sum(rew_history) / max(len(rew_history), 1)
            now = time.time()
            real_fps = frames_count / max(now - fps_t0, 1e-6)
            kl_str = (
                f"kl={last_ppo_stats.get('approx_kl', 0.0):.4f} "
                f"ent={last_ppo_stats.get('entropy', 0.0):.2f} "
                f"ev={last_ppo_stats.get('explained_var', 0.0):+.2f}"
                if last_ppo_stats else "upd=0"
            )
            stage_code = curr_tracker.get_8stage_code()
            air_str = f" air={env.aerial_state.code_str()}" if curr_tracker.stage_level >= 5 else ""
            print(f"Ep {ep_num:6d} upd={ppo_updates_count:4d} S{curr_tracker.stage_level}[{stage_code}]{air_str} "
                  f"touch={tr:6.2%} step={mean_t:5.1f}({mean_t*4:3.0f}f) spd={mean_spd:4.0f} r={mean_r:6.2f} "
                  f"fps={real_fps:5.0f} | {kl_str} lr={lr_now:.2e} | "
                  f"tk={done_touch} to={done_timeout}{eval_line}")
            done_touch = done_timeout = 0
            fps_t0 = time.time(); frames_count = 0
        elif eval_line and not eval_line.startswith(" | eval-skip"):
            print(f"Ep {ep_num:6d}{eval_line}")

        if ep_num % 50 == 0:
            save_ckpt(
                latest_save_path, model, ep_num,
                mode=args.mode, note="ppo_latest",
                extra=_build_ckpt_extra(),
            )

    try:
        if vec_env is not None:
            while args.episodes <= 0 or (last_episode - start_episode) < args.episodes:
                lr_now = cosine_lr_cyclic(max(1, last_episode), lr_cycle, args.lr_max, args.lr_min)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

                buffer.reset()
                worker_chunks = vec_env.collect_rollout(model, args.rollout_steps, args.sticky_prob)

                all_infos = []
                for (c_obs, c_act, c_logp, c_rew, c_done, c_val, last_val, w_ticks, infos) in worker_chunks:
                    frames_count += w_ticks
                    adv_arr, ret_arr = compute_gae_1d(
                        c_rew, c_val, c_done, last_val,
                        gamma=GAMMA, gae_lambda=GAE_LAMBDA, reward_scale=REWARD_SCALE,
                    )
                    buffer.add_trajectory(c_obs, c_act, c_logp, c_rew, c_done, c_val, adv_arr, ret_arr)
                    all_infos.extend(infos)

                is_vf_warmup = bool(loaded_ok and ppo_updates_count < args.vf_warmup)
                last_ppo_stats = ppo_update(
                    model, optimizer, buffer,
                    ppo_epochs=PPO_EPOCHS,
                    minibatch_size=MINIBATCH_SIZE,
                    clip_coef=CLIP_COEF,
                    ent_coef=args.ent_coef,
                    vf_coef=VF_COEF,
                    sym_coef=args.sym_coef,
                    max_grad_norm=MAX_GRAD_NORM,
                    vf_only=is_vf_warmup,
                    device=device,
                )
                ppo_updates_count += 1

                for info in all_infos:
                    last_episode += 1
                    _handle_episode_end(
                        last_episode,
                        touched=info["touched"],
                        step_taken=info["step"],
                        ep_rew=info["ep_reward"],
                        car_spd=info.get("car_speed", 0.0),
                        stage_idx=info.get("stage_idx", 0),
                        is_high_band=info.get("is_high_band_sample", False),
                        lr_now=lr_now,
                    )
                    if args.episodes > 0 and (last_episode - start_episode) >= args.episodes:
                        break
        else:
            ep_iter = (
                itertools.count(start_episode + 1)
                if args.episodes <= 0
                else range(start_episode + 1, start_episode + args.episodes + 1)
            )
            for episode in ep_iter:
                last_episode = episode
                lr_now = cosine_lr_cyclic(episode, lr_cycle, args.lr_max, args.lr_min)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

                env.reset(sticky_init_prob=args.sticky_prob)
                obs_v3 = env.last_obs_v3
                prev_snap = env.snapshot()

                ep_reward = 0.0
                touched = False
                touched_at = float(max_ep_decisions)
                touch_car_speed = 0.0
                done = False
                step = 0
                ep_ticks = 0

                while step < max_ep_decisions:
                    if device.type == "cpu":
                        act_np, logp_val, val_scalar = model.act_step_fast(obs_v3, deterministic=False)
                    else:
                        with torch.no_grad():
                            x_t = torch.from_numpy(obs_v3).to(device)
                            act_t, logp_t, _, val_t = model.get_action_and_value(x_t, deterministic=False)
                            act_np = act_t.cpu().numpy()
                            logp_val = float(logp_t.cpu().item())
                            val_scalar = float(val_t.cpu().item())

                    ctrl = multi_discrete_to_controls(act_np)
                    ticks_run = 0
                    boost_used_step = 0.0
                    boost_pickups_step = 0
                    flip_triggered_step = False
                    flip_speed_step = 0.0

                    for _ in range(TICKS_PER_DECISION):
                        b_u, f_trig, f_spd = env.step_tick(ctrl)
                        boost_used_step += b_u
                        boost_pickups_step += env.last_boost_pickups
                        if f_trig and not flip_triggered_step:
                            flip_triggered_step = True
                            flip_speed_step = f_spd
                        ticks_run += 1
                        if env._touched:
                            break
                    step += 1
                    ep_ticks += ticks_run
                    frames_count += ticks_run

                    snap = env.snapshot()
                    touched_now = snap["touched"]
                    timeout_now = (step == max_ep_decisions) and not touched_now

                    if touched_now and POST_TOUCH_STEPS > 0:
                        for _ in range(POST_TOUCH_STEPS * TICKS_PER_DECISION):
                            env.car.set_controls(ctrl)
                            arena.step(1)

                    next_obs_v3, _, _ = build_obs_v3(
                        env.car, arena,
                        target_pos=env.target_pos, target_time=env.target_time,
                    )

                    exact_step = ep_ticks / float(TICKS_PER_DECISION)
                    r, _ = compute_step_reward(
                        cfg,
                        prev_dist=prev_snap["dist"],
                        new_dist=snap["dist"],
                        prev_vec_dist=prev_snap["vec_dist"],
                        new_vec_dist=snap["vec_dist"],
                        boost_used=boost_used_step,
                        boost_pickups=boost_pickups_step,
                        flip_triggered=flip_triggered_step,
                        flip_trigger_speed=flip_speed_step,
                        touched=touched_now,
                        timeout=timeout_now,
                        ticks_elapsed=ticks_run,
                    )
                    ep_reward += r

                    if touched_now:
                        done = True
                        touched = True
                        touched_at = exact_step
                        touch_car_speed = snap["car_speed"]
                    elif timeout_now:
                        done = True

                    buffer.add(obs_v3, act_np, logp_val, r, float(done), val_scalar)

                    obs_v3 = next_obs_v3
                    prev_snap = snap

                    if buffer.is_full():
                        with torch.no_grad():
                            if done:
                                last_val = 0.0
                            else:
                                last_val = float(model.get_value(torch.from_numpy(obs_v3).to(device)).cpu().item())
                        buffer.compute_returns_and_advantages(
                            last_val, gamma=GAMMA, gae_lambda=GAE_LAMBDA, reward_scale=REWARD_SCALE
                        )
                        is_vf_warmup = bool(loaded_ok and ppo_updates_count < args.vf_warmup)
                        last_ppo_stats = ppo_update(
                            model, optimizer, buffer,
                            ppo_epochs=PPO_EPOCHS,
                            minibatch_size=MINIBATCH_SIZE,
                            clip_coef=CLIP_COEF,
                            ent_coef=args.ent_coef,
                            vf_coef=VF_COEF,
                            sym_coef=args.sym_coef,
                            max_grad_norm=MAX_GRAD_NORM,
                            vf_only=is_vf_warmup,
                            device=device,
                        )
                        buffer.reset()
                        ppo_updates_count += 1

                    if done:
                        break

                _handle_episode_end(
                    episode,
                    touched=touched,
                    step_taken=touched_at,
                    ep_rew=ep_reward,
                    car_spd=touch_car_speed,
                    stage_idx=env.current_stage_idx,
                    is_high_band=env.is_high_band_sample,
                    lr_now=lr_now,
                )

    except KeyboardInterrupt:
        print(f"\n[ppo] Training interrupted by user at episode {last_episode}. Saving latest checkpoint...")
    finally:
        if vec_env is not None:
            vec_env.close()
        tracker.wait_and_close()

    save_ckpt(
        latest_save_path, model, last_episode,
        mode=args.mode, note="ppo_final",
        extra=_build_ckpt_extra(),
    )
    print(f"[ppo] Saved {latest_save_path}")


if __name__ == "__main__":
    main()
