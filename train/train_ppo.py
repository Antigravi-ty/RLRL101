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
)
from core.nn import ActorCritic, multi_discrete_to_controls
from core.obs_v3 import build_obs_v3
from core.env import BallReachEnv, FocusRegionConfig, save_focus_config, load_focus_config
from core.reward import RewardConfig, compute_step_reward
from train.ppo import RolloutBuffer, ppo_update, get_device
from train.checkpoint import (
    ensure_dir, save_ckpt, load_ckpt, BestTracker, ckpt_paths,
)

GAMMA            = 0.997
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


def _worker_loop(conn, mode, seed):
    random.seed(seed)
    np.random.seed(seed)
    env = BallReachEnv(mode=mode)
    cfg = RewardConfig(max_decisions=MAX_DECISIONS_STAGE0)

    sticky_ticks_left = 0
    sticky_steer = 1
    sticky_jump_last = False

    def _reset_ep():
        nonlocal sticky_ticks_left, sticky_steer, sticky_jump_last
        env.reset()
        sticky_ticks_left = 0
        sticky_steer = 1
        sticky_jump_last = False
        return env.last_obs_v3, env.snapshot()

    obs_v3, prev_snap = _reset_ep()
    step = 0
    ep_ticks = 0
    ep_reward = 0.0

    while True:
        cmd, data = conn.recv()
        if cmd == "close":
            conn.close()
            break
        elif cmd == "get_obs":
            conn.send(obs_v3)
        elif cmd == "step":
            act_np, sticky_prob = data
            act_np = np.array(act_np, dtype=np.int64, copy=True)

            if sticky_ticks_left <= 0 and sticky_prob > 0.0 and random.random() < sticky_prob:
                sticky_ticks_left = STICKY_DURATION_TICKS
                sticky_steer = random.choice((0, 1, 2))
                sticky_jump_last = (random.random() < 0.5)

            is_sticky_now = (sticky_ticks_left > 0)
            if is_sticky_now:
                act_np[0] = 0
                act_np[1] = sticky_steer
                if sticky_ticks_left <= TICKS_PER_DECISION and sticky_jump_last:
                    act_np[3] = 1
                else:
                    act_np[3] = 0

            ticks_run = 0
            boost_used_step = 0.0
            flip_triggered_step = False
            flip_speed_step = 0.0

            for _ in range(TICKS_PER_DECISION):
                if is_sticky_now:
                    c_tick = multi_discrete_to_controls(act_np)
                    if sticky_ticks_left == 1 and sticky_jump_last:
                        c_tick.jump = True
                    elif is_sticky_now:
                        c_tick.jump = False
                    sticky_ticks_left = max(0, sticky_ticks_left - 1)
                else:
                    c_tick = multi_discrete_to_controls(act_np)

                b_u, f_trig, f_spd = env.step_tick(c_tick)
                boost_used_step += b_u
                if f_trig and not flip_triggered_step:
                    flip_triggered_step = True
                    flip_speed_step = f_spd
                ticks_run += 1
                if env._touched:
                    break

            step += 1
            ep_ticks += ticks_run
            snap = env.snapshot()
            touched_now = snap["touched"]
            timeout_now = (step >= MAX_DECISIONS_STAGE0) and not touched_now
            done = touched_now or timeout_now

            r, _ = compute_step_reward(
                cfg,
                prev_dist=prev_snap["dist"],
                new_dist=snap["dist"],
                boost_used=boost_used_step,
                flip_triggered=flip_triggered_step,
                flip_trigger_speed=flip_speed_step,
                touched=touched_now,
                timeout=timeout_now,
                ticks_elapsed=ticks_run,
            )
            ep_reward += r
            exact_step = ep_ticks / float(TICKS_PER_DECISION)

            info = None
            if done:
                info = {
                    "touched": touched_now,
                    "step": exact_step,
                    "ep_reward": ep_reward,
                }
                obs_v3, prev_snap = _reset_ep()
                step = 0
                ep_ticks = 0
                ep_reward = 0.0
            else:
                obs_v3, _, _ = build_obs_v3(
                    env.car, env.arena,
                    target_pos=env.target_pos, target_time=env.target_time,
                )
                prev_snap = snap

            conn.send((obs_v3, act_np, r, done, ticks_run, info))


class MultiCoreVecEnv:
    def __init__(self, n_workers, mode="all", base_seed=42):
        import multiprocessing as mp
        ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
        self.n_workers = n_workers
        self.conns = []
        self.procs = []
        for i in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(target=_worker_loop, args=(child_conn, mode, base_seed + i * 1009), daemon=True)
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)

    def get_obs_batch(self):
        for c in self.conns:
            c.send(("get_obs", None))
        return np.stack([c.recv() for c in self.conns], axis=0)

    def step_batch(self, actions_np, sticky_prob):
        for c, act in zip(self.conns, actions_np):
            c.send(("step", (act, sticky_prob)))
        results = [c.recv() for c in self.conns]
        next_obs = np.stack([r[0] for r in results], axis=0)
        exec_acts = np.stack([r[1] for r in results], axis=0)
        rewards = np.array([r[2] for r in results], dtype=np.float32)
        dones = np.array([r[3] for r in results], dtype=np.float32)
        ticks_list = [r[4] for r in results]
        infos = [r[5] for r in results if r[5] is not None]
        return next_obs, exec_acts, rewards, dones, ticks_list, infos

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1.0)


def parse_args():
    p = argparse.ArgumentParser(description="PPO Multi-Discrete Keyboard Training")
    p.add_argument("--prefix", default=None)
    p.add_argument("--from", "--from-ckpt", dest="from_ckpt", default="latest",
                   choices=["latest", "best", "none"])
    p.add_argument("--save-prefix", "--save_prefix", dest="save_prefix", default=None)
    p.add_argument("--mode", default=os.environ.get("TRAIN_MODE", "all"),
                   choices=["all", "ground", "target", "static_aerial", "aerial", "mixed", "stage0", "stage1"])
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")),
                   help="Number of parallel CPU env workers (default: 1, set 4 or 8 for multi-core)")
    p.add_argument("--sticky-prob", type=float, default=STICKY_EXPLORE_PROB,
                   help="30-frame backward sticky exploration probability (default: 0.05)")
    p.add_argument("--focus-prob", type=float, default=None,
                   help="Optional focus region sampling probability (e.g. 0.5)")
    p.add_argument("--focus-angle", type=str, default=None,
                   help="Optional focus region relative angle range deg_min,deg_max (e.g. 135,180)")
    p.add_argument("--focus-dist", type=str, default=None,
                   help="Optional focus region horizontal distance range min,max (e.g. 150,850)")
    p.add_argument("--lr-max", "--lr_max", dest="lr_max", type=float, default=2e-4)
    p.add_argument("--lr-min", "--lr_min", dest="lr_min", type=float, default=2e-5)
    p.add_argument("--lr-period", type=int, default=50000)
    p.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    p.add_argument("--ent-coef", type=float, default=ENT_COEF)
    p.add_argument("--sym-coef", type=float, default=SYM_COEF)
    p.add_argument("--vf-warmup", type=int, default=1)
    p.add_argument("--reset-jump", default="auto", choices=["auto", "yes", "no"])
    p.add_argument("--episodes", type=int, default=int(os.environ.get("TRAIN_EPISODES", "0")))
    p.add_argument("--eval-every", type=int, default=int(os.environ.get("CKPT_EVAL_EVERY", "200")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    return p.parse_args()


def main():
    args = parse_args()
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
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    device = get_device()

    read_prefix = args.prefix if args.prefix is not None else f"{args.mode}_v0"
    save_prefix = args.save_prefix if args.save_prefix is not None else read_prefix
    latest_save_path, best_save_path = ckpt_paths(save_prefix, default_mode=args.mode)
    read_latest_path, read_best_path = ckpt_paths(read_prefix, default_mode=args.mode)

    env = BallReachEnv(mode=args.mode)
    arena = env.arena

    model = ActorCritic(obs_dim=OBS_V3_DIM, hidden=PPO_HIDDEN_DIM, action_dims=PPO_ACTION_DIMS).to(device)

    src = None
    if args.from_ckpt == "best":
        src = read_best_path
    elif args.from_ckpt == "latest":
        src = read_latest_path

    loaded_ok = False
    if src and os.path.exists(src):
        try:
            ckpt = load_ckpt(src, model, strict_dims=False)
            ep_info = ckpt.get("episode", "?")
            loaded_ok = True
            print(f"[ppo] loaded {src} (ep={ep_info}) on device={device}")
            if args.reset_jump != "no":
                with torch.no_grad():
                    b_diff = float((model.actor[-1].bias[9] - model.actor[-1].bias[10]).item())
                if args.reset_jump == "yes" or abs(b_diff) > 0.5:
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

    tracker = BestTracker(
        save_path=best_save_path,
        mode=args.mode,
    )
    if loaded_ok:
        tracker.init_baseline(model, env)
        print(f"[ppo] baseline 16-bin eval: {tracker.last_duel_info}")

    touch_history = deque(maxlen=500)
    time_history  = deque(maxlen=500)
    rew_history   = deque(maxlen=200)

    fps_t0 = time.time(); fps_dec = 0
    done_touch = done_timeout = 0
    last_ppo_stats = {}
    ppo_updates_count = 0

    ep_limit_str = "INF (Ctrl+C to stop)" if args.episodes <= 0 else str(args.episodes)
    lr_cycle = args.episodes if args.episodes > 0 else args.lr_period

    print(f"[ppo] mode={args.mode} workers={args.workers} sticky={args.sticky_prob:.0%} "
          f"max_dec={max_ep_decisions} ({max_ep_decisions*TICKS_PER_DECISION} ticks) "
          f"| episodes={ep_limit_str} rollout={args.rollout_steps} ent={args.ent_coef:.4f} sym={args.sym_coef:.2f} lr={args.lr_max:.2e}->{args.lr_min:.2e} "
          f"| save: latest={latest_save_path}, best={best_save_path}")

    last_episode = 0
    vec_env = MultiCoreVecEnv(args.workers, mode=args.mode, base_seed=seed) if args.workers > 1 else None

    try:
        if vec_env is not None:
            obs_batch = vec_env.get_obs_batch()
            while args.episodes <= 0 or last_episode < args.episodes:
                lr_now = cosine_lr_cyclic(max(1, last_episode), lr_cycle, args.lr_max, args.lr_min)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

                with torch.no_grad():
                    x_t = torch.from_numpy(obs_batch).to(device)
                    act_t, _, _, val_t = model.get_action_and_value(x_t, deterministic=False)
                    acts_np = act_t.cpu().numpy()
                    vals_np = val_t.cpu().numpy()

                next_obs_batch, exec_acts_np, rewards_np, dones_np, _, infos = vec_env.step_batch(
                    acts_np, args.sticky_prob
                )
                with torch.no_grad():
                    _, logp_exec_t, _, _ = model.get_action_and_value(
                        torch.from_numpy(obs_batch).to(device),
                        torch.from_numpy(exec_acts_np).to(device),
                    )
                    logp_exec_np = logp_exec_t.cpu().numpy()

                for i in range(args.workers):
                    buffer.add(obs_batch[i], exec_acts_np[i], logp_exec_np[i], rewards_np[i], dones_np[i], vals_np[i])
                    fps_dec += 1
                    if buffer.is_full():
                        with torch.no_grad():
                            last_val = 0.0 if dones_np[i] > 0.5 else float(
                                model.get_value(torch.from_numpy(next_obs_batch[i]).to(device)).cpu().item()
                            )
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

                obs_batch = next_obs_batch

                for info in infos:
                    last_episode += 1
                    touched = info["touched"]
                    touch_history.append(1 if touched else 0)
                    rew_history.append(info["ep_reward"])
                    if touched:
                        time_history.append(info["step"])
                        done_touch += 1
                    else:
                        done_timeout += 1

                    PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50" if last_episode <= 500 else "200"))
                    eval_line = ""
                    if last_episode % args.eval_every == 0:
                        saved, _, _, _ = tracker.maybe_save(model, last_episode, env)
                        eval_line = f" | {tracker.last_duel_info}" + (" *BEST*" if saved else "")

                    if last_episode % PRINT_EVERY == 0:
                        tr = sum(touch_history) / max(len(touch_history), 1)
                        mean_t = (sum(time_history) / max(len(time_history), 1)) if time_history else 0.0
                        mean_r = sum(rew_history) / max(len(rew_history), 1)
                        now = time.time()
                        dec_fps = fps_dec / max(now - fps_t0, 1e-6)
                        kl_str = f"kl={last_ppo_stats.get('approx_kl', 0.0):.4f} ev={last_ppo_stats.get('explained_var', 0.0):+.2f}" if last_ppo_stats else "upd=0"
                        print(f"Ep {last_episode:6d} upd={ppo_updates_count:4d} touch={tr:6.2%} "
                              f"step={mean_t:5.1f}({mean_t*4:3.0f}f) r={mean_r:6.2f} "
                              f"fps={dec_fps:5.0f} | {kl_str} lr={lr_now:.2e} | "
                              f"tk={done_touch} to={done_timeout}{eval_line}")
                        done_touch = done_timeout = 0
                        fps_t0 = time.time(); fps_dec = 0
                    elif eval_line:
                        print(f"Ep {last_episode:6d}{eval_line}")

                    if last_episode % 50 == 0:
                        save_ckpt(latest_save_path, model, last_episode, mode=args.mode, note="ppo_latest")
        else:
            ep_iter = itertools.count(1) if args.episodes <= 0 else range(1, args.episodes + 1)
            for episode in ep_iter:
                last_episode = episode
                lr_now = cosine_lr_cyclic(episode, lr_cycle, args.lr_max, args.lr_min)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

                env.reset()
                obs_v3 = env.last_obs_v3
                prev_snap = env.snapshot()

                ep_reward = 0.0
                touched = False
                touched_at = float(max_ep_decisions)
                done = False
                step = 0
                ep_ticks = 0

                sticky_ticks_left = 0
                sticky_steer = 1
                sticky_jump_last = False

                while step < max_ep_decisions:
                    if sticky_ticks_left <= 0 and args.sticky_prob > 0.0 and random.random() < args.sticky_prob:
                        sticky_ticks_left = STICKY_DURATION_TICKS
                        sticky_steer = random.choice((0, 1, 2))
                        sticky_jump_last = (random.random() < 0.5)

                    overrides = None
                    is_sticky_now = (sticky_ticks_left > 0)
                    if is_sticky_now:
                        jump_ch = 1 if (sticky_ticks_left <= TICKS_PER_DECISION and sticky_jump_last) else 0
                        overrides = {0: 0, 1: sticky_steer, 3: jump_ch}

                    if device.type == "cpu":
                        act_np, logp_val, val_scalar = model.act_step_fast(
                            obs_v3, deterministic=False, forced_action_overrides=overrides
                        )
                    else:
                        with torch.no_grad():
                            x_t = torch.from_numpy(obs_v3).to(device)
                            act_t, logp_t, _, val_t = model.get_action_and_value(x_t, deterministic=False)
                            if overrides is not None:
                                for ch_k, ch_v in overrides.items():
                                    act_t[ch_k] = int(ch_v)
                                _, logp_t, _, _ = model.get_action_and_value(x_t, act_t)
                            act_np = act_t.cpu().numpy()
                            logp_val = float(logp_t.cpu().item())
                            val_scalar = float(val_t.cpu().item())

                    ticks_run = 0
                    boost_used_step = 0.0
                    flip_triggered_step = False
                    flip_speed_step = 0.0

                    for _ in range(TICKS_PER_DECISION):
                        ctrl = multi_discrete_to_controls(act_np)
                        if is_sticky_now:
                            if sticky_ticks_left == 1 and sticky_jump_last:
                                ctrl.jump = True
                            else:
                                ctrl.jump = False
                            sticky_ticks_left = max(0, sticky_ticks_left - 1)

                        b_u, f_trig, f_spd = env.step_tick(ctrl)
                        boost_used_step += b_u
                        if f_trig and not flip_triggered_step:
                            flip_triggered_step = True
                            flip_speed_step = f_spd
                        ticks_run += 1
                        if env._touched:
                            break
                    step += 1
                    ep_ticks += ticks_run
                    fps_dec += 1

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
                        boost_used=boost_used_step,
                        flip_triggered=flip_triggered_step,
                        flip_trigger_speed=flip_speed_step,
                        touched=touched_now,
                        timeout=timeout_now,
                        ticks_elapsed=ticks_run,
                    )
                    ep_reward += r

                    if touched_now:
                        done = True; touched = True; touched_at = exact_step; done_touch += 1
                    elif timeout_now:
                        done = True; done_timeout += 1

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

                touch_history.append(1 if touched else 0)
                rew_history.append(ep_reward)
                if touched:
                    time_history.append(touched_at)

                PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50" if episode <= 500 else "200"))

                eval_line = ""
                if episode % args.eval_every == 0:
                    saved, _, _, _ = tracker.maybe_save(model, episode, env)
                    eval_line = f" | {tracker.last_duel_info}" + (" *BEST*" if saved else "")

                if episode % PRINT_EVERY == 0:
                    tr = sum(touch_history) / max(len(touch_history), 1)
                    mean_t = (sum(time_history) / max(len(time_history), 1)) if time_history else 0.0
                    mean_r = sum(rew_history) / max(len(rew_history), 1)
                    now = time.time()
                    dec_fps = fps_dec / max(now - fps_t0, 1e-6)
                    kl_str = f"kl={last_ppo_stats.get('approx_kl', 0.0):.4f} ev={last_ppo_stats.get('explained_var', 0.0):+.2f}" if last_ppo_stats else "upd=0"
                    print(f"Ep {episode:6d} upd={ppo_updates_count:4d} touch={tr:6.2%} "
                          f"step={mean_t:5.1f}({mean_t*4:3.0f}f) r={mean_r:6.2f} "
                          f"fps={dec_fps:5.0f} | {kl_str} lr={lr_now:.2e} | "
                          f"tk={done_touch} to={done_timeout}{eval_line}")
                    done_touch = done_timeout = 0
                    fps_t0 = time.time(); fps_dec = 0
                elif eval_line:
                    print(f"Ep {episode:6d}{eval_line}")

                if episode % 50 == 0:
                    save_ckpt(latest_save_path, model, episode, mode=args.mode, note="ppo_latest")

    except KeyboardInterrupt:
        print(f"\n[ppo] Training interrupted by user at episode {last_episode}. Saving latest checkpoint...")
    finally:
        if vec_env is not None:
            vec_env.close()

    save_ckpt(latest_save_path, model, last_episode, mode=args.mode, note="ppo_final")
    print(f"[ppo] Saved {latest_save_path}")


if __name__ == "__main__":
    main()
