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
)
from core.nn import ActorCritic, multi_discrete_to_controls
from core.obs_v3 import build_obs_v3
from core.env import BallReachEnv
from core.reward import RewardConfig, compute_step_reward
from train.ppo import RolloutBuffer, ppo_update, get_device
from train.checkpoint import (
    ensure_dir, save_ckpt, load_ckpt, BestTracker, ckpt_paths,
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


def parse_args():
    p = argparse.ArgumentParser(description="PPO Multi-Discrete Keyboard Training")
    p.add_argument("--prefix", default=None,
                   help="Checkpoint prefix to load from (default: <mode>_v0 -> checkpoints/<mode>_v0_latest.pt)")
    p.add_argument("--from", "--from-ckpt", dest="from_ckpt", default="latest",
                   choices=["latest", "best", "none"],
                   help="Which checkpoint to load: latest / best / none (default: latest)")
    p.add_argument("--save-prefix", "--save_prefix", dest="save_prefix", default=None,
                   help="Prefix for saving checkpoints (defaults to --prefix or <mode>_v0)")
    p.add_argument("--mode", default=os.environ.get("TRAIN_MODE", "ground"),
                   choices=["ground", "target", "static_aerial", "aerial", "mixed", "stage0", "stage1"],
                   help="Training mode: ground (alias stage0) / target (alias stage1) / static_aerial / aerial / mixed (default: ground)")
    p.add_argument("--lr-max", "--lr_max", dest="lr_max", type=float, default=2e-4,
                   help="Cosine LR start (default: 2e-4)")
    p.add_argument("--lr-min", "--lr_min", dest="lr_min", type=float, default=2e-5,
                   help="Cosine LR end (default: 2e-5)")
    p.add_argument("--lr-period", type=int, default=50000,
                   help="Cosine LR cycle period in episodes when running indefinitely (default: 50000)")
    p.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS,
                   help="Steps per PPO rollout update (default: 2048)")
    p.add_argument("--ent-coef", type=float, default=ENT_COEF,
                   help="Entropy bonus coefficient (default: 0.005, recommended <= 0.008)")
    p.add_argument("--sym-coef", type=float, default=SYM_COEF,
                   help="Left-Right mirror symmetry distillation coefficient (default: 0.02)")
    p.add_argument("--vf-warmup", type=int, default=1,
                   help="Number of Critic-only warmup updates when loading from a checkpoint (default: 1)")
    p.add_argument("--reset-jump", default="auto", choices=["auto", "yes", "no"],
                   help="Revive dead Jump head (ch3) when loading checkpoint: auto / yes / no (default: auto)")
    p.add_argument("--episodes", type=int, default=int(os.environ.get("TRAIN_EPISODES", "0")),
                   help="Total episodes to train (default: 0 = infinite until Ctrl+C)")
    p.add_argument("--eval-every", type=int, default=int(os.environ.get("CKPT_EVAL_EVERY", "200")),
                   help="Evaluate scenarios every N episodes (default: 200)")
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")),
                   help="Random seed (default: 42)")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir()

    if args.mode == "stage0":
        args.mode = "ground"
    elif args.mode == "stage1":
        args.mode = "target"

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
            ckpt = load_ckpt(src, model)
            ep_info = ckpt.get("episode", "?")
            loaded_ok = True
            print(f"[ppo] loaded {src} (ep={ep_info}) on device={device}")
            if args.reset_jump != "no":
                with torch.no_grad():
                    b_diff = float((model.actor[-1].bias[9] - model.actor[-1].bias[10]).item())
                if args.reset_jump == "yes" or b_diff > 1.5:
                    model.revive_jump_head(off_bias=0.0, weight_std=0.01)
                    print(f"[ppo] revived ch3 (Jump) head for Flip exploration (prev bias_diff={b_diff:.2f} -> 0.0)")
        except Exception as e:
            print(f"[ppo] ckpt incompatible ({e}), starting fresh PPO weights")
    else:
        print(f"[ppo] starting fresh PPO weights on device={device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max, eps=1e-5)
    buffer = RolloutBuffer(args.rollout_steps, OBS_V3_DIM, action_channels=len(PPO_ACTION_DIMS))

    max_ep_decisions = MAX_DECISIONS_STAGE0
    cfg = RewardConfig(
        max_decisions=max_ep_decisions,
        target_cpa_bonus=120.0 if args.mode in ("target", "stage1") else 0.0,
    )

    tracker = BestTracker(
        save_path=best_save_path,
        mode=args.mode,
    )
    if loaded_ok:
        tracker.init_baseline(model, env)
        print(f"[ppo] baseline eval ({args.mode}) {tracker.last_duel_info}")

    touch_history = deque(maxlen=500)
    time_history  = deque(maxlen=500)
    gvel_history  = deque(maxlen=500)
    rew_history   = deque(maxlen=200)

    fps_t0 = time.time(); fps_dec = 0
    done_touch = done_timeout = 0
    last_ppo_stats = {}
    ppo_updates_count = 0

    ep_limit_str = "INF (Ctrl+C to stop)" if args.episodes <= 0 else str(args.episodes)
    lr_cycle = args.episodes if args.episodes > 0 else args.lr_period

    print(f"[ppo] mode={args.mode} "
          f"max_dec={max_ep_decisions} ({max_ep_decisions*TICKS_PER_DECISION} ticks) "
          f"| episodes={ep_limit_str} rollout={args.rollout_steps} ent={args.ent_coef:.4f} sym={args.sym_coef:.2f} lr={args.lr_max:.2e}->{args.lr_min:.2e} "
          f"| save: latest={latest_save_path}, best={best_save_path}")

    ep_iter = itertools.count(1) if args.episodes <= 0 else range(1, args.episodes + 1)
    last_episode = 0

    try:
        for episode in ep_iter:
            last_episode = episode
            lr_now = cosine_lr_cyclic(episode, lr_cycle, args.lr_max, args.lr_min)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            env.reset()
            obs_v3 = env.last_obs_v3
            prev_snap = env.snapshot()
            min_dist = prev_snap["dist"]
            peak_closing = 0.0
            closing_accum = 0.0

            ep_reward = 0.0
            touched = False
            touched_at = float(max_ep_decisions)
            touch_gvel = 0.0
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
                wasted_ticks = 0.0
                boost_used_step = 0.0
                for _ in range(TICKS_PER_DECISION):
                    wasted_ticks += float(env.step_with_waste_check(ctrl))
                    boost_used_step += float(getattr(env, "last_boost_used", 0.0))
                    ticks_run += 1
                    if env._touched:
                        break
                step += 1
                ep_ticks += ticks_run
                fps_dec += 1

                snap = env.snapshot()
                touched_now = snap["touched"]
                timeout_now = (step == max_ep_decisions) and not touched_now

                cpa_dist_val = None
                cpa_time_err_val = 0.0
                if touched_now and POST_TOUCH_STEPS > 0:
                    for _ in range(POST_TOUCH_STEPS * TICKS_PER_DECISION):
                        env.car.set_controls(ctrl)
                        arena.step(1)
                    snap_post = env.snapshot()
                    best_gvel = max(snap["ball_vel_to_goal"], snap_post["ball_vel_to_goal"])
                    snap["ball_vel_to_goal"] = best_gvel
                    touch_gvel = best_gvel
                elif touched_now:
                    touch_gvel = snap["ball_vel_to_goal"]

                if touched_now and cfg.target_cpa_bonus > 0.0:
                    cpa_dist_val, _, cpa_time_err_val = env.evaluate_shot_cpa()

                next_obs_v3, _, _ = build_obs_v3(
                    env.car, arena,
                    target_pos=env.target_pos, target_time=env.target_time,
                )

                exact_step = ep_ticks / float(TICKS_PER_DECISION)
                r, detail = compute_step_reward(
                    cfg,
                    prev_dist=prev_snap["dist"], new_dist=snap["dist"],
                    step=exact_step,
                    touched=touched_now, timeout=timeout_now,
                    air_time=snap["air_time"], align=snap["align"],
                    prev_dist_h=prev_snap["dist_h"], new_dist_h=snap["dist_h"],
                    closing=snap["closing"],
                    peak_closing=peak_closing, closing_accum=closing_accum,
                    ball_vel_to_goal=snap["ball_vel_to_goal"],
                    ticks_elapsed=ticks_run,
                    boost_wasted_ticks=wasted_ticks,
                    boost_used=boost_used_step,
                    min_dist=min_dist,
                    cpa_dist=cpa_dist_val,
                    cpa_time_err=cpa_time_err_val,
                )
                min_dist      = detail["min_dist"]
                peak_closing  = detail["peak_closing"]
                closing_accum = detail["closing_accum"]
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
                gvel_history.append(touch_gvel)

            PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50" if episode <= 500 else "200"))

            eval_line = ""
            if episode % args.eval_every == 0:
                saved, _, _, _ = tracker.maybe_save(model, episode, env)
                eval_line = f" | {tracker.last_duel_info}" + (" *BEST*" if saved else "")

            if episode % PRINT_EVERY == 0:
                tr = sum(touch_history) / max(len(touch_history), 1)
                mean_t = (sum(time_history) / max(len(time_history), 1)) if time_history else 0.0
                mean_gv = (sum(gvel_history) / max(len(gvel_history), 1)) if gvel_history else 0.0
                mean_r = sum(rew_history) / max(len(rew_history), 1)
                now = time.time()
                dec_fps = fps_dec / max(now - fps_t0, 1e-6)
                kl_str = f"kl={last_ppo_stats.get('approx_kl', 0.0):.4f} ev={last_ppo_stats.get('explained_var', 0.0):+.2f}" if last_ppo_stats else "upd=0"
                print(f"Ep {episode:6d} upd={ppo_updates_count:4d} touch={tr:6.2%} "
                      f"step={mean_t:5.1f}({mean_t*4:3.0f}f) gvel={mean_gv:5.0f} r={mean_r:6.2f} "
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

    save_ckpt(latest_save_path, model, last_episode, mode=args.mode, note="ppo_final")
    print(f"[ppo] Saved {latest_save_path}")


if __name__ == "__main__":
    main()
