"""train/trainer.py —— 1v1 最快开球 (Fastest Kickoff Duel) 8-Worker 并行自博弈 + MPS 梯度更新主循环。"""
import os, sys, time, random, math, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from collections import deque

from core.constants import (
    OBS_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS,
    TICKS_PER_DECISION, MAX_DECISIONS, SETTLE_TICKS,
    LOGIT_CLAMP_RANGE, DEFAULT_GAMMA, DEFAULT_GAE_LAMBDA, DEFAULT_WORKERS,
)
from core.policy_net import ActorCritic, multi_discrete_to_controls
from core.env import KickoffEnv
from core.reward import RewardConfig, compute_step_reward
from train.ppo_loss import RolloutBuffer, ppo_update, get_device, compute_gae_1d
from train.checkpoint import (
    ensure_dir, save_ckpt, load_ckpt, BestTracker, ckpt_paths,
)
from train.vec_env import MultiCoreVecEnv, _relax_cpu_affinity
from train.logger import TrainingLogger

GAMMA          = float(os.environ.get("GAMMA", str(DEFAULT_GAMMA)))
GAE_LAMBDA     = float(os.environ.get("GAE_LAMBDA", str(DEFAULT_GAE_LAMBDA)))
ROLLOUT_STEPS  = int(os.environ.get("ROLLOUT_STEPS", "4096"))
PPO_EPOCHS     = int(os.environ.get("PPO_EPOCHS", "4"))
MINIBATCH_SIZE = int(os.environ.get("MINIBATCH_SIZE", "512"))
CLIP_COEF      = 0.2
ENT_COEF       = 0.005
VF_COEF        = 0.5
SYM_COEF       = float(os.environ.get("SYM_COEF", "0.02"))
MAX_GRAD_NORM  = 0.5
REWARD_SCALE   = 0.05


def cosine_lr_cyclic(ep, period, lr_max, lr_min):
    t = ((ep - 1) % max(period, 1)) / max(period, 1)
    return lr_min + (lr_max - lr_min) * 0.5 * (1.0 + math.cos(math.pi * t))


def parse_args():
    p = argparse.ArgumentParser(description="PPO 1v1 Fastest Kickoff Self-Play Trainer (8 Workers + MPS Backprop)")
    p.add_argument("--prefix", default=None)
    p.add_argument("--from", "--from-ckpt", dest="from_ckpt", default="latest",
                   choices=["latest", "best", "none"])
    p.add_argument("--save-prefix", "--save_prefix", dest="save_prefix", default="ko_v1")
    p.add_argument("--save-path", dest="save_path", default=None)
    p.add_argument("--best-path", dest="best_path", default=None)
    p.add_argument("--log-file", dest="log_file", default=None)
    p.add_argument("--mode", default="kickoff")
    p.add_argument("--slot", type=int, default=None, help="Lock training to a single kickoff slot (0..4) or None for all 5 slots")
    p.add_argument("--workers", "--num-workers", dest="workers", type=int, default=int(os.environ.get("WORKERS", str(DEFAULT_WORKERS))))
    p.add_argument("--logit-clamp", type=float, default=LOGIT_CLAMP_RANGE)
    p.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor gamma (default: 0.998 for 6s kickoff)")
    p.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=GAE_LAMBDA, help="GAE lambda (default: 0.975)")
    p.add_argument("--lr-max", "--lr_max", "--lr", dest="lr_max", type=float, default=3e-4)
    p.add_argument("--lr-min", "--lr_min", "--lr-end", dest="lr_min", type=float, default=3e-5)
    p.add_argument("--lr-period", "--lr-decay-episodes", dest="lr_period", type=int, default=300000)
    p.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    p.add_argument("--batch-size", "--minibatch-size", dest="batch_size", type=int, default=MINIBATCH_SIZE)
    p.add_argument("--epochs", "--ppo-epochs", dest="epochs", type=int, default=PPO_EPOCHS)
    p.add_argument("--ent-coef", type=float, default=ENT_COEF)
    p.add_argument("--sym-coef", type=float, default=SYM_COEF)
    p.add_argument("--vf-warmup", type=int, default=1)
    p.add_argument("--episodes", type=int, default=int(os.environ.get("TRAIN_EPISODES", "0")))
    p.add_argument("--eval-every", type=int, default=int(os.environ.get("CKPT_EVAL_EVERY", "100")))
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    return p.parse_args()


def main():
    args = parse_args()
    _relax_cpu_affinity()
    ensure_dir()

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    default_main_threads = max(2, min(int(args.workers), os.cpu_count() or 4))
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", str(default_main_threads))))
    device = get_device(prefer_mps=True)

    read_prefix = args.prefix if args.prefix is not None else (args.save_prefix or "ko_v1")
    save_prefix = args.save_prefix if args.save_prefix is not None else read_prefix
    if args.save_path:
        base_name = os.path.basename(args.save_path)
        if base_name.endswith("_latest.pt"):
            save_prefix = base_name[:-len("_latest.pt")]
        elif base_name.endswith(".pt"):
            save_prefix = base_name[:-3]

    latest_save_path, best_save_path = ckpt_paths(save_prefix, default_mode=args.mode)
    if args.save_path:
        latest_save_path = args.save_path
        os.makedirs(os.path.dirname(os.path.abspath(latest_save_path)), exist_ok=True)
    if args.best_path:
        best_save_path = args.best_path
        os.makedirs(os.path.dirname(os.path.abspath(best_save_path)), exist_ok=True)

    read_latest_path, read_best_path = ckpt_paths(read_prefix, default_mode=args.mode)

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_txt_path = args.log_file if args.log_file else os.path.join(log_dir, f"{save_prefix}.log")
    os.makedirs(os.path.dirname(os.path.abspath(log_txt_path)), exist_ok=True)

    train_logger = TrainingLogger(log_txt_path=log_txt_path, save_prefix=save_prefix)

    def _log_print(msg: str, metrics_dict: dict = None):
        train_logger.log(msg, metrics_dict)

    eval_env = KickoffEnv(mode=args.mode, slot_idx=args.slot, seed=seed)

    model = ActorCritic(
        obs_dim=OBS_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=args.logit_clamp,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    src = None
    if args.from_ckpt == "best":
        src = read_best_path
    elif args.from_ckpt == "latest":
        src = read_latest_path

    loaded_ok = False
    start_episode = 0
    if src and os.path.exists(src):
        try:
            ckpt = load_ckpt(src, model, strict_dims=True)
            ep_info = ckpt.get("episode", 0)
            if isinstance(ep_info, int) and ep_info > 0:
                start_episode = ep_info
            loaded_ok = True
            _log_print(f"[ppo] loaded {src} (ep={ep_info}, params={n_params:,}) on device={device}")
        except Exception as e:
            print(f"[ppo] ckpt incompatible ({e}), starting fresh PPO weights")
    else:
        print(f"[ppo] starting fresh PPO weights (obs_dim={OBS_DIM}, hidden={PPO_HIDDEN_DIM}, params={n_params:,}) on device={device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max, eps=1e-5)
    buffer = RolloutBuffer(args.rollout_steps, OBS_DIM, action_channels=len(PPO_ACTION_DIMS))

    max_ep_decisions = MAX_DECISIONS
    cfg = RewardConfig(max_decisions=max_ep_decisions)

    tracker = BestTracker(
        save_path=best_save_path,
        mode=args.mode,
        load_existing=(args.from_ckpt != "none"),
    )

    touch_history  = deque(maxlen=400)
    frames_history = deque(maxlen=400)
    spd_history    = deque(maxlen=400)
    rew_history    = deque(maxlen=200)

    fps_t0 = time.time(); frames_count = 0
    done_touch = done_timeout = 0
    last_ppo_stats = {}
    ppo_updates_count = 0

    ep_limit_str = "INF (Ctrl+C to stop)" if args.episodes <= 0 else str(args.episodes)
    lr_cycle = args.lr_period

    _log_print(
        f"[ppo] 1v1-Kickoff-Duel settle={SETTLE_TICKS}f max={max_ep_decisions}dec({max_ep_decisions*TICKS_PER_DECISION}f=6.0s) "
        f"| device={device} workers={args.workers} params={n_params:,} "
        f"| gamma={args.gamma} lam={args.gae_lambda} rollout={args.rollout_steps} mb={args.batch_size} ep_ppo={args.epochs} "
        f"ent={args.ent_coef:.4f} sym={args.sym_coef:.2f} lr={args.lr_max:.2e}->{args.lr_min:.2e} "
        f"| save: latest={latest_save_path}, best={best_save_path}, log={log_txt_path}"
    )

    last_episode = start_episode
    vec_env = (
        MultiCoreVecEnv(
            args.workers, mode=args.mode, base_seed=seed,
            logit_clamp=args.logit_clamp, reward_cfg=cfg,
        )
        if args.workers > 1 else None
    )

    def _handle_episode_end(ep_num, touched, ep_frames, ep_rew, car_spd, lr_now):
        nonlocal done_touch, done_timeout, fps_t0, frames_count
        touch_history.append(1 if touched else 0)
        rew_history.append(ep_rew)
        if touched:
            frames_history.append(ep_frames)
            spd_history.append(car_spd)
            done_touch += 1
        else:
            done_timeout += 1

        PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50" if (ep_num - start_episode) <= 500 else "100"))
        eval_line = ""
        if ep_num % args.eval_every == 0:
            saved, c_tc, _, c_ticks, c_res = tracker.maybe_save(model, ep_num, eval_env)
            eval_line = f" | {tracker.last_duel_info}"
            _log_print(f"Ep {ep_num:6d}{eval_line}", {
                "event": "eval_done",
                "ep": ep_num,
                "cand_touch": c_tc,
                "cand_avg_frames": c_ticks,
                "best_avg_frames": tracker.best_all_ticks,
                "is_new_best": saved,
                "slot_frames": c_res["slot_frames"],
                "msg": tracker.last_duel_info,
            })

        if ep_num % PRINT_EVERY == 0:
            tr = sum(touch_history) / max(len(touch_history), 1)
            mean_f = (sum(frames_history) / max(len(frames_history), 1)) if frames_history else 720.0
            min_f = min(frames_history) if frames_history else 720.0
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
            line_msg = (
                f"Ep {ep_num:6d} upd={ppo_updates_count:4d} "
                f"touch={tr:6.2%} avg={mean_f:5.1f}f({mean_f/120.0:4.2f}s) min={min_f:3.0f}f({min_f/120.0:4.2f}s) "
                f"spd={mean_spd:4.0f} r={mean_r:6.2f} fps={real_fps:5.0f} | {kl_str} lr={lr_now:.2e} | "
                f"tk={done_touch} to={done_timeout}"
            )
            _log_print(line_msg, {
                "event": "train",
                "ep": ep_num,
                "upd": ppo_updates_count,
                "touch_rate": round(tr, 4),
                "mean_frames": round(mean_f, 1),
                "min_frames": round(min_f, 1),
                "mean_spd": round(mean_spd, 1),
                "mean_rew": round(mean_r, 2),
                "fps": round(real_fps, 0),
                "kl": round(last_ppo_stats.get("approx_kl", 0.0), 5) if last_ppo_stats else 0.0,
                "ent": round(last_ppo_stats.get("entropy", 0.0), 3) if last_ppo_stats else 0.0,
                "ev": round(last_ppo_stats.get("explained_var", 0.0), 3) if last_ppo_stats else 0.0,
                "lr": round(lr_now, 7),
                "tk": done_touch,
                "to": done_timeout,
            })
            done_touch = done_timeout = 0
            fps_t0 = time.time(); frames_count = 0

        if ep_num % 50 == 0:
            save_ckpt(latest_save_path, model, ep_num, mode=args.mode, note="ppo_latest")

    try:
        if vec_env is not None:
            while args.episodes <= 0 or (last_episode - start_episode) < args.episodes:
                lr_now = cosine_lr_cyclic(max(1, last_episode - start_episode), lr_cycle, args.lr_max, args.lr_min)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

                buffer.reset()
                worker_chunks = vec_env.collect_rollout(model, args.rollout_steps)

                all_infos = []
                for (c_obs, c_act, c_logp, c_rew, c_done, c_val, last_val, w_ticks, infos) in worker_chunks:
                    frames_count += w_ticks
                    adv_arr, ret_arr = compute_gae_1d(
                        c_rew, c_val, c_done, last_val,
                        gamma=args.gamma, gae_lambda=args.gae_lambda, reward_scale=REWARD_SCALE,
                    )
                    buffer.add_trajectory(c_obs, c_act, c_logp, c_rew, c_done, c_val, adv_arr, ret_arr)
                    all_infos.extend(infos)

                is_vf_warmup = bool(loaded_ok and ppo_updates_count < args.vf_warmup)
                last_ppo_stats = ppo_update(
                    model, optimizer, buffer,
                    ppo_epochs=args.epochs,
                    minibatch_size=args.batch_size,
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
                        ep_frames=info.get("frames", info["step"] * TICKS_PER_DECISION),
                        ep_rew=info["ep_reward"],
                        car_spd=info.get("car_speed", 0.0),
                        lr_now=lr_now,
                    )
                    if args.episodes > 0 and (last_episode - start_episode) >= args.episodes:
                        break
        else:
            single_env = KickoffEnv(mode=args.mode, slot_idx=args.slot, seed=seed)
            ep_iter = (
                itertools.count(start_episode + 1)
                if args.episodes <= 0
                else range(start_episode + 1, start_episode + args.episodes + 1)
            )
            for episode in ep_iter:
                last_episode = episode
                lr_now = cosine_lr_cyclic(max(1, episode - start_episode), lr_cycle, args.lr_max, args.lr_min)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

                single_env.reset()
                obs_b, obs_o = single_env.last_obs_pair
                prev_snap_b, prev_snap_o = single_env.snapshot_pair()

                ep_rew_b = 0.0
                ep_rew_o = 0.0
                touched = False
                touch_car_speed = 0.0
                step = 0
                ep_ticks = 0

                while step < max_ep_decisions:
                    with torch.no_grad():
                        xb = torch.from_numpy(obs_b).to(device)
                        xo = torch.from_numpy(obs_o).to(device)
                        act_b_t, logp_b_t, _, val_b_t = model.get_action_and_value(xb, deterministic=False)
                        act_o_t, logp_o_t, _, val_o_t = model.get_action_and_value(xo, deterministic=False)
                        act_b = act_b_t.cpu().numpy()
                        act_o = act_o_t.cpu().numpy()
                        logp_b, val_b = float(logp_b_t.cpu().item()), float(val_b_t.cpu().item())
                        logp_o, val_o = float(logp_o_t.cpu().item()), float(val_o_t.cpu().item())

                    ctrl_b = multi_discrete_to_controls(act_b)
                    ctrl_o = multi_discrete_to_controls(act_o)
                    ticks_run = 0
                    for _ in range(TICKS_PER_DECISION):
                        single_env.step_duel_tick(ctrl_b, ctrl_o)
                        ticks_run += 1
                        if single_env._touched:
                            break

                    step += 1
                    ep_ticks += ticks_run
                    frames_count += ticks_run

                    snap_b, snap_o = single_env.snapshot_pair()
                    b_touched = snap_b["touched"]
                    o_touched = snap_o["touched"]
                    any_touched = b_touched or o_touched
                    timeout_now = (step == max_ep_decisions) and not any_touched
                    done = any_touched or timeout_now

                    r_b, _ = compute_step_reward(
                        cfg,
                        prev_dist=prev_snap_b["dist"],
                        new_dist=snap_b["dist"],
                        init_dist=snap_b["init_dist"],
                        touched=b_touched,
                        opp_touched=o_touched,
                        timeout=timeout_now,
                        ticks_elapsed=ticks_run,
                        ep_ticks=ep_ticks,
                        max_decisions_ep=max_ep_decisions,
                    )
                    r_o, _ = compute_step_reward(
                        cfg,
                        prev_dist=prev_snap_o["dist"],
                        new_dist=snap_o["dist"],
                        init_dist=snap_o["init_dist"],
                        touched=o_touched,
                        opp_touched=b_touched,
                        timeout=timeout_now,
                        ticks_elapsed=ticks_run,
                        ep_ticks=ep_ticks,
                        max_decisions_ep=max_ep_decisions,
                    )
                    ep_rew_b += r_b
                    ep_rew_o += r_o

                    buffer.add(obs_b, act_b, logp_b, r_b, float(done), val_b)
                    buffer.add(obs_o, act_o, logp_o, r_o, float(done), val_o)

                    if any_touched:
                        touched = True
                        touch_car_speed = snap_b["car_speed"] if b_touched else snap_o["car_speed"]

                    obs_b, obs_o = single_env.get_obs_pair()
                    prev_snap_b, prev_snap_o = snap_b, snap_o

                    if buffer.is_full():
                        with torch.no_grad():
                            last_val = 0.0 if done else float(model.get_value(torch.from_numpy(obs_o).to(device)).cpu().item())
                        buffer.compute_returns_and_advantages(
                            last_val, gamma=args.gamma, gae_lambda=args.gae_lambda, reward_scale=REWARD_SCALE
                        )
                        is_vf_warmup = bool(loaded_ok and ppo_updates_count < args.vf_warmup)
                        last_ppo_stats = ppo_update(
                            model, optimizer, buffer,
                            ppo_epochs=args.epochs,
                            minibatch_size=args.batch_size,
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

                win_rew = ep_rew_b if snap_b["touched"] else (ep_rew_o if snap_o["touched"] else 0.5 * (ep_rew_b + ep_rew_o))
                _handle_episode_end(
                    episode,
                    touched=touched,
                    ep_frames=ep_ticks,
                    ep_rew=win_rew,
                    car_spd=touch_car_speed,
                    lr_now=lr_now,
                )

    except KeyboardInterrupt:
        print(f"\n[ppo] Training interrupted by user at episode {last_episode}. Saving latest checkpoint...")
    finally:
        if vec_env is not None:
            vec_env.close()
        tracker.wait_and_close()

    save_ckpt(latest_save_path, model, last_episode, mode=args.mode, note="ppo_final")
    print(f"[ppo] Saved {latest_save_path}")


if __name__ == "__main__":
    main()
