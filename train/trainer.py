"""train/trainer.py —— 静态球大王 PPO 多核 4-Phase 课程训练主循环 (70/30 双池 + 5向独立门控 + 状态修饰码)。"""
import os, sys, time, json, random, math, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
from collections import deque

from core.constants import (
    OBS_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS,
    TICKS_PER_DECISION, MAX_DECISIONS,
    POST_TOUCH_STEPS,
    LOGIT_CLAMP_RANGE, NUM_CURRICULUM_STAGES,
    POOL_FRONTIER,
)
from core.policy_net import ActorCritic, multi_discrete_to_controls
from core.env import BallReachEnv
from core.curriculum import (
    NUM_CURRICULUM_STEPS, get_curriculum_step_spec, stage_level_to_step_idx,
    format_milestone_tag, format_phase_coord_code,
)
from core.reward import RewardConfig, compute_step_reward
from train.ppo_loss import RolloutBuffer, ppo_update, get_device, compute_gae_1d
from train.checkpoint import (
    ensure_dir, save_ckpt, load_ckpt, BestTracker, ckpt_paths,
)
from train.vec_env import MultiCoreVecEnv, _relax_cpu_affinity
from train.curriculum_tracker import CurriculumTracker
from train.logger import TrainingLogger

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


def _parse_stage_or_step(val_str):
    if val_str is None:
        return None, None
    s = str(val_str).strip().lower()
    if s in ("", "auto", "default", "curriculum", "none"):
        return None, None
    if s in ("all", "done", "max", "full"):
        return NUM_CURRICULUM_STAGES, NUM_CURRICULUM_STEPS - 1
    if s.startswith("p"):
        try:
            body = s[1:]
            if "." in body:
                p_str, sub_str = body.split(".", 1)
                p_num, sub_num = int(p_str), int(sub_str)
            else:
                p_num, sub_num = int(body), 1
            for idx in range(NUM_CURRICULUM_STEPS):
                sp = get_curriculum_step_spec(idx)
                if sp.phase == p_num and sp.substep == sub_num:
                    return p_num, idx
        except Exception:
            pass
    v = int(float(s))
    if 1 <= v <= 5:
        return v, stage_level_to_step_idx(v)
    step_idx = max(0, min(NUM_CURRICULUM_STEPS - 1, v))
    return get_curriculum_step_spec(step_idx).phase, step_idx


def parse_args():
    p = argparse.ArgumentParser(description="PPO Static Ball Master Training (4-Phase Car-First Curriculum + BoostPad Attention)")
    p.add_argument("--prefix", default=None)
    p.add_argument("--from", "--from-ckpt", dest="from_ckpt", default="latest",
                   choices=["latest", "best", "none"])
    p.add_argument("--save-prefix", "--save_prefix", dest="save_prefix", default="stm_v1")
    p.add_argument("--save-path", dest="save_path", default=None)
    p.add_argument("--best-path", dest="best_path", default=None)
    p.add_argument("--log-file", dest="log_file", default=None)
    p.add_argument("--mode", default=os.environ.get("TRAIN_MODE", "curriculum"),
                   choices=["curriculum", "auto", "all", "mixed"])
    p.add_argument("--stage-level", "--stage", dest="stage_level", type=str, default=None,
                   help="Initial curriculum phase (1..4, 'auto', 'P2.03', or step index 0..36)")
    p.add_argument("--step", "--step-idx", dest="step_idx", type=int, default=None,
                   help="Initial curriculum step index (0..36)")
    p.add_argument("--no-auto-unlock", action="store_true")
    p.add_argument("--workers", "--num-workers", dest="workers", type=int, default=int(os.environ.get("WORKERS", "6")))
    p.add_argument("--logit-clamp", type=float, default=LOGIT_CLAMP_RANGE)
    p.add_argument("--eval-min-touch", "--success-gate", dest="eval_min_touch", type=float, default=0.70)
    p.add_argument("--eval-warmup-eps", type=int, default=60)
    p.add_argument("--sync-eval", action="store_true")
    p.add_argument("--lr-max", "--lr_max", "--lr", dest="lr_max", type=float, default=3e-4)
    p.add_argument("--lr-min", "--lr_min", "--lr-end", dest="lr_min", type=float, default=3e-5)
    p.add_argument("--lr-period", "--lr-decay-episodes", dest="lr_period", type=int, default=500000)
    p.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    p.add_argument("--batch-size", "--minibatch-size", dest="batch_size", type=int, default=MINIBATCH_SIZE)
    p.add_argument("--epochs", "--ppo-epochs", dest="epochs", type=int, default=PPO_EPOCHS)
    p.add_argument("--ent-coef", type=float, default=ENT_COEF)
    p.add_argument("--sym-coef", type=float, default=SYM_COEF)
    p.add_argument("--vf-warmup", type=int, default=1)
    p.add_argument("--revive-jump", action="store_true", help="Force soft-revive of saturated Jump/Boost output heads on load")
    p.add_argument("--no-revive-jump", action="store_true", help="Disable automatic soft-revive of saturated Jump/Boost heads")
    p.add_argument("--episodes", type=int, default=int(os.environ.get("TRAIN_EPISODES", "0")))
    p.add_argument("--eval-every", type=int, default=int(os.environ.get("CKPT_EVAL_EVERY", "200")))
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
    device = get_device()

    read_prefix = args.prefix if args.prefix is not None else (args.save_prefix or "stm_v1")
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
    milestone_dir = os.path.join(os.path.dirname(os.path.abspath(latest_save_path)), "milestones")
    os.makedirs(milestone_dir, exist_ok=True)

    train_logger = TrainingLogger(log_txt_path=log_txt_path, save_prefix=save_prefix)

    def _log_print(msg: str, metrics_dict: dict = None):
        train_logger.log(msg, metrics_dict)

    parsed_stage, parsed_step = _parse_stage_or_step(args.stage_level)
    if args.step_idx is not None:
        init_step_idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(args.step_idx)))
        init_stage_lv = get_curriculum_step_spec(init_step_idx).phase
    elif parsed_step is not None:
        init_step_idx = parsed_step
        init_stage_lv = parsed_stage
    elif args.mode in ("all", "mixed"):
        init_step_idx = NUM_CURRICULUM_STEPS - 1
        init_stage_lv = NUM_CURRICULUM_STAGES
    else:
        init_step_idx = 0
        init_stage_lv = 1

    env = BallReachEnv(
        mode=args.mode,
        stage_level=init_stage_lv,
        step_idx=init_step_idx,
    )
    arena = env.arena

    model = ActorCritic(
        obs_dim=OBS_DIM,
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
            ckpt = load_ckpt(src, model, strict_dims=True)
            ep_info = ckpt.get("episode", 0)
            if isinstance(ep_info, int) and ep_info > 0:
                start_episode = ep_info
            if args.step_idx is None and parsed_step is None:
                if "step_idx" in ckpt:
                    init_step_idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(ckpt["step_idx"])))
                    init_stage_lv = get_curriculum_step_spec(init_step_idx).phase
                elif "stage_level" in ckpt:
                    init_stage_lv = max(1, min(NUM_CURRICULUM_STAGES, int(ckpt["stage_level"])))
                    init_step_idx = stage_level_to_step_idx(init_stage_lv)
                env.set_curriculum_state(init_stage_lv, step_idx=init_step_idx)
            loaded_ok = True
            with torch.no_grad():
                last_lin = model.actor_mlp[-1]
                j_norm = float(last_lin.weight[9:11].norm().item())
                b_norm = float(last_lin.weight[11:13].norm().item())
            if args.revive_jump or ((not args.no_revive_jump) and j_norm > 0.85):
                model.revive_jump_head()
                _log_print(f"[ppo] soft-revived saturated Jump/Boost heads (j_norm={j_norm:.2f}, b_norm={b_norm:.2f}) to unpin ground exploration")
            _log_print(f"[ppo] loaded {src} (ep={ep_info}, step={init_step_idx+1}/{NUM_CURRICULUM_STEPS} {format_phase_coord_code(init_step_idx)}) on device={device}")
        except Exception as e:
            print(f"[ppo] ckpt incompatible ({e}), starting fresh PPO weights")
    else:
        print(f"[ppo] starting fresh PPO weights (obs_dim={OBS_DIM}) on device={device}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr_max, eps=1e-5)
    buffer = RolloutBuffer(args.rollout_steps, OBS_DIM, action_channels=len(PPO_ACTION_DIMS))

    max_ep_decisions = MAX_DECISIONS
    cfg = RewardConfig(max_decisions=max_ep_decisions)

    curr_tracker = CurriculumTracker(
        init_stage_level=init_stage_lv,
        init_step_idx=init_step_idx,
        unlock_threshold=args.eval_min_touch,
        curriculum_state=env.curriculum_state,
    )

    tracker = BestTracker(
        save_path=best_save_path,
        mode=args.mode,
        load_existing=(args.from_ckpt != "none"),
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

    _log_print(
        f"[ppo] mode={args.mode} step={curr_tracker.step_idx+1}/{NUM_CURRICULUM_STEPS} "
        f"{format_phase_coord_code(curr_tracker.step_idx)} pool=70F/30M workers={args.workers} "
        f"clamp=±{args.logit_clamp:.2f} max_dec={max_ep_decisions} ({max_ep_decisions*TICKS_PER_DECISION} ticks) "
        f"| episodes={ep_limit_str} rollout={args.rollout_steps} mb={args.batch_size} ep_ppo={args.epochs} "
        f"ent={args.ent_coef:.4f} sym={args.sym_coef:.2f} "
        f"lr={args.lr_max:.2e}->{args.lr_min:.2e} | gate=70% "
        f"| save: latest={latest_save_path}, best={best_save_path}, log={log_txt_path}"
    )

    last_episode = start_episode
    vec_env = (
        MultiCoreVecEnv(
            args.workers, mode=args.mode, base_seed=seed,
            logit_clamp=args.logit_clamp, reward_cfg=cfg, init_step_idx=curr_tracker.step_idx,
        )
        if args.workers > 1 else None
    )
    if vec_env is not None:
        vec_env.sync_curriculum(
            curr_tracker.stage_level,
            curr_tracker.get_dir5_rates(),
            step_idx=curr_tracker.step_idx,
        )

    def _build_ckpt_extra():
        return {
            "step_idx": curr_tracker.step_idx,
            "stage_level": curr_tracker.stage_level,
            "grid_code": curr_tracker.get_5dir_code(),
            "status_code": curr_tracker.format_status_code(),
        }

    def _handle_episode_end(
        ep_num, touched, step_taken, ep_rew, car_spd, stage_idx, lr_now,
        pool_id=POOL_FRONTIER, dir5_idx=0, ep_step_idx=None,
    ):
        nonlocal done_touch, done_timeout, fps_t0, frames_count
        touch_history.append(1 if touched else 0)
        rew_history.append(ep_rew)

        prev_step_idx = curr_tracker.step_idx
        curr_tracker.record_episode(
            stage_idx, touched, pool_id=pool_id, dir5_idx=dir5_idx,
            ep_step_idx=ep_step_idx,
        )
        env.stage_rates = curr_tracker.get_dir5_rates()
        if touched:
            time_history.append(step_taken)
            spd_history.append(car_spd)
            done_touch += 1
        else:
            done_timeout += 1

        completed_eval = tracker.pop_completed_result()
        if completed_eval is not None:
            _, _, _, _, async_msg = completed_eval
            _log_print(async_msg, {"event": "eval_done", "ep": ep_num, "msg": async_msg})

        auto_unlock = (not args.no_auto_unlock) and (args.mode in ("curriculum", "auto"))
        adv_msg = curr_tracker.maybe_advance(auto_unlock=auto_unlock)
        if adv_msg or (ep_num % 50 == 0 and vec_env is not None):
            env.set_curriculum_state(curr_tracker.stage_level, step_idx=curr_tracker.step_idx)
            if vec_env is not None:
                vec_env.sync_curriculum(
                    curr_tracker.stage_level,
                    curr_tracker.get_dir5_rates(),
                    step_idx=curr_tracker.step_idx,
                )
        if adv_msg:
            ms_tag = format_milestone_tag(prev_step_idx, ep_num)
            ms_path = os.path.join(milestone_dir, f"{save_prefix}_{ms_tag}.pt")
            save_ckpt(
                ms_path, model, ep_num,
                mode=args.mode, note=f"milestone:{adv_msg}",
                extra=_build_ckpt_extra(),
            )
            _log_print(f"{adv_msg} | saved milestone -> {os.path.basename(ms_path)}", {
                "event": "advance",
                "ep": ep_num,
                "prev_step_idx": prev_step_idx,
                "new_step_idx": curr_tracker.step_idx,
                "phase": curr_tracker.current_spec.phase,
                "msg": adv_msg,
                "milestone_ckpt": os.path.basename(ms_path),
            })

        PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "50" if (ep_num - start_episode) <= 500 else "200"))
        eval_line = ""
        if ep_num % args.eval_every == 0:
            gate_ok = curr_tracker.check_eval_gate(
                min_touch=None,
                warmup_eps=args.eval_warmup_eps,
                total_eps=(ep_num - start_episode),
            )
            if gate_ok:
                if args.sync_eval:
                    saved, _, _, _ = tracker.maybe_save(
                        model, ep_num, env,
                        active_stages=curr_tracker.step_idx + 1,
                        extra_meta=_build_ckpt_extra(),
                    )
                    eval_line = f" | {tracker.last_duel_info}" + (" *BEST*" if saved else "")
                else:
                    started = tracker.trigger_async_eval(
                        model, ep_num,
                        active_stages=curr_tracker.step_idx + 1,
                        aerial_state=env.curriculum_state,
                        extra_meta=_build_ckpt_extra(),
                    )
                    eval_line = f" | eval-async(P{curr_tracker.current_spec.phase}.{curr_tracker.current_spec.substep:02d})" if started else " | eval-busy"
            else:
                eval_line = f" | eval-skip(Dir5[{curr_tracker.get_5dir_code()}]<{curr_tracker.current_spec.gate:.0%})"

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
                f"pad={last_ppo_stats.get('pad_ent', 0.0):.2f}/{last_ppo_stats.get('pad_max', 0.0):.0%} "
                f"ev={last_ppo_stats.get('explained_var', 0.0):+.2f}"
                if last_ppo_stats else "upd=0"
            )
            status_code = curr_tracker.format_status_code()
            line_msg = (
                f"Ep {ep_num:6d} upd={ppo_updates_count:4d} {status_code} "
                f"touch={tr:6.2%} step={mean_t:5.1f}({mean_t*4:3.0f}f) spd={mean_spd:4.0f} r={mean_r:6.2f} "
                f"fps={real_fps:5.0f} | {kl_str} lr={lr_now:.2e} | "
                f"tk={done_touch} to={done_timeout}{eval_line}"
            )
            _log_print(line_msg, {
                "event": "train",
                "ep": ep_num,
                "upd": ppo_updates_count,
                "step_idx": curr_tracker.step_idx,
                "phase": curr_tracker.current_spec.phase,
                "status_code": status_code,
                "dir5_code": curr_tracker.get_5dir_code(),
                "dir5_rates": curr_tracker.get_dir5_rates(),
                "pool_rates": [round(x, 4) for x in curr_tracker.get_pool_rates()],
                "touch_rate": round(tr, 4),
                "mean_step": round(mean_t, 2),
                "mean_spd": round(mean_spd, 1),
                "mean_rew": round(mean_r, 2),
                "fps": round(real_fps, 0),
                "kl": round(last_ppo_stats.get("approx_kl", 0.0), 5) if last_ppo_stats else 0.0,
                "ent": round(last_ppo_stats.get("entropy", 0.0), 3) if last_ppo_stats else 0.0,
                "pad_ent": round(last_ppo_stats.get("pad_ent", 0.0), 4) if last_ppo_stats else 0.0,
                "pad_max": round(last_ppo_stats.get("pad_max", 0.0), 4) if last_ppo_stats else 0.0,
                "ev": round(last_ppo_stats.get("explained_var", 0.0), 3) if last_ppo_stats else 0.0,
                "lr": round(lr_now, 7),
                "tk": done_touch,
                "to": done_timeout,
            })
            done_touch = done_timeout = 0
            fps_t0 = time.time(); frames_count = 0
        elif eval_line and not eval_line.startswith(" | eval-skip"):
            _log_print(f"Ep {ep_num:6d}{eval_line}")

        if ep_num % 50 == 0:
            save_ckpt(
                latest_save_path, model, ep_num,
                mode=args.mode, note="ppo_latest",
                extra=_build_ckpt_extra(),
            )

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
                        gamma=GAMMA, gae_lambda=GAE_LAMBDA, reward_scale=REWARD_SCALE,
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
                        step_taken=info["step"],
                        ep_rew=info["ep_reward"],
                        car_spd=info.get("car_speed", 0.0),
                        stage_idx=info.get("stage_idx", 0),
                        lr_now=lr_now,
                        pool_id=info.get("pool_id", POOL_FRONTIER),
                        dir5_idx=info.get("dir5_idx", 0),
                        ep_step_idx=info.get("step_idx", curr_tracker.step_idx),
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
                lr_now = cosine_lr_cyclic(max(1, episode - start_episode), lr_cycle, args.lr_max, args.lr_min)
                for g in optimizer.param_groups:
                    g["lr"] = lr_now

                env.reset()
                obs_vec = env.last_obs
                prev_snap = env.snapshot()
                ep_max_dec = min(max_ep_decisions, prev_snap.get("ep_max_decisions", max_ep_decisions))

                ep_reward = 0.0
                touched = False
                touched_at = float(ep_max_dec)
                touch_car_speed = 0.0
                done = False
                step = 0
                ep_ticks = 0

                while step < ep_max_dec:
                    if device.type == "cpu":
                        act_np, logp_val, val_scalar = model.act_step_fast(obs_vec, deterministic=False)
                    else:
                        with torch.no_grad():
                            x_t = torch.from_numpy(obs_vec).to(device)
                            act_t, logp_t, _, val_t = model.get_action_and_value(x_t, deterministic=False)
                            act_np = act_t.cpu().numpy()
                            logp_val = float(logp_t.cpu().item())
                            val_scalar = float(val_t.cpu().item())

                    ctrl = multi_discrete_to_controls(act_np)
                    ticks_run = 0
                    boost_used_step = 0.0
                    boost_pickups_step = 0
                    flip_triggered_step = False
                    djump_triggered_step = False
                    jump_held_ticks = 0
                    handbrake_held_ticks = 0

                    for _ in range(TICKS_PER_DECISION):
                        b_u, f_trig, dj_trig = env.step_tick(ctrl)
                        boost_used_step += b_u
                        boost_pickups_step += int(env.last_boost_pickups)
                        if env.last_jump_held:
                            jump_held_ticks += 1
                        if env.last_handbrake_held:
                            handbrake_held_ticks += 1
                        if f_trig:
                            flip_triggered_step = True
                        if dj_trig:
                            djump_triggered_step = True
                        ticks_run += 1
                        if env._touched:
                            break
                    step += 1
                    ep_ticks += ticks_run
                    frames_count += ticks_run

                    snap = env.snapshot()
                    touched_now = snap["touched"]
                    timeout_now = (step == ep_max_dec) and not touched_now

                    if touched_now and POST_TOUCH_STEPS > 0:
                        for _ in range(POST_TOUCH_STEPS * TICKS_PER_DECISION):
                            env.car.set_controls(ctrl)
                            arena.step(1)

                    next_obs_vec, _, _ = env.get_obs()

                    exact_step = ep_ticks / float(TICKS_PER_DECISION)
                    r, _ = compute_step_reward(
                        cfg,
                        prev_dist=prev_snap["dist"],
                        new_dist=snap["dist"],
                        init_dist=snap["init_dist"],
                        min_dist=snap["min_dist"],
                        on_ground=snap["on_ground"],
                        boost_used=boost_used_step,
                        boost_pickups=boost_pickups_step,
                        jump_held_ticks=jump_held_ticks,
                        handbrake_held_ticks=handbrake_held_ticks,
                        djump_triggered=djump_triggered_step,
                        flip_triggered=flip_triggered_step,
                        touched=touched_now,
                        timeout=timeout_now,
                        ticks_elapsed=ticks_run,
                        prev_snap=prev_snap,
                        new_snap=snap,
                        ep_ticks=ep_ticks,
                        max_decisions_ep=ep_max_dec,
                    )
                    ep_reward += r

                    if touched_now:
                        done = True
                        touched = True
                        touched_at = exact_step
                        touch_car_speed = snap["car_speed"]
                    elif timeout_now:
                        done = True

                    buffer.add(obs_vec, act_np, logp_val, r, float(done), val_scalar)

                    obs_vec = next_obs_vec
                    prev_snap = snap

                    if buffer.is_full():
                        with torch.no_grad():
                            if done:
                                last_val = 0.0
                            else:
                                last_val = float(model.get_value(torch.from_numpy(obs_vec).to(device)).cpu().item())
                        buffer.compute_returns_and_advantages(
                            last_val, gamma=GAMMA, gae_lambda=GAE_LAMBDA, reward_scale=REWARD_SCALE
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

                _handle_episode_end(
                    episode,
                    touched=touched,
                    step_taken=touched_at,
                    ep_rew=ep_reward,
                    car_spd=touch_car_speed,
                    stage_idx=env.current_stage_idx,
                    lr_now=lr_now,
                    pool_id=env.current_pool_id,
                    dir5_idx=env.current_dir5_idx,
                    ep_step_idx=env.current_step_idx,
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
