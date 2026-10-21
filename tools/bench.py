#!/usr/bin/env python3
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    torch = None

from core.constants import OBS_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS, TICKS_PER_DECISION, MAX_DECISIONS
from core.policy_net import ActorCritic, multi_discrete_to_controls
from core.env import KickoffEnv
from train.vec_env import MultiCoreVecEnv, _relax_cpu_affinity


def bench_single_worker(steps=2048):
    _relax_cpu_affinity()
    if torch is not None:
        torch.set_num_threads(1)
    env = KickoffEnv()
    model = ActorCritic(obs_dim=OBS_DIM, hidden=PPO_HIDDEN_DIM, action_dims=PPO_ACTION_DIMS)
    if torch is not None and hasattr(model, "to"):
        model = model.to("cpu")
        model.eval()

    env.reset()
    obs_b, _, _ = env.get_obs(0)
    obs_o, _, _ = env.get_obs(1)
    ep_step = 0
    total_ticks = 0

    t0 = time.perf_counter()
    for _ in range(steps):
        act_b, _, _ = model.act_step_fast(obs_b, deterministic=False)
        act_o, _, _ = model.act_step_fast(obs_o, deterministic=False)
        ctrl_b = multi_discrete_to_controls(act_b)
        ctrl_o = multi_discrete_to_controls(act_o)
        for _ in range(TICKS_PER_DECISION):
            env.step_duel_tick(ctrl_b, ctrl_o)
            total_ticks += 1
            if env._touched:
                break
        ep_step += 1
        if env._touched_by_car[0] or env._touched_by_car[1] or ep_step >= MAX_DECISIONS:
            env.reset()
            obs_b, _, _ = env.get_obs(0)
            obs_o, _, _ = env.get_obs(1)
            ep_step = 0
        else:
            obs_b, _, _ = env.get_obs(0)
            obs_o, _, _ = env.get_obs(1)
    dt = max(1e-6, time.perf_counter() - t0)
    return (steps * 2) / dt, total_ticks / dt, dt


def bench_multi_worker(n_workers, steps=4096):
    _relax_cpu_affinity()
    if torch is not None:
        torch.set_num_threads(1)
    model = ActorCritic(obs_dim=OBS_DIM, hidden=PPO_HIDDEN_DIM, action_dims=PPO_ACTION_DIMS)
    if torch is not None and hasattr(model, "to"):
        model = model.to("cpu")
        model.eval()
    vec_env = MultiCoreVecEnv(n_workers=n_workers, base_seed=42)
    try:
        vec_env.collect_rollout(model, total_steps=n_workers * 32, sticky_prob=0.0)
        t0 = time.perf_counter()
        chunks = vec_env.collect_rollout(model, total_steps=steps, sticky_prob=0.02)
        dt = max(1e-6, time.perf_counter() - t0)
        total_dec = sum(len(c[0]) for c in chunks)
        # Each worker has 2 streams (Blue & Orange) sharing the same arena physics ticks
        total_ticks = sum(c[7] for c in chunks) // 2
        return total_dec / dt, total_ticks / dt, dt
    finally:
        vec_env.close()


def main():
    ap = argparse.ArgumentParser(description="Benchmark 1v1 Kickoff Self-Play rollout speed (DPS & FPS)")
    ap.add_argument("--steps", type=int, default=4096, help="Rollout decision steps per test (default: 4096)")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8], help="Worker counts to test (default: 1 4 8)")
    args = ap.parse_args()

    print("==========================================================================")
    print(f"  RLRL101 1v1 Kickoff Multi-Worker Benchmark (steps={args.steps}, cpu_count={os.cpu_count()})")
    print("==========================================================================")
    print(f"  {'Workers':<10} | {'Car-Dec/s':>14} | {'Arena FPS':>16} | {'Elapsed':>10} | {'Speedup':>9}")
    print("-" * 74)

    base_dps = None
    for w in args.workers:
        if w <= 1:
            dps, fps, dt = bench_single_worker(steps=args.steps // 2)
        else:
            dps, fps, dt = bench_multi_worker(w, steps=args.steps)
        if base_dps is None:
            base_dps = dps
        speedup = dps / max(1e-6, base_dps)
        print(f"  {w:<10d} | {dps:>14.0f} | {fps:>16.0f} | {dt:>9.3f}s | {speedup:>8.2f}x")
    print("==========================================================================")


if __name__ == "__main__":
    main()
