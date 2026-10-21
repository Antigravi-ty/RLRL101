#!/usr/bin/env python3
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from core.constants import OBS_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS, TICKS_PER_DECISION
from core.policy_net import ActorCritic, multi_discrete_to_controls
from core.env import BallReachEnv
from train.vec_env import MultiCoreVecEnv, _relax_cpu_affinity


def bench_single_worker(steps=2048):
    _relax_cpu_affinity()
    torch.set_num_threads(1)
    env = BallReachEnv(mode="curriculum", stage_level=1)
    model = ActorCritic(obs_dim=OBS_DIM, hidden=PPO_HIDDEN_DIM, action_dims=PPO_ACTION_DIMS).to("cpu")
    model.eval()

    env.reset()
    obs_vec = env.last_obs
    ep_step = 0
    total_ticks = 0

    t0 = time.perf_counter()
    for _ in range(steps):
        act_np, _, _ = model.act_step_fast(obs_vec, deterministic=False)
        ctrl = multi_discrete_to_controls(act_np)
        for _ in range(TICKS_PER_DECISION):
            env.step_tick(ctrl)
            total_ticks += 1
            if env._touched:
                break
        ep_step += 1
        if env._touched or ep_step >= 180:
            env.reset()
            obs_vec = env.last_obs
            ep_step = 0
        else:
            obs_vec, _, _ = env.get_obs()
    dt = max(1e-6, time.perf_counter() - t0)
    return steps / dt, total_ticks / dt, dt


def bench_multi_worker(n_workers, steps=2048):
    _relax_cpu_affinity()
    torch.set_num_threads(1)
    model = ActorCritic(obs_dim=OBS_DIM, hidden=PPO_HIDDEN_DIM, action_dims=PPO_ACTION_DIMS).to("cpu")
    model.eval()
    vec_env = MultiCoreVecEnv(n_workers=n_workers, mode="curriculum", base_seed=42)
    try:
        vec_env.collect_rollout(model, total_steps=n_workers * 16, sticky_prob=0.0)
        t0 = time.perf_counter()
        chunks = vec_env.collect_rollout(model, total_steps=steps, sticky_prob=0.02)
        dt = max(1e-6, time.perf_counter() - t0)
        total_dec = sum(len(c[0]) for c in chunks)
        total_ticks = sum(c[7] for c in chunks)
        return total_dec / dt, total_ticks / dt, dt
    finally:
        vec_env.close()


def main():
    ap = argparse.ArgumentParser(description="Benchmark single-core vs multi-core rollout speed (DPS & FPS)")
    ap.add_argument("--steps", type=int, default=2048, help="Rollout decision steps per test (default: 2048)")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4], help="Worker counts to test (default: 1 2 4)")
    args = ap.parse_args()

    print("==========================================================================")
    print(f"  RLRL101 Multi-Core Throughput Benchmark (steps={args.steps}, cpu_count={os.cpu_count()})")
    print("==========================================================================")
    print(f"  {'Workers':<10} | {'DPS (Dec/s)':>14} | {'FPS (Frames/s)':>16} | {'Elapsed':>10} | {'Speedup':>9}")
    print("-" * 74)

    base_fps = None
    for w in args.workers:
        if w <= 1:
            dps, fps, dt = bench_single_worker(steps=args.steps)
        else:
            dps, fps, dt = bench_multi_worker(w, steps=args.steps)
        if base_fps is None:
            base_fps = fps
        speedup = fps / max(1e-6, base_fps)
        print(f"  {w:<10d} | {dps:>14.0f} | {fps:>16.0f} | {dt:>9.3f}s | {speedup:>8.2f}x")
    print("==========================================================================")


if __name__ == "__main__":
    main()
