"""train/vec_env.py —— 多核并行 Rollout 采样器 (MultiCoreVecEnv for 4-Phase Static Target Master)。"""
import os
import sys
import random
import numpy as np
import torch

from core.constants import (
    OBS_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS,
    TICKS_PER_DECISION, MAX_DECISIONS, LOGIT_CLAMP_RANGE,
    POOL_FRONTIER,
)
from core.policy_net import ActorCritic, multi_discrete_to_controls
from core.env import BallReachEnv
from core.reward import RewardConfig, compute_step_reward


def _relax_cpu_affinity():
    if hasattr(os, "sched_setaffinity"):
        try:
            n_cpus = os.cpu_count() or 4
            os.sched_setaffinity(0, set(range(n_cpus)))
        except Exception:
            pass


def _worker_loop(conn, mode, seed, logit_clamp, reward_cfg=None, init_step_idx=0):
    _relax_cpu_affinity()
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = BallReachEnv(mode=mode, stage_level=1, step_idx=init_step_idx)
    cfg = reward_cfg if reward_cfg is not None else RewardConfig(max_decisions=MAX_DECISIONS)
    local_model = ActorCritic(
        obs_dim=OBS_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=logit_clamp,
    ).to("cpu")
    local_model.eval()

    def _reset_ep():
        env.reset()
        return env.last_obs, env.snapshot()

    obs_vec, prev_snap = _reset_ep()
    step = 0
    ep_ticks = 0
    ep_reward = 0.0

    while True:
        cmd, data = conn.recv()
        if cmd == "close":
            conn.close()
            break
        elif cmd == "sync_curriculum":
            stage_lv, stage_rates, step_idx = data
            step_changed = (int(step_idx) != int(env.current_step_idx))
            env.set_curriculum_state(
                stage_lv, stage_rates=stage_rates, step_idx=step_idx,
            )
            if step_changed:
                obs_vec, prev_snap = _reset_ep()
                step = 0
                ep_ticks = 0
                ep_reward = 0.0
        elif cmd == "sync_reward_cfg":
            cfg = data
        elif cmd == "collect_chunk":
            state_dict, chunk_steps = data
            if state_dict is not None:
                local_model.load_state_dict(state_dict)

            c_obs  = np.zeros((chunk_steps, OBS_DIM), dtype=np.float32)
            c_act  = np.zeros((chunk_steps, len(PPO_ACTION_DIMS)), dtype=np.int64)
            c_logp = np.zeros(chunk_steps, dtype=np.float32)
            c_rew  = np.zeros(chunk_steps, dtype=np.float32)
            c_done = np.zeros(chunk_steps, dtype=np.float32)
            c_val  = np.zeros(chunk_steps, dtype=np.float32)
            infos  = []
            total_ticks = 0

            for t in range(chunk_steps):
                act_np, logp_val, val_scalar = local_model.act_step_fast(obs_vec, deterministic=False)
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
                total_ticks += ticks_run
                snap = env.snapshot()
                touched_now = snap["touched"]
                max_dec_cfg = getattr(cfg, "max_decisions", MAX_DECISIONS)
                ep_max_dec = min(max_dec_cfg, snap.get("ep_max_decisions", max_dec_cfg))
                timeout_now = (step >= ep_max_dec) and not touched_now
                done = touched_now or timeout_now

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
                exact_step = ep_ticks / float(TICKS_PER_DECISION)

                c_obs[t]  = obs_vec
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
                        "step_idx": env.current_step_idx,
                        "stage_idx": env.current_stage_idx,
                        "pool_id": env.current_pool_id,
                        "dir5_idx": env.current_dir5_idx,
                    })
                    obs_vec, prev_snap = _reset_ep()
                    step = 0
                    ep_ticks = 0
                    ep_reward = 0.0
                else:
                    obs_vec, _, _ = env.get_obs()
                    prev_snap = snap

            if c_done[-1] > 0.5:
                last_val = 0.0
            else:
                _, _, last_val = local_model.act_step_fast(obs_vec, deterministic=False)

            conn.send((c_obs, c_act, c_logp, c_rew, c_done, c_val, float(last_val), total_ticks, infos))


class MultiCoreVecEnv:
    def __init__(
        self,
        n_workers,
        mode="curriculum",
        base_seed=42,
        logit_clamp=LOGIT_CLAMP_RANGE,
        reward_cfg=None,
        init_step_idx=0,
    ):
        import multiprocessing as mp
        ctx = mp.get_context("spawn" if sys.platform == "darwin" else "fork")
        self.n_workers = n_workers
        self.conns = []
        self.procs = []
        for i in range(n_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker_loop,
                args=(child_conn, mode, base_seed + i * 1009, logit_clamp, reward_cfg, init_step_idx),
                daemon=True,
            )
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)

    def sync_curriculum(
        self,
        stage_level: int,
        stage_rates=None,
        step_idx: int = None,
    ):
        payload = (int(stage_level), stage_rates, step_idx)
        for c in self.conns:
            c.send(("sync_curriculum", payload))

    def sync_reward_cfg(self, cfg):
        for c in self.conns:
            c.send(("sync_reward_cfg", cfg))

    def collect_rollout(self, model, total_steps: int):
        chunk_steps = max(1, (total_steps + self.n_workers - 1) // self.n_workers)
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        for c in self.conns:
            c.send(("collect_chunk", (cpu_sd, chunk_steps)))
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
