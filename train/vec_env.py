import os
import sys
import random
import numpy as np
try:
    import torch
except ImportError:
    torch = None

from core.constants import (
    OBS_DIM, PPO_HIDDEN_DIM, PPO_ACTION_DIMS,
    TICKS_PER_DECISION, MAX_DECISIONS, LOGIT_CLAMP_RANGE,
)
from core.policy_net import ActorCritic, multi_discrete_to_controls
from core.env import KickoffEnv
from core.reward import RewardConfig, compute_step_reward


def _relax_cpu_affinity():
    if hasattr(os, "sched_setaffinity"):
        try:
            n_cpus = os.cpu_count() or 4
            os.sched_setaffinity(0, set(range(n_cpus)))
        except Exception:
            pass


def _worker_loop(conn, mode, seed, logit_clamp, reward_cfg=None, init_slot_idx=None):
    _relax_cpu_affinity()
    if torch is not None:
        torch.set_num_threads(1)
        torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = KickoffEnv(mode=mode, slot_idx=init_slot_idx, seed=seed)
    cfg = reward_cfg if reward_cfg is not None else RewardConfig(max_decisions=MAX_DECISIONS)
    local_model = ActorCritic(
        obs_dim=OBS_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=logit_clamp,
    )
    if hasattr(local_model, "to"):
        local_model = local_model.to("cpu")
    if hasattr(local_model, "eval"):
        local_model.eval()

    def _reset_ep():
        env.reset()
        obs_b, obs_o = env.last_obs_pair
        snap_b, snap_o = env.snapshot_pair()
        return obs_b, obs_o, snap_b, snap_o

    obs_b, obs_o, prev_snap_b, prev_snap_o = _reset_ep()
    step = 0
    ep_ticks = 0
    ep_rew_b = 0.0
    ep_rew_o = 0.0

    while True:
        cmd, data = conn.recv()
        if cmd == "close":
            conn.close()
            break
        elif cmd == "sync_reward_cfg":
            cfg = data
        elif cmd == "collect_chunk":
            state_dict, chunk_steps = data
            if state_dict is not None:
                local_model.load_state_dict(state_dict)

            # Simultaneously record both Blue (0) and Orange (1) 1v1 duel trajectories
            c_obs_b  = np.zeros((chunk_steps, OBS_DIM), dtype=np.float32)
            c_act_b  = np.zeros((chunk_steps, len(PPO_ACTION_DIMS)), dtype=np.int64)
            c_logp_b = np.zeros(chunk_steps, dtype=np.float32)
            c_rew_b  = np.zeros(chunk_steps, dtype=np.float32)
            c_done_b = np.zeros(chunk_steps, dtype=np.float32)
            c_val_b  = np.zeros(chunk_steps, dtype=np.float32)

            c_obs_o  = np.zeros((chunk_steps, OBS_DIM), dtype=np.float32)
            c_act_o  = np.zeros((chunk_steps, len(PPO_ACTION_DIMS)), dtype=np.int64)
            c_logp_o = np.zeros(chunk_steps, dtype=np.float32)
            c_rew_o  = np.zeros(chunk_steps, dtype=np.float32)
            c_done_o = np.zeros(chunk_steps, dtype=np.float32)
            c_val_o  = np.zeros(chunk_steps, dtype=np.float32)

            infos = []
            total_ticks = 0

            for t in range(chunk_steps):
                act_b, logp_b, val_b = local_model.act_step_fast(obs_b, deterministic=False)
                act_o, logp_o, val_o = local_model.act_step_fast(obs_o, deterministic=False)
                ctrl_b = multi_discrete_to_controls(act_b)
                ctrl_o = multi_discrete_to_controls(act_o)

                ticks_run = 0
                for _ in range(TICKS_PER_DECISION):
                    env.step_duel_tick(ctrl_b, ctrl_o)
                    ticks_run += 1
                    if env._touched:
                        break

                step += 1
                ep_ticks += ticks_run
                total_ticks += ticks_run

                snap_b, snap_o = env.snapshot_pair()
                b_touched = snap_b["touched"]
                o_touched = snap_o["touched"]
                any_touched = b_touched or o_touched

                max_dec_cfg = getattr(cfg, "max_decisions", MAX_DECISIONS)
                timeout_now = (step >= max_dec_cfg) and (not any_touched)
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
                    max_decisions_ep=max_dec_cfg,
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
                    max_decisions_ep=max_dec_cfg,
                )

                ep_rew_b += r_b
                ep_rew_o += r_o

                c_obs_b[t]  = obs_b
                c_act_b[t]  = act_b
                c_logp_b[t] = logp_b
                c_rew_b[t]  = r_b
                c_done_b[t] = float(done)
                c_val_b[t]  = val_b

                c_obs_o[t]  = obs_o
                c_act_o[t]  = act_o
                c_logp_o[t] = logp_o
                c_rew_o[t]  = r_o
                c_done_o[t] = float(done)
                c_val_o[t]  = val_o

                if done:
                    exact_step = ep_ticks / float(TICKS_PER_DECISION)
                    if b_touched and (not o_touched or snap_b["car_speed"] >= snap_o["car_speed"]):
                        win_spd = snap_b["car_speed"]
                        win_rew = ep_rew_b
                    elif o_touched:
                        win_spd = snap_o["car_speed"]
                        win_rew = ep_rew_o
                    else:
                        win_spd = max(snap_b["car_speed"], snap_o["car_speed"])
                        win_rew = 0.5 * (ep_rew_b + ep_rew_o)

                    infos.append({
                        "touched": any_touched,
                        "blue_touched": b_touched,
                        "orange_touched": o_touched,
                        "winner": env.winner_car_idx,
                        "step": exact_step,
                        "frames": ep_ticks,
                        "ep_reward": win_rew,
                        "mean_reward": 0.5 * (ep_rew_b + ep_rew_o),
                        "car_speed": win_spd,
                        "slot_idx": env.current_slot_idx,
                    })
                    obs_b, obs_o, prev_snap_b, prev_snap_o = _reset_ep()
                    step = 0
                    ep_ticks = 0
                    ep_rew_b = 0.0
                    ep_rew_o = 0.0
                else:
                    obs_b, obs_o = env.get_obs_pair()
                    prev_snap_b, prev_snap_o = snap_b, snap_o

            if c_done_b[-1] > 0.5:
                last_val_b = 0.0
                last_val_o = 0.0
            else:
                _, _, last_val_b = local_model.act_step_fast(obs_b, deterministic=False)
                _, _, last_val_o = local_model.act_step_fast(obs_o, deterministic=False)

            traj_b = (c_obs_b, c_act_b, c_logp_b, c_rew_b, c_done_b, c_val_b, float(last_val_b), total_ticks, infos)
            traj_o = (c_obs_o, c_act_o, c_logp_o, c_rew_o, c_done_o, c_val_o, float(last_val_o), 0, [])
            conn.send((traj_b, traj_o))


class MultiCoreVecEnv:
    def __init__(
        self,
        n_workers,
        mode="kickoff",
        base_seed=42,
        logit_clamp=LOGIT_CLAMP_RANGE,
        reward_cfg=None,
        **kwargs,
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
                args=(child_conn, mode, base_seed + i * 1009, logit_clamp, reward_cfg, i % 5),
                daemon=True,
            )
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)

    def sync_reward_cfg(self, cfg):
        for c in self.conns:
            c.send(("sync_reward_cfg", cfg))

    def collect_rollout(self, model, total_steps: int, **kwargs):
        # Each worker produces 2 trajectories (Blue + Orange), so divide by 2 * n_workers
        n_streams = 2 * self.n_workers
        chunk_steps = max(1, (total_steps + n_streams - 1) // n_streams)
        cpu_sd = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in model.state_dict().items()}
        for c in self.conns:
            c.send(("collect_chunk", (cpu_sd, chunk_steps)))
        results = []
        for c in self.conns:
            traj_b, traj_o = c.recv()
            results.append(traj_b)
            results.append(traj_o)
        return results

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=1.0)
