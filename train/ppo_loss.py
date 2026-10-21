"""train/ppo_loss.py —— RolloutBuffer、GAE 优势估计与 PPO 裁剪损失 + 镜像对称损失更新。"""
import numpy as np
import torch
import torch.nn as nn


def get_device(prefer_mps=True):
    if torch.cuda.is_available():
        return torch.device("cuda")
    import os
    if prefer_mps and os.environ.get("USE_MPS", "0") == "1":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_channels=6):
        self.n_steps = int(n_steps)
        self.obs_dim = int(obs_dim)
        self.action_channels = int(action_channels)
        self.reset()

    def reset(self):
        self.obs        = np.zeros((self.n_steps, self.obs_dim), dtype=np.float32)
        self.actions    = np.zeros((self.n_steps, self.action_channels), dtype=np.int64)
        self.logprobs   = np.zeros(self.n_steps, dtype=np.float32)
        self.rewards    = np.zeros(self.n_steps, dtype=np.float32)
        self.dones      = np.zeros(self.n_steps, dtype=np.float32)
        self.values     = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns    = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, logprob, reward, done, value):
        if self.ptr >= self.n_steps:
            return
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.logprobs[self.ptr] = float(logprob)
        self.rewards[self.ptr]  = float(reward)
        self.dones[self.ptr]    = float(done)
        self.values[self.ptr]   = float(value)
        self.ptr += 1

    def add_trajectory(self, obs_arr, act_arr, logp_arr, rew_arr, done_arr, val_arr, adv_arr, ret_arr):
        k = len(rew_arr)
        rem = self.n_steps - self.ptr
        if rem <= 0:
            return
        take = min(k, rem)
        s, e = self.ptr, self.ptr + take
        self.obs[s:e]        = obs_arr[:take]
        self.actions[s:e]    = act_arr[:take]
        self.logprobs[s:e]   = logp_arr[:take]
        self.rewards[s:e]    = rew_arr[:take]
        self.dones[s:e]      = done_arr[:take]
        self.values[s:e]     = val_arr[:take]
        self.advantages[s:e] = adv_arr[:take]
        self.returns[s:e]    = ret_arr[:take]
        self.ptr = e

    def is_full(self):
        return self.ptr >= self.n_steps

    def compute_returns_and_advantages(self, last_value, gamma=0.995, gae_lambda=0.95, reward_scale=0.05):
        last_gae_lam = 0.0
        n = self.ptr
        scaled_rewards = self.rewards[:n] * reward_scale
        for t in reversed(range(n)):
            next_non_terminal = 1.0 - self.dones[t]
            next_val = float(last_value) if t == n - 1 else self.values[t + 1]
            delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns[:n] = self.advantages[:n] + self.values[:n]


def compute_gae_1d(rewards, values, dones, last_value, gamma=0.995, gae_lambda=0.95, reward_scale=0.05):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    scaled_rewards = rewards * reward_scale
    last_gae_lam = 0.0
    for t in reversed(range(n)):
        next_non_terminal = 1.0 - dones[t]
        next_val = float(last_value) if t == n - 1 else float(values[t + 1])
        delta = scaled_rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam
    returns = advantages + values
    return advantages, returns


def _channel_kl_loss(student_logits, teacher_logits_detached, slices):
    import torch.nn.functional as F
    kl_sum = 0.0
    for (s, e) in slices:
        log_p_student = F.log_softmax(student_logits[:, s:e], dim=-1)
        log_p_teacher = F.log_softmax(teacher_logits_detached[:, s:e], dim=-1)
        p_teacher = log_p_teacher.exp()
        kl_ch = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)
        kl_sum = kl_sum + kl_ch
    return kl_sum


def ppo_update(
    model,
    optimizer,
    buffer: RolloutBuffer,
    ppo_epochs=4,
    minibatch_size=256,
    clip_coef=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    sym_coef=0.02,
    augment_symmetry=True,
    max_grad_norm=0.5,
    vf_only=False,
    device=torch.device("cpu"),
):
    from core.obs import mirror_obs_torch, mirror_logits_torch

    n = buffer.ptr
    b_obs      = torch.from_numpy(buffer.obs[:n]).to(device)
    b_actions  = torch.from_numpy(buffer.actions[:n]).to(device)
    b_logprobs = torch.from_numpy(buffer.logprobs[:n]).to(device)
    b_adv      = torch.from_numpy(buffer.advantages[:n]).to(device)
    b_returns  = torch.from_numpy(buffer.returns[:n]).to(device)
    b_values   = torch.from_numpy(buffer.values[:n]).to(device)

    if augment_symmetry and sym_coef > 0.0:
        b_obs_m = mirror_obs_torch(b_obs)

    inds = np.arange(n)
    clipfracs = []
    pg_losses, v_losses, ent_losses, sym_losses, approx_kls = [], [], [], [], []
    pad_ents, pad_maxs = [], []

    for _ in range(ppo_epochs):
        np.random.shuffle(inds)
        for start in range(0, n, minibatch_size):
            end = start + minibatch_size
            mb_inds = inds[start:end]

            mb_obs = b_obs[mb_inds]
            mb_act = b_actions[mb_inds]
            mb_adv_raw = b_adv[mb_inds]

            if vf_only:
                with torch.no_grad():
                    feat, _ = model._encode(mb_obs)
                newvalue = model.critic_mlp(feat).squeeze(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.critic_mlp.parameters(), max_grad_norm)
                optimizer.step()
                pg_losses.append(0.0)
                v_losses.append(v_loss.item())
                ent_losses.append(0.0)
                approx_kls.append(0.0)
                clipfracs.append(0.0)
                continue

            _, newlogprob, entropy, newvalue = model.get_action_and_value(mb_obs, mb_act)
            p_ent, p_max, _ = model.get_attention_stats()
            pad_ents.append(p_ent)
            pad_maxs.append(p_max)

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp().clamp(max=10.0)

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - logratio).mean().item()
                approx_kls.append(approx_kl)
                clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

            if mb_adv_raw.numel() > 1:
                mb_advantages = (mb_adv_raw - mb_adv_raw.mean()) / (mb_adv_raw.std() + 1e-8)
            else:
                mb_advantages = mb_adv_raw

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            raw_logits_orig = model.actor(mb_obs)
            sym_loss = torch.tensor(0.0, device=device)
            if augment_symmetry and sym_coef > 0.0:
                mb_obs_m = b_obs_m[mb_inds]

                logits_orig = model.apply_action_mask(raw_logits_orig, mb_obs, ch4_action=mb_act[:, 4])
                logits_mirr = model.apply_action_mask(model.actor(mb_obs_m), mb_obs_m, ch4_action=mb_act[:, 4])
                v_orig = newvalue
                v_mirr = model.get_value(mb_obs_m)

                off_center = ((mb_obs[:, 0] - mb_obs[:, 1]).abs() > 0.015).float()

                orig_better = (v_orig.detach() >= v_mirr.detach()).float()
                mirr_better = 1.0 - orig_better

                target_for_mirr = mirror_logits_torch(logits_orig.detach())
                target_for_orig = mirror_logits_torch(logits_mirr.detach())

                kl_to_mirr = _channel_kl_loss(logits_mirr, target_for_mirr, model._slices)
                kl_to_orig = _channel_kl_loss(logits_orig, target_for_orig, model._slices)
                sym_kl = (off_center * (orig_better * kl_to_mirr + mirr_better * kl_to_orig)).sum() / (off_center.sum() + 1e-8)

                v_sym = 0.5 * (
                    off_center * (
                        orig_better * ((v_mirr - v_orig.detach()) ** 2)
                        + mirr_better * ((v_orig - v_mirr.detach()) ** 2)
                    )
                ).sum() / (off_center.sum() + 1e-8)

                sym_loss = sym_kl + 0.25 * v_sym
                sym_losses.append(sym_loss.item())

            entropy_loss = entropy.mean()
            bound_loss = model.compute_logit_bound_loss(raw_logits_orig, obs=mb_obs, actions=mb_act)
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss + sym_coef * sym_loss + 0.1 * bound_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            ent_losses.append(entropy_loss.item())

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))

    return {
        "pg_loss":       float(np.mean(pg_losses)),
        "v_loss":        float(np.mean(v_losses)),
        "entropy":       float(np.mean(ent_losses)),
        "sym_loss":      float(np.mean(sym_losses)) if sym_losses else 0.0,
        "approx_kl":     float(np.mean(approx_kls)),
        "clipfrac":      float(np.mean(clipfracs)),
        "explained_var": explained_var,
        "pad_ent":       float(np.mean(pad_ents)) if pad_ents else 1.0,
        "pad_max":       float(np.mean(pad_maxs)) if pad_maxs else (1.0 / 34.0),
    }
