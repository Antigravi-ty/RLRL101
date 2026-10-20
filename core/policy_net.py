"""core/policy_net.py —— Boost Pad 多头注意力 (Entity Cross-Attention) + PPO Actor-Critic 网络架构与动作映射。"""
import math
import numpy as np
import RocketSim as rs

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    _TORCH_AVAILABLE = False

from .constants import (
    OBS_DIM, TRUNK_OBS_DIM, NUM_BOOST_PADS, PAD_TOKEN_DIM,
    ATTN_NUM_HEADS, ATTN_HEAD_DIM, ATTN_OUT_DIM,
    IDX_ON_GROUND, IDX_BOOST,
    PPO_ACTION_DIMS, PPO_LOGITS_DIM, PPO_HIDDEN_DIM,
    LOGIT_CLAMP_RANGE,
)
from .obs import PAD_IS_BIG_MASK


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


_BaseModule = nn.Module if _TORCH_AVAILABLE else object


if _TORCH_AVAILABLE:
    class _ProjectedClampSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, limit):
            ctx.save_for_backward(x)
            ctx.limit = float(limit)
            return torch.clamp(x, -ctx.limit, ctx.limit)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            lim = ctx.limit
            allow = (
                ((x >= -lim) & (x <= lim))
                | ((x < -lim) & (grad_output < 0.0))
                | ((x > lim) & (grad_output > 0.0))
            )
            return grad_output * allow.to(grad_output.dtype), None


class BoostPadAttention(_BaseModule):
    """独立处理 34 个 Boost Pads 的多头交叉注意力模块 (无虚拟 Null Token)。
    - Query: 来自 42 维车/静态目标/环境主干特征 (Trunk)
    - Key / Value: 来自 34 个 Boost Pad 实体特征 [rel_fwd, rel_right, is_active, cooldown_norm, is_big]
    - 输出: 32 维浓缩氮气上下文向量 + (B, num_heads, 34) 注意力权重矩阵
    """
    def __init__(
        self,
        trunk_dim=TRUNK_OBS_DIM,
        pad_in_dim=PAD_TOKEN_DIM + 1,
        num_heads=ATTN_NUM_HEADS,
        head_dim=ATTN_HEAD_DIM,
    ):
        super().__init__()
        self.trunk_dim = int(trunk_dim)
        self.pad_in_dim = int(pad_in_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.out_dim = self.num_heads * self.head_dim
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

        self.register_buffer(
            "pad_is_big",
            torch.tensor(PAD_IS_BIG_MASK, dtype=torch.float32).view(1, NUM_BOOST_PADS, 1),
        )
        self.q_proj = layer_init(nn.Linear(self.trunk_dim, self.out_dim), std=1.0)
        self.k_proj = layer_init(nn.Linear(self.pad_in_dim, self.out_dim), std=1.0)
        self.v_proj = layer_init(nn.Linear(self.pad_in_dim, self.out_dim), std=1.0)
        self.out_proj = layer_init(nn.Linear(self.out_dim, self.out_dim), std=1.0)
        self.last_attn_weights = None

    def forward(self, trunk, pads_raw):
        B = trunk.shape[0]
        is_big = self.pad_is_big.expand(B, NUM_BOOST_PADS, 1)
        pad_tokens = torch.cat([pads_raw, is_big], dim=-1)  # (B, 34, 5)

        q = self.q_proj(trunk).view(B, self.num_heads, 1, self.head_dim)
        k = self.k_proj(pad_tokens).view(B, NUM_BOOST_PADS, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(pad_tokens).view(B, NUM_BOOST_PADS, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, 1, 34)
        attn_weights = F.softmax(scores, dim=-1).squeeze(-2)        # (B, H, 34)
        self.last_attn_weights = attn_weights

        ctx = torch.matmul(attn_weights.unsqueeze(-2), v).squeeze(-2)  # (B, H, head_dim)
        ctx_flat = ctx.reshape(B, self.out_dim)
        pad_embed = torch.tanh(self.out_proj(ctx_flat))
        return pad_embed, attn_weights

    def get_attention_stats(self, x=None):
        with torch.no_grad():
            if x is not None and x.dim() in (1, 2) and x.shape[-1] >= self.trunk_dim + NUM_BOOST_PADS * PAD_TOKEN_DIM:
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                trunk = x[:, :self.trunk_dim]
                pads_raw = x[:, self.trunk_dim:self.trunk_dim + NUM_BOOST_PADS * PAD_TOKEN_DIM].view(-1, NUM_BOOST_PADS, PAD_TOKEN_DIM)
                _, w = self.forward(trunk, pads_raw)
            else:
                w = self.last_attn_weights
            if w is None:
                return torch.tensor(1.0), torch.tensor(1.0 / NUM_BOOST_PADS)
            p = w.mean(dim=1)
            ent = -(p * torch.log(p.clamp(min=1e-8))).sum(dim=-1) / math.log(float(NUM_BOOST_PADS))
            max_w = p.max(dim=-1)[0]
            return ent.mean(), max_w.mean()


class ActorCritic(_BaseModule):
    def __init__(
        self,
        obs_dim=OBS_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=LOGIT_CLAMP_RANGE,
        attn_heads=ATTN_NUM_HEADS,
        attn_head_dim=ATTN_HEAD_DIM,
    ):
        if _TORCH_AVAILABLE:
            super().__init__()
        self.obs_dim = int(obs_dim)
        self.trunk_dim = TRUNK_OBS_DIM
        self.hidden_dim = int(hidden)
        self.action_dims = list(action_dims)
        self.logits_dim = sum(self.action_dims)
        self.logit_clamp = float(logit_clamp)
        self.attn_heads = int(attn_heads)
        self.attn_head_dim = int(attn_head_dim)
        self._slices = []
        cur = 0
        for d in self.action_dims:
            self._slices.append((cur, cur + d))
            cur += d

        self._np_state = None
        self.last_attn_weights = None

        if not _TORCH_AVAILABLE:
            attn_out = self.attn_heads * self.attn_head_dim
            pad_in = PAD_TOKEN_DIM + 1
            combined_dim = self.trunk_dim + attn_out
            self._np_state = {
                "pad_attn.q_proj.weight": np.zeros((attn_out, self.trunk_dim), dtype=np.float32),
                "pad_attn.q_proj.bias": np.zeros((attn_out,), dtype=np.float32),
                "pad_attn.k_proj.weight": np.zeros((attn_out, pad_in), dtype=np.float32),
                "pad_attn.k_proj.bias": np.zeros((attn_out,), dtype=np.float32),
                "pad_attn.v_proj.weight": np.zeros((attn_out, pad_in), dtype=np.float32),
                "pad_attn.v_proj.bias": np.zeros((attn_out,), dtype=np.float32),
                "pad_attn.out_proj.weight": np.zeros((attn_out, attn_out), dtype=np.float32),
                "pad_attn.out_proj.bias": np.zeros((attn_out,), dtype=np.float32),
                "actor_mlp.0.weight": np.zeros((self.hidden_dim, combined_dim), dtype=np.float32),
                "actor_mlp.0.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "actor_mlp.2.weight": np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32),
                "actor_mlp.2.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "actor_mlp.4.weight": np.zeros((self.logits_dim, self.hidden_dim), dtype=np.float32),
                "actor_mlp.4.bias": np.zeros((self.logits_dim,), dtype=np.float32),
                "critic_mlp.0.weight": np.zeros((self.hidden_dim, combined_dim), dtype=np.float32),
                "critic_mlp.0.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "critic_mlp.2.weight": np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32),
                "critic_mlp.2.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "critic_mlp.4.weight": np.zeros((1, self.hidden_dim), dtype=np.float32),
                "critic_mlp.4.bias": np.zeros((1,), dtype=np.float32),
            }
            return

        self.pad_attn = BoostPadAttention(
            trunk_dim=self.trunk_dim,
            pad_in_dim=PAD_TOKEN_DIM + 1,
            num_heads=attn_heads,
            head_dim=attn_head_dim,
        )
        combined_dim = self.trunk_dim + self.pad_attn.out_dim  # 42 + 32 = 74

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(combined_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.logits_dim), std=0.01, bias_const=0.0),
        )

        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(combined_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0, bias_const=0.0),
        )

    def to(self, *args, **kwargs):
        if _TORCH_AVAILABLE:
            return super().to(*args, **kwargs)
        return self

    def eval(self):
        if _TORCH_AVAILABLE:
            return super().eval()
        return self

    def load_state_dict(self, state_dict, strict=True):
        if _TORCH_AVAILABLE:
            return super().load_state_dict(state_dict, strict=strict)
        self._np_state = {k: np.asarray(v, dtype=np.float32) for k, v in state_dict.items()}
        return self

    def state_dict(self, *args, **kwargs):
        if _TORCH_AVAILABLE:
            return super().state_dict(*args, **kwargs)
        return dict(self._np_state or {})

    def _encode_np(self, obs_np):
        sd = self._np_state
        arr = np.asarray(obs_np, dtype=np.float32)
        single = (arr.ndim == 1)
        if single:
            arr = arr[None, :]
        B = arr.shape[0]
        trunk = arr[:, :self.trunk_dim]
        pads_raw = arr[:, self.trunk_dim:self.obs_dim].reshape(B, NUM_BOOST_PADS, PAD_TOKEN_DIM)
        is_big = np.broadcast_to(PAD_IS_BIG_MASK.reshape(1, NUM_BOOST_PADS, 1), (B, NUM_BOOST_PADS, 1))
        pad_tokens = np.concatenate([pads_raw, is_big], axis=-1)

        q = (trunk @ sd["pad_attn.q_proj.weight"].T + sd["pad_attn.q_proj.bias"]).reshape(B, self.attn_heads, 1, self.attn_head_dim)
        k = (pad_tokens @ sd["pad_attn.k_proj.weight"].T + sd["pad_attn.k_proj.bias"]).reshape(B, NUM_BOOST_PADS, self.attn_heads, self.attn_head_dim).transpose(0, 2, 1, 3)
        v = (pad_tokens @ sd["pad_attn.v_proj.weight"].T + sd["pad_attn.v_proj.bias"]).reshape(B, NUM_BOOST_PADS, self.attn_heads, self.attn_head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(float(self.attn_head_dim))
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores)
        attn_w = (exp_s / np.sum(exp_s, axis=-1, keepdims=True)).squeeze(-2)
        self.last_attn_weights = attn_w

        ctx = np.matmul(attn_w[:, :, None, :], v).squeeze(-2).reshape(B, self.attn_heads * self.attn_head_dim)
        pad_embed = np.tanh(ctx @ sd["pad_attn.out_proj.weight"].T + sd["pad_attn.out_proj.bias"])
        feat = np.concatenate([trunk, pad_embed], axis=-1)
        return feat, single

    def _forward_np(self, obs_np):
        sd = self._np_state
        feat, single = self._encode_np(obs_np)
        h1 = np.tanh(feat @ sd["actor_mlp.0.weight"].T + sd["actor_mlp.0.bias"])
        h2 = np.tanh(h1 @ sd["actor_mlp.2.weight"].T + sd["actor_mlp.2.bias"])
        logits = h2 @ sd["actor_mlp.4.weight"].T + sd["actor_mlp.4.bias"]

        vh1 = np.tanh(feat @ sd["critic_mlp.0.weight"].T + sd["critic_mlp.0.bias"])
        vh2 = np.tanh(vh1 @ sd["critic_mlp.2.weight"].T + sd["critic_mlp.2.bias"])
        val = (vh2 @ sd["critic_mlp.4.weight"].T + sd["critic_mlp.4.bias"]).squeeze(-1)
        if single:
            return logits[0].copy(), float(val[0])
        return logits.copy(), val

    def _encode(self, x):
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)
        trunk = x[:, :self.trunk_dim]
        pads_raw = x[:, self.trunk_dim:self.obs_dim].view(-1, NUM_BOOST_PADS, PAD_TOKEN_DIM)
        pad_embed, attn_w = self.pad_attn(trunk, pads_raw)
        self.last_attn_weights = attn_w
        feat = torch.cat([trunk, pad_embed], dim=-1)
        return feat, single

    def actor(self, x):
        feat, single = self._encode(x)
        out = self.actor_mlp(feat)
        return out.squeeze(0) if single else out

    def critic(self, x):
        feat, single = self._encode(x)
        out = self.critic_mlp(feat)
        return out.squeeze(0) if single else out

    def forward(self, x):
        if not _TORCH_AVAILABLE:
            return self._forward_np(x)
        feat, single = self._encode(x)
        logits = self.actor_mlp(feat)
        val = self.critic_mlp(feat)
        if single:
            return logits.squeeze(0), val.squeeze(0)
        return logits, val

    def get_value(self, x):
        if not _TORCH_AVAILABLE:
            _, val = self._forward_np(x)
            return val
        return self.critic(x).squeeze(-1)

    def get_attention_stats(self, attn_weights=None):
        """计算 Boost Pad 注意力监控指标:
        - pad_ent: 归一化注意力熵 in [0, 1] (0=完全聚焦单垫子, 1=34个垫子完全均匀分布)
        - pad_max: 平均最高单垫子注意力权重 in [0, 1]
        - mean_probs: (34,) numpy 数组 (最近一次前向传播的 34 个垫子注意力均值)
        """
        w = attn_weights if attn_weights is not None else self.last_attn_weights
        if w is None:
            return 1.0, 1.0 / NUM_BOOST_PADS, np.full(NUM_BOOST_PADS, 1.0 / NUM_BOOST_PADS, dtype=np.float32)
        if not _TORCH_AVAILABLE or isinstance(w, np.ndarray):
            w_np = np.asarray(w, dtype=np.float32)
            p = w_np.mean(axis=1)
            p_clip = np.clip(p, 1e-8, 1.0)
            ent = -(p_clip * np.log(p_clip)).sum(axis=-1) / math.log(float(NUM_BOOST_PADS))
            max_w = p.max(axis=-1)
            mean_vec = p.mean(axis=0).astype(np.float32)
            return float(np.mean(ent)), float(np.mean(max_w)), mean_vec
        with torch.no_grad():
            p = w.mean(dim=1)
            ent = -(p * torch.log(p.clamp(min=1e-8))).sum(dim=-1) / math.log(float(NUM_BOOST_PADS))
            max_w = p.max(dim=-1)[0]
            mean_vec = p.mean(dim=0).detach().cpu().numpy().astype(np.float32)
            return float(ent.mean().item()), float(max_w.mean().item()), mean_vec

    def revive_jump_head(self, off_bias=0.0, weight_std=0.01):
        if not _TORCH_AVAILABLE:
            return
        with torch.no_grad():
            last_linear = self.actor_mlp[-1]
            nn.init.orthogonal_(last_linear.weight[9:11], weight_std)
            last_linear.bias[9] = float(off_bias)
            last_linear.bias[10] = 0.0

    def _clamp_raw_logits(self, raw_logits):
        if self.logit_clamp <= 0.0:
            return raw_logits
        parts = []
        for (s, e) in self._slices:
            ch_l = raw_logits[:, s:e]
            ch_c = ch_l - ch_l.mean(dim=-1, keepdim=True)
            parts.append(_ProjectedClampSTE.apply(ch_c, self.logit_clamp))
        return torch.cat(parts, dim=-1)

    def compute_logit_bound_loss(self, raw_logits):
        if self.logit_clamp <= 0.0:
            return torch.tensor(0.0, device=raw_logits.device)
        losses = []
        for (s, e) in self._slices:
            ch_l = raw_logits[:, s:e]
            ch_c = ch_l - ch_l.mean(dim=-1, keepdim=True)
            excess = F.relu(ch_c.abs() - self.logit_clamp)
            losses.append((excess * excess).mean())
        return torch.stack(losses).mean()

    def apply_action_mask(self, raw_logits, obs, ch4_action=None, clamp_logits=True):
        if clamp_logits and self.logit_clamp > 0.0:
            masked = self._clamp_raw_logits(raw_logits)
        else:
            masked = raw_logits.clone()

        on_ground = obs[:, IDX_ON_GROUND] > 0.5
        in_air    = ~on_ground
        has_nitro = obs[:, IDX_BOOST] > 1e-4
        no_nitro  = ~has_nitro

        masked[no_nitro, 12] = -1e9
        masked[on_ground, 6] = -1e9
        masked[on_ground, 8] = -1e9
        masked[in_air, 14] = -1e9

        if ch4_action is not None:
            force_w = on_ground & (ch4_action == 1) & has_nitro
            masked[force_w, 0] = -1e9
            masked[force_w, 1] = -1e9
        return masked

    @((torch.no_grad()) if _TORCH_AVAILABLE else (lambda f: f))
    def act_step_fast(self, obs_np, deterministic=False, forced_action_overrides=None):
        if _TORCH_AVAILABLE and self._np_state is None:
            x = torch.from_numpy(obs_np)
            feat, _ = self._encode(x)
            logits = self.actor_mlp(feat).squeeze(0).numpy().copy()
            val = float(self.critic_mlp(feat).item())
        else:
            logits, val = self._forward_np(obs_np)

        if not deterministic and self.logit_clamp > 0.0:
            for (s, e) in self._slices:
                ch_l = logits[s:e]
                logits[s:e] = np.clip(ch_l - np.mean(ch_l), -self.logit_clamp, self.logit_clamp)

        on_ground = bool(obs_np[IDX_ON_GROUND] > 0.5)
        has_nitro = bool(obs_np[IDX_BOOST] > 1e-4)

        if not has_nitro:
            logits[12] = -1e9
        if on_ground:
            logits[6] = -1e9
            logits[8] = -1e9
        else:
            logits[14] = -1e9

        act = np.empty(len(self._slices), dtype=np.int64)
        logp_total = 0.0

        def _sample_ch(ch_idx):
            s, e = self._slices[ch_idx]
            l = logits[s:e]
            l_max = np.max(l)
            exp_l = np.exp(l - l_max)
            sum_exp = np.sum(exp_l)
            probs = exp_l / sum_exp
            if forced_action_overrides is not None and ch_idx in forced_action_overrides:
                a = int(forced_action_overrides[ch_idx])
            elif deterministic:
                a = int(np.argmax(l))
            else:
                r = np.random.random()
                c = 0.0
                a = len(probs) - 1
                for k, p in enumerate(probs):
                    c += p
                    if r <= c:
                        a = k
                        break
            act[ch_idx] = a
            return float(l[a] - l_max - np.log(sum_exp))

        # 1. Sample ch4 (Boost) first
        logp_total += _sample_ch(4)

        # 2. If on_ground and Boost==On and has_nitro, mask out S(0) and None(1) -> force W(2)
        if on_ground and act[4] == 1 and has_nitro:
            logits[0] = -1e9
            logits[1] = -1e9

        # 3. Sample remaining channels (0, 1, 2, 3, 5)
        for i in (0, 1, 2, 3, 5):
            logp_total += _sample_ch(i)

        return act, logp_total, val

    def get_action_and_value(self, x, action=None, deterministic=False):
        if not _TORCH_AVAILABLE or self._np_state is not None:
            obs_np = np.asarray(x, dtype=np.float32)
            act, logp, val = self.act_step_fast(obs_np, deterministic=deterministic)
            return act, logp, 0.0, val
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        feat, _ = self._encode(x)
        raw_logits = self.actor_mlp(feat)
        value = self.critic_mlp(feat).squeeze(-1)

        use_clamp = (not deterministic) or (action is not None)
        logits = self.apply_action_mask(raw_logits, x, ch4_action=None, clamp_logits=use_clamp)

        s4, e4 = self._slices[4]
        l4 = logits[:, s4:e4]
        log_p4 = F.log_softmax(l4, dim=-1)
        p4 = log_p4.exp()
        if action is None:
            if deterministic:
                a4 = torch.argmax(l4, dim=-1)
            else:
                a4 = torch.multinomial(p4, 1).squeeze(-1)
        else:
            a4 = action[:, 4]

        on_ground = x[:, IDX_ON_GROUND] > 0.5
        has_nitro = x[:, IDX_BOOST] > 1e-4
        force_w = on_ground & (a4 == 1) & has_nitro
        if force_w.any():
            logits = logits.clone()
            logits[force_w, 0] = -1e9
            logits[force_w, 1] = -1e9

        acts = [None] * len(self._slices)
        logprobs = [None] * len(self._slices)
        entropies = [None] * len(self._slices)

        acts[4] = a4
        logprobs[4] = log_p4.gather(-1, a4.unsqueeze(-1)).squeeze(-1)
        entropies[4] = -(p4 * log_p4).sum(dim=-1)

        for i in (0, 1, 2, 3, 5):
            s, e = self._slices[i]
            l = logits[:, s:e]
            log_p = F.log_softmax(l, dim=-1)
            p = log_p.exp()
            if action is None:
                if deterministic:
                    a = torch.argmax(l, dim=-1)
                else:
                    a = torch.multinomial(p, 1).squeeze(-1)
                acts[i] = a
            else:
                a = action[:, i]
            logprobs[i] = log_p.gather(-1, a.unsqueeze(-1)).squeeze(-1)
            entropies[i] = -(p * log_p).sum(dim=-1)

        if action is None:
            action = torch.stack(acts, dim=-1)

        logprob_sum = torch.stack(logprobs, dim=-1).sum(dim=-1)
        entropy_sum = torch.stack(entropies, dim=-1).sum(dim=-1)

        if single:
            return action.squeeze(0), logprob_sum.squeeze(0), entropy_sum.squeeze(0), value.squeeze(0)
        return action, logprob_sum, entropy_sum, value

    def get_channel_probs(self, x):
        if not _TORCH_AVAILABLE or self._np_state is not None:
            obs_np = np.asarray(x, dtype=np.float32)
            if obs_np.ndim > 1:
                obs_np = obs_np[0]
            logits, value = self._forward_np(obs_np)
            for (s, e) in self._slices:
                ch_l = logits[s:e]
                logits[s:e] = np.clip(ch_l - np.mean(ch_l), -self.logit_clamp, self.logit_clamp)
            on_ground = bool(obs_np[IDX_ON_GROUND] > 0.5)
            has_nitro = bool(obs_np[IDX_BOOST] > 1e-4)
            if not has_nitro:
                logits[12] = -1e9
            if on_ground:
                logits[6] = -1e9
                logits[8] = -1e9
            else:
                logits[14] = -1e9
            a4 = int(np.argmax(logits[11:13]))
            if on_ground and a4 == 1 and has_nitro:
                logits[0] = -1e9
                logits[1] = -1e9
            probs = []
            for (s, e) in self._slices:
                l = logits[s:e]
                l_max = np.max(l)
                exp_l = np.exp(l - l_max)
                probs.append((exp_l / np.sum(exp_l)).astype(np.float32))
            return probs, float(value)

        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            feat, _ = self._encode(x)
            raw_logits = self.actor_mlp(feat)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=None, clamp_logits=True)
            a4 = torch.argmax(logits[:, 11:13], dim=-1)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=a4, clamp_logits=True).squeeze(0)
            value = float(self.critic_mlp(feat).squeeze().item())
            probs = [F.softmax(logits[s:e], dim=-1).cpu().numpy() for (s, e) in self._slices]
        return probs, value


def multi_discrete_to_controls(action_array):
    c = rs.CarControls()
    thr_val = float(int(action_array[0]) - 1)
    str_val = float(int(action_array[1]) - 1)
    rol_val = float(int(action_array[2]) - 1)

    c.throttle  = thr_val
    c.pitch     = -thr_val
    c.steer     = str_val
    c.yaw       = str_val
    c.roll      = rol_val
    c.jump      = bool(int(action_array[3]) == 1)
    c.boost     = bool(int(action_array[4]) == 1)
    c.handbrake = bool(int(action_array[5]) == 1)
    return c


def controls_to_multi_discrete(ctrl_dict):
    thr = float(ctrl_dict.get("thr", 0.0))
    ptc = float(ctrl_dict.get("ptc", 0.0))
    if thr > 0.3 or ptc < -0.3:
        ch0 = 2
    elif thr < -0.3 or ptc > 0.3:
        ch0 = 0
    else:
        ch0 = 1

    st = float(ctrl_dict.get("str", 0.0))
    yw = float(ctrl_dict.get("yaw", 0.0))
    if st > 0.3 or yw > 0.3:
        ch1 = 2
    elif st < -0.3 or yw < -0.3:
        ch1 = 0
    else:
        ch1 = 1

    rl = float(ctrl_dict.get("roll", 0.0))
    if rl > 0.3:
        ch2 = 2
    elif rl < -0.3:
        ch2 = 0
    else:
        ch2 = 1

    ch3 = 1 if bool(ctrl_dict.get("jump", False)) else 0
    ch4 = 1 if bool(ctrl_dict.get("boost", False)) else 0
    ch5 = 1 if bool(ctrl_dict.get("handbrake", False) or ctrl_dict.get("hb", False)) else 0
    return np.array([ch0, ch1, ch2, ch3, ch4, ch5], dtype=np.int64)


def multi_discrete_to_key_str(act_arr):
    keys = []
    if act_arr[0] == 2: keys.append("W")
    elif act_arr[0] == 0: keys.append("S")
    if act_arr[1] == 0: keys.append("A")
    elif act_arr[1] == 2: keys.append("D")
    if act_arr[2] == 0: keys.append("Q")
    elif act_arr[2] == 2: keys.append("E")
    if act_arr[3] == 1: keys.append("Jump")
    if act_arr[4] == 1: keys.append("Boost")
    if act_arr[5] == 1: keys.append("Slide")
    return "+".join(keys) if keys else "None"
