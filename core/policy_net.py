"""core/policy_net.py —— 1v1 最快开球极简纯 MLP Actor-Critic 网络架构与动作掩码 (无 Attention 开销，极速推理与训练)。"""
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
    OBS_DIM,
    IDX_ON_GROUND, IDX_HAS_FLIP_OR_JUMP, IDX_IS_FLIPPING, IDX_IS_JUMPING, IDX_BOOST,
    PPO_ACTION_DIMS, PPO_HIDDEN_DIM,
    LOGIT_CLAMP_RANGE,
)


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


class ActorCritic(_BaseModule):
    """精简双层 Tanh MLP Actor-Critic (默认 42 -> 128 -> 128 -> 15/1，参数量仅 ~36K)。
    支持 PyTorch (CPU / MPS / CUDA) 训练与零依赖纯 NumPy 极速推理。
    """
    def __init__(
        self,
        obs_dim=OBS_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=LOGIT_CLAMP_RANGE,
        **kwargs,
    ):
        if _TORCH_AVAILABLE:
            super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden)
        self.action_dims = list(action_dims)
        self.logits_dim = sum(self.action_dims)
        self.logit_clamp = float(logit_clamp)
        self._slices = []
        cur = 0
        for d in self.action_dims:
            self._slices.append((cur, cur + d))
            cur += d

        self._np_state = None

        if not _TORCH_AVAILABLE:
            self._np_state = {
                "actor_mlp.0.weight": np.zeros((self.hidden_dim, self.obs_dim), dtype=np.float32),
                "actor_mlp.0.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "actor_mlp.2.weight": np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32),
                "actor_mlp.2.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "actor_mlp.4.weight": np.zeros((self.logits_dim, self.hidden_dim), dtype=np.float32),
                "actor_mlp.4.bias": np.zeros((self.logits_dim,), dtype=np.float32),
                "critic_mlp.0.weight": np.zeros((self.hidden_dim, self.obs_dim), dtype=np.float32),
                "critic_mlp.0.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "critic_mlp.2.weight": np.zeros((self.hidden_dim, self.hidden_dim), dtype=np.float32),
                "critic_mlp.2.bias": np.zeros((self.hidden_dim,), dtype=np.float32),
                "critic_mlp.4.weight": np.zeros((1, self.hidden_dim), dtype=np.float32),
                "critic_mlp.4.bias": np.zeros((1,), dtype=np.float32),
            }
            return

        self.actor_mlp = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.logits_dim), std=0.01, bias_const=0.0),
        )

        self.critic_mlp = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
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

    def _forward_np(self, obs_np):
        sd = self._np_state
        arr = np.asarray(obs_np, dtype=np.float32)
        single = (arr.ndim == 1)
        if single:
            arr = arr[None, :]
        h1 = np.tanh(arr @ sd["actor_mlp.0.weight"].T + sd["actor_mlp.0.bias"])
        h2 = np.tanh(h1 @ sd["actor_mlp.2.weight"].T + sd["actor_mlp.2.bias"])
        logits = h2 @ sd["actor_mlp.4.weight"].T + sd["actor_mlp.4.bias"]

        vh1 = np.tanh(arr @ sd["critic_mlp.0.weight"].T + sd["critic_mlp.0.bias"])
        vh2 = np.tanh(vh1 @ sd["critic_mlp.2.weight"].T + sd["critic_mlp.2.bias"])
        val = (vh2 @ sd["critic_mlp.4.weight"].T + sd["critic_mlp.4.bias"]).squeeze(-1)
        if single:
            return logits[0].copy(), float(val[0])
        return logits.copy(), val

    def actor(self, x):
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)
        out = self.actor_mlp(x)
        return out.squeeze(0) if single else out

    def critic(self, x):
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)
        out = self.critic_mlp(x)
        return out.squeeze(0) if single else out

    def forward(self, x):
        if not _TORCH_AVAILABLE:
            return self._forward_np(x)
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)
        logits = self.actor_mlp(x)
        val = self.critic_mlp(x)
        if single:
            return logits.squeeze(0), val.squeeze(0)
        return logits, val

    def get_value(self, x):
        if not _TORCH_AVAILABLE:
            _, val = self._forward_np(x)
            return val
        return self.critic(x).squeeze(-1)

    def _clamp_raw_logits(self, raw_logits):
        if self.logit_clamp <= 0.0:
            return raw_logits
        parts = []
        for (s, e) in self._slices:
            ch_l = raw_logits[:, s:e]
            ch_c = ch_l - ch_l.mean(dim=-1, keepdim=True)
            parts.append(_ProjectedClampSTE.apply(ch_c, self.logit_clamp))
        return torch.cat(parts, dim=-1)

    def compute_logit_bound_loss(self, raw_logits, obs=None, actions=None):
        if self.logit_clamp > 0.0:
            losses = []
            for (s, e) in self._slices:
                ch_l = raw_logits[:, s:e]
                ch_c = ch_l - ch_l.mean(dim=-1, keepdim=True)
                excess = F.relu(ch_c.abs() - self.logit_clamp)
                losses.append((excess * excess).mean())
            base_loss = torch.stack(losses).mean()
        else:
            base_loss = torch.tensor(0.0, device=raw_logits.device)

        if obs is not None and actions is not None:
            on_ground = obs[:, IDX_ON_GROUND] > 0.5
            has_nitro = obs[:, IDX_BOOST] > 1e-4
            force_w = on_ground & (actions[:, 4] == 1) & has_nitro
            if force_w.any():
                ch0_logp_w = F.log_softmax(raw_logits[force_w, 0:3], dim=-1)[:, 2]
                base_loss = base_loss - 0.15 * ch0_logp_w.mean()
        return base_loss

    def apply_action_mask(self, raw_logits, obs, ch4_action=None, clamp_logits=True):
        if clamp_logits and self.logit_clamp > 0.0:
            masked = self._clamp_raw_logits(raw_logits)
        else:
            masked = raw_logits.clone()

        on_ground        = obs[:, IDX_ON_GROUND] > 0.5
        in_air           = ~on_ground
        has_nitro        = obs[:, IDX_BOOST] > 1e-4
        no_nitro         = ~has_nitro
        has_flip_or_jump = obs[:, IDX_HAS_FLIP_OR_JUMP] > 0.5
        is_flipping      = obs[:, IDX_IS_FLIPPING] > 0.5
        is_jumping       = obs[:, IDX_IS_JUMPING] > 0.5
        prev_jump        = obs[:, 30] > 0.5
        cz               = obs[:, 4] * 5120.0
        b_up             = obs[:, 18] * 5000.0
        dir_fwd          = obs[:, 13]

        dead_air_jump         = in_air & (~has_flip_or_jump) & (~is_jumping)
        expired_hold_jump     = in_air & (~is_jumping) & prev_jump
        low_ball_auto_release = is_jumping & prev_jump & (cz > 34.0) & (b_up < 95.0)
        inverted_flip_boost   = is_flipping & (dir_fwd < 0.25)

        masked[no_nitro | inverted_flip_boost, 12] = -1e9
        masked[on_ground, 6] = -1e9
        masked[on_ground, 8] = -1e9
        masked[in_air, 14] = -1e9
        masked[dead_air_jump | expired_hold_jump | low_ball_auto_release, 10] = -1e9

        if ch4_action is not None:
            force_w = on_ground & (ch4_action == 1) & has_nitro
            masked[force_w, 0] = -1e9
            masked[force_w, 1] = -1e9
        return masked

    @((torch.no_grad()) if _TORCH_AVAILABLE else (lambda f: f))
    def act_step_fast(self, obs_np, deterministic=False, forced_action_overrides=None):
        if _TORCH_AVAILABLE and self._np_state is None:
            x = torch.from_numpy(obs_np)
            logits = self.actor_mlp(x).numpy().copy()
            val = float(self.critic_mlp(x).item())
        else:
            logits, val = self._forward_np(obs_np)

        if not deterministic and self.logit_clamp > 0.0:
            for (s, e) in self._slices:
                ch_l = logits[s:e]
                logits[s:e] = np.clip(ch_l - np.mean(ch_l), -self.logit_clamp, self.logit_clamp)

        on_ground = bool(obs_np[IDX_ON_GROUND] > 0.5)
        has_nitro = bool(obs_np[IDX_BOOST] > 1e-4)
        has_flip_or_jump = bool(obs_np[IDX_HAS_FLIP_OR_JUMP] > 0.5)
        is_flipping = bool(obs_np[IDX_IS_FLIPPING] > 0.5)
        is_jumping = bool(obs_np[IDX_IS_JUMPING] > 0.5)
        prev_jump = bool(obs_np[30] > 0.5)
        cz = float(obs_np[4]) * 5120.0
        b_up = float(obs_np[18]) * 5000.0
        dir_fwd = float(obs_np[13])

        if (not has_nitro) or (is_flipping and dir_fwd < 0.25):
            logits[12] = -1e9
        if on_ground:
            logits[6] = -1e9
            logits[8] = -1e9
            if is_jumping and prev_jump and cz > 34.0 and b_up < 95.0:
                logits[10] = -1e9
        else:
            logits[14] = -1e9
            if ((not has_flip_or_jump) and (not is_jumping)) or ((not is_jumping) and prev_jump) or (is_jumping and prev_jump and cz > 34.0 and b_up < 95.0):
                logits[10] = -1e9

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

        logp_total += _sample_ch(4)
        if on_ground and act[4] == 1 and has_nitro:
            logits[0] = -1e9
            logits[1] = -1e9
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

        raw_logits = self.actor_mlp(x)
        value = self.critic_mlp(x).squeeze(-1)

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
            has_flip_or_jump = bool(obs_np[IDX_HAS_FLIP_OR_JUMP] > 0.5)
            is_flipping = bool(obs_np[IDX_IS_FLIPPING] > 0.5)
            is_jumping = bool(obs_np[IDX_IS_JUMPING] > 0.5)
            prev_jump = bool(obs_np[30] > 0.5)
            cz = float(obs_np[4]) * 5120.0
            b_up = float(obs_np[18]) * 5000.0
            dir_fwd = float(obs_np[13])
            if (not has_nitro) or (is_flipping and dir_fwd < 0.25):
                logits[12] = -1e9
            if on_ground:
                logits[6] = -1e9
                logits[8] = -1e9
                if is_jumping and prev_jump and cz > 34.0 and b_up < 95.0:
                    logits[10] = -1e9
            else:
                logits[14] = -1e9
                if ((not has_flip_or_jump) and (not is_jumping)) or ((not is_jumping) and prev_jump) or (is_jumping and prev_jump and cz > 34.0 and b_up < 95.0):
                    logits[10] = -1e9
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
            raw_logits = self.actor_mlp(x)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=None, clamp_logits=True)
            a4 = torch.argmax(logits[:, 11:13], dim=-1)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=a4, clamp_logits=True).squeeze(0)
            value = float(self.critic_mlp(x).squeeze().item())
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
