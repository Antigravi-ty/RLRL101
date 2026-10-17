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
    OBS_V3_DIM, PPO_ACTION_DIMS, PPO_LOGITS_DIM, PPO_HIDDEN_DIM,
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
    def __init__(
        self,
        obs_dim=OBS_V3_DIM,
        hidden=PPO_HIDDEN_DIM,
        action_dims=PPO_ACTION_DIMS,
        logit_clamp=LOGIT_CLAMP_RANGE,
    ):
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

        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.logits_dim), std=0.01, bias_const=0.0),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0, bias_const=0.0),
        )

        # Balance initial gradient SNR for 187d obs:
        # 136 dimensions [46:182] are the 34 boost pads; scaling their initial input
        # weights by 0.15 prevents 34 pads from drowning out the 15 kinematics/ball dims at ep 0.
        if self.obs_dim == 187:
            with torch.no_grad():
                self.actor[0].weight[:, 46:182].mul_(0.15)
                self.critic[0].weight[:, 46:182].mul_(0.15)
                z_cols = [4, 11, 15, 19, 31, 32, 33, 35]
                self.actor[0].weight[:, z_cols].mul_(3.0)
                self.critic[0].weight[:, z_cols].mul_(3.0)

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def revive_jump_head(self, off_bias=0.0, weight_std=0.01):
        with torch.no_grad():
            last_linear = self.actor[-1]
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

        on_ground = obs[:, 24] > 0.5
        in_air    = ~on_ground
        has_nitro = obs[:, 29] > 1e-4
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
        x = torch.from_numpy(obs_np)
        logits = self.actor(x).numpy().copy()
        val = float(self.critic(x).item())

        if not deterministic and self.logit_clamp > 0.0:
            for (s, e) in self._slices:
                ch_l = logits[s:e]
                logits[s:e] = np.clip(ch_l - np.mean(ch_l), -self.logit_clamp, self.logit_clamp)

        on_ground = bool(obs_np[24] > 0.5)
        has_nitro = bool(obs_np[29] > 1e-4)

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
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        raw_logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

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

        on_ground = x[:, 24] > 0.5
        has_nitro = x[:, 29] > 1e-4
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
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            raw_logits = self.actor(x)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=None, clamp_logits=True)
            a4 = torch.argmax(logits[:, 11:13], dim=-1)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=a4, clamp_logits=True).squeeze(0)
            value = float(self.critic(x).squeeze().item())
            probs = [F.softmax(logits[s:e], dim=-1).cpu().numpy() for (s, e) in self._slices]
        return probs, value


def adapt_state_dict_input_dim(state_dict, model):
    from .obs_v3 import BIG_PAD_INDICES
    target_sd = model.state_dict()
    adapted = {}
    for k, v in state_dict.items():
        if k in target_sd and v.shape != target_sd[k].shape:
            t_v = target_sd[k].clone()
            if v.dim() == 2 and v.shape[0] == t_v.shape[0]:
                t_v.zero_()
                if v.shape[1] == 75 and t_v.shape[1] == 187:
                    t_v[:, :46] = v[:, :46]
                    for idx_k, pad_idx in enumerate(BIG_PAD_INDICES):
                        src_s = 46 + 4 * idx_k
                        dst_s = 46 + 4 * pad_idx
                        t_v[:, dst_s:dst_s + 4] = v[:, src_s:src_s + 4]
                    t_v[:, -5:] = v[:, 70:75]
                else:
                    min_in = min(v.shape[1], t_v.shape[1])
                    t_v[:, :min_in] = v[:, :min_in]
                    t_v[:, -5:] = v[:, -5:]
                adapted[k] = t_v
            else:
                adapted[k] = t_v
        else:
            adapted[k] = v
    return adapted


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
