"""神经网络定义：
纯 PPO 多离散 6 通道键盘 Actor-Critic 网络 (已彻底移除旧版 DQN 与固定宏依赖)。
"""
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
)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


_BaseModule = nn.Module if _TORCH_AVAILABLE else object


class ActorCritic(_BaseModule):
    """PPO 多离散通道 Actor-Critic 网络。

    6 个独立键盘通道 (PPO_ACTION_DIMS = [3, 3, 3, 2, 2, 2]):
      ch0: thr_pitch  (0 -> S:-1, 1 -> None:0, 2 -> W:+1)  [W/S 同时控制地面油门与空中俯仰]
      ch1: steer_yaw  (0 -> A:-1, 1 -> None:0, 2 -> D:+1)  [A/D 同时控制地面转向与空中偏航]
      ch2: roll       (0 -> Q:-1, 1 -> None:0, 2 -> E:+1)  [Q/E 控制空中横滚]
      ch3: jump       (0 -> Off,  1 -> On)
      ch4: boost      (0 -> Off,  1 -> On)
      ch5: handbrake  (0 -> Off,  1 -> On)
    """
    def __init__(self, obs_dim=OBS_V3_DIM, hidden=PPO_HIDDEN_DIM, action_dims=PPO_ACTION_DIMS):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_dim = int(hidden)
        self.action_dims = list(action_dims)
        self.logits_dim = sum(self.action_dims)
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
            layer_init(nn.Linear(self.hidden_dim, self.logits_dim), std=0.01),
        )
        # 动作输出层初始偏置先验（加速初期探索）：
        #   - ch0 (S/0/W): W (+0.8) 轻微正向先验
        #   - ch3 (Jump):  Off (+1.5) 先验，避免开局原地胡乱连跳飞天
        #   - ch4 (Boost): On (+0.5) 轻微先验
        #   - ch5 (Slide): Off (+1.5) 先验，避免全程锁死手刹打滑
        with torch.no_grad():
            b = self.actor[-1].bias
            b[2]  = 0.8   # W
            b[9]  = 1.5   # Jump Off
            b[12] = 0.5   # Boost On
            b[13] = 1.5   # Handbrake Off

        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
        )

    def forward(self, x):
        """前向返回 (logits, value)，方便直接导出 ONNX 给浏览器推理。"""
        return self.actor(x), self.critic(x)

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def revive_jump_head(self, off_bias=0.0, weight_std=0.01):
        """当从旧版饱和权重（如 P(jump=1) 极低的死跳跃头）恢复训练时，
        重置 ch3 (Jump, 索引 9:11) 的输出层权重与偏置，恢复 50/50 的跳跃与翻滚探索概率。
        """
        with torch.no_grad():
            last_linear = self.actor[-1]
            nn.init.orthogonal_(last_linear.weight[9:11], weight_std)
            last_linear.bias[9] = float(off_bias)
            last_linear.bias[10] = 0.0

    def apply_action_mask(self, raw_logits, obs, ch4_action=None):
        """对批量或单步 raw_logits 施加精确的物理空通道与物理非法动作掩码 (Invalid Action Masking)：
          1. ch4 (Boost, 11:13): 当无氮气 (obs[:, 29] <= 1e-4) 时，喷气为空通道，禁止 Boost=On (12 -> -1e9)；
          2. ch0 (S/0/W, 0:3)  : 当在地面 (obs[:, 24] > 0.5) & 按住 Boost (ch4 == 1) & 有氮气时，
                                 RocketSim 物理引擎底层强制满油门 (throttle=1.0)，ch0 为纯空通道。
                                 禁止 S(0) 与 None(1) (-> -1e9)，强制输出 W(2)，
                                 彻底消除空通道上的熵漂移 (Null-Channel Entropy Drift)；
          3. ch2 (Q/0/E, 6:9)  : 当在地面时，地面横滚为空通道，禁止 Q(6) 与 E(8) (-> -1e9)；
          4. ch5 (Slide, 13:15): 当在空中 (~on_ground) 时，手刹为空通道，禁止 Slide=On(14 -> -1e9)。
        """
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
    def act_step_fast(self, obs_np, deterministic=False):
        """单步极速无锁推理（内置条件动作空间 Mask，与 get_action_and_value 100% 严格对齐）：
        输入: obs_np (obs_dim,)
        输出: (act_np [6,], logprob_float, val_float)
        """
        x = torch.from_numpy(obs_np)
        logits = self.actor(x).numpy().copy()
        val = float(self.critic(x).item())

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
            if deterministic:
                a = int(np.argmax(probs))
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

        # 1. 先采样 ch4 (Boost)
        logp_total += _sample_ch(4)

        # 2. 若在地面且按住 Boost 且有氮气，强制 ch0 为 W(2)
        if on_ground and act[4] == 1 and has_nitro:
            logits[0] = -1e9
            logits[1] = -1e9

        # 3. 采样其余通道 (0, 1, 2, 3, 5)
        for i in (0, 1, 2, 3, 5):
            logp_total += _sample_ch(i)

        return act, logp_total, val

    def get_action_and_value(self, x, action=None, deterministic=False):
        """批量或单步计算带条件动作空间 Mask 的 logprob / entropy / value。"""
        single = (x.dim() == 1)
        if single:
            x = x.unsqueeze(0)

        raw_logits = self.actor(x)
        value = self.critic(x).squeeze(-1)

        # 第一阶段：先对静态通道施加掩码，并确定 ch4 (Boost) 动作 a4
        logits = self.apply_action_mask(raw_logits, x, ch4_action=None)
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

        # 第二阶段：根据 a4 对 ch0 (S/0/W) 施加条件掩码 (on_ground & boost==1 & nitro>0 -> 强制 W)
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
        """返回 6 个通道各自带条件掩码的概率分布列表与 Critic Value（供 HUD 实时展示）。"""
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            raw_logits = self.actor(x)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=None)
            a4 = torch.argmax(logits[:, 11:13], dim=-1)
            logits = self.apply_action_mask(raw_logits, x, ch4_action=a4).squeeze(0)
            value = float(self.critic(x).squeeze().item())
            probs = [F.softmax(logits[s:e], dim=-1).cpu().numpy() for (s, e) in self._slices]
        return probs, value


def multi_discrete_to_controls(action_array):
    """将 6 维整型动作数组 [ch0, ch1, ch2, ch3, ch4, ch5] 转换为 RocketSim CarControls。
    严格遵循键盘绑定物理规律：
      W (ch0==2) -> throttle = +1.0, pitch = -1.0 (空中前倾)
      S (ch0==0) -> throttle = -1.0, pitch = +1.0 (空中后仰)
      None(ch0==1)-> throttle = 0.0, pitch = 0.0
    """
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
    """将控制字典逆映射回 6 通道离散索引 [0..2, 0..2, 0..2, 0..1, 0..1, 0..1]。"""
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
    """将 6 通道离散动作格式化为直观的键盘按键字符串，如 'W+D+Boost'。"""
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
