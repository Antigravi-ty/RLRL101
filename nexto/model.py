"""nexto/model.py —— 零 PyTorch 强依赖的 Nexto (EARLPerceiver + ControlsPredictorDot) 纯 NumPy 批量推理引擎。"""
import math
import os
import pickle
import zipfile
import numpy as np
import RocketSim as rs

from core.policy_net import controls_to_multi_discrete, multi_discrete_to_key_str

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nexto-model.pt")


def _load_torchscript_archive_np(pt_path: str):
    """直接从 TorchScript 归档 (.pt zip) 解析全部权重张量与常量表，无需安装 PyTorch。"""
    with zipfile.ZipFile(pt_path, "r") as z:
        names = z.namelist()
        data_pkl = next((n for n in names if n.endswith("/data.pkl") or n == "data.pkl"), None)
        const_pkl = next((n for n in names if n.endswith("/constants.pkl") or n == "constants.pkl"), None)
        if data_pkl is None or const_pkl is None:
            raise RuntimeError(f"Invalid TorchScript archive: {pt_path}")

        storages = {
            n.split("/data/")[1]: z.read(n)
            for n in names
            if "/data/" in n and not n.endswith("/")
        }
        const_storages = {
            n.split("/constants/")[1]: z.read(n)
            for n in names
            if "/constants/" in n and not n.endswith("/")
        }

        class _DummyModule:
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                if isinstance(state, dict):
                    self.__dict__.update(state)
                else:
                    self._state = state

        class _DataUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        arr = np.frombuffer(storage, dtype=np.float32)
                        if len(size) == 0:
                            return arr[storage_offset].copy()
                        total = int(np.prod(size))
                        return arr[storage_offset:storage_offset + total].reshape(size).copy()
                    return _rebuild
                if module.startswith("__torch__"):
                    return type(name, (_DummyModule,), {})
                if module == "torch" and "Storage" in name:
                    return name
                return super().find_class(module, name)

            def persistent_load(self, pid):
                return storages[str(pid[2])]

        class _ConstUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        dt, raw = storage
                        arr = np.frombuffer(raw, dtype=dt)
                        if len(size) == 0:
                            return arr[storage_offset].copy()
                        total = int(np.prod(size))
                        return arr[storage_offset:storage_offset + total].reshape(size).copy()
                    return _rebuild
                if module == "torch":
                    return {
                        "FloatStorage": np.float32,
                        "LongStorage": np.int64,
                        "DoubleStorage": np.float64,
                    }.get(name, name)
                return super().find_class(module, name)

            def persistent_load(self, pid):
                return (pid[1], const_storages[str(pid[2])])

        root_obj = _DataUnpickler(z.open(data_pkl)).load()
        consts = _ConstUnpickler(z.open(const_pkl)).load()

    state_dict = {}

    def _collect(obj, prefix=""):
        if isinstance(obj, np.ndarray):
            state_dict[prefix] = obj
        elif hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if not k.startswith("_"):
                    _collect(v, f"{prefix}.{k}" if prefix else k)

    _collect(root_obj)
    return state_dict, consts


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


class NextoAgent:
    """Nexto 预训练策略批量推理器，支持任意数量车辆 (B >= 1) 与动态实体序列长度 (N_entities)。"""

    def __init__(self, model_path: str = None, deterministic: bool = True, stochastic_kickoffs: bool = True):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.deterministic = bool(deterministic)
        self.stochastic_kickoffs = bool(stochastic_kickoffs)

        sd, consts = _load_torchscript_archive_np(self.model_path)
        self.sd = sd
        self.num_heads = int(consts[0]) if len(consts) > 0 else 4
        self.scale = float(consts[1]) if len(consts) > 1 else math.sqrt(32.0)
        self.lookup_table = np.asarray(consts[2], dtype=np.float32) if len(consts) > 2 else self.make_lookup_table()

        # Precompute constant 90-action embeddings through ControlsPredictorDot.net (8 -> 32 -> 32 -> 32)
        h = np.maximum(0.0, self.lookup_table @ sd["net.output.net.0.weight"].T + sd["net.output.net.0.bias"])
        h = np.maximum(0.0, h @ sd["net.output.net.2.weight"].T + sd["net.output.net.2.bias"])
        self.act_emb = np.maximum(0.0, h @ sd["net.output.net.4.weight"].T + sd["net.output.net.4.bias"])  # (90, 32)

        # Precompute mapping from 90 discrete actions to 6 multi-discrete channels [3, 3, 3, 2, 2, 2]
        self._ch_masks = self._build_channel_projection_masks(self.lookup_table)

    @staticmethod
    def make_lookup_table() -> np.ndarray:
        actions = []
        # Ground (27 actions)
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial (63 actions)
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:
                                continue
                            if pitch == roll == jump == 0:
                                continue
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, float(handbrake)])
        return np.array(actions, dtype=np.float32)

    @staticmethod
    def _build_channel_projection_masks(lookup_table: np.ndarray):
        """构造 (90,) -> [3, 3, 3, 2, 2, 2] 各通道边缘概率投影矩阵。"""
        dims = (3, 3, 3, 2, 2, 2)
        masks = [np.zeros((d, len(lookup_table)), dtype=np.float32) for d in dims]
        for k, a in enumerate(lookup_table):
            ctrl_dict = {
                "thr": float(a[0]),
                "str": float(a[1]),
                "ptc": float(a[2]),
                "yaw": float(a[3]),
                "roll": float(a[4]),
                "jump": bool(a[5] > 0.5),
                "boost": bool(a[6] > 0.5),
                "handbrake": bool(a[7] > 0.5),
            }
            md = controls_to_multi_discrete(ctrl_dict)
            for ch_i, val_i in enumerate(md):
                masks[ch_i][int(val_i), k] = 1.0
        return masks

    def _mha_block(self, q: np.ndarray, kv: np.ndarray, mask: np.ndarray, prefix: str):
        sd = self.sd
        q_norm = _layer_norm(q, sd[f"{prefix}.norm1.weight"], sd[f"{prefix}.norm1.bias"])
        kv_norm = _layer_norm(kv, sd[f"{prefix}.norm2.weight"], sd[f"{prefix}.norm2.bias"])

        wq, wk, wv = np.split(sd[f"{prefix}.attention.in_proj_weight"], 3, axis=0)
        bq, bk, bv = np.split(sd[f"{prefix}.attention.in_proj_bias"], 3, axis=0)

        B, Lq, D = q.shape
        N = kv.shape[1]
        H = self.num_heads
        Hd = D // H

        Q = (q_norm @ wq.T + bq).reshape(B, Lq, H, Hd).transpose(0, 2, 1, 3)  # (B, H, 1, Hd)
        K = (kv_norm @ wk.T + bk).reshape(B, N, H, Hd).transpose(0, 2, 1, 3)   # (B, H, N, Hd)
        V = (kv_norm @ wv.T + bv).reshape(B, N, H, Hd).transpose(0, 2, 1, 3)   # (B, H, N, Hd)

        scores = np.matmul(Q / self.scale, K.transpose(0, 1, 3, 2))            # (B, H, 1, N)
        if mask is not None:
            scores = scores + np.where(mask[:, None, None, :] > 0.5, -1e9, 0.0)

        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - scores_max)
        attn_w = exp_s / np.sum(exp_s, axis=-1, keepdims=True)                 # (B, H, 1, N)

        ctx = np.matmul(attn_w, V).transpose(0, 2, 1, 3).reshape(B, Lq, D)
        attn_out = ctx @ sd[f"{prefix}.attention.out_proj.weight"].T + sd[f"{prefix}.attention.out_proj.bias"]
        x = q + attn_out

        x_norm = _layer_norm(x, sd[f"{prefix}.norm3.weight"], sd[f"{prefix}.norm3.bias"])
        ff = np.maximum(0.0, x_norm @ sd[f"{prefix}.linear1.weight"].T + sd[f"{prefix}.linear1.bias"])
        ff = ff @ sd[f"{prefix}.linear2.weight"].T + sd[f"{prefix}.linear2.bias"]
        # Return block output (B, 1, D) and head-averaged attention weights (B, N)
        return x + ff, np.mean(attn_w[:, :, 0, :], axis=1)

    def forward_batch(self, q: np.ndarray, kv: np.ndarray, m: np.ndarray = None):
        """批量前向传播:
        - q:  (B, 1, 32)
        - kv: (B, N_entities, 24)
        - m:  (B, N_entities)
        返回: (logits_90: (B, 90), mean_attn_weights: (B, N_entities))
        """
        sd = self.sd
        qh = np.maximum(0.0, q @ sd["net.earl.query_preprocess.0.weight"].T + sd["net.earl.query_preprocess.0.bias"])
        qh = np.maximum(0.0, qh @ sd["net.earl.query_preprocess.2.weight"].T + sd["net.earl.query_preprocess.2.bias"])

        kvh = np.maximum(0.0, kv @ sd["net.earl.key_value_preprocess.0.weight"].T + sd["net.earl.key_value_preprocess.0.bias"])
        kvh = np.maximum(0.0, kvh @ sd["net.earl.key_value_preprocess.2.weight"].T + sd["net.earl.key_value_preprocess.2.bias"])

        qh, w0 = self._mha_block(qh, kvh, m, "net.earl.blocks.0")
        qh, w1 = self._mha_block(qh, kvh, m, "net.earl.blocks.1")

        qh_relu = np.maximum(0.0, qh[:, 0, :])  # (B, 128)
        player_emb = qh_relu @ sd["net.output.emb_convertor.weight"].T + sd["net.output.emb_convertor.bias"]  # (B, 32)
        logits = player_emb @ self.act_emb.T    # (B, 90)
        mean_attn = 0.5 * (w0 + w1)             # (B, N_entities)
        return logits, mean_attn

    def act_batch(self, q: np.ndarray, kv: np.ndarray, m: np.ndarray = None, is_kickoff: bool = False, rng=None):
        """对 B 辆车同时推理并采样动作。
        返回: (actions_8d: (B, 8), act_indices: (B,), probs_90: (B, 90), mean_attn: (B, N_entities))
        """
        logits, mean_attn = self.forward_batch(q, kv, m)
        B = logits.shape[0]

        beta = 1.0 if self.deterministic else 0.5
        if is_kickoff and self.stochastic_kickoffs and self.deterministic:
            beta = 0.5

        scaled_logits = logits.copy()
        if 0.0 < abs(beta) < 1.0:
            scaled_logits = scaled_logits * math.log((beta + 1.0) / (1.0 - beta), 3.0)

        l_max = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_l = np.exp(scaled_logits - l_max)
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)

        if beta >= 1.0:
            act_indices = np.argmax(logits, axis=-1).astype(np.int64)
        elif beta <= -1.0:
            act_indices = np.argmin(logits, axis=-1).astype(np.int64)
        else:
            act_indices = np.empty(B, dtype=np.int64)
            for b_i in range(B):
                r = rng.random() if rng is not None else np.random.random()
                cdf = np.cumsum(probs[b_i])
                idx = int(np.searchsorted(cdf, r, side="right"))
                act_indices[b_i] = min(idx, probs.shape[1] - 1)

        actions_8d = self.lookup_table[act_indices].copy()
        return actions_8d, act_indices, probs, mean_attn

    def marginal_channel_probs(self, probs_90_single: np.ndarray):
        """将单车的 90 维动作概率分布边缘化为 6 个控制通道的概率分布列表。"""
        p = np.asarray(probs_90_single, dtype=np.float32)
        out = []
        for mask in self._ch_masks:
            ch_p = mask @ p
            s = float(np.sum(ch_p))
            if s > 1e-8:
                ch_p = ch_p / s
            out.append(ch_p.astype(np.float32))
        return out

    @staticmethod
    def attention_stats(attn_single: np.ndarray):
        """计算单车的实体注意力归一化熵与最高注意力权重。"""
        p = np.asarray(attn_single, dtype=np.float32)
        n = max(2, len(p))
        p_clip = np.clip(p, 1e-8, 1.0)
        p_clip = p_clip / np.sum(p_clip)
        ent = float(-np.sum(p_clip * np.log(p_clip)) / math.log(float(n)))
        max_w = float(np.max(p_clip))
        return ent, max_w


def action_8d_to_car_controls(act_8d: np.ndarray) -> rs.CarControls:
    c = rs.CarControls()
    c.throttle  = float(act_8d[0])
    c.steer     = float(act_8d[1])
    c.pitch     = float(act_8d[2])
    c.yaw       = float(act_8d[3])
    c.roll      = float(act_8d[4])
    c.jump      = bool(act_8d[5] > 0.5)
    c.boost     = bool(act_8d[6] > 0.5)
    c.handbrake = bool(act_8d[7] > 0.5)
    return c


def action_8d_to_input_dict(act_8d: np.ndarray, act_idx: int = 0) -> dict:
    d = {
        "thr": float(act_8d[0]),
        "str": float(act_8d[1]),
        "ptc": float(act_8d[2]),
        "yaw": float(act_8d[3]),
        "roll": float(act_8d[4]),
        "jump": bool(act_8d[5] > 0.5),
        "boost": bool(act_8d[6] > 0.5),
        "handbrake": bool(act_8d[7] > 0.5),
    }
    md = controls_to_multi_discrete(d)
    d["ppo_channels"] = [int(x) for x in md]
    d["nexto_enum_idx"] = int(act_idx)
    d["keys_str"] = f"#{int(act_idx):02d}:" + multi_discrete_to_key_str(md)
    return d
