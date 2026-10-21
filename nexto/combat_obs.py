"""nexto/combat_obs.py —— 双参考系 + 双注意力 (CarEntityAttention & 7维距离反比例 BoostPadAttention) + 四角色受控策略网络。

核心设计完全对齐你的架构规范:
1. 主干特征 (Trunk, 57 维):
   - A.face_dists    [0..5]   (6)  自身到 6 个球场平面的世界轴向距离 (阵营对称归一化 / 5120): [-x, +x, -y(自门), +y(敌门), floor, ceil]
   - B.car_ang_b     [6..8]   (3)  自身局部坐标系下的三轴角速度 (pitch, yaw, roll) / 5.5
   - C.car_vel_b     [9..12]  (4)  自身局部坐标系下的速度单位矢量 (v_dir_fwd, v_dir_right, v_dir_up) + 速度标量 |v| / 2300
   - D.ball_rel_b    [13..19] (7)  球在车体局部坐标系下的单位方向矢量 (3) + 相对坐标 (3) + 标量距离 (1)
   - E.flags         [20..25] (6)  on_ground, flip_or_jump, is_flipping, is_jumping, air_time, boost
   - F.closing       [26..26] (1)  车朝向球的主动接近速度 / 2300
   - G.prev_act      [27..32] (6)  上一决策步动作 (6 通道)
   - H.car_orient    [33..35] (3)  车体三轴的世界 Z 分量 (fwd_z, right_z, up_z)
   - I.ball_faces    [36..41] (6)  球体到 6 个球场平面的世界轴向距离 (阵营对称归一化 / 5120)
   - K.up_ref_kin    [42..52] (11) 对抗重力的 Z 轴恒定朝上参考系 (Car-Origin Yaw-Aligned Up-Z Frame):
                                   自身速度单位矢量(3) + 球的相对位置(3) + 球的相对速度(3) + 车头俯仰/横滚 sin_pitch, sin_roll (2)
   - L.role_onehot   [53..56] (4)  四角色战术标签 One-Hot: [Chaser, Optimistic, Pessimistic, Freelancer]

2. 第一套注意力: 可变长其他车辆注意力 (`CarEntityAttention`, 支持任意数量车辆, 3v3 下为 2 队友 + 3 对手 = 5 Token):
   - 每个车辆实体 Token (24 维, 在自身原点 + 自身车头水平朝向为正方向、Z 轴朝上的相对绝对参考系下表达):
     - [0..1]   (2)  阵营标识: [is_teammate, is_opponent]
     - [2..4]   (3)  相对方向单位矢量 (dir_x, dir_y, dir_z)
     - [5..7]   (3)  相对位置坐标 (rel_x, rel_y, rel_z) / 5120
     - [8]      (1)  相对欧氏距离 dist / 5120
     - [9..11]  (3)  该车速度单位矢量 (vx_dir, vy_dir, vz_dir)
     - [12]     (1)  该车速度标量 |v| / 2300
     - [13..15] (3)  该车三轴角速度 (wx, wy, wz) / 5.5
     - [16..18] (3)  该车车头朝向单位矢量 (fwd_x, fwd_y, fwd_z)
     - [19]     (1)  该车氮气量 boost / 100
     - [20..23] (4)  状态与战术角色: [on_ground, has_flip, is_demoed, is_chaser_flag]
   - 支持蒸馏阶段的 Information Bottleneck (50% 概率遮蔽队友 Token，逼迫网络读取自身的 L.role_onehot 标签)。
   - 同时内置距离反比例偏置 (Distance Decay Bias)，优先关注近距离对抗/配合车辆。

3. 第二套注意力: 34 个氮气垫注意力 (`BoostPadAttention`, 7 维完整特征 + 距离反比例偏置):
   - 每个 Boost Pad Token (7 维, 在自身原点 + 车头朝向参考系下):
     - [0..2] (3) 相对方向单位矢量 (dir_x, dir_y, dir_z)
     - [3]    (1) 绝对欧氏距离 dist / 5120
     - [4]    (1) 可拾取状态 is_active (bool: 1.0=able to collect, 0.0=not able to collect)
     - [5]    (1) 归一化冷却倒计时 cooldown (down counter: remaining_cooldown / max_cooldown)
     - [6]    (1) 大小垫标识 is_big (bool: 1.0=100%大氮气, 0.0=12%小氮气)
   - 内置可学习距离反比例函数偏置: score_j = (Q K_j^T) / sqrt(d) - alpha_h * log(1 + dist_j / D0)，确保初始化时注意力天然聚焦于附近的氮气垫！
"""
import math
import numpy as np

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

from core.obs import (
    BOOST_PAD_LOCATIONS, PAD_COOLDOWN_SCALES, PAD_IS_BIG_MASK,
    FIELD_X, FIELD_Y, FIELD_Z, S_FACE, S_CAR_ANG, S_CAR_VEL, S_BALL_POS, S_CLOSING,
    format_prev_action_vec,
)
from core.constants import NUM_BOOST_PADS, PPO_ACTION_DIMS, PPO_HIDDEN_DIM
from .role_labeler import ROLE_ONE_HOTS, ROLE_CHASER

COMBAT_TRUNK_DIM     = 57
CAR_TOKEN_DIM        = 24
BOOST_PAD_TOKEN_DIM  = 7
DEFAULT_MAX_OTHER_CARS = 5  # 3v3: 2 teammates + 3 opponents


def _rot_xy_to_car_yaw(vec_3d: np.ndarray, cos_t: float, sin_t: float) -> np.ndarray:
    """将世界/阵营对齐坐标系下的 3D 向量绕世界 Z 轴旋转至【以自身车头水平朝向为 +Y 正前方、+X 为右方、+Z 竖直朝上】的重力参考系。
    完美对齐 Nexto `_convert_to_relative` 的几何定义:
      x' =  cos_t * x - sin_t * y
      y' =  sin_t * x + cos_t * y
      z' =  z (严格保持竖直朝上对抗重力!)
    """
    x = vec_3d[..., 0]
    y = vec_3d[..., 1]
    z = vec_3d[..., 2]
    nx = cos_t * x - sin_t * y
    ny = sin_t * x + cos_t * y
    return np.stack([nx, ny, z], axis=-1).astype(np.float32)


def _unit_and_mag(vec_3d: np.ndarray, eps: float = 1e-6):
    mag = float(np.linalg.norm(vec_3d))
    if mag > eps:
        return (vec_3d / mag).astype(np.float32), mag
    return np.zeros(3, dtype=np.float32), 0.0


class DualAttnObsBuilder:
    """构建新双注意力模型所需的完整单车/批量观测字典:
      - trunk:      (B, 57)
      - car_tokens: (B, N_other, 24)
      - car_mask:   (B, N_other)      1.0 表示被 Mask 遮蔽 (如无车槽位或蒸馏期 50% 队友遮蔽), 0.0 表示有效可见
      - pad_tokens: (B, 34, 7)        7 维 Boost Pad 特征 [dir_x, dir_y, dir_z, dist_norm, is_active, cooldown_norm, is_big]
    """

    @classmethod
    def build_car_combat_obs(
        cls,
        car_idx: int,
        cars: list,
        car_teams: np.ndarray,
        arena,
        pads: list,
        prev_action_vec: np.ndarray = None,
        roles_info: list = None,
        mask_teammates: bool = False,
        max_other_cars: int = DEFAULT_MAX_OTHER_CARS,
    ) -> dict:
        n_cars = len(cars)
        car = cars[car_idx]
        team = int(car_teams[car_idx]) if car_idx < len(car_teams) else 0
        team_sign = -1.0 if team == 1 else 1.0  # 橙队翻转 X, Y，使本方永远向 +Y 进攻

        cs = car.get_state()
        bs = arena.ball.get_state()

        # 1. 阵营对齐的世界坐标 (蓝队=1, 橙队反转 X,Y)
        inv3 = np.array([team_sign, team_sign, 1.0], dtype=np.float32)
        c_pos = np.array([float(cs.pos.x), float(cs.pos.y), float(cs.pos.z)], dtype=np.float32) * inv3
        c_vel = np.array([float(cs.vel.x), float(cs.vel.y), float(cs.vel.z)], dtype=np.float32) * inv3
        c_ang = np.array([float(cs.ang_vel.x), float(cs.ang_vel.y), float(cs.ang_vel.z)], dtype=np.float32) * inv3

        fw_raw = car.get_forward_dir()
        rg_raw = car.get_right_dir()
        up_raw = car.get_up_dir()
        fwd   = np.array([float(fw_raw.x), float(fw_raw.y), float(fw_raw.z)], dtype=np.float32) * inv3
        right = np.array([float(rg_raw.x), float(rg_raw.y), float(rg_raw.z)], dtype=np.float32) * inv3
        up    = np.array([float(up_raw.x), float(up_raw.y), float(up_raw.z)], dtype=np.float32) * inv3

        b_pos = np.array([float(bs.pos.x), float(bs.pos.y), float(bs.pos.z)], dtype=np.float32) * inv3
        b_vel = np.array([float(bs.vel.x), float(bs.vel.y), float(bs.vel.z)], dtype=np.float32) * inv3

        # 2. 车头水平偏航旋转角 (Nexto 同款重力参考系: 原点在自身，+Y 对齐车头水平朝向，+Z 恒定竖直朝上对抗重力)
        horiz_norm = math.hypot(float(fwd[0]), float(fwd[1]))
        if horiz_norm > 1e-5:
            # theta = atan2(fwd_x, fwd_y) -> sin_t = fwd_x / h, cos_t = fwd_y / h
            sin_t = float(fwd[0]) / horiz_norm
            cos_t = float(fwd[1]) / horiz_norm
        else:
            # 当车体完全垂直朝天/朝地时，用车顶向量的水平分量平滑退化，避免 atan2 奇点跳变
            up_h = math.hypot(float(up[0]), float(up[1])) + 1e-6
            sin_t = -float(up[0]) / up_h
            cos_t = -float(up[1]) / up_h

        # 3. 固定主干特征 (Trunk, 57 维)
        # A. 自身距六个球场平面的世界轴向距离 (阵营对称归一化)
        faces = np.array([
            c_pos[0] + FIELD_X,
            FIELD_X - c_pos[0],
            c_pos[1] + FIELD_Y,
            FIELD_Y - c_pos[1],
            c_pos[2],
            FIELD_Z - c_pos[2],
        ], dtype=np.float32) / S_FACE

        # B. 自身局部坐标系下的三轴角速度
        car_ang_b = np.array([
            float(np.dot(c_ang, right)),
            float(np.dot(c_ang, up)),
            float(np.dot(c_ang, fwd)),
        ], dtype=np.float32) / S_CAR_ANG

        # C. 自身速度在自身局部坐标系下的【单位矢量 (3) + 标量 (1)】
        v_b = np.array([
            float(np.dot(c_vel, fwd)),
            float(np.dot(c_vel, right)),
            float(np.dot(c_vel, up)),
        ], dtype=np.float32)
        v_unit_b, v_mag = _unit_and_mag(v_b)
        car_vel_b_4d = np.array([
            v_unit_b[0], v_unit_b[1], v_unit_b[2], v_mag / S_CAR_VEL
        ], dtype=np.float32)

        # D. 球在自身局部坐标系下的单位矢量 (3) + 相对位置 (3) + 距离标量 (1)
        rel_b_world = b_pos - c_pos
        b_local = np.array([
            float(np.dot(rel_b_world, fwd)),
            float(np.dot(rel_b_world, right)),
            float(np.dot(rel_b_world, up)),
        ], dtype=np.float32)
        b_dir_local, b_dist = _unit_and_mag(b_local)
        ball_rel_b_7d = np.array([
            b_dir_local[0], b_dir_local[1], b_dir_local[2],
            b_local[0] / S_BALL_POS, b_local[1] / S_BALL_POS, b_local[2] / S_BALL_POS,
            b_dist / S_BALL_POS,
        ], dtype=np.float32)

        # E & F & G & H & I
        inv_dist = 1.0 / max(b_dist, 1e-6)
        closing_spd = float(np.dot(rel_b_world, c_vel)) * inv_dist
        hf = cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump
        flags_6d = np.array([
            1.0 if bool(cs.is_on_ground) else 0.0,
            1.0 if bool(hf) else 0.0,
            1.0 if bool(cs.is_flipping) else 0.0,
            1.0 if bool(cs.is_jumping) else 0.0,
            min(float(cs.air_time), 2.0) / 2.0,
            float(cs.boost) / 100.0,
        ], dtype=np.float32)
        prev_act_6d = format_prev_action_vec(prev_action_vec)
        car_orient_z = np.array([fwd[2], right[2], up[2]], dtype=np.float32)
        ball_faces = np.array([
            b_pos[0] + FIELD_X,
            FIELD_X - b_pos[0],
            b_pos[1] + FIELD_Y,
            FIELD_Y - b_pos[1],
            b_pos[2],
            FIELD_Z - b_pos[2],
        ], dtype=np.float32) / S_FACE

        # K. 对抗重力的 Z 轴朝上参考系特征 (11 维):
        #    自身速度在重力系下的单位方向(3) + 球在重力系下的相对位置(3) + 球在重力系下的相对速度(3) + 俯仰/横滚姿态(2)
        v_up_ref = _rot_xy_to_car_yaw(c_vel, cos_t, sin_t)
        v_up_dir, _ = _unit_and_mag(v_up_ref)
        b_pos_up_ref = _rot_xy_to_car_yaw(rel_b_world, cos_t, sin_t) / S_BALL_POS
        b_vel_up_ref = _rot_xy_to_car_yaw(b_vel - c_vel, cos_t, sin_t) / S_CAR_VEL
        up_ref_11d = np.concatenate([
            v_up_dir,                                        # (3)
            b_pos_up_ref,                                    # (3)
            b_vel_up_ref,                                    # (3)
            np.array([fwd[2], right[2]], dtype=np.float32),  # (2)
        ]).astype(np.float32)

        # L. 四角色 One-Hot 标签 (4 维)
        if roles_info is not None and car_idx < len(roles_info) and roles_info[car_idx] is not None:
            role_oh = np.asarray(roles_info[car_idx]["role_onehot"], dtype=np.float32)
        else:
            role_oh = ROLE_ONE_HOTS[ROLE_CHASER].copy()

        trunk = np.concatenate([
            faces,                         # 0..5   (6)
            car_ang_b,                     # 6..8   (3)
            car_vel_b_4d,                  # 9..12  (4)
            ball_rel_b_7d,                 # 13..19 (7)
            flags_6d,                      # 20..25 (6)
            [closing_spd / S_CLOSING],     # 26     (1)
            prev_act_6d,                   # 27..32 (6)
            car_orient_z,                  # 33..35 (3)
            ball_faces,                    # 36..41 (6)
            up_ref_11d,                    # 42..52 (11)
            role_oh,                       # 53..56 (4)
        ]).astype(np.float32)

        # 4. 可变长其他车辆 Token 序列 (按先队友、后对手排序，支持任意数量，填充至 max(n_other, max_other_cars))
        mate_indices = [j for j in range(n_cars) if j != car_idx and int(car_teams[j]) == team]
        opp_indices  = [j for j in range(n_cars) if j != car_idx and int(car_teams[j]) != team]
        ordered_others = mate_indices + opp_indices
        n_slots = max(1, max(len(ordered_others), int(max_other_cars)))

        car_tokens = np.zeros((n_slots, CAR_TOKEN_DIM), dtype=np.float32)
        car_mask = np.ones(n_slots, dtype=np.float32)  # 1.0 = masked out, 0.0 = valid

        for slot_k, other_idx in enumerate(ordered_others):
            is_mate = (int(car_teams[other_idx]) == team)
            if is_mate and mask_teammates:
                # 蒸馏期 50% 概率触发的信息瓶颈屏蔽: 遮蔽队友 Token，逼迫模型依赖 L.role_onehot
                car_mask[slot_k] = 1.0
                continue

            car_mask[slot_k] = 0.0
            oc = cars[other_idx]
            ocs = oc.get_state()
            ofw_raw = oc.get_forward_dir()

            o_pos = np.array([float(ocs.pos.x), float(ocs.pos.y), float(ocs.pos.z)], dtype=np.float32) * inv3
            o_vel = np.array([float(ocs.vel.x), float(ocs.vel.y), float(ocs.vel.z)], dtype=np.float32) * inv3
            o_ang = np.array([float(ocs.ang_vel.x), float(ocs.ang_vel.y), float(ocs.ang_vel.z)], dtype=np.float32) * inv3
            o_fwd = np.array([float(ofw_raw.x), float(ofw_raw.y), float(ofw_raw.z)], dtype=np.float32) * inv3

            # 转换到自身原点 + 自身车头水平朝向的重力参考系
            rel_pos_up = _rot_xy_to_car_yaw(o_pos - c_pos, cos_t, sin_t)
            rel_dir_up, rel_dist = _unit_and_mag(rel_pos_up)

            o_vel_up = _rot_xy_to_car_yaw(o_vel, cos_t, sin_t)
            o_vel_dir, o_spd = _unit_and_mag(o_vel_up)

            o_ang_up = _rot_xy_to_car_yaw(o_ang, cos_t, sin_t) / S_CAR_ANG
            o_fwd_up = _rot_xy_to_car_yaw(o_fwd, cos_t, sin_t)

            ohf = ocs.has_flip_or_jump() if callable(ocs.has_flip_or_jump) else ocs.has_flip_or_jump
            o_is_chaser = 0.0
            if roles_info is not None and other_idx < len(roles_info) and roles_info[other_idx] is not None:
                o_is_chaser = 1.0 if roles_info[other_idx]["role_id"] == ROLE_CHASER else 0.0

            car_tokens[slot_k, 0]     = 1.0 if is_mate else 0.0
            car_tokens[slot_k, 1]     = 0.0 if is_mate else 1.0
            car_tokens[slot_k, 2:5]   = rel_dir_up
            car_tokens[slot_k, 5:8]   = rel_pos_up / S_FACE
            car_tokens[slot_k, 8]     = rel_dist / S_FACE
            car_tokens[slot_k, 9:12]  = o_vel_dir
            car_tokens[slot_k, 12]    = o_spd / S_CAR_VEL
            car_tokens[slot_k, 13:16] = o_ang_up
            car_tokens[slot_k, 16:19] = o_fwd_up
            car_tokens[slot_k, 19]    = float(ocs.boost) / 100.0
            car_tokens[slot_k, 20]    = 1.0 if bool(ocs.is_on_ground) else 0.0
            car_tokens[slot_k, 21]    = 1.0 if bool(ohf) else 0.0
            car_tokens[slot_k, 22]    = 1.0 if bool(ocs.is_demoed) else 0.0
            car_tokens[slot_k, 23]    = o_is_chaser

        # 5. 34 个 Boost Pad 实体特征 (每个 7 维: [dir_x, dir_y, dir_z, dist_norm, is_active, cooldown_norm, is_big])
        #    对于橙队 (team==1)，氮气垫世界坐标同样乘 inv3，且顺序按对称槽位排列
        pads_world = BOOST_PAD_LOCATIONS * inv3[None, :]
        rel_pads_world = pads_world - c_pos[None, :]
        rel_pads_up = _rot_xy_to_car_yaw(rel_pads_world, cos_t, sin_t)  # (34, 3)
        pad_dists = np.linalg.norm(rel_pads_up, axis=-1, keepdims=True)  # (34, 1)
        pad_dirs = rel_pads_up / np.maximum(pad_dists, 1e-6)             # (34, 3)
        pad_dists_norm = (pad_dists / S_FACE).astype(np.float32)         # (34, 1)

        pad_active = np.empty((NUM_BOOST_PADS, 1), dtype=np.float32)
        pad_cd = np.empty((NUM_BOOST_PADS, 1), dtype=np.float32)
        for k_pad, p_obj in enumerate(pads):
            ps = p_obj.get_state()
            pad_active[k_pad, 0] = 1.0 if ps.is_active else 0.0
            pad_cd[k_pad, 0] = min(1.0, max(0.0, float(ps.cooldown) / float(PAD_COOLDOWN_SCALES[k_pad])))

        pad_is_big = PAD_IS_BIG_MASK.reshape(NUM_BOOST_PADS, 1)
        pad_tokens = np.concatenate([
            pad_dirs,        # [0..2] (3) 绝对重力系下的方向单位矢量
            pad_dists_norm,  # [3]    (1) 绝对距离 / 5120
            pad_active,      # [4]    (1) enable 状态 (bool: able to collect)
            pad_cd,          # [5]    (1) 冷却倒计时 down counter (cooldown_norm)
            pad_is_big,      # [6]    (1) 是否为大氮气垫 is_big
        ], axis=-1).astype(np.float32)

        return {
            "trunk": trunk,
            "car_tokens": car_tokens,
            "car_mask": car_mask,
            "pad_tokens": pad_tokens,
        }


if _TORCH_AVAILABLE:
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    class DistanceDecayAttention(nn.Module):
        """带可学习距离反比例偏置 (Inverse-Distance Bias) 的多头交叉注意力模块:
        解决初始化时远距离坐标内积膨胀导致注意力分散的问题，使注意力天然聚焦于附近的样本 (车辆或氮气垫)。
        支持动态序列长度 N 与掩码 mask (1.0 表示屏蔽，若全被屏蔽则优雅回退为 0 向量)。
        """
        def __init__(
            self,
            q_dim: int,
            kv_dim: int,
            dist_idx: int,
            num_heads: int = 2,
            head_dim: int = 16,
            init_dist_scale: float = 1.8,
        ):
            super().__init__()
            self.q_dim = int(q_dim)
            self.kv_dim = int(kv_dim)
            self.dist_idx = int(dist_idx)
            self.num_heads = int(num_heads)
            self.head_dim = int(head_dim)
            self.out_dim = self.num_heads * self.head_dim
            self.scale = 1.0 / math.sqrt(float(self.head_dim))

            self.q_proj = _layer_init(nn.Linear(self.q_dim, self.out_dim), std=1.0)
            self.k_proj = _layer_init(nn.Linear(self.kv_dim, self.out_dim), std=1.0)
            self.v_proj = _layer_init(nn.Linear(self.kv_dim, self.out_dim), std=1.0)
            self.out_proj = _layer_init(nn.Linear(self.out_dim, self.out_dim), std=1.0)

            # 每个注意力头独立的可学习距离衰减系数 alpha_h > 0
            # 反比例核函数: inv_prox = 1.0 / (1.0 + dist_norm * 5.0) ∈ (0, 1]
            self.dist_alpha = nn.Parameter(torch.full((1, self.num_heads, 1, 1), float(init_dist_scale)))
            self.last_attn_weights = None

        def forward(self, trunk_q: torch.Tensor, kv_tokens: torch.Tensor, mask: torch.Tensor = None):
            B, N, _ = kv_tokens.shape
            q = self.q_proj(trunk_q).view(B, self.num_heads, 1, self.head_dim)
            k = self.k_proj(kv_tokens).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = self.v_proj(kv_tokens).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, 1, N)

            # 注入距离反比例偏置: 距离越近 (dist_norm -> 0)，1/(1 + 5*d) 越接近 1，加分越高；距离越远权重越小
            dist_norm = kv_tokens[:, :, self.dist_idx].clamp(min=0.0).view(B, 1, 1, N)
            inv_dist_bonus = 1.0 / (1.0 + 5.0 * dist_norm)
            alpha_pos = F.softplus(self.dist_alpha)
            scores = scores + alpha_pos * inv_dist_bonus

            if mask is not None:
                m_bool = (mask > 0.5).view(B, 1, 1, N)
                scores = scores.masked_fill(m_bool, -1e9)
                all_masked = m_bool.all(dim=-1, keepdim=True)  # (B, 1, 1, 1)
            else:
                all_masked = None

            attn_w = F.softmax(scores, dim=-1)  # (B, H, 1, N)
            if all_masked is not None:
                attn_w = attn_w.masked_fill(all_masked, 0.0)

            self.last_attn_weights = attn_w.squeeze(-2)  # (B, H, N)
            ctx = torch.matmul(attn_w, v).squeeze(-2).reshape(B, self.out_dim)
            embed = torch.tanh(self.out_proj(ctx))
            return embed, self.last_attn_weights

    class DualAttentionCombatNet(nn.Module):
        """全新多车实战受控策略网络:
        - 主干 (57 维: 含车体 3D 局部系 + Z 轴朝上重力系 + 4 维角色 One-Hot)
        - CarEntityAttention (支持任意数量其他车辆 + 距离反比例偏置 + 队友 Mask)
        - BoostPadAttention  (34 个 7 维氮气垫 + 距离反比例偏置)
        - 两级条件自回归动作头 (Autoregressive Action Head):
          先输出二值动作开关 (Jump, Boost, Handbrake)，再将采样出的开关状态条件注入摇杆方向头 (Throttle/Pitch, Steer/Yaw, Roll)，
          彻底消除翻滚 (Dodge/Stall) 时方向通道与跳跃键的独立边缘化脱节问题！
        """
        def __init__(
            self,
            trunk_dim: int = COMBAT_TRUNK_DIM,
            car_token_dim: int = CAR_TOKEN_DIM,
            pad_token_dim: int = BOOST_PAD_TOKEN_DIM,
            hidden_dim: int = PPO_HIDDEN_DIM,
            action_dims: list = PPO_ACTION_DIMS,
        ):
            super().__init__()
            self.trunk_dim = int(trunk_dim)
            self.car_token_dim = int(car_token_dim)
            self.pad_token_dim = int(pad_token_dim)
            self.hidden_dim = int(hidden_dim)
            self.action_dims = list(action_dims)

            # 1) 其他车辆注意力 (dist_norm 在索引 8)
            self.car_attn = DistanceDecayAttention(
                q_dim=self.trunk_dim,
                kv_dim=self.car_token_dim,
                dist_idx=8,
                num_heads=2,
                head_dim=16,
                init_dist_scale=1.5,
            )
            # 2) 34 氮气垫注意力 (dist_norm 在索引 3)
            self.pad_attn = DistanceDecayAttention(
                q_dim=self.trunk_dim,
                kv_dim=self.pad_token_dim,
                dist_idx=3,
                num_heads=2,
                head_dim=16,
                init_dist_scale=2.2,
            )

            combined_dim = self.trunk_dim + self.car_attn.out_dim + self.pad_attn.out_dim  # 57 + 32 + 32 = 121
            self.shared_encoder = nn.Sequential(
                _layer_init(nn.Linear(combined_dim, self.hidden_dim)),
                nn.Tanh(),
                _layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.Tanh(),
            )

            # 第一级动作头: 先决策二值开关通道 ch3(Jump:2), ch4(Boost:2), ch5(Slide:2) -> 共 6 logits
            self.bin_head = _layer_init(nn.Linear(self.hidden_dim, 6), std=0.01)
            # 第二级条件动作头: 接收 [hidden_feat (256) + 确定的 (jump, boost, slide) (3)] -> 输出方向轴 ch0(3), ch1(3), ch2(3) -> 共 9 logits
            self.dir_cond_head = _layer_init(nn.Linear(self.hidden_dim + 3, 9), std=0.01)

            # Critic 价值头
            self.critic_mlp = nn.Sequential(
                _layer_init(nn.Linear(combined_dim, self.hidden_dim)),
                nn.Tanh(),
                _layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                nn.Tanh(),
                _layer_init(nn.Linear(self.hidden_dim, 1), std=1.0),
            )

        def encode(self, trunk: torch.Tensor, car_tokens: torch.Tensor, pad_tokens: torch.Tensor, car_mask: torch.Tensor = None):
            car_ctx, car_w = self.car_attn(trunk, car_tokens, mask=car_mask)
            pad_ctx, pad_w = self.pad_attn(trunk, pad_tokens, mask=None)
            combined = torch.cat([trunk, car_ctx, pad_ctx], dim=-1)
            h = self.shared_encoder(combined)
            return h, combined, car_w, pad_w

        def forward(
            self,
            trunk: torch.Tensor,
            car_tokens: torch.Tensor,
            pad_tokens: torch.Tensor,
            car_mask: torch.Tensor = None,
            cond_bin_actions: torch.Tensor = None,
        ):
            """前向计算 15 维动作 Logits [ch0(3), ch1(3), ch2(3), ch3_jump(2), ch4_boost(2), ch5_slide(2)] 与 Value。
            若未提供 cond_bin_actions，则使用 bin_head 的预测概率期望值作为软条件输入。
            """
            h, combined, _, _ = self.encode(trunk, car_tokens, pad_tokens, car_mask=car_mask)
            bin_logits = self.bin_head(h)  # (B, 6) -> [jump(2), boost(2), slide(2)]

            if cond_bin_actions is not None:
                # Teacher-Forcing 或采样后的精确条件: (B, 3) ∈ {0.0, 1.0}
                bin_cond = cond_bin_actions.float()
            else:
                p_jump  = F.softmax(bin_logits[:, 0:2], dim=-1)[:, 1:2]
                p_boost = F.softmax(bin_logits[:, 2:4], dim=-1)[:, 1:2]
                p_slide = F.softmax(bin_logits[:, 4:6], dim=-1)[:, 1:2]
                bin_cond = torch.cat([p_jump, p_boost, p_slide], dim=-1)

            dir_logits = self.dir_cond_head(torch.cat([h, bin_cond], dim=-1))  # (B, 9)
            full_logits = torch.cat([dir_logits, bin_logits], dim=-1)          # (B, 15)
            value = self.critic_mlp(combined).squeeze(-1)
            return full_logits, value
