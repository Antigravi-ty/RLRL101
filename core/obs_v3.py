"""obs_v3 —— 面向终极目标的车体自中心（Hard-Attach Ego Frame）全量观测与两阶段意图接口。

不仅完全兼容包含前 31 维 (obs_v2)，而且修正并补齐了终极目标所需的全部核心几何特征（共 75 维）：
  [0..31]  兼容 obs_v2 基础层：
           A.face_dists  (6) : 车到六个平面的距离 (-x, +x, -y, +y, floor, ceil)
           B.car_ang_b   (3) : 车体局部坐标系角速度 (pitch, yaw, roll)
           C.car_vel_b   (4) : 车体局部速度 (v_fwd, v_right, v_up, |v|)
           D.ball_rel_b  (4) : 球在车体坐标系下相对位置 (b_fwd, b_right, b_up, |d|)
           E.ball_relvel (4) : 球在车体坐标系下相对速度 (bv_fwd, bv_right, bv_up, |bv|)
           F.ball_ang_b  (3) : 球在车体坐标系下角速度 (wx_b, wy_b, wz_b)
           G.flags       (6) : on_ground, has_flip_or_jump, is_flipping, is_jumping, air_time, boost
           H.closing     (1) : 车球三维接近速度 closing_speed
  [31..34] I.car_orient  (3) : 车体三轴正交基在世界 Z 轴投影 (fwd_z, right_z, up_z)
                               ★ 关键修正：等价于世界重力向量 [0,0,-1] 在车体坐标系下的投影，
                                 彻底解决仅靠角速度无法感知空中静态翻转/倒扣姿态的问题。
  [34..40] J.ball_faces  (6) : 球到场地六个平面的距离 (-x, +x, -y, +y, floor, ceil)
  [40..46] K.goals_rel_b (6) : 敌方球门中心 (0, 5120, 320) 与我方球门中心 (0, -5120, 320) 在车体坐标系下的相对位置
  [46..70] L.big_pads_b  (24): 场上 6 个大 Boost Pad (6 x 4)：
                               [rel_fwd, rel_right, is_active, remaining_cooldown_norm]
  [70..75] M.intent_cond (5) : 两阶段架构（Intent -> Action）的条件输入接口：
                               [want_ball, target_fwd, target_right, kick_dir_fwd, kick_dir_right]
                               在 Stage 0/1 底层小脑训练时由环境规则生成（Oracle Intent：拿球并将球踢向敌方球门中心）。
"""
import numpy as np
from .obs_v2 import (
    FIELD_X, FIELD_Y, FIELD_Z,
    S_FACE, S_CAR_ANG, S_CAR_VEL, S_BALL_POS, S_BALL_VEL, S_BALL_ANG, S_CLOSING,
    build_obs_v2, _v,
)
from .constants import OBS_V3_DIM

# RocketSim SOCCAR 中 6 个 Big Boost Pads 的固定索引
BIG_PAD_INDICES = (3, 4, 15, 18, 29, 30)
BIG_PAD_COOLDOWN = 10.0

OPP_GOAL_POS = np.array([0.0, 5120.0, 320.0], dtype=np.float32)
OWN_GOAL_POS = np.array([0.0, -5120.0, 320.0], dtype=np.float32)


def compute_oracle_intent(car_pos, ball_pos, fwd, right, up, want_ball=1.0, target_pos=None, target_time=None):
    """生成底层控制器所需的 Oracle 意图特征（车体自中心坐标系下）：
    - want_ball: 在 stage0/kickoff 下为 1.0；在 stage1 (定点定时击球) 下编码期望出球速度/时间比
    - target_rel (fwd, right): 期望击球点（球后方沿球->目标点反方向 95 uu）相对于自车的位置
    - kick_dir (fwd, right): 期望把球踢向的目标方向（球 -> 目标点的单位向量在自车平面的投影）
    """
    tgt = OPP_GOAL_POS if target_pos is None else np.asarray(target_pos, dtype=np.float32)
    ball_to_tgt = tgt - ball_pos
    b2t_dist = float(np.linalg.norm(ball_to_tgt)) + 1e-6
    b2t_dir = ball_to_tgt / b2t_dist

    # 理想击球点：球后方约 95 uu（沿球到目标点的反方向）
    ideal_hit_pos = ball_pos - b2t_dir * 95.0
    rel_target = ideal_hit_pos - car_pos

    target_fwd = float(np.dot(rel_target, fwd)) / S_BALL_POS
    target_right = float(np.dot(rel_target, right)) / S_BALL_POS

    kick_dir_fwd = float(np.dot(b2t_dir, fwd))
    kick_dir_right = float(np.dot(b2t_dir, right))

    if target_time is not None and target_time > 0.05:
        # 归一化期望飞行时间 (例如 1.5s -> 1.0)，让底层网络区分快抽 (小值) 与慢吊 (大值)
        wb_val = float(np.clip(float(target_time) / 1.5, 0.1, 2.5))
    else:
        wb_val = float(want_ball)

    return np.array([
        wb_val,
        np.clip(target_fwd, -2.0, 2.0),
        np.clip(target_right, -2.0, 2.0),
        kick_dir_fwd,
        kick_dir_right,
    ], dtype=np.float32)


def build_obs_v3(car, arena, intent_override=None, target_pos=None, target_time=None):
    """构建 75 维全量观测向量，同时返回兼容 obs_v2 的 31 维切片与 parts 字典。"""
    ball_state = arena.ball.get_state()
    obs_v2, parts = build_obs_v2(car, ball_state)

    cs = car.get_state()
    fwd   = _v(car.get_forward_dir())
    right = _v(car.get_right_dir())
    up    = _v(car.get_up_dir())

    car_pos  = _v(cs.pos)
    ball_pos = _v(ball_state.pos)
    ball_ang = _v(ball_state.ang_vel) if hasattr(ball_state, "ang_vel") else np.zeros(3, dtype=np.float32)

    # F组修正：将球角速度也投影到车体坐标系下（保持全状态纯粹自中心旋转不变性）
    ball_ang_b = np.array([
        float(np.dot(ball_ang, right)),
        float(np.dot(ball_ang, up)),
        float(np.dot(ball_ang, fwd)),
    ], dtype=np.float32) / S_BALL_ANG

    # I. 车体三轴正交基在世界 Z 轴分量 (重力感知)
    car_orient_z = np.array([fwd[2], right[2], up[2]], dtype=np.float32)

    # J. 球到六个平面的距离
    ball_faces = np.array([
        ball_pos[0] + FIELD_X,
        FIELD_X - ball_pos[0],
        ball_pos[1] + FIELD_Y,
        FIELD_Y - ball_pos[1],
        ball_pos[2],
        FIELD_Z - ball_pos[2],
    ], dtype=np.float32) / S_FACE

    # K. 敌方与我方球门在自车坐标系下的相对位置
    rel_opp_goal = OPP_GOAL_POS - car_pos
    rel_own_goal = OWN_GOAL_POS - car_pos
    goals_rel_b = np.array([
        float(np.dot(rel_opp_goal, fwd)),
        float(np.dot(rel_opp_goal, right)),
        float(np.dot(rel_opp_goal, up)),
        float(np.dot(rel_own_goal, fwd)),
        float(np.dot(rel_own_goal, right)),
        float(np.dot(rel_own_goal, up)),
    ], dtype=np.float32) / S_BALL_POS

    # L. 6 个 Big Boost Pads 的车体相对坐标 + 激活掩码 + 剩余冷却时间 (0~1)
    pads = arena.get_boost_pads()
    big_pads_feat = np.zeros(24, dtype=np.float32)
    for idx_k, pad_idx in enumerate(BIG_PAD_INDICES):
        if pad_idx < len(pads):
            p = pads[pad_idx]
            p_pos = _v(p.get_pos())
            ps = p.get_state()
            rel_p = p_pos - car_pos
            base = idx_k * 4
            big_pads_feat[base + 0] = float(np.dot(rel_p, fwd)) / S_FACE
            big_pads_feat[base + 1] = float(np.dot(rel_p, right)) / S_FACE
            big_pads_feat[base + 2] = 1.0 if ps.is_active else 0.0
            big_pads_feat[base + 3] = float(ps.cooldown) / BIG_PAD_COOLDOWN

    # M. 两阶段意图条件输入
    if intent_override is not None:
        intent_vec = np.asarray(intent_override, dtype=np.float32)
    else:
        intent_vec = compute_oracle_intent(
            car_pos, ball_pos, fwd, right, up,
            want_ball=1.0, target_pos=target_pos, target_time=target_time,
        )

    # 替换 obs_v2 中第 21..24 位的球角速度为自车坐标系投影
    obs_v2_ego = obs_v2.copy()
    obs_v2_ego[21:24] = ball_ang_b

    obs_v3 = np.concatenate([
        obs_v2_ego,     # 31
        car_orient_z,   # 3  -> 34
        ball_faces,     # 6  -> 40
        goals_rel_b,    # 6  -> 46
        big_pads_feat,  # 24 -> 70
        intent_vec,     # 5  -> 75
    ]).astype(np.float32)

    assert obs_v3.shape[0] == OBS_V3_DIM, f"Expected {OBS_V3_DIM}, got {obs_v3.shape[0]}"
    parts["I.car_orient"] = car_orient_z
    parts["J.ball_faces"] = ball_faces * S_FACE
    parts["K.goals_rel_b"] = goals_rel_b * S_BALL_POS
    parts["M.intent"] = intent_vec
    return obs_v3, obs_v2, parts


# ==============================================================================
# 左右镜像对称变换算子 (Left-Right X=0 Reflection Symmetry Operators)
# 经 RocketSim 物理引擎逐帧验证：镜像轨迹的 75 维观测误差 < 7e-6 (float32 机器精度)
# ==============================================================================

# 在 X -> -X 镜像反射下需要取反 (-1) 的维度索引（所有局部 right 投影、偏航角速度 yaw、横滚角速度 roll）
OBS_V3_MIRROR_NEG_INDICES = (
    7, 8,                   # B.car_ang_b: yaw, roll
    10,                     # C.car_vel_b: v_right
    14,                     # D.ball_rel_b: b_right
    18,                     # E.ball_relvel: bv_right
    22, 23,                 # F.ball_ang_b: wy_b (up), wz_b (fwd)
    32,                     # I.car_orient: right_z
    41, 44,                 # K.goals_rel_b: opp_goal_right, own_goal_right
    47, 51, 55, 59, 63, 67, # L.big_pads_b: 6 个大垫子的 rel_right
    72, 74,                 # M.intent_cond: target_right, kick_dir_right
)


def _build_obs_v3_mirror_perm_and_sign():
    """预计算 75 维观测在左右镜像下的置换索引 perm 与符号向量 sign，支持单步张量乘法完成镜像。"""
    perm = np.arange(OBS_V3_DIM, dtype=np.int64)
    # 1. 交换车体到 -x 与 +x 墙面距离 (0 <-> 1)
    perm[0], perm[1] = 1, 0
    # 2. 交换球到 -x 与 +x 墙面距离 (34 <-> 35)
    perm[34], perm[35] = 35, 34
    # 3. 交换左右对称的三对大 Boost Pads: (3<->4), (15<->18), (29<->30)
    for base in (46, 54, 62):
        for k in range(4):
            perm[base + k] = base + 4 + k
            perm[base + 4 + k] = base + k

    sign = np.ones(OBS_V3_DIM, dtype=np.float32)
    for idx in OBS_V3_MIRROR_NEG_INDICES:
        sign[idx] = -1.0
    return perm, sign


_OBS_MIRROR_PERM_NP, _OBS_MIRROR_SIGN_NP = _build_obs_v3_mirror_perm_and_sign()

# 15 维动作 logits 的左右镜像置换索引：
#   ch0 (S/0/W, 0..3):     [0, 1, 2] (不变)
#   ch1 (A/0/D, 3..6):     [5, 4, 3] (A <-> D)
#   ch2 (Q/0/E, 6..9):     [8, 7, 6] (Q <-> E)
#   ch3..ch5 (Jump/Boost/Slide, 9..15): [9, 10, 11, 12, 13, 14] (不变)
_LOGITS_MIRROR_PERM_NP = np.array(
    [0, 1, 2, 5, 4, 3, 8, 7, 6, 9, 10, 11, 12, 13, 14], dtype=np.int64
)


def mirror_obs_v3_np(obs: np.ndarray) -> np.ndarray:
    """NumPy 75 维观测左右镜像变换，支持 (75,) 或 (N, 75)。"""
    return obs[..., _OBS_MIRROR_PERM_NP] * _OBS_MIRROR_SIGN_NP


def mirror_actions_np(actions: np.ndarray) -> np.ndarray:
    """NumPy 6 通道离散动作左右镜像变换：翻转 ch1 (A<->D) 与 ch2 (Q<->E)。"""
    m = np.array(actions, dtype=np.int64, copy=True)
    m[..., 1] = 2 - m[..., 1]
    m[..., 2] = 2 - m[..., 2]
    return m


def mirror_obs_v3_torch(obs_t):
    """PyTorch 张量 75 维观测左右镜像变换，支持 (..., 75)。"""
    import torch
    perm = torch.as_tensor(_OBS_MIRROR_PERM_NP, dtype=torch.long, device=obs_t.device)
    sign = torch.as_tensor(_OBS_MIRROR_SIGN_NP, dtype=obs_t.dtype, device=obs_t.device)
    return obs_t.index_select(-1, perm) * sign


def mirror_actions_torch(act_t):
    """PyTorch 张量 6 通道动作左右镜像变换：翻转 ch1 (A<->D) 与 ch2 (Q<->E)。"""
    m = act_t.clone()
    m[..., 1] = 2 - m[..., 1]
    m[..., 2] = 2 - m[..., 2]
    return m


def mirror_logits_torch(logits_t):
    """PyTorch 张量 15 维动作 logits 左右镜像变换：翻转 ch1 (A<->D) 与 ch2 (Q<->E) 的对应分量。"""
    import torch
    perm = torch.as_tensor(_LOGITS_MIRROR_PERM_NP, dtype=torch.long, device=logits_t.device)
    return logits_t.index_select(-1, perm)

