"""core/obs.py —— 1v1 最快开球 42 维极简对称观测空间 (无冗余 BoostPad Token，纯向量极速推理)。

观测特征结构 (共 42 维, 索引 0..41):
  --- 面板可视化特征 (OBS_PANEL_DIM = 33, 索引 0..32) ---
  A.face_dists   [0..5]   (6)  车体到左/右/后/前墙壁与地面/天花板的距离 (已做蓝橙队伍对称对齐)
  B.car_ang_b    [6..8]   (3)  车体局部角速度 (pitch, yaw, roll)
  C.car_vel_b    [9..12]  (4)  车体局部速度 (v_fwd, v_right, v_up, |v|)
  D.ball_rel_b   [13..19] (7)  球相对方向单位向量 (dir_fwd, dir_right, dir_up) + 局部相对坐标 (b_fwd, b_right, b_up, |d|)
  E.flags        [20..25] (6)  on_ground, flip_or_jump, is_flipping, is_jumping, air_time, boost
  F.closing      [26..26] (1)  车体朝向球的三维接近速度 closing_spd
  G.prev_act     [27..32] (6)  上一决策步动作 (thr_pitch, steer_yaw, roll, jump, boost, slide)
  --- 扩展主干特征 (索引 33..41) ---
  H.car_orient   [33..35] (3)  车体三轴世界 Z 分量 (fwd_z, right_z, up_z)
  I.opp_rel_b    [36..41] (6)  对手相对位置与抢球态势 (opp_fwd, opp_right, opp_up, opp_dist, opp_vfwd, opp_closing)
"""
import numpy as np
from .constants import OBS_DIM, OBS_PANEL_DIM

FIELD_X = 4096.0
FIELD_Y = 5120.0
FIELD_Z = 2000.0

S_FACE     = 5120.0
S_CAR_ANG  = 5.5
S_CAR_VEL  = 2300.0
S_BALL_POS = 5000.0
S_CLOSING  = 2300.0

OBS_GROUPS = {
    "A.face_dists": (0,  6,  ["-x", "+x", "-y", "+y", "floor", "ceil"]),
    "B.car_ang_b":  (6,  9,  ["pitch", "yaw", "roll"]),
    "C.car_vel_b":  (9,  13, ["v_fwd", "v_right", "v_up", "|v|"]),
    "D.ball_rel_b": (13, 20, ["dir_fwd", "dir_right", "dir_up", "b_fwd", "b_right", "b_up", "|d|"]),
    "E.flags":      (20, 26, ["on_ground", "flip_or_jump", "is_flipping", "is_jumping", "air_time", "boost"]),
    "F.closing":    (26, 27, ["closing_spd"]),
    "G.prev_act":   (27, 33, ["p_thr_ptc", "p_str_yaw", "p_roll", "p_jump", "p_boost", "p_slide"]),
}
GROUP_ORDER = list(OBS_GROUPS.keys())


def _v(x):
    return np.array([x.x, x.y, x.z], dtype=np.float32)


def _hfj(cs):
    h = cs.has_flip_or_jump
    return bool(h() if callable(h) else h)


def format_prev_action_vec(prev_action=None) -> np.ndarray:
    """将 6 通道动作转换为标准化 6 维特征 [-1..+1, -1..+1, -1..+1, 0/1, 0/1, 0/1]。"""
    if prev_action is None:
        return np.zeros(6, dtype=np.float32)
    arr = np.asarray(prev_action, dtype=np.float32)
    if arr.shape[0] != 6:
        return np.zeros(6, dtype=np.float32)
    return np.array([
        float(arr[0]),
        float(arr[1]),
        float(arr[2]),
        float(arr[3]),
        float(arr[4]),
        float(arr[5]),
    ], dtype=np.float32)


def action_indices_to_prev_vec(act_indices) -> np.ndarray:
    """将离散动作索引 [0..2, 0..2, 0..2, 0..1, 0..1, 0..1] 转换为 [-1..+1, -1..+1, -1..+1, 0/1, 0/1, 0/1]。"""
    if act_indices is None:
        return np.zeros(6, dtype=np.float32)
    return np.array([
        float(int(act_indices[0]) - 1),
        float(int(act_indices[1]) - 1),
        float(int(act_indices[2]) - 1),
        float(int(act_indices[3])),
        float(int(act_indices[4])),
        float(int(act_indices[5])),
    ], dtype=np.float32)


def build_obs(car, arena, prev_action_vec=None, opp_car=None, is_orange: bool = None):
    """构建 42 维自我中心观测向量 (蓝橙完全对称)，返回 (obs_42, obs_panel_33, parts_dict)。"""
    cs = car.get_state()
    ball_state = arena.ball.get_state()

    if is_orange is None:
        is_orange = (int( getattr(car, "team", 0) ) == 1)

    fwd   = _v(car.get_forward_dir())
    right = _v(car.get_right_dir())
    up    = _v(car.get_up_dir())

    car_pos = _v(cs.pos)
    car_vel = _v(cs.vel)
    car_ang = _v(cs.ang_vel)

    # Team-aligned coordinates so Orange (at +Y facing -Y) sees identical wall distances to Blue (at -Y facing +Y)
    team_sign = -1.0 if is_orange else 1.0
    tx = team_sign * float(car_pos[0])
    ty = team_sign * float(car_pos[1])
    tz = float(car_pos[2])

    faces = np.array([
        tx + FIELD_X,
        FIELD_X - tx,
        ty + FIELD_Y,
        FIELD_Y - ty,
        tz,
        FIELD_Z - tz,
    ], dtype=np.float32)

    car_ang_b = np.array([
        float(np.dot(car_ang, right)),
        float(np.dot(car_ang, up)),
        float(np.dot(car_ang, fwd)),
    ], dtype=np.float32)

    v_b = np.array([
        float(np.dot(car_vel, fwd)),
        float(np.dot(car_vel, right)),
        float(np.dot(car_vel, up)),
    ], dtype=np.float32)
    v_mag = float(np.linalg.norm(car_vel))

    ball_pos = _v(ball_state.pos)

    rel = ball_pos - car_pos
    b_fwd   = float(np.dot(rel, fwd))
    b_right = float(np.dot(rel, right))
    b_up    = float(np.dot(rel, up))
    ball_dist = float(np.linalg.norm(rel))
    inv_dist = 1.0 / max(ball_dist, 1e-6)
    dir_fwd   = b_fwd * inv_dist
    dir_right = b_right * inv_dist
    dir_up    = b_up * inv_dist

    if ball_dist > 1e-6:
        closing_speed = float(np.dot(rel, car_vel)) * inv_dist
    else:
        closing_speed = 0.0

    on_ground        = 1.0 if bool(cs.is_on_ground) else 0.0
    has_flip_or_jump = 1.0 if _hfj(cs) else 0.0
    is_flipping      = 1.0 if bool(cs.is_flipping) else 0.0
    is_jumping       = 1.0 if bool(cs.is_jumping) else 0.0
    air_time         = float(cs.air_time)
    boost_raw        = float(cs.boost)

    prev_act = format_prev_action_vec(prev_action_vec)
    car_orient_z = np.array([fwd[2], right[2], up[2]], dtype=np.float32)

    # Opponent relative kinematics in car local frame (6 dims)
    if opp_car is not None:
        ocs = opp_car.get_state()
        opp_pos = _v(ocs.pos)
        opp_vel = _v(ocs.vel)
    else:
        # Fallback symmetric mirror shadow if solo
        opp_pos = np.array([-car_pos[0], -car_pos[1], car_pos[2]], dtype=np.float32)
        opp_vel = np.array([-car_vel[0], -car_vel[1], car_vel[2]], dtype=np.float32)

    opp_rel = opp_pos - car_pos
    opp_fwd   = float(np.dot(opp_rel, fwd))
    opp_right = float(np.dot(opp_rel, right))
    opp_up    = float(np.dot(opp_rel, up))
    opp_dist  = float(np.linalg.norm(opp_rel))
    opp_vfwd  = float(np.dot(opp_vel, fwd))
    opp_to_ball = ball_pos - opp_pos
    opp_ball_d = float(np.linalg.norm(opp_to_ball))
    opp_closing = float(np.dot(opp_to_ball, opp_vel)) / max(opp_ball_d, 1e-6) if opp_ball_d > 1e-6 else 0.0

    opp_feat_norm = np.array([
        opp_fwd / S_BALL_POS,
        opp_right / S_BALL_POS,
        opp_up / S_BALL_POS,
        opp_dist / S_BALL_POS,
        opp_vfwd / S_CAR_VEL,
        opp_closing / S_CLOSING,
    ], dtype=np.float32)

    obs_panel = np.concatenate([
        faces / S_FACE,                                                          # 0..5   (6)
        car_ang_b / S_CAR_ANG,                                                   # 6..8   (3)
        v_b / S_CAR_VEL,                                                         # 9..11  (3)
        [v_mag / S_CAR_VEL],                                                     # 12     (1)
        [dir_fwd, dir_right, dir_up],                                            # 13..15 (3)
        [b_fwd / S_BALL_POS, b_right / S_BALL_POS, b_up / S_BALL_POS, ball_dist / S_BALL_POS],  # 16..19 (4)
        [on_ground, has_flip_or_jump, is_flipping, is_jumping,
         min(air_time, 2.0) / 2.0, boost_raw / 100.0],                           # 20..25 (6)
        [closing_speed / S_CLOSING],                                             # 26     (1)
        prev_act,                                                                # 27..32 (6)
    ]).astype(np.float32)

    obs_full = np.concatenate([
        obs_panel,      # 0..32 (33)
        car_orient_z,   # 33..35 (3)
        opp_feat_norm,  # 36..41 (6) -> Total = 42
    ]).astype(np.float32)

    assert obs_full.shape[0] == OBS_DIM, f"Expected {OBS_DIM}, got {obs_full.shape[0]}"

    parts = {
        "A.face_dists": faces,
        "B.car_ang_b":  car_ang_b,
        "C.car_vel_b":  np.concatenate([v_b, [v_mag]]).astype(np.float32),
        "D.ball_rel_b": np.array([dir_fwd, dir_right, dir_up, b_fwd, b_right, b_up, ball_dist], dtype=np.float32),
        "E.flags":      np.array([on_ground, has_flip_or_jump, is_flipping, is_jumping, air_time, boost_raw], dtype=np.float32),
        "F.closing":    np.array([closing_speed], dtype=np.float32),
        "G.prev_act":   prev_act,
        "H.car_orient": car_orient_z,
        "I.opp_rel_b":  np.array([opp_fwd, opp_right, opp_up, opp_dist, opp_vfwd, opp_closing], dtype=np.float32),
    }
    return obs_full, obs_panel, parts


# ---- 左右镜像对称变换 (Bilateral Left-Right Symmetry for OBS_DIM = 42) ----
OBS_MIRROR_NEG_INDICES = (
    7, 8,          # B.car_ang_b: yaw, roll
    10,            # C.car_vel_b: v_right
    14, 17,        # D.ball_rel_b: dir_right, b_right
    28, 29,        # G.prev_act: p_str_yaw, p_roll
    34,            # H.car_orient: right_z
    37,            # I.opp_rel_b: opp_right
)


def _build_obs_mirror_perm_and_sign():
    perm = np.arange(OBS_DIM, dtype=np.int64)
    perm[0], perm[1] = 1, 0     # car -x <-> +x
    sign = np.ones(OBS_DIM, dtype=np.float32)
    for idx in OBS_MIRROR_NEG_INDICES:
        sign[idx] = -1.0
    return perm, sign


_OBS_MIRROR_PERM_NP, _OBS_MIRROR_SIGN_NP = _build_obs_mirror_perm_and_sign()

_LOGITS_MIRROR_PERM_NP = np.array(
    [0, 1, 2, 5, 4, 3, 8, 7, 6, 9, 10, 11, 12, 13, 14], dtype=np.int64
)


def mirror_obs_np(obs: np.ndarray) -> np.ndarray:
    return obs[..., _OBS_MIRROR_PERM_NP] * _OBS_MIRROR_SIGN_NP


def mirror_actions_np(actions: np.ndarray) -> np.ndarray:
    m = np.array(actions, dtype=np.int64, copy=True)
    m[..., 1] = 2 - m[..., 1]
    m[..., 2] = 2 - m[..., 2]
    return m


def mirror_obs_torch(obs_t):
    import torch
    perm = torch.as_tensor(_OBS_MIRROR_PERM_NP, dtype=torch.long, device=obs_t.device)
    sign = torch.as_tensor(_OBS_MIRROR_SIGN_NP, dtype=obs_t.dtype, device=obs_t.device)
    return obs_t.index_select(-1, perm) * sign


def mirror_actions_torch(act_t):
    m = act_t.clone()
    m[..., 1] = 2 - m[..., 1]
    m[..., 2] = 2 - m[..., 2]
    return m


def mirror_logits_torch(logits_t):
    import torch
    perm = torch.as_tensor(_LOGITS_MIRROR_PERM_NP, dtype=torch.long, device=logits_t.device)
    return logits_t.index_select(-1, perm)
