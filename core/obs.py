"""core/obs.py —— Freestyle Air-Dribble 190 维观测空间 (54 维主干 Trunk + 34×4 维 Boost Pad 实体集)。

主干特征 (Trunk, 54 维, 索引 0..53):
  A.face_dists   [0..5]   (6)  车体到 -x/+x/-y/+y/floor/ceil 的距离
  B.car_ang_b    [6..8]   (3)  车体局部角速度 (pitch, yaw, roll)
  C.car_vel_b    [9..12]  (4)  车体局部速度 (v_fwd, v_right, v_up, |v|)
  D.ball_rel_b   [13..19] (7)  球相对方向单位向量 (dir_fwd, dir_right, dir_up) + 相对位置 (b_fwd, b_right, b_up, |d|)
  E.flags        [20..25] (6)  on_ground, flip_or_jump, is_flipping, is_jumping, air_time, boost
  F.closing      [26..26] (1)  车体朝向球的三维主动接近速度 closing_spd
  G.prev_act     [27..32] (6)  上一决策步动作 (thr_pitch, steer_yaw, roll, jump, boost, slide)
  H.ball_vel_b   [33..38] (6)  球在车局部系速度 (bv_fwd, bv_right, bv_up) + 球与车相对速度 (rv_fwd, rv_right, rv_up)
  I.air_combo    [39..44] (6)  球世界系垂直速度 ball_vz, 球速率 |bv|, 离平面净空 surf_clear, 盒内标志 in_air_box, 连击数 touches, 连击路程 chain_dist
  J.car_orient   [45..47] (3)  车体三轴世界 Z 分量 (fwd_z, right_z, up_z)
  K.ball_faces   [48..53] (6)  球体到六面墙壁距离 (-x, +x, -y, +y, floor, ceil)

实体集特征 (Boost Pad Tokens, 136 维 = 34×4, 索引 54..189):
  L.boost_pads   [54..189] (136) 每个氮气垫 [rel_fwd, rel_right, is_active, cooldown_norm]
"""
import numpy as np
from .constants import (
    OBS_DIM, OBS_PANEL_DIM, TRUNK_OBS_DIM, NUM_BOOST_PADS,
    FIELD_X, FIELD_Y, FIELD_Z, CORNER_SUM,
    AIR_BOX_MIN_Z, AIR_BOX_CLEARANCE,
)

S_FACE     = 5120.0
S_CAR_ANG  = 5.5
S_CAR_VEL  = 2300.0
S_BALL_POS = 5000.0
S_BALL_VEL = 2300.0
S_CLOSING  = 2300.0

OBS_GROUPS = {
    "A.face_dists": (0,  6,  ["-x", "+x", "-y", "+y", "floor", "ceil"]),
    "B.car_ang_b":  (6,  9,  ["pitch", "yaw", "roll"]),
    "C.car_vel_b":  (9,  13, ["v_fwd", "v_right", "v_up", "|v|"]),
    "D.ball_rel_b": (13, 20, ["dir_fwd", "dir_right", "dir_up", "b_fwd", "b_right", "b_up", "|d|"]),
    "E.flags":      (20, 26, ["on_ground", "flip_or_jump", "is_flipping", "is_jumping", "air_time", "boost"]),
    "F.closing":    (26, 27, ["closing_spd"]),
    "G.prev_act":   (27, 33, ["p_thr_ptc", "p_str_yaw", "p_roll", "p_jump", "p_boost", "p_slide"]),
    "H.ball_vel_b": (33, 39, ["bv_fwd", "bv_right", "bv_up", "rv_fwd", "rv_right", "rv_up"]),
    "I.air_combo":  (39, 45, ["ball_vz", "|bv|", "surf_clr", "in_box", "touches", "chain_d"]),
}
GROUP_ORDER = list(OBS_GROUPS.keys())

BOOST_PAD_LOCATIONS = np.array([
    (    0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    ( 1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    ( 3072.0, -4096.0, 73.0),
    ( -940.0, -3308.0, 70.0),
    (  940.0, -3308.0, 70.0),
    (    0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    ( 3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    ( 1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (    0.0, -1024.0, 70.0),
    ( 2048.0, -1036.0, 70.0),
    (-3584.0,     0.0, 73.0),
    (-1024.0,     0.0, 70.0),
    ( 1024.0,     0.0, 70.0),
    ( 3584.0,     0.0, 73.0),
    (-2048.0,  1036.0, 70.0),
    (    0.0,  1024.0, 70.0),
    ( 2048.0,  1036.0, 70.0),
    (-1788.0,  2300.0, 70.0),
    ( 1788.0,  2300.0, 70.0),
    (-3584.0,  2484.0, 70.0),
    ( 3584.0,  2484.0, 70.0),
    (    0.0,  2816.0, 70.0),
    ( -940.0,  3308.0, 70.0),
    (  940.0,  3308.0, 70.0),
    (-3072.0,  4096.0, 73.0),
    ( 3072.0,  4096.0, 73.0),
    (-1792.0,  4184.0, 70.0),
    ( 1792.0,  4184.0, 70.0),
    (    0.0,  4240.0, 70.0),
], dtype=np.float32)

BIG_PAD_INDICES = (3, 4, 15, 18, 29, 30)
BIG_PAD_COOLDOWN = 10.0
SMALL_PAD_COOLDOWN = 4.0

PAD_COOLDOWN_SCALES = np.array([
    BIG_PAD_COOLDOWN if i in BIG_PAD_INDICES else SMALL_PAD_COOLDOWN
    for i in range(NUM_BOOST_PADS)
], dtype=np.float32)

PAD_IS_BIG_MASK = np.array([
    1.0 if i in BIG_PAD_INDICES else 0.0
    for i in range(NUM_BOOST_PADS)
], dtype=np.float32)

PAD_MIRROR_PAIRS = (
    (1, 2), (3, 4), (5, 6), (8, 9), (10, 11), (12, 14), (15, 18),
    (16, 17), (19, 21), (22, 23), (24, 25), (27, 28), (29, 30), (31, 32),
)


def compute_surface_clearance(bx: float, by: float, bz: float) -> float:
    """计算球心 (bx, by, bz) 到球场任意平面 (地面、天花板、两侧边墙、两端后墙、45°斜角墙) 的最小垂直净空距离。"""
    d_floor  = float(bz)
    d_ceil   = float(FIELD_Z - bz)
    d_wall_x = float(FIELD_X - abs(bx))
    d_wall_y = float(FIELD_Y - abs(by))
    d_corner = float((CORNER_SUM - (abs(bx) + abs(by))) * 0.70710678)
    return max(0.0, min(d_floor, d_ceil, d_wall_x, d_wall_y, d_corner))


def is_in_air_box(bx: float, by: float, bz: float) -> bool:
    """判断球是否处于脱离所有平面的中央计分盒区 (Air-Box)。"""
    if float(bz) < AIR_BOX_MIN_Z:
        return False
    return compute_surface_clearance(bx, by, bz) >= AIR_BOX_CLEARANCE


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


def build_obs(
    car,
    arena,
    prev_action_vec=None,
    combo_touches: int = 0,
    combo_chain_dist: float = 0.0,
):
    """构建 190 维观测向量 (54 维主干 + 136 维 Boost Pads)，返回 (obs_190, obs_panel_45, parts_dict)。"""
    cs = car.get_state()
    ball_state = arena.ball.get_state()

    fwd   = _v(car.get_forward_dir())
    right = _v(car.get_right_dir())
    up    = _v(car.get_up_dir())

    car_pos = _v(cs.pos)
    car_vel = _v(cs.vel)
    car_ang = _v(cs.ang_vel)

    faces = np.array([
        car_pos[0] + FIELD_X,
        FIELD_X - car_pos[0],
        car_pos[1] + FIELD_Y,
        FIELD_Y - car_pos[1],
        car_pos[2],
        FIELD_Z - car_pos[2],
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
    ball_vel = _v(ball_state.vel)

    rel = ball_pos - car_pos
    b_fwd   = float(np.dot(rel, fwd))
    b_right = float(np.dot(rel, right))
    b_up    = float(np.dot(rel, up))
    ball_dist = float(np.linalg.norm(rel))
    inv_dist = 1.0 / max(ball_dist, 1e-6)
    dir_fwd   = b_fwd * inv_dist
    dir_right = b_right * inv_dist
    dir_up    = b_up * inv_dist

    rel_vel = ball_vel - car_vel
    if ball_dist > 1e-6:
        # Active closing speed of car toward moving ball: -dot(rel, ball_vel - car_vel)/dist
        closing_speed = float(np.dot(rel, car_vel - ball_vel)) * inv_dist
    else:
        closing_speed = 0.0

    # Ball velocity & relative velocity in car local frame (H.ball_vel_b, 6 dims)
    bv_fwd   = float(np.dot(ball_vel, fwd))
    bv_right = float(np.dot(ball_vel, right))
    bv_up    = float(np.dot(ball_vel, up))
    rv_fwd   = float(np.dot(rel_vel, fwd))
    rv_right = float(np.dot(rel_vel, right))
    rv_up    = float(np.dot(rel_vel, up))
    ball_vel_b_raw = np.array([bv_fwd, bv_right, bv_up, rv_fwd, rv_right, rv_up], dtype=np.float32)

    # Air-Box clearance & Combo state (I.air_combo, 6 dims)
    bx, by, bz = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])
    ball_vz_w = float(ball_vel[2])
    ball_spd = float(np.linalg.norm(ball_vel))
    surf_clr = compute_surface_clearance(bx, by, bz)
    in_box = 1.0 if (bz >= AIR_BOX_MIN_Z and surf_clr >= AIR_BOX_CLEARANCE) else 0.0
    c_touches = float(max(0, int(combo_touches)))
    c_dist = float(max(0.0, float(combo_chain_dist)))
    air_combo_raw = np.array([ball_vz_w, ball_spd, surf_clr, in_box, c_touches, c_dist], dtype=np.float32)
    air_combo_norm = np.array([
        np.clip(ball_vz_w / S_BALL_VEL, -1.0, 1.0),
        min(ball_spd / 3000.0, 1.5),
        min(surf_clr / 1500.0, 1.5),
        in_box,
        min(c_touches / 10.0, 2.0),
        min(c_dist / 5000.0, 2.0),
    ], dtype=np.float32)

    on_ground        = 1.0 if bool(cs.is_on_ground) else 0.0
    has_flip_or_jump = 1.0 if _hfj(cs) else 0.0
    is_flipping      = 1.0 if bool(cs.is_flipping) else 0.0
    is_jumping       = 1.0 if bool(cs.is_jumping) else 0.0
    air_time         = float(cs.air_time)
    boost_raw        = float(cs.boost)

    prev_act = format_prev_action_vec(prev_action_vec)

    car_orient_z = np.array([fwd[2], right[2], up[2]], dtype=np.float32)

    ball_faces = np.array([
        ball_pos[0] + FIELD_X,
        FIELD_X - ball_pos[0],
        ball_pos[1] + FIELD_Y,
        FIELD_Y - ball_pos[1],
        ball_pos[2],
        FIELD_Z - ball_pos[2],
    ], dtype=np.float32) / S_FACE

    pads = arena.get_boost_pads()
    rel_pads = BOOST_PAD_LOCATIONS - car_pos[None, :]
    pads_fwd = (rel_pads @ fwd) / S_FACE
    pads_rgt = (rel_pads @ right) / S_FACE
    pads_active = np.empty(NUM_BOOST_PADS, dtype=np.float32)
    pads_cd = np.empty(NUM_BOOST_PADS, dtype=np.float32)
    for idx_k, p in enumerate(pads):
        ps = p.get_state()
        pads_active[idx_k] = 1.0 if ps.is_active else 0.0
        pads_cd[idx_k] = min(1.0, max(0.0, float(ps.cooldown) / PAD_COOLDOWN_SCALES[idx_k]))

    all_pads_feat = np.empty(NUM_BOOST_PADS * 4, dtype=np.float32)
    all_pads_feat[0::4] = pads_fwd
    all_pads_feat[1::4] = pads_rgt
    all_pads_feat[2::4] = pads_active
    all_pads_feat[3::4] = pads_cd

    obs_panel = np.concatenate([
        faces / S_FACE,                                                          # 0..5   (6)
        car_ang_b / S_CAR_ANG,                                                   # 6..8   (3)
        v_b / S_CAR_VEL,                                                         # 9..11  (3)
        [v_mag / S_CAR_VEL],                                                     # 12     (1)
        [dir_fwd, dir_right, dir_up],                                            # 13..15 (3)
        [b_fwd / S_BALL_POS, b_right / S_BALL_POS, b_up / S_BALL_POS, ball_dist / S_BALL_POS],  # 16..19 (4)
        [on_ground, has_flip_or_jump, is_flipping, is_jumping,
         min(air_time, 4.0) / 4.0, boost_raw / 100.0],                           # 20..25 (6)
        [closing_speed / S_CLOSING],                                             # 26     (1)
        prev_act,                                                                # 27..32 (6)
        ball_vel_b_raw / S_BALL_VEL,                                             # 33..38 (6)
        air_combo_norm,                                                          # 39..44 (6)
    ]).astype(np.float32)

    obs_full = np.concatenate([
        obs_panel,      # 0..44  (45)
        car_orient_z,   # 45..47 (3)
        ball_faces,     # 48..53 (6)  -> Total Trunk = 54
        all_pads_feat,  # 54..189 (136) -> Total = 190
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
        "H.ball_vel_b": ball_vel_b_raw,
        "I.air_combo":  air_combo_raw,
        "J.car_orient": car_orient_z,
        "K.ball_faces": ball_faces * S_FACE,
        "L.boost_pads": all_pads_feat,
    }
    return obs_full, obs_panel, parts


# ---- 左右镜像对称变换 (Bilateral Symmetry for OBS_DIM = 190) ----
OBS_MIRROR_NEG_INDICES = (
    7, 8,          # B.car_ang_b: yaw, roll
    10,            # C.car_vel_b: v_right
    14, 17,        # D.ball_rel_b: dir_right, b_right
    28, 29,        # G.prev_act: p_str_yaw, p_roll
    34, 37,        # H.ball_vel_b: bv_right, rv_right
    46,            # J.car_orient: right_z
    *(TRUNK_OBS_DIM + 4 * i + 1 for i in range(NUM_BOOST_PADS)),  # L.boost_pads: pad_right
)


def _build_obs_mirror_perm_and_sign():
    perm = np.arange(OBS_DIM, dtype=np.int64)
    perm[0], perm[1] = 1, 0       # car -x <-> +x
    perm[48], perm[49] = 49, 48   # ball -x <-> +x
    for i_left, i_right in PAD_MIRROR_PAIRS:
        b_left = TRUNK_OBS_DIM + 4 * i_left
        b_right = TRUNK_OBS_DIM + 4 * i_right
        for k in range(4):
            perm[b_left + k] = b_right + k
            perm[b_right + k] = b_left + k

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
