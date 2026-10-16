import numpy as np
from .obs_v2 import (
    FIELD_X, FIELD_Y, FIELD_Z,
    S_FACE, S_CAR_ANG, S_CAR_VEL, S_BALL_POS, S_BALL_VEL, S_BALL_ANG, S_CLOSING,
    build_obs_v2, _v,
)
from .constants import OBS_V3_DIM

BIG_PAD_INDICES = (3, 4, 15, 18, 29, 30)
BIG_PAD_COOLDOWN = 10.0

OPP_GOAL_POS = np.array([0.0, 5120.0, 320.0], dtype=np.float32)
OWN_GOAL_POS = np.array([0.0, -5120.0, 320.0], dtype=np.float32)


def compute_oracle_intent(car_pos, ball_pos, fwd, right, up, want_ball=1.0, target_pos=None, target_time=None):
    tgt = OPP_GOAL_POS if target_pos is None else np.asarray(target_pos, dtype=np.float32)
    ball_to_tgt = tgt - ball_pos
    b2t_dist = float(np.linalg.norm(ball_to_tgt)) + 1e-6
    b2t_dir = ball_to_tgt / b2t_dist

    ideal_hit_pos = ball_pos - b2t_dir * 95.0
    rel_target = ideal_hit_pos - car_pos

    target_fwd = float(np.dot(rel_target, fwd)) / S_BALL_POS
    target_right = float(np.dot(rel_target, right)) / S_BALL_POS

    kick_dir_fwd = float(np.dot(b2t_dir, fwd))
    kick_dir_right = float(np.dot(b2t_dir, right))

    if target_time is not None and target_time > 0.05:
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
    ball_state = arena.ball.get_state()
    obs_v2, parts = build_obs_v2(car, ball_state)

    cs = car.get_state()
    fwd   = _v(car.get_forward_dir())
    right = _v(car.get_right_dir())
    up    = _v(car.get_up_dir())

    car_pos  = _v(cs.pos)
    ball_pos = _v(ball_state.pos)
    ball_ang = _v(ball_state.ang_vel) if hasattr(ball_state, "ang_vel") else np.zeros(3, dtype=np.float32)

    ball_ang_b = np.array([
        float(np.dot(ball_ang, right)),
        float(np.dot(ball_ang, up)),
        float(np.dot(ball_ang, fwd)),
    ], dtype=np.float32) / S_BALL_ANG

    car_orient_z = np.array([fwd[2], right[2], up[2]], dtype=np.float32)

    ball_faces = np.array([
        ball_pos[0] + FIELD_X,
        FIELD_X - ball_pos[0],
        ball_pos[1] + FIELD_Y,
        FIELD_Y - ball_pos[1],
        ball_pos[2],
        FIELD_Z - ball_pos[2],
    ], dtype=np.float32) / S_FACE

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

    if intent_override is not None:
        intent_vec = np.asarray(intent_override, dtype=np.float32)
    else:
        intent_vec = compute_oracle_intent(
            car_pos, ball_pos, fwd, right, up,
            want_ball=1.0, target_pos=target_pos, target_time=target_time,
        )

    obs_v2_ego = obs_v2.copy()
    obs_v2_ego[21:24] = ball_ang_b

    obs_v3 = np.concatenate([
        obs_v2_ego,
        car_orient_z,
        ball_faces,
        goals_rel_b,
        big_pads_feat,
        intent_vec,
    ]).astype(np.float32)

    assert obs_v3.shape[0] == OBS_V3_DIM, f"Expected {OBS_V3_DIM}, got {obs_v3.shape[0]}"
    parts["I.car_orient"] = car_orient_z
    parts["J.ball_faces"] = ball_faces * S_FACE
    parts["K.goals_rel_b"] = goals_rel_b * S_BALL_POS
    parts["M.intent"] = intent_vec
    return obs_v3, obs_v2, parts


OBS_V3_MIRROR_NEG_INDICES = (
    7, 8,
    10,
    14,
    18,
    22, 23,
    32,
    41, 44,
    47, 51, 55, 59, 63, 67,
    72, 74,
)


def _build_obs_v3_mirror_perm_and_sign():
    perm = np.arange(OBS_V3_DIM, dtype=np.int64)
    perm[0], perm[1] = 1, 0
    perm[34], perm[35] = 35, 34
    for base in (46, 54, 62):
        for k in range(4):
            perm[base + k] = base + 4 + k
            perm[base + 4 + k] = base + k

    sign = np.ones(OBS_V3_DIM, dtype=np.float32)
    for idx in OBS_V3_MIRROR_NEG_INDICES:
        sign[idx] = -1.0
    return perm, sign


_OBS_MIRROR_PERM_NP, _OBS_MIRROR_SIGN_NP = _build_obs_v3_mirror_perm_and_sign()

_LOGITS_MIRROR_PERM_NP = np.array(
    [0, 1, 2, 5, 4, 3, 8, 7, 6, 9, 10, 11, 12, 13, 14], dtype=np.int64
)


def mirror_obs_v3_np(obs: np.ndarray) -> np.ndarray:
    return obs[..., _OBS_MIRROR_PERM_NP] * _OBS_MIRROR_SIGN_NP


def mirror_actions_np(actions: np.ndarray) -> np.ndarray:
    m = np.array(actions, dtype=np.int64, copy=True)
    m[..., 1] = 2 - m[..., 1]
    m[..., 2] = 2 - m[..., 2]
    return m


def mirror_obs_v3_torch(obs_t):
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
