"""obs_v2 —— 车体绝对坐标系下的观测。训练和可视化共用。

维度 31：
  A.face_dists  6   到 -x/+x/-y/+y/floor/ceil 的距离
  B.car_ang_b   3   车体角速度 (pitch=yaw 绕 up, yaw=绕 up, roll=绕 fwd)
  C.car_vel_b   4   车体速度 3 分量 + 速度标量
  D.ball_rel_b  4   球在车体下的相对位置 3 分量 + 距离
  E.ball_relvel 4   球相对车体速度 3 分量 + 标量
  F.ball_ang_w  3   球世界角速度
  G.flags       6   on_ground, has_flip_or_jump, is_flipping, is_jumping, air_time, boost
  H.closing     1   closing speed (3D)
"""
import numpy as np

FIELD_X = 4096.0
FIELD_Y = 5120.0
FIELD_Z = 2000.0

S_FACE     = 5120.0   # 统一成最大边长
S_CAR_ANG  = 5.5
S_CAR_VEL  = 2300.0
S_BALL_POS = 5000.0
S_BALL_VEL = 2500.0
S_BALL_ANG = 6.0
S_CLOSING  = 5000.0   # 车 2300 + 球 3000

OBS_V2_DIM = 31

OBS_V2_GROUPS = {
    "A.face_dists":  (0,  6,  ["-x", "+x", "-y", "+y", "floor", "ceil"]),
    "B.car_ang_b":   (6,  9,  ["pitch", "yaw", "roll"]),
    "C.car_vel_b":   (9,  13, ["v_fwd", "v_right", "v_up", "|v|"]),
    "D.ball_rel_b":  (13, 17, ["b_fwd", "b_right", "b_up", "|d|"]),
    "E.ball_relvel": (17, 21, ["bv_fwd", "bv_right", "bv_up", "|bv|"]),
    "F.ball_ang_w":  (21, 24, ["wx", "wy", "wz"]),
    "G.flags":       (24, 30, ["on_ground", "flip_or_jump", "is_flipping",
                                "is_jumping", "air_time", "boost"]),
    "H.closing":     (30, 31, ["closing_spd"]),
}
GROUP_ORDER = list(OBS_V2_GROUPS.keys())


def _v(x):
    return np.array([x.x, x.y, x.z], dtype=np.float32)


def _hfj(cs):
    h = cs.has_flip_or_jump
    return bool(h() if callable(h) else h)


def build_obs_v2(car, ball_state):
    cs = car.get_state()
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

    # 车体角速度：pitch 轴=right, yaw 轴=up, roll 轴=forward
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
    ball_ang = _v(ball_state.ang_vel) if hasattr(ball_state, "ang_vel") \
               else np.zeros(3, dtype=np.float32)

    rel = ball_pos - car_pos
    rel_b = np.array([
        float(np.dot(rel, fwd)),
        float(np.dot(rel, right)),
        float(np.dot(rel, up)),
    ], dtype=np.float32)
    ball_dist = float(np.linalg.norm(rel))

    rel_vel = ball_vel - car_vel
    rel_vel_b = np.array([
        float(np.dot(rel_vel, fwd)),
        float(np.dot(rel_vel, right)),
        float(np.dot(rel_vel, up)),
    ], dtype=np.float32)
    rel_vel_mag = float(np.linalg.norm(rel_vel))

    # closing_speed > 0: 在靠近  < 0: 在远离
    if ball_dist > 1e-6:
        closing_speed = -float(np.dot(rel, rel_vel)) / ball_dist
    else:
        closing_speed = 0.0

    on_ground        = 1.0 if bool(cs.is_on_ground) else 0.0
    has_flip_or_jump = 1.0 if _hfj(cs) else 0.0
    is_flipping      = 1.0 if bool(cs.is_flipping) else 0.0
    is_jumping       = 1.0 if bool(cs.is_jumping) else 0.0
    air_time         = float(cs.air_time)
    boost_raw        = float(cs.boost)

    parts = {
        "A.face_dists":  faces,
        "B.car_ang_b":   car_ang_b,
        "C.car_vel_b":   np.concatenate([v_b, [v_mag]]).astype(np.float32),
        "D.ball_rel_b":  np.concatenate([rel_b, [ball_dist]]).astype(np.float32),
        "E.ball_relvel": np.concatenate([rel_vel_b, [rel_vel_mag]]).astype(np.float32),
        "F.ball_ang_w":  ball_ang.astype(np.float32),
        "G.flags":       np.array([on_ground, has_flip_or_jump,
                                    is_flipping, is_jumping,
                                    air_time, boost_raw], dtype=np.float32),
        "H.closing":     np.array([closing_speed], dtype=np.float32),
    }

    obs = np.concatenate([
        faces        / S_FACE,
        car_ang_b    / S_CAR_ANG,
        v_b          / S_CAR_VEL,
        [v_mag       / S_CAR_VEL],
        rel_b        / S_BALL_POS,
        [ball_dist   / S_BALL_POS],
        rel_vel_b    / S_BALL_VEL,
        [rel_vel_mag / S_BALL_VEL],
        ball_ang     / S_BALL_ANG,
        [on_ground, has_flip_or_jump, is_flipping, is_jumping,
         min(air_time, 2.0) / 2.0, boost_raw / 100.0],
        [closing_speed / S_CLOSING],
    ]).astype(np.float32)
    assert obs.shape[0] == OBS_V2_DIM, obs.shape
    return obs, parts