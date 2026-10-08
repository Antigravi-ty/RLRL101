import os
import numpy as np
import RocketSim as rs

# 自动定位并初始化碰撞网格
mesh_path = os.path.join(os.path.dirname(__file__), "collision_meshes")
if os.path.exists(mesh_path):
    try:
        rs.init(mesh_path)
    except Exception:
        pass
else:
    try:
        rs.init()
    except Exception:
        pass

DECISION_HZ = 30
PHYSICS_HZ = 120
TICKS_PER_DECISION = PHYSICS_HZ // DECISION_HZ  # 4

MAX_DECISIONS = 150       # 5s 上限
TOUCH_DIST = 200.0
BALL_RADIUS = 93.0

OBS_DIM = 20
N_ACTIONS = 7
N_INTENTS = 28
HIDDEN_DIM = 128

# (throttle, steer)
ACTIONS = [
    (1.0,  0.0),
    (1.0, -1.0),
    (1.0,  1.0),
    (-1.0, 0.0),
    (-1.0, -1.0),
    (-1.0, 1.0),
    (0.0,  0.0),
]


def build_obs(car, ball_state, team=rs.Team.BLUE):
    cs = car.get_state()
    fwd = car.get_forward_dir()
    right = car.get_right_dir()
    up = car.get_up_dir()

    sign = 1.0 if team == rs.Team.BLUE else -1.0

    cp = np.array([sign * cs.pos.x, sign * cs.pos.y, cs.pos.z], dtype=np.float32)
    cv = np.array([sign * cs.vel.x, sign * cs.vel.y, cs.vel.z], dtype=np.float32)
    cw = np.array([sign * cs.ang_vel.x, sign * cs.ang_vel.y, cs.ang_vel.z], dtype=np.float32)

    cf = np.array([sign * fwd.x, sign * fwd.y, fwd.z], dtype=np.float32)
    cu = np.array([sign * up.x, sign * up.y, up.z], dtype=np.float32)
    cr = np.array([sign * right.x, sign * right.y, right.z], dtype=np.float32)

    bp = np.array([sign * ball_state.pos.x, sign * ball_state.pos.y, ball_state.pos.z], dtype=np.float32)
    bv = np.array([sign * ball_state.vel.x, sign * ball_state.vel.y, ball_state.vel.z], dtype=np.float32)

    rel = bp - cp
    dist = float(np.linalg.norm(rel))

    rel_local = np.array([np.dot(rel, cr), np.dot(rel, cf), np.dot(rel, cu)], dtype=np.float32) / 5000.0
    bv_local  = np.array([np.dot(bv, cr), np.dot(bv, cf), np.dot(bv, cu)], dtype=np.float32) / 2000.0

    obs = np.concatenate([
        rel_local,
        bv_local,
        cf,
        cu,
        cv / 2300.0,
        cw / 10.0,
        [cs.boost / 100.0],
        [dist / 5000.0],
    ]).astype(np.float32)

    assert obs.shape[0] == OBS_DIM
    return obs, dist