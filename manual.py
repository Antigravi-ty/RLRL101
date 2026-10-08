"""
teleop.py — 手动操控 RocketSim 并录制遥测（精简 CSV 版）

操作:
  W/A/S/D     地面: 油门/转向   空中: pitch/yaw
  Space 或 /  跳跃
  .           加速 boost
  Shift       手刹/power slide
  Q / E       空中 roll
  1           重置到地面静止
  2           重置到高空 (z=2000)
  3           重置到高速 (vx=1500)
  4           Reset kickoff
  R           开始/停止录制
  P           保存录制到 CSV
"""
import sys, json, time, math
import numpy as np
import RocketSim as rs
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets


# =====================================================
# 坐标系转换：RocketSim (x, y, z) → 渲染 (-x, y, z)
# =====================================================
def r2r(p):
    return np.array([-p[0], p[1], p[2]], dtype=np.float32)


def vec_r2r(v):
    return r2r([v.x, v.y, v.z])


# =====================================================
# 精简录制：字段定义
# =====================================================
SLIM_FIELDS = [
    "i",
    "px", "py", "pz",
    "vx", "vy", "vz",
    "wx", "wy", "wz",
    "qw", "qx", "qy", "qz",
    "boost", "air",
    "thr", "str", "ptc", "yaw", "roll",
    "flg", "whl",
]

FLAG_NAMES = [
    "on_ground", "flipping", "jumping", "auto_flip",
    "supersonic", "flipped", "jumped", "double_jumped",
    "flip_or_jump", "flip_reset", "got_flip_reset", "world_contact",
    "jump_in", "boost_in", "handbrake_in",
]

WHEEL_NAMES = ["wheel0", "wheel1", "wheel2", "wheel3"]

RECORD_FPS = 120


# =====================================================
# 全局状态
# =====================================================
class S:
    keys = {
        'W': False, 'A': False, 'S': False, 'D': False,
        'Space': False, 'Shift': False,
        'Q': False, 'E': False,
        'Period': False, 'Slash': False,
    }
    recording = False
    buffer = []
    frame = 0

_state = S()


# =====================================================
# RocketSim 世界
# =====================================================
import os
_mesh_path = os.path.join(os.path.dirname(__file__), "collision_meshes")
if os.path.exists(_mesh_path):
    try:
        rs.init(_mesh_path)
    except Exception:
        pass
else:
    try:
        rs.init()
    except Exception:
        pass

arena = rs.Arena(rs.GameMode.SOCCAR)
car = arena.add_car(rs.Team.BLUE, rs.CarConfig.OCTANE)

pads = arena.get_boost_pads()
PAD_POS_ROCKET = []
PAD_IS_BIG = []
for p in pads:
    pos = p.get_pos()
    PAD_POS_ROCKET.append((pos.x, pos.y, pos.z))
    PAD_IS_BIG.append(bool(p.is_big))

PAD_POS_RENDER = np.array([r2r(p) for p in PAD_POS_ROCKET], dtype=np.float32)
BIG_IDX = [i for i, b in enumerate(PAD_IS_BIG) if b]
SMALL_IDX = [i for i, b in enumerate(PAD_IS_BIG) if not b]
BIG_XYZ = PAD_POS_RENDER[BIG_IDX]
SMALL_XYZ = PAD_POS_RENDER[SMALL_IDX]
print(f"[setup] pads: {len(BIG_IDX)} big + {len(SMALL_IDX)} small = {len(PAD_POS_RENDER)}")


def set_scene(name):
    cs = rs.CarState()
    if name == 'ground':
        cs.pos = rs.Vec(3000, 0, 17)
    elif name == 'air':
        cs.pos = rs.Vec(3000, 0, 2000)
    elif name == 'high_speed':
        cs.pos = rs.Vec(3000, 0, 17)
        cs.vel = rs.Vec(1500, 0, 0)
    elif name == 'kickoff':
        arena.reset_kickoff()
        return
    car.set_state(cs)
    arena.step(1)
    print(f"[scene] {name}")


# =====================================================
# 键 → CarControls
# =====================================================
def build_ctrl():
    c = rs.CarControls()
    if _state.keys['W']:
        c.throttle = 1.0
        c.pitch    = -1.0
    if _state.keys['S']:
        c.throttle = -1.0
        c.pitch    = 1.0
    if _state.keys['A']:
        c.steer = -1.0
        c.yaw   = -1.0
    if _state.keys['D']:
        c.steer = 1.0
        c.yaw   = 1.0
    c.jump = _state.keys['Space'] or _state.keys['Slash']
    c.handbrake = _state.keys['Shift']
    c.boost = _state.keys['Period']
    if _state.keys['Q']:
        c.roll = -1.0
    if _state.keys['E']:
        c.roll = 1.0
    return c


# =====================================================
# 精简快照
# =====================================================
def quat_from_rotmat(m):
    """rot_mat -> 四元数 (w, x, y, z)，统一到 [+w] 半球"""
    R = np.array(m, dtype=np.float64).T
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    if qw < 0.0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz

    return qw, qx, qy, qz


def wheels_mask(cs):
    w = cs.wheels_with_contact
    if isinstance(w, str):
        s = w.strip().strip('()')
        return sum((1 << i) for i, x in enumerate(s.split(',')) if x.strip() == 'True')
    try:
        return sum((1 << i) for i, v in enumerate(w) if v)
    except TypeError:
        return 0


def pack_flags(cs, ctrl):
    hfj = cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump

    flags = [
        cs.is_on_ground, cs.is_flipping, cs.is_jumping, cs.is_auto_flipping,
        cs.is_supersonic, cs.has_flipped, cs.has_jumped, cs.has_double_jumped,
        hfj, cs.has_flip_reset, cs.got_flip_reset, cs.has_world_contact,
        bool(ctrl.jump), bool(ctrl.boost), bool(ctrl.handbrake),
    ]
    v = 0
    for i, b in enumerate(flags):
        if b:
            v |= (1 << i)
    return v


def slim_snapshot(ctrl):
    cs = car.get_state()
    qw, qx, qy, qz = quat_from_rotmat(cs.rot_mat)

    return [
        _state.frame,
        round(cs.pos.x, 2), round(cs.pos.y, 2), round(cs.pos.z, 2),
        round(cs.vel.x, 2), round(cs.vel.y, 2), round(cs.vel.z, 2),
        round(cs.ang_vel.x, 3), round(cs.ang_vel.y, 3), round(cs.ang_vel.z, 3),
        round(qw, 4), round(qx, 4), round(qy, 4), round(qz, 4),
        round(cs.boost, 1), round(cs.air_time, 3),
        round(ctrl.throttle, 2), round(ctrl.steer, 2),
        round(ctrl.pitch, 2), round(ctrl.yaw, 2), round(ctrl.roll, 2),
        pack_flags(cs, ctrl),
        wheels_mask(cs),
    ]


# =====================================================
# CSV 保存
# =====================================================
def write_csv(filename, frames, extra_meta=None):
    lines = []
    lines.append(f"# teleop slim v1  fps={RECORD_FPS}  frames={len(frames)}")
    lines.append("# coord: RocketSim (x,y,z); render flips x -> -x")
    lines.append("# quat: (qw,qx,qy,qz), w>=0 hemisphere")
    if extra_meta:
        for k, v in extra_meta.items():
            lines.append(f"# {k}: {v}")
    lines.append(f"# flags LSB order: {','.join(FLAG_NAMES)}")
    lines.append(f"# whl bit order: {','.join(WHEEL_NAMES)}")
    lines.append(','.join(SLIM_FIELDS))
    for fr in frames:
        lines.append(','.join(str(x) for x in fr))

    with open(filename, 'w') as f:
        f.write('\n'.join(lines))


# =====================================================
# Qt 3D 视图
# =====================================================
class View(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.hud = QtWidgets.QLabel(self)
        self.hud.setStyleSheet(
            "color:#00ffcc;font-size:13px;font-weight:bold;"
            "background:rgba(0,0,0,200);padding:6px;")
        self.hud.setGeometry(10, 10, 620, 170)
        self.refresh_hud()

    def refresh_hud(self):
        rec = "●REC" if _state.recording else "  --"
        cs = car.get_state()
        speed = np.sqrt(cs.vel.x**2 + cs.vel.y**2 + cs.vel.z**2)
        hfj = cs.has_flip_or_jump() if callable(cs.has_flip_or_jump) else cs.has_flip_or_jump
        msg = (
            f"{rec}  frames={len(_state.buffer)}  idx={_state.frame}\n"
            f"pos=({cs.pos.x:6.0f},{cs.pos.y:6.0f},{cs.pos.z:5.0f})  "
            f"speed={speed:5.0f}  boost={cs.boost:5.1f}\n"
            f"ground={int(cs.is_on_ground)}  flip={int(cs.is_flipping)}  "
            f"jump={int(cs.is_jumping)}  canJump={int(hfj)}  "
            f"air_t={cs.air_time:5.2f}s\n"
            f"[R]ec [P]save(csv)  [1]ground [2]air [3]high_speed [4]kickoff\n"
            f"WASD=drive/pitch/yaw  Space//=jump  .=boost  Shift=slide  Q/E=roll"
        )
        self.hud.setText(msg)

    def keyPressEvent(self, e):
        k = e.key()
        kmap = {
            QtCore.Qt.Key_W: 'W', QtCore.Qt.Key_A: 'A',
            QtCore.Qt.Key_S: 'S', QtCore.Qt.Key_D: 'D',
            QtCore.Qt.Key_Space: 'Space',
            QtCore.Qt.Key_Shift: 'Shift',
            QtCore.Qt.Key_Q: 'Q', QtCore.Qt.Key_E: 'E',
            QtCore.Qt.Key_Period: 'Period', QtCore.Qt.Key_Slash: 'Slash',
        }
        if k in kmap:
            _state.keys[kmap[k]] = True
        elif k == QtCore.Qt.Key_R:
            _state.recording = not _state.recording
            if _state.recording:
                _state.buffer = []
                _state.frame = 0
                print("[REC] START")
            else:
                print(f"[REC] STOP ({len(_state.buffer)} frames)")
        elif k == QtCore.Qt.Key_P:
            fn = f"teleop_{int(time.time())}.csv"
            write_csv(fn, _state.buffer, extra_meta={"recorded_at": int(time.time())})
            print(f"[SAVE] {fn} ({len(_state.buffer)} frames)")
        elif k == QtCore.Qt.Key_1:
            set_scene('ground')
        elif k == QtCore.Qt.Key_2:
            set_scene('air')
        elif k == QtCore.Qt.Key_3:
            set_scene('high_speed')
        elif k == QtCore.Qt.Key_4:
            set_scene('kickoff')
        self.refresh_hud()
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        k = e.key()
        kmap = {
            QtCore.Qt.Key_W: 'W', QtCore.Qt.Key_A: 'A',
            QtCore.Qt.Key_S: 'S', QtCore.Qt.Key_D: 'D',
            QtCore.Qt.Key_Space: 'Space',
            QtCore.Qt.Key_Shift: 'Shift',
            QtCore.Qt.Key_Q: 'Q', QtCore.Qt.Key_E: 'E',
            QtCore.Qt.Key_Period: 'Period', QtCore.Qt.Key_Slash: 'Slash',
        }
        if k in kmap:
            _state.keys[kmap[k]] = False
        self.refresh_hud()
        super().keyReleaseEvent(e)


# =====================================================
# 主程序
# =====================================================
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
view = View()
view.setWindowTitle('Teleop Calibration (slim CSV)')
view.setGeometry(80, 80, 1000, 750)
view.setCameraPosition(distance=6000, elevation=35, azimuth=45)
view.show()

# ---------- 地板网格 ----------
floor = gl.GLGridItem()
floor.setSize(8000, 10240, 0)
floor.setSpacing(500, 500, 0)
floor.setColor((255, 255, 255, 35))
view.addItem(floor)

# ---------- 场地边界 ----------
X_MIN, X_MAX = -4000, 4000
Y_MIN, Y_MAX = -5120, 5120
Z_MIN, Z_MAX = 0, 2000

corners = np.array([
    [X_MIN, Y_MIN, Z_MIN], [X_MAX, Y_MIN, Z_MIN],
    [X_MAX, Y_MAX, Z_MIN], [X_MIN, Y_MAX, Z_MIN],
    [X_MIN, Y_MIN, Z_MAX], [X_MAX, Y_MIN, Z_MAX],
    [X_MAX, Y_MAX, Z_MAX], [X_MIN, Y_MAX, Z_MAX],
], dtype=np.float32)
edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]
lines = []
for a, b in edges:
    lines.append(corners[a]); lines.append(corners[b])
border = gl.GLLinePlotItem(antialias=True)
border.setData(pos=np.array(lines), color=(255, 255, 255, 90), width=2, mode='lines')
view.addItem(border)

# ---------- 球门 ----------
GOAL_W = 1786.0
GOAL_H = 642.0
for sign_y in (-1, 1):
    y0 = sign_y * Y_MAX
    gx0, gx1 = -GOAL_W/2, GOAL_W/2
    gz0, gz1 = 0, GOAL_H
    goal_pts = np.array([
        [gx0, y0, gz0], [gx1, y0, gz0],
        [gx1, y0, gz1], [gx0, y0, gz1],
    ], dtype=np.float32)
    goal_lines = [
        goal_pts[0], goal_pts[1],
        goal_pts[1], goal_pts[2],
        goal_pts[2], goal_pts[3],
        goal_pts[3], goal_pts[0],
        goal_pts[0], goal_pts[2],
    ]
    color = (1.0, 0.5, 0.0, 1.0) if sign_y > 0 else (0.3, 0.6, 1.0, 1.0)
    goal_item = gl.GLLinePlotItem(antialias=True)
    goal_item.setData(pos=np.array(goal_lines), color=color, width=4, mode='lines')
    view.addItem(goal_item)

# ---------- Boost pads ----------
big_scatter = gl.GLScatterPlotItem()
big_scatter.setData(pos=BIG_XYZ, color=(1.0, 0.95, 0.0, 1.0), size=22)
view.addItem(big_scatter)

small_scatter = gl.GLScatterPlotItem()
small_scatter.setData(pos=SMALL_XYZ, color=(1.0, 0.85, 0.0, 0.6), size=12)
view.addItem(small_scatter)

# ---------- 车 ----------
car_scatter = gl.GLScatterPlotItem()
car_scatter.setData(pos=np.array([[0, 0, 0]]), color=(1, 1, 1, 1), size=18)
view.addItem(car_scatter)

# ---------- 轨迹 ----------
trail = gl.GLLinePlotItem(antialias=True)
view.addItem(trail)
trail_pts = []

# ---------- 朝向指示 ----------
fwd_line = gl.GLLinePlotItem(antialias=True); view.addItem(fwd_line)
up_line  = gl.GLLinePlotItem(antialias=True); view.addItem(up_line)
rgt_line = gl.GLLinePlotItem(antialias=True); view.addItem(rgt_line)

# ---------- 球 ----------
ball_scatter = gl.GLScatterPlotItem()
view.addItem(ball_scatter)

set_scene('ground')

_ticks = 0


def tick():
    global _ticks, trail_pts
    _ticks += 1

    ctrl = build_ctrl()
    car.set_controls(ctrl)
    arena.step(1)
    _state.frame += 1

    if _state.recording:
        _state.buffer.append(slim_snapshot(ctrl))

    cs = car.get_state()
    pos_r = r2r([cs.pos.x, cs.pos.y, cs.pos.z])
    car_scatter.setData(pos=pos_r.reshape(1, 3), color=(1, 1, 1, 1), size=18)

    trail_pts.append(pos_r)
    if len(trail_pts) > 600:
        trail_pts.pop(0)
    if len(trail_pts) > 1:
        trail.setData(pos=np.array(trail_pts), color=(0, 0.8, 1, 0.6), width=3)

    fwd = vec_r2r(car.get_forward_dir())
    up  = vec_r2r(car.get_up_dir())
    rgt = vec_r2r(car.get_right_dir())
    L = 300
    fwd_line.setData(pos=np.array([pos_r, pos_r + L*fwd]),
                     color=(0, 1, 0, 1), width=4)
    up_line.setData(pos=np.array([pos_r, pos_r + L*up]),
                    color=(1, 0, 0, 1), width=4)
    rgt_line.setData(pos=np.array([pos_r, pos_r + 0.5*L*rgt]),
                     color=(0, 0.5, 1, 1), width=4)

    bs = arena.ball.get_state()
    ball_r = r2r([bs.pos.x, bs.pos.y, bs.pos.z])
    ball_scatter.setData(pos=ball_r.reshape(1, 3), color=(1, 0.2, 0.2, 1), size=26)

    pad_states = [p.get_state().is_active for p in pads]
    big_colors = np.zeros((len(BIG_IDX), 4), dtype=np.float32)
    for k, idx in enumerate(BIG_IDX):
        if pad_states[idx]:
            big_colors[k] = (1.0, 0.95, 0.0, 1.0)
        else:
            big_colors[k] = (0.3, 0.3, 0.1, 0.4)
    big_scatter.setData(pos=BIG_XYZ, color=big_colors, size=22)

    small_colors = np.zeros((len(SMALL_IDX), 4), dtype=np.float32)
    for k, idx in enumerate(SMALL_IDX):
        if pad_states[idx]:
            small_colors[k] = (1.0, 0.85, 0.0, 0.6)
        else:
            small_colors[k] = (0.2, 0.2, 0.1, 0.25)
    small_scatter.setData(pos=SMALL_XYZ, color=small_colors, size=12)

    if _ticks % 10 == 0:
        view.refresh_hud()


timer = QtCore.QTimer()
timer.timeout.connect(tick)
timer.start(8)  # ~120Hz

if __name__ == '__main__':
    sys.exit(app.exec_())