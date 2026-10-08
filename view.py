import sys
import socket
import struct
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

# ---- Boost pad positions from RocketSim (no fallback) ----
import RocketSim as _rs

_arena = _rs.Arena(_rs.GameMode.SOCCAR)
_pads = _arena.get_boost_pads()

PAD_POSITIONS = []   # 顺序与 RocketSim 返回一致
PAD_IS_BIG = []
for p in _pads:
    pos = p.get_pos()
    PAD_POSITIONS.append((-pos.x, pos.y))
    PAD_IS_BIG.append(bool(p.is_big))

NUM_PADS = len(PAD_POSITIONS)
BIG_INDICES = [i for i, b in enumerate(PAD_IS_BIG) if b]
SMALL_INDICES = [i for i, b in enumerate(PAD_IS_BIG) if not b]

BIG_XYZ = np.array([[PAD_POSITIONS[i][0], PAD_POSITIONS[i][1], 30.0] for i in BIG_INDICES])
SMALL_XYZ = np.array([[PAD_POSITIONS[i][0], PAD_POSITIONS[i][1], 30.0] for i in SMALL_INDICES])

del _arena
print(f"[view] pads: {len(BIG_INDICES)} big, {len(SMALL_INDICES)} small, total {NUM_PADS}")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.bind(('127.0.0.1', 9999))
client_socket.setblocking(False)

CONTROL_ADDRESS = ('127.0.0.1', 8888)
keyboard_controls = {"throttle": 0.0, "steer": 0.0, "boost": 0.0}


class ControlGlView(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.hud = QtWidgets.QLabel(self)
        self.hud.setStyleSheet(
            "color: #00ffcc; font-size: 20px; font-weight: bold; "
            "background: rgba(0,0,0,100); padding: 5px;"
        )
        self.hud.setGeometry(10, 10, 380, 40)
        self.hud.setText("Boost: 0.0 | Speed: 0 | --")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_W:
            keyboard_controls["throttle"] = 1.0
        elif event.key() == QtCore.Qt.Key_S:
            keyboard_controls["throttle"] = -1.0
        elif event.key() == QtCore.Qt.Key_D:
            keyboard_controls["steer"] = -1.0
        elif event.key() == QtCore.Qt.Key_A:
            keyboard_controls["steer"] = 1.0
        elif event.key() == QtCore.Qt.Key_Space:
            keyboard_controls["boost"] = 1.0
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() in [QtCore.Qt.Key_W, QtCore.Qt.Key_S]:
            keyboard_controls["throttle"] = 0.0
        elif event.key() in [QtCore.Qt.Key_D, QtCore.Qt.Key_A]:
            keyboard_controls["steer"] = 0.0
        elif event.key() == QtCore.Qt.Key_Space:
            keyboard_controls["boost"] = 0.0
        super().keyReleaseEvent(event)


app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

view = ControlGlView()
view.setWindowTitle('RocketSim - Cyber Arena v104')
view.setGeometry(100, 100, 900, 700)
view.setCameraPosition(distance=7000, elevation=30, azimuth=45)
view.show()

# ---- Floor ----
floor = gl.GLGridItem()
floor.setSize(8000, 10240, 0)
floor.setSpacing(400, 400, 0)
floor.setColor((255, 255, 255, 35))
view.addItem(floor)

# ---- Field border ----
points = np.array([
    [-4000, -5120, 0], [4000, -5120, 0], [4000, 5120, 0], [-4000, 5120, 0],
    [-4000, -5120, 2000], [4000, -5120, 2000],
    [4000, 5120, 2000], [-4000, 5120, 2000],
])
box_lines = np.array([
    points[0], points[1], points[1], points[2], points[2], points[3], points[3], points[0],
    points[4], points[5], points[5], points[6], points[6], points[7], points[7], points[4],
    points[0], points[4], points[1], points[5], points[2], points[6], points[3], points[7],
])
box_edges = gl.GLLinePlotItem(antialias=True)
box_edges.setData(pos=box_lines, color=(255, 255, 255, 80), width=2, mode='lines')
view.addItem(box_edges)

# ---- Boost pads ----
big_scatter = gl.GLScatterPlotItem()
big_scatter.setData(pos=BIG_XYZ, color=(1.0, 0.95, 0.0, 1.0), size=22)
view.addItem(big_scatter)

small_scatter = gl.GLScatterPlotItem()
small_scatter.setData(pos=SMALL_XYZ, color=(1.0, 0.85, 0.0, 0.5), size=10)
view.addItem(small_scatter)

# ---- Dynamic items ----
car_scatter = gl.GLScatterPlotItem()
view.addItem(car_scatter)

ball_scatter = gl.GLScatterPlotItem()
view.addItem(ball_scatter)

trail_line = gl.GLLinePlotItem(antialias=True)
view.addItem(trail_line)

forward_indicator = gl.GLLinePlotItem(antialias=True)
view.addItem(forward_indicator)
up_indicator = gl.GLLinePlotItem(antialias=True)
view.addItem(up_indicator)
right_indicator = gl.GLLinePlotItem(antialias=True)
view.addItem(right_indicator)

trajectory_points = []

EXPECTED_PKT = 'f' * (20 + NUM_PADS)


def sync_loop():
    global trajectory_points

    control_packet = struct.pack(
        'fff',
        keyboard_controls["throttle"],
        keyboard_controls["steer"],
        keyboard_controls["boost"],
    )
    try:
        client_socket.sendto(control_packet, CONTROL_ADDRESS)
    except Exception:
        pass

    last_packet = None
    while True:
        try:
            message, _ = client_socket.recvfrom(4096)
            last_packet = message
        except BlockingIOError:
            break

    if last_packet is None:
        return

    try:
        data = struct.unpack(EXPECTED_PKT, last_packet)
        x, y, z = data[0], data[1], data[2]
        fx, fy, fz = data[3], data[4], data[5]
        ux, uy, uz = data[6], data[7], data[8]
        rx, ry, rz = data[9], data[10], data[11]
        reset_flag = data[12]
        bx, by, bz = data[13], data[14], data[15]
        boost_amount = data[16]
        is_boosting = data[17]
        is_on_ground = data[18]
        speed_uu = data[19]
        pad_act = data[20:]

        ground_txt = "Ground" if is_on_ground > 0.5 else "Air"
        boost_txt = " [BOOST]" if is_boosting > 0.5 else ""
        view.hud.setText(
            f"Boost: {boost_amount:5.1f} | Speed: {speed_uu:6.0f} uu/s | {ground_txt}{boost_txt}"
        )

        if reset_flag > 0.5:
            trajectory_points = []

        # 每个点： (x, y, z, is_boosting)
        trajectory_points.append([x, y, z, is_boosting])
        if len(trajectory_points) > 400:
            trajectory_points.pop(0)

        car_scatter.setData(
            pos=np.array([[x, y, z]]),
            color=(1.0, 1.0, 1.0, 1.0),
            size=14,
        )
        ball_scatter.setData(
            pos=np.array([[bx, by, bz]]),
            color=(1.0, 0.2, 0.2, 1.0),
            size=24,
        )

        if len(trajectory_points) > 1:
            tp = np.array(trajectory_points, dtype=np.float32)  # (N, 4)
            pos = tp[:, :3]
            col = np.zeros((len(tp), 4), dtype=np.float32)
            boosting_mask = tp[:, 3] > 0.5
            col[~boosting_mask] = (0.0, 0.8, 1.0, 0.55)   # 青蓝 = 正常
            col[boosting_mask]  = (1.0, 0.8, 0.0, 0.95)   # 金黄 = boost
            trail_line.setData(pos=pos, color=col, width=4)

        # ---- Pad colors ----
        if len(pad_act) == NUM_PADS:
            big_colors = np.zeros((len(BIG_INDICES), 4), dtype=np.float32)
            for k, idx in enumerate(BIG_INDICES):
                if pad_act[idx] > 0.5:
                    big_colors[k] = (1.0, 0.95, 0.0, 1.0)
                else:
                    big_colors[k] = (0.3, 0.3, 0.1, 0.4)
            big_scatter.setData(pos=BIG_XYZ, color=big_colors, size=22)

            small_colors = np.zeros((len(SMALL_INDICES), 4), dtype=np.float32)
            for k, idx in enumerate(SMALL_INDICES):
                if pad_act[idx] > 0.5:
                    small_colors[k] = (1.0, 0.85, 0.0, 0.6)
                else:
                    small_colors[k] = (0.2, 0.2, 0.1, 0.25)
            small_scatter.setData(pos=SMALL_XYZ, color=small_colors, size=10)

        base_len = 350
        forward_indicator.setData(
            pos=np.array([[x, y, z],
                        [x + fx * base_len, y + fy * base_len, z + fz * base_len]]),
            color=(0.0, 1.0, 0.0, 1.0),
            width=4,
        )
        up_indicator.setData(
            pos=np.array([[x, y, z],
                        [x + ux * base_len, y + uy * base_len, z + uz * base_len]]),
            color=(1.0, 0.0, 0.0, 1.0),
            width=4,
        )
        right_indicator.setData(
            pos=np.array([[x, y, z],
                        [x + rx * base_len * 0.33,
                        y + ry * base_len * 0.33,
                        z + rz * base_len * 0.33]]),
            color=(0.0, 0.5, 1.0, 1.0),
            width=4,
        )
    except Exception as e:
        print("parse error:", e)

timer = QtCore.QTimer()
timer.timeout.connect(sync_loop)
timer.start(4)

if __name__ == '__main__':
    sys.exit(app.exec_())