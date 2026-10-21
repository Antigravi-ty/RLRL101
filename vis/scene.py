"""vis/scene.py —— 3D 球场渲染组件、轨迹点线与 3D 车辆碰撞箱半透明容器 (CarHitboxRenderer)。"""
import math
import numpy as np
import pyqtgraph.opengl as gl
from .camera import TunerGLViewWidget


def r2r(p):
    return np.array([-p[0], p[1], p[2]], dtype=np.float32)


def vec_r2r(v):
    return np.array([-v.x, v.y, v.z], dtype=np.float32)


def vec_r2r_tuple(t):
    return np.array([-t[0], t[1], t[2]], dtype=np.float32)


def add_floor(view, size=(8000, 10240), spacing=(500, 500)):
    grid = gl.GLGridItem()
    grid.setSize(size[0], size[1], 0)
    grid.setSpacing(spacing[0], spacing[1], 0)
    grid.setColor((255, 255, 255, 35))
    view.addItem(grid)


def add_field_border(view, x=4000, y=5120, z=2000):
    corners = np.array([
        [-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
    ], dtype=np.float32)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in edges:
        pts.append(corners[a]); pts.append(corners[b])
    item = gl.GLLinePlotItem(antialias=True)
    item.setData(pos=np.array(pts), color=(255, 255, 255, 90), width=2, mode='lines')
    view.addItem(item)


def add_goals(view, y=5120, w=1786.0, h=642.0):
    for sign, color in [(-1, (0.3, 0.6, 1.0, 1.0)), (1, (1.0, 0.5, 0.0, 1.0))]:
        y0 = sign * y
        gx0, gx1 = -w/2, w/2
        gz0, gz1 = 0, h
        p = np.array([
            [gx0, y0, gz0], [gx1, y0, gz0], [gx1, y0, gz1], [gx0, y0, gz1],
        ], dtype=np.float32)
        pts = [p[0],p[1], p[1],p[2], p[2],p[3], p[3],p[0], p[0],p[2]]
        item = gl.GLLinePlotItem(antialias=True)
        item.setData(pos=np.array(pts), color=color, width=4, mode='lines')
        view.addItem(item)


def add_pads(view, arena):
    pads = arena.get_boost_pads()
    big_xyz, small_xyz = [], []
    big_idx, small_idx = [], []
    for i, p in enumerate(pads):
        pos = p.get_pos()
        xyz = [-pos.x, pos.y, 30.0]
        if p.is_big:
            big_xyz.append(xyz); big_idx.append(i)
        else:
            small_xyz.append(xyz); small_idx.append(i)
    big_xyz = np.array(big_xyz, dtype=np.float32) if big_xyz else np.zeros((0,3), np.float32)
    small_xyz = np.array(small_xyz, dtype=np.float32) if small_xyz else np.zeros((0,3), np.float32)

    big = gl.GLScatterPlotItem()
    big.setData(pos=big_xyz, color=(1.0, 0.95, 0.0, 1.0), size=22)
    view.addItem(big)

    small = gl.GLScatterPlotItem()
    small.setData(pos=small_xyz, color=(1.0, 0.85, 0.0, 0.5), size=10)
    view.addItem(small)

    return big, small, big_idx, small_idx, pads


def update_pad_colors(big, small, big_idx, small_idx, pads):
    pad_states = [p.get_state().is_active for p in pads]
    if getattr(update_pad_colors, "_last_pad_states", None) == pad_states:
        return
    update_pad_colors._last_pad_states = list(pad_states)
    bc = np.zeros((len(big_idx), 4), dtype=np.float32)
    for k, idx in enumerate(big_idx):
        bc[k] = (1.0, 0.95, 0.0, 1.0) if pad_states[idx] else (0.3, 0.3, 0.1, 0.4)
    big.setData(pos=big.pos, color=bc, size=22)

    sc = np.zeros((len(small_idx), 4), dtype=np.float32)
    for k, idx in enumerate(small_idx):
        sc[k] = (1.0, 0.85, 0.0, 0.6) if pad_states[idx] else (0.2, 0.2, 0.1, 0.25)
    small.setData(pos=small.pos, color=sc, size=10)


class BallGuideLinesRenderer:
    """Draws 6 spatial reference line segments from the Ball to the 6 arena boundary planes."""

    def __init__(self, view, x_bound=4000.0, y_bound=5120.0, z_min=0.0, z_max=2000.0):
        self.x_bound = float(x_bound)
        self.y_bound = float(y_bound)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.line_color = (0.78, 0.82, 0.86, 0.28)
        self.dot_color = (0.82, 0.86, 0.90, 0.48)
        self.lines = gl.GLLinePlotItem(antialias=True)
        view.addItem(self.lines)
        self.dots = gl.GLScatterPlotItem()
        self.dots.setData(pos=np.zeros((0, 3), dtype=np.float32), color=self.dot_color, size=7)
        view.addItem(self.dots)

    def update(self, ball_pos_r):
        bx = float(np.clip(ball_pos_r[0], -self.x_bound, self.x_bound))
        by = float(np.clip(ball_pos_r[1], -self.y_bound, self.y_bound))
        bz = float(np.clip(ball_pos_r[2], self.z_min, self.z_max))
        b = [bx, by, bz]

        endpoints = np.array([
            [-self.x_bound, by, bz],
            [ self.x_bound, by, bz],
            [bx, -self.y_bound, bz],
            [bx,  self.y_bound, bz],
            [bx, by, self.z_min],
            [bx, by, self.z_max],
        ], dtype=np.float32)

        seg_pts = np.array([
            b, endpoints[0],
            b, endpoints[1],
            b, endpoints[2],
            b, endpoints[3],
            b, endpoints[4],
            b, endpoints[5],
        ], dtype=np.float32)

        self.lines.setData(pos=seg_pts, color=self.line_color, width=1.5, mode='lines')
        self.dots.setData(pos=endpoints, color=self.dot_color, size=7)


class TrailRenderer:
    """Renders 3D trajectory trail (Cyan/Team=coasting, Gold=boost) + rising-edge impulse dots for Jump & DoubleJump/Flip.
    Uses bounded deques with synchronized step & wall-clock expiration so jump_pts, djump_pts, and trail line pts
    share the exact same lifetime horizon and automatically expire without memory overflow.
    """

    def __init__(self, view, max_len=600, max_age_s=6.0, max_age_sec=None, coast_color=(0.0, 0.8, 1.0, 0.60)):
        from collections import deque
        import time as _time
        self._time_fn = _time.monotonic
        self.line = gl.GLLinePlotItem(antialias=True)
        view.addItem(self.line)
        self.jump_scatter = gl.GLScatterPlotItem()
        self.jump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32), color=(0.1, 1.0, 0.3, 1.0), size=14)
        view.addItem(self.jump_scatter)
        self.djump_scatter = gl.GLScatterPlotItem()
        self.djump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32), color=(1.0, 0.15, 0.95, 1.0), size=18)
        view.addItem(self.djump_scatter)

        self.max_len = max(2, int(max_len))
        age_val = max_age_sec if max_age_sec is not None else max_age_s
        self.max_age_s = float(age_val) if age_val is not None else None
        self.coast_color = tuple(float(c) for c in coast_color)

        self.pts = deque(maxlen=self.max_len)
        self._pts_meta = deque(maxlen=self.max_len)      # (step_id, timestamp)
        self.jump_pts = deque(maxlen=self.max_len)
        self._jump_meta = deque(maxlen=self.max_len)     # (step_id, timestamp)
        self.djump_pts = deque(maxlen=self.max_len)
        self._djump_meta = deque(maxlen=self.max_len)    # (step_id, timestamp)

        self._step_counter = 0
        self._prev_jumping = False
        self._prev_djump = False

    def _evict_expired(self, now_t: float):
        min_step = self._step_counter - self.max_len
        min_time = (now_t - self.max_age_s) if (self.max_age_s is not None and self.max_age_s > 0.0) else None

        pts_changed = False
        while self._pts_meta and (
            self._pts_meta[0][0] <= min_step
            or (min_time is not None and self._pts_meta[0][1] < min_time)
        ):
            self._pts_meta.popleft()
            self.pts.popleft()
            pts_changed = True

        jump_changed = False
        while self._jump_meta and (
            self._jump_meta[0][0] <= min_step
            or (min_time is not None and self._jump_meta[0][1] < min_time)
        ):
            self._jump_meta.popleft()
            self.jump_pts.popleft()
            jump_changed = True

        djump_changed = False
        while self._djump_meta and (
            self._djump_meta[0][0] <= min_step
            or (min_time is not None and self._djump_meta[0][1] < min_time)
        ):
            self._djump_meta.popleft()
            self.djump_pts.popleft()
            djump_changed = True

        return pts_changed, jump_changed, djump_changed

    def _sync_jump_scatter(self):
        if self.jump_pts:
            self.jump_scatter.setData(
                pos=np.array(self.jump_pts, dtype=np.float32),
                color=(0.1, 1.0, 0.3, 1.0),
                size=14,
            )
        else:
            self.jump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def _sync_djump_scatter(self):
        if self.djump_pts:
            self.djump_scatter.setData(
                pos=np.array(self.djump_pts, dtype=np.float32),
                color=(1.0, 0.15, 0.95, 1.0),
                size=18,
            )
        else:
            self.djump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def _sync_line(self):
        if len(self.pts) > 1:
            tp = np.array(self.pts, dtype=np.float32)
            col = np.zeros((len(tp), 4), dtype=np.float32)
            col[:] = self.coast_color
            bm = tp[:, 3] > 0.5
            col[bm] = (1.0, 0.85, 0.0, 0.95)
            self.line.setData(pos=tp[:, :3], color=col, width=4)
        else:
            self.line.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def push(self, xyz, boosting=0.0, jumping=False, djump_or_flip=False, now=None):
        self._step_counter += 1
        now_t = self._time_fn() if now is None else float(now)
        b_flag = 1.0 if float(boosting) > 0.5 else 0.0

        pts_evicted, jump_evicted, djump_evicted = self._evict_expired(now_t)

        jump_added = False
        if jumping and not self._prev_jumping:
            self.jump_pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            self._jump_meta.append((self._step_counter, now_t))
            jump_added = True

        djump_added = False
        if djump_or_flip and not self._prev_djump:
            self.djump_pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            self._djump_meta.append((self._step_counter, now_t))
            djump_added = True

        self._prev_jumping = bool(jumping)
        self._prev_djump = bool(djump_or_flip)

        if jump_added or jump_evicted:
            self._sync_jump_scatter()
        if djump_added or djump_evicted:
            self._sync_djump_scatter()

        if self.pts:
            last = self.pts[-1]
            if (abs(xyz[0] - last[0]) < 0.1 and
                abs(xyz[1] - last[1]) < 0.1 and
                abs(xyz[2] - last[2]) < 0.1 and
                abs(b_flag - last[3]) < 0.1):
                # Refresh timestamp/step of stationary car tail and redraw if old points expired
                self._pts_meta[-1] = (self._step_counter, now_t)
                if pts_evicted:
                    self._sync_line()
                return

        self.pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2]), b_flag))
        self._pts_meta.append((self._step_counter, now_t))
        self._sync_line()

    def clear(self):
        self.pts.clear()
        self._pts_meta.clear()
        self.jump_pts.clear()
        self._jump_meta.clear()
        self.djump_pts.clear()
        self._djump_meta.clear()
        self._step_counter = 0
        self._prev_jumping = False
        self._prev_djump = False
        self.line.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.jump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.djump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))


class CarHitboxRenderer:
    """Renders Octane 3D collision hitbox as a semi-transparent container + crisp wireframe edges
    (matching the brightness of BallGuideLinesRenderer) + 3D orientation axes to eliminate POV optical illusions.
    Octane hitbox: size=(120.51, 86.70, 38.66), offset=(13.88, 0.0, 20.75) in car local frame.
    """
    HALF_SIZE = (60.255, 43.35, 19.33)
    OFFSET = (13.88, 0.0, 20.75)

    # 8 vertices in local (fwd, rgt, up) signs
    _LOCAL_SIGNS = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float32)

    # 12 box edges
    _EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    # 12 triangles (6 faces)
    _FACES = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 6, 5], [4, 7, 6],  # Top
        [1, 5, 6], [1, 6, 2],  # Front (+fwd)
        [0, 3, 7], [0, 7, 4],  # Back (-fwd)
        [3, 2, 6], [3, 6, 7],  # Right (+rgt)
        [0, 4, 5], [0, 5, 1],  # Left (-rgt)
    ], dtype=np.int32)

    def __init__(self, view, team=None):
        init_verts = np.zeros((8, 3), dtype=np.float32)
        self._face_colors = self._make_face_colors(team)

        self.mesh_item = gl.GLMeshItem(
            vertexes=init_verts,
            faces=self._FACES,
            faceColors=self._face_colors,
            smooth=False,
            drawEdges=False,
            glOptions="translucent",
        )
        view.addItem(self.mesh_item)

        if team == 0:
            self.edge_color = (0.35, 0.80, 1.0, 0.80)
        elif team == 1:
            self.edge_color = (1.0, 0.58, 0.22, 0.80)
        else:
            self.edge_color = (0.82, 0.86, 0.90, 0.72)
        self.edge_lines = gl.GLLinePlotItem(antialias=True)
        view.addItem(self.edge_lines)

        self.fwd = gl.GLLinePlotItem(antialias=True); view.addItem(self.fwd)
        self.up  = gl.GLLinePlotItem(antialias=True); view.addItem(self.up)
        self.rgt = gl.GLLinePlotItem(antialias=True); view.addItem(self.rgt)

    @staticmethod
    def _make_face_colors(team=None):
        if team == 0:
            fc = np.full((12, 4), (0.15, 0.52, 0.95, 0.16), dtype=np.float32)
            fc[4] = (0.10, 0.95, 1.0, 0.30)
            fc[5] = (0.10, 0.95, 1.0, 0.30)
            fc[2] = (0.30, 0.70, 1.0, 0.22)
            fc[3] = (0.30, 0.70, 1.0, 0.22)
            return fc
        if team == 1:
            fc = np.full((12, 4), (0.95, 0.42, 0.12, 0.16), dtype=np.float32)
            fc[4] = (1.0, 0.85, 0.15, 0.30)
            fc[5] = (1.0, 0.85, 0.15, 0.30)
            fc[2] = (1.0, 0.60, 0.25, 0.22)
            fc[3] = (1.0, 0.60, 0.25, 0.22)
            return fc
        fc = np.full((12, 4), (0.78, 0.82, 0.86, 0.14), dtype=np.float32)
        fc[4] = (0.20, 0.90, 0.85, 0.24)
        fc[5] = (0.20, 0.90, 0.85, 0.24)
        fc[2] = (0.82, 0.86, 0.90, 0.18)
        fc[3] = (0.82, 0.86, 0.90, 0.18)
        return fc

    def update(self, pos_r, fwd_r, up_r, rgt_r, length=240.0, edge_color_override=None, edge_width=2.0):
        p = np.asarray(pos_r, dtype=np.float32)
        f = np.asarray(fwd_r, dtype=np.float32)
        u = np.asarray(up_r, dtype=np.float32)
        r = np.asarray(rgt_r, dtype=np.float32)

        ox, oy, oz = self.OFFSET
        hx, hy, hz = self.HALF_SIZE
        center = p + ox * f + oy * r + oz * u

        s = self._LOCAL_SIGNS
        verts = (
            center[None, :]
            + (s[:, 0:1] * hx) * f[None, :]
            + (s[:, 1:2] * hy) * r[None, :]
            + (s[:, 2:3] * hz) * u[None, :]
        ).astype(np.float32)

        self.mesh_item.setMeshData(
            vertexes=verts,
            faces=self._FACES,
            faceColors=self._face_colors,
        )

        edge_pts = np.empty((len(self._EDGES) * 2, 3), dtype=np.float32)
        for idx, (i0, i1) in enumerate(self._EDGES):
            edge_pts[2 * idx] = verts[i0]
            edge_pts[2 * idx + 1] = verts[i1]
        ec = edge_color_override if edge_color_override is not None else self.edge_color
        self.edge_lines.setData(pos=edge_pts, color=ec, width=float(edge_width), mode="lines")

        self.fwd.setData(pos=np.array([center, center + length * f]), color=(0, 1, 0, 0.9), width=3)
        self.up.setData( pos=np.array([center, center + 0.75 * length * u]), color=(1, 0, 0, 0.9), width=3)
        self.rgt.setData(pos=np.array([center, center + 0.50 * length * r]), color=(0, 0.5, 1, 0.9), width=3)

    def hide(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        self.edge_lines.setData(pos=empty)
        self.fwd.setData(pos=empty)
        self.up.setData(pos=empty)
        self.rgt.setData(pos=empty)


# Keep CarAxesRenderer alias pointing to CarHitboxRenderer for drop-in compatibility
CarAxesRenderer = CarHitboxRenderer


class VelocityArrowRenderer:
    def __init__(self, view, color=(1.0, 0.9, 0.1, 0.95), width=4):
        self.color = color
        self.width = width
        self.line = gl.GLLinePlotItem(antialias=True)
        view.addItem(self.line)

    def update(self, pos_r, vel_r, fallback_dir_r=None, min_len=0.0, max_len=1200.0, scale=0.45, show=True):
        if not show:
            self.hide()
            return
        spd = float(np.linalg.norm(vel_r))
        if spd > 1.0:
            dir_u = vel_r / spd
        elif fallback_dir_r is not None:
            fn = float(np.linalg.norm(fallback_dir_r))
            if fn < 1e-6:
                self.hide()
                return
            dir_u = fallback_dir_r / fn
        else:
            self.hide()
            return

        length = min(max_len, max(min_len, min_len + spd * scale))
        if length < 1.0:
            self.hide()
            return

        tip = pos_r + dir_u * length
        up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        side = np.cross(dir_u, up_ref)
        sn = float(np.linalg.norm(side))
        if sn < 1e-4:
            side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            side = side / sn

        head_len = min(140.0, max(45.0, length * 0.25))
        head_w   = head_len * 0.45
        w1 = tip - dir_u * head_len + side * head_w
        w2 = tip - dir_u * head_len - side * head_w

        pts = np.array([pos_r, tip, tip, w1, tip, w2], dtype=np.float32)
        self.line.setData(pos=pts, color=self.color, width=self.width, mode='lines')

    def hide(self):
        self.line.setData(pos=np.zeros((0, 3), dtype=np.float32))


def setup_standard_camera(view, distance=5500):
    view.setCameraPosition(distance=distance, elevation=30, azimuth=45)
    if hasattr(view, "center_params"):
        view.center_params["distance"] = float(distance)
        view.center_params["elevation"] = 30.0
        view.center_params["azimuth"] = 45.0

