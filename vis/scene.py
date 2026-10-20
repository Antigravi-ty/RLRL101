"""vis/scene.py —— 3D 球场渲染组件、轨迹点线、车辆碰撞箱、地面 1s/2s/3s 可达等时线与场外 3D 非规则平滑曲面立体物。"""
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


class IrregularSmoothBlobRenderer:
    """Renders a 3D non-spherical, organic smooth surface outside the arena to verify 3D spatial depth perception.

    Combines:
      1. Smooth vertex-shaded translucent 3D mesh (with directional/normal-based face shading so curvature is vivid)
      2. Surface-hugging latitude/longitude contour rings so human eyes immediately perceive 3D volume from any angle
      3. Ground pedestal and bounding reference lines connecting the floating 3D volume to the z=0 plane
    """

    def __init__(self, view, center_r=(-5400.0, 0.0, 950.0), base_radius=680.0, n_lat=32, n_lon=64):
        self.center_r = np.asarray(center_r, dtype=np.float32)
        self.base_radius = float(base_radius)
        self.n_lat = int(n_lat)
        self.n_lon = int(n_lon)

        verts, faces, face_colors, contour_segments = self._build_organic_geometry()
        self.mesh_item = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=face_colors,
            smooth=True,
            drawEdges=False,
            glOptions="translucent",
        )
        view.addItem(self.mesh_item)

        self.contour_lines = gl.GLLinePlotItem(antialias=True)
        self.contour_lines.setData(
            pos=contour_segments,
            color=(0.35, 0.95, 1.0, 0.52),
            width=1.6,
            mode="lines",
        )
        view.addItem(self.contour_lines)

        # Pedestal / anchor lines to ground so its 3D position outside the field is crystal clear
        cx, cy, cz = float(self.center_r[0]), float(self.center_r[1]), float(self.center_r[2])
        anchor_pts = np.array([
            [cx, cy, 0.0], [cx, cy, cz],
            [cx - 750.0, cy, 2.0], [cx + 750.0, cy, 2.0],
            [cx, cy - 750.0, 2.0], [cx, cy + 750.0, 2.0],
        ], dtype=np.float32)
        self.pedestal_lines = gl.GLLinePlotItem(antialias=True)
        self.pedestal_lines.setData(
            pos=anchor_pts,
            color=(0.45, 0.85, 0.95, 0.40),
            width=2.0,
            mode="lines",
        )
        view.addItem(self.pedestal_lines)

    def _radius_fn(self, theta, phi):
        """Anisotropic 3D organic envelope (resembling a multi-lobed asymmetric 3D reachable cloud)."""
        # theta in [0, pi] (polar angle from +Z), phi in [0, 2*pi) (azimuth)
        st = np.sin(theta)
        ct = np.cos(theta)
        # Forward/backward asymmetry + kidney/teardrop lobes + vertical twist
        r = (
            1.00
            + 0.34 * st * np.cos(phi)
            - 0.22 * (st ** 2) * np.cos(2.0 * phi)
            + 0.18 * (st ** 3) * np.sin(3.0 * phi + 0.4)
            + 0.20 * np.sin(2.0 * theta) * np.cos(phi - 0.5)
            + 0.12 * np.cos(3.0 * theta)
        )
        return self.base_radius * np.maximum(0.35, r)

    def _build_organic_geometry(self):
        thetas = np.linspace(0.0, math.pi, self.n_lat + 1, dtype=np.float32)
        phis = np.linspace(0.0, 2.0 * math.pi, self.n_lon, endpoint=False, dtype=np.float32)
        T, P = np.meshgrid(thetas, phis, indexing="ij")  # (n_lat+1, n_lon)
        R = self._radius_fn(T, P)

        # Asymmetric forward stretch (x/y/z scaling & shear) so it is unmistakably non-spherical
        X = R * np.sin(T) * np.cos(P) * 1.15
        Y = R * np.sin(T) * np.sin(P) * 1.35 + 140.0 * np.cos(T)
        Z = R * np.cos(T) * 0.88 + 90.0 * np.sin(T) * np.cos(2.0 * P)

        grid_pts = np.stack([
            self.center_r[0] + X,
            self.center_r[1] + Y,
            self.center_r[2] + Z,
        ], axis=-1).astype(np.float32)  # (n_lat+1, n_lon, 3)

        verts = grid_pts.reshape(-1, 3)
        faces = []
        for i in range(self.n_lat):
            for j in range(self.n_lon):
                jn = (j + 1) % self.n_lon
                v00 = i * self.n_lon + j
                v01 = i * self.n_lon + jn
                v10 = (i + 1) * self.n_lon + j
                v11 = (i + 1) * self.n_lon + jn
                if i > 0:
                    faces.append([v00, v10, v11])
                if i < self.n_lat - 1:
                    faces.append([v00, v11, v01])
        faces = np.asarray(faces, dtype=np.int32)

        # Compute per-triangle 3D lighting (key light + sky hemisphere + rim effect) for strong depth perception
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(n_len, 1e-6)

        light_dir = np.array([0.45, -0.55, 0.70], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        diff = np.clip(normals @ light_dir, 0.0, 1.0)
        hemi = 0.5 * (normals[:, 2] + 1.0)
        shade = 0.25 + 0.55 * diff + 0.20 * hemi

        face_colors = np.zeros((len(faces), 4), dtype=np.float32)
        # Rich cyan-emerald-violet gradient by normal orientation & shading
        face_colors[:, 0] = np.clip(0.10 + 0.45 * (1.0 - hemi) + 0.25 * diff, 0.05, 1.0)
        face_colors[:, 1] = np.clip(0.55 * shade + 0.25 * hemi, 0.15, 1.0)
        face_colors[:, 2] = np.clip(0.85 * shade + 0.15, 0.25, 1.0)
        face_colors[:, 3] = 0.62

        # Build surface contour rings (latitude slices + 8 meridian ribs) for crisp 3D curvature cues
        seg_list = []
        for i in range(2, self.n_lat - 1, 3):
            ring = grid_pts[i]
            for j in range(self.n_lon):
                seg_list.append(ring[j])
                seg_list.append(ring[(j + 1) % self.n_lon])
        for j in range(0, self.n_lon, 8):
            mer = grid_pts[:, j, :]
            for i in range(self.n_lat):
                seg_list.append(mer[i])
                seg_list.append(mer[i + 1])

        contour_segments = np.asarray(seg_list, dtype=np.float32)
        return verts, faces, face_colors, contour_segments


class GroundIsochroneRenderer:
    """Renders 1s, 2s, and 3s ground reachable boundary contours (isochrones) + translucent ground fan bands."""

    COLORS_LINE = (
        (0.05, 1.00, 0.65, 0.95),  # 1.0s: Vivid Spring Cyan-Green
        (1.00, 0.85, 0.10, 0.92),  # 2.0s: Warm Gold
        (1.00, 0.28, 0.65, 0.88),  # 3.0s: Electric Rose-Magenta
    )
    COLORS_FILL = (
        (0.05, 0.95, 0.65, 0.11),  # 1.0s inner disc
        (1.00, 0.82, 0.10, 0.08),  # 1.0s -> 2.0s band
        (1.00, 0.25, 0.62, 0.06),  # 2.0s -> 3.0s band
    )

    def __init__(self, view):
        self.lines = []
        self.fills = []
        for h in range(3):
            mesh = gl.GLMeshItem(
                vertexes=np.zeros((3, 3), dtype=np.float32),
                faces=np.zeros((1, 3), dtype=np.int32),
                color=self.COLORS_FILL[h],
                smooth=False,
                drawEdges=False,
                glOptions="translucent",
            )
            view.addItem(mesh)
            self.fills.append(mesh)

        for h in range(3):
            item = gl.GLLinePlotItem(antialias=True)
            item.setData(pos=np.zeros((0, 3), dtype=np.float32), color=self.COLORS_LINE[h], width=3.0 - 0.3 * h)
            view.addItem(item)
            self.lines.append(item)

        self.visible = True

    def update(self, car_pos_w, contours_w, on_ground=True, up_dir_w=None):
        if not self.visible or (not on_ground) or not contours_w:
            self.hide()
            return

        car_r = r2r(car_pos_w)
        if up_dir_w is not None:
            up_r = vec_r2r_tuple(up_dir_w)
            un = float(np.linalg.norm(up_r))
            if un > 1e-4:
                car_r += (up_r / un) * 6.0
        else:
            car_r[2] = max(car_r[2], 18.0) + 6.0
        prev_ring_r = None

        for h in range(3):
            c_w = contours_w[h]
            c_r = np.empty_like(c_w)
            c_r[:, 0] = -c_w[:, 0]
            c_r[:, 1] = c_w[:, 1]
            c_r[:, 2] = c_w[:, 2]
            self.lines[h].setData(pos=c_r, color=self.COLORS_LINE[h], width=3.2 - 0.35 * h)

            ring_r = c_r[:-1]  # (N, 3) open ring
            n_pts = len(ring_r)
            if n_pts >= 3:
                if prev_ring_r is None:
                    # Triangle fan from car_r to 1s ring
                    verts = np.vstack([car_r.reshape(1, 3), ring_r]).astype(np.float32)
                    idx_i = np.arange(1, n_pts + 1, dtype=np.int32)
                    idx_next = np.roll(idx_i, -1)
                    faces = np.stack([np.zeros(n_pts, dtype=np.int32), idx_i, idx_next], axis=1)
                else:
                    # Annular strip between prev_ring_r and ring_r
                    verts = np.vstack([prev_ring_r, ring_r]).astype(np.float32)
                    i0 = np.arange(n_pts, dtype=np.int32)
                    i1 = np.roll(i0, -1)
                    j0 = i0 + n_pts
                    j1 = i1 + n_pts
                    f1 = np.stack([i0, j0, j1], axis=1)
                    f2 = np.stack([i0, j1, i1], axis=1)
                    faces = np.vstack([f1, f2])
                self.fills[h].setMeshData(vertexes=verts, faces=faces, color=self.COLORS_FILL[h])
                prev_ring_r = ring_r

    def hide(self):
        empty_pos = np.zeros((0, 3), dtype=np.float32)
        empty_v = np.zeros((3, 3), dtype=np.float32)
        empty_f = np.zeros((0, 3), dtype=np.int32)
        for h in range(3):
            self.lines[h].setData(pos=empty_pos)
            self.fills[h].setMeshData(vertexes=empty_v, faces=empty_f)


class AerialIsochroneRenderer:
    """Renders 1s/2s/3s continuous 3D reachability funnel / horn or discrete bubbles with normal lighting & rings."""

    COLORS_RIB = (
        (0.10, 1.00, 0.70, 0.85),  # 1.0s: Spring Cyan
        (1.00, 0.85, 0.15, 0.80),  # 2.0s: Warm Gold
        (1.00, 0.30, 0.70, 0.75),  # 3.0s: Electric Rose
    )

    def __init__(self, view):
        self.view = view
        self.meshes = []
        self.rib_lines = []

        for h in range(3):
            mesh = gl.GLMeshItem(
                vertexes=np.zeros((3, 3), dtype=np.float32),
                faces=np.zeros((1, 3), dtype=np.int32),
                smooth=False,
                drawEdges=False,
                glOptions="translucent",
            )
            view.addItem(mesh)
            self.meshes.append(mesh)

            rib = gl.GLLinePlotItem(antialias=True)
            rib.setData(pos=np.zeros((0, 3), dtype=np.float32), color=self.COLORS_RIB[h], width=2.0)
            view.addItem(rib)
            self.rib_lines.append(rib)

        # Continuous 3D Funnel / Horn items
        self.funnel_mesh = gl.GLMeshItem(
            vertexes=np.zeros((3, 3), dtype=np.float32),
            faces=np.zeros((1, 3), dtype=np.int32),
            smooth=False,
            drawEdges=False,
            glOptions="translucent",
        )
        view.addItem(self.funnel_mesh)

        self.funnel_ring_lines = []
        for h in range(3):
            ring_line = gl.GLLinePlotItem(antialias=True)
            ring_line.setData(pos=np.zeros((0, 3), dtype=np.float32), color=self.COLORS_RIB[h], width=2.8 - 0.3 * h)
            view.addItem(ring_line)
            self.funnel_ring_lines.append(ring_line)

        self.funnel_rail_line = gl.GLLinePlotItem(antialias=True)
        self.funnel_rail_line.setData(pos=np.zeros((0, 3), dtype=np.float32), color=(0.85, 0.90, 0.95, 0.35), width=1.5, mode="lines")
        view.addItem(self.funnel_rail_line)

        self.visible = True

    def hide_funnel(self):
        empty_v = np.zeros((3, 3), dtype=np.float32)
        empty_f = np.zeros((0, 3), dtype=np.int32)
        empty_p = np.zeros((0, 3), dtype=np.float32)
        self.funnel_mesh.setMeshData(vertexes=empty_v, faces=empty_f)
        for rl in self.funnel_ring_lines:
            rl.setData(pos=empty_p)
        self.funnel_rail_line.setData(pos=empty_p)

    def hide_bubbles(self):
        empty_v = np.zeros((3, 3), dtype=np.float32)
        empty_f = np.zeros((0, 3), dtype=np.int32)
        empty_p = np.zeros((0, 3), dtype=np.float32)
        for h in range(3):
            self.meshes[h].setMeshData(vertexes=empty_v, faces=empty_f)
            self.rib_lines[h].setData(pos=empty_p)

    def hide(self):
        self.hide_bubbles()
        self.hide_funnel()

    def update_funnel(self, funnel_data):
        if not self.visible or funnel_data is None:
            self.hide()
            return
        self.hide_bubbles()

        w_verts = funnel_data["verts"]
        gl_verts = w_verts.copy()
        gl_verts[:, 0] = -gl_verts[:, 0]
        self.funnel_mesh.setMeshData(
            vertexes=gl_verts,
            faces=funnel_data["faces"],
            faceColors=funnel_data["face_colors"],
        )

        for h, ring_w in enumerate(funnel_data.get("rings", [])):
            if h < len(self.funnel_ring_lines):
                gl_ring = ring_w.copy()
                gl_ring[:, 0] = -gl_ring[:, 0]
                self.funnel_ring_lines[h].setData(pos=gl_ring, color=self.COLORS_RIB[h], width=2.8 - 0.3 * h)

        rails_w = funnel_data.get("rails")
        if rails_w is not None and len(rails_w) > 0:
            gl_rails = rails_w.copy()
            gl_rails[:, 0] = -gl_rails[:, 0]
            self.funnel_rail_line.setData(pos=gl_rails, color=(0.85, 0.90, 0.95, 0.35), width=1.5, mode="lines")
        else:
            self.funnel_rail_line.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def update_bubbles(self, bubble_data_list, mode="ALL"):
        if not self.visible or mode == "OFF" or not bubble_data_list:
            self.hide()
            return
        self.hide_funnel()

        active_indices = (0, 1, 2) if mode in ("ALL", "BUBBLES") else ((0, 1) if mode == "1s+2s" else (0,))

        for h in range(3):
            if h in active_indices and h < len(bubble_data_list):
                bdata = bubble_data_list[h]
                w_verts = bdata["verts"]
                gl_verts = w_verts.copy()
                gl_verts[:, 0] = -gl_verts[:, 0]
                self.meshes[h].setMeshData(
                    vertexes=gl_verts,
                    faces=bdata["faces"],
                    faceColors=bdata["face_colors"],
                )

                w_ribs = bdata["ribs"]
                gl_ribs = w_ribs.copy()
                gl_ribs[:, 0] = -gl_ribs[:, 0]
                self.rib_lines[h].setData(pos=gl_ribs, color=self.COLORS_RIB[h], width=2.0)
            else:
                empty_v = np.zeros((3, 3), dtype=np.float32)
                empty_f = np.zeros((0, 3), dtype=np.int32)
                self.meshes[h].setMeshData(vertexes=empty_v, faces=empty_f)
                self.rib_lines[h].setData(pos=np.zeros((0, 3), dtype=np.float32))

    def update(self, bubble_data_list, mode="ALL"):
        self.update_bubbles(bubble_data_list, mode=mode)


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
    """Renders 3D trajectory trail (Cyan=coasting, Gold=boost) + rising-edge impulse dots for Jump & DoubleJump/Flip."""

    def __init__(self, view, max_len=600):
        self.line = gl.GLLinePlotItem(antialias=True)
        view.addItem(self.line)
        self.jump_scatter = gl.GLScatterPlotItem()
        self.jump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32), color=(0.1, 1.0, 0.3, 1.0), size=14)
        view.addItem(self.jump_scatter)
        self.djump_scatter = gl.GLScatterPlotItem()
        self.djump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32), color=(1.0, 0.15, 0.95, 1.0), size=18)
        view.addItem(self.djump_scatter)

        self.pts = []
        self.jump_pts = []
        self.djump_pts = []
        self._prev_jumping = False
        self._prev_djump = False
        self.max_len = max_len

    def push(self, xyz, boosting=0.0, jumping=False, djump_or_flip=False):
        b_flag = 1.0 if float(boosting) > 0.5 else 0.0

        if jumping and not self._prev_jumping:
            self.jump_pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            self.jump_scatter.setData(
                pos=np.array(self.jump_pts, dtype=np.float32),
                color=(0.1, 1.0, 0.3, 1.0),
                size=14,
            )
        if djump_or_flip and not self._prev_djump:
            self.djump_pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
            self.djump_scatter.setData(
                pos=np.array(self.djump_pts, dtype=np.float32),
                color=(1.0, 0.15, 0.95, 1.0),
                size=18,
            )
        self._prev_jumping = bool(jumping)
        self._prev_djump = bool(djump_or_flip)

        if self.pts:
            last = self.pts[-1]
            if (abs(xyz[0] - last[0]) < 0.1 and
                abs(xyz[1] - last[1]) < 0.1 and
                abs(xyz[2] - last[2]) < 0.1 and
                abs(b_flag - last[3]) < 0.1):
                return

        self.pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2]), b_flag))
        if len(self.pts) > self.max_len:
            self.pts.pop(0)
        if len(self.pts) > 1:
            tp = np.array(self.pts, dtype=np.float32)
            col = np.zeros((len(tp), 4), dtype=np.float32)
            col[:] = (0.0, 0.8, 1.0, 0.60)
            bm = tp[:, 3] > 0.5
            col[bm] = (1.0, 0.85, 0.0, 0.95)
            self.line.setData(pos=tp[:, :3], color=col, width=4)

    def clear(self):
        self.pts = []
        self.jump_pts = []
        self.djump_pts = []
        self._prev_jumping = False
        self._prev_djump = False
        self.line.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.jump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))
        self.djump_scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))


class CarHitboxRenderer:
    """Renders Octane 3D collision hitbox as a semi-transparent container + crisp wireframe edges + 3D axes."""
    HALF_SIZE = (60.255, 43.35, 19.33)
    OFFSET = (13.88, 0.0, 20.75)

    _LOCAL_SIGNS = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=np.float32)

    _EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    _FACES = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [1, 5, 6], [1, 6, 2],
        [0, 3, 7], [0, 7, 4],
        [3, 2, 6], [3, 6, 7],
        [0, 4, 5], [0, 5, 1],
    ], dtype=np.int32)

    def __init__(self, view):
        init_verts = np.zeros((8, 3), dtype=np.float32)
        face_colors = np.full((12, 4), (0.78, 0.82, 0.86, 0.14), dtype=np.float32)
        face_colors[4] = (0.20, 0.90, 0.85, 0.24)
        face_colors[5] = (0.20, 0.90, 0.85, 0.24)
        face_colors[2] = (0.82, 0.86, 0.90, 0.18)
        face_colors[3] = (0.82, 0.86, 0.90, 0.18)
        self._face_colors = face_colors

        self.mesh_item = gl.GLMeshItem(
            vertexes=init_verts,
            faces=self._FACES,
            faceColors=self._face_colors,
            smooth=False,
            drawEdges=False,
            glOptions="translucent",
        )
        view.addItem(self.mesh_item)

        self.edge_color = (0.82, 0.86, 0.90, 0.72)
        self.edge_lines = gl.GLLinePlotItem(antialias=True)
        view.addItem(self.edge_lines)

        self.fwd = gl.GLLinePlotItem(antialias=True); view.addItem(self.fwd)
        self.up  = gl.GLLinePlotItem(antialias=True); view.addItem(self.up)
        self.rgt = gl.GLLinePlotItem(antialias=True); view.addItem(self.rgt)

    def update(self, pos_r, fwd_r, up_r, rgt_r, length=240.0):
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
        self.edge_lines.setData(pos=edge_pts, color=self.edge_color, width=2.0, mode="lines")

        self.fwd.setData(pos=np.array([center, center + length * f]), color=(0, 1, 0, 0.9), width=3)
        self.up.setData( pos=np.array([center, center + 0.75 * length * u]), color=(1, 0, 0, 0.9), width=3)
        self.rgt.setData(pos=np.array([center, center + 0.50 * length * r]), color=(0, 0.5, 1, 0.9), width=3)


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
