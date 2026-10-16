"""
所有可视化工具共用的场景构建。
- 坐标转换
- 地板/边界/球门
- Boost pads
- 轨迹（带 boost 高亮）
- 车头三轴指示器
"""
import numpy as np
import pyqtgraph.opengl as gl


def r2r(p):
    """RocketSim (x,y,z) → render (-x,y,z)"""
    return np.array([-p[0], p[1], p[2]], dtype=np.float32)


def vec_r2r(v):
    return np.array([-v.x, v.y, v.z], dtype=np.float32)

def vec_r2r_tuple(t):
    """(x,y,z) 元组 → render 坐标"""
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
    """返回 (big_scatter, small_scatter, big_idx, small_idx, pad_refs)"""
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


class TrailRenderer:
    """带 boost 金色高亮的轨迹。"""
    def __init__(self, view, max_len=400):
        self.line = gl.GLLinePlotItem(antialias=True)
        view.addItem(self.line)
        self.pts = []   # (x, y, z, boosting)
        self.max_len = max_len

    def push(self, xyz, boosting):
        if self.pts:
            last = self.pts[-1]
            if (abs(xyz[0] - last[0]) < 0.1 and
                abs(xyz[1] - last[1]) < 0.1 and
                abs(xyz[2] - last[2]) < 0.1 and
                abs(boosting - last[3]) < 0.1):
                return
        self.pts.append((float(xyz[0]), float(xyz[1]), float(xyz[2]), float(boosting)))
        if len(self.pts) > self.max_len:
            self.pts.pop(0)
        if len(self.pts) > 1:
            tp = np.array(self.pts, dtype=np.float32)
            col = np.zeros((len(tp), 4), dtype=np.float32)
            bm = tp[:, 3] > 0.5
            col[~bm] = (0.0, 0.8, 1.0, 0.55)
            col[bm]  = (1.0, 0.8, 0.0, 0.95)
            self.line.setData(pos=tp[:, :3], color=col, width=4)

    def clear(self):
        self.pts = []
        self.line.setData(pos=np.zeros((0, 3)))


class CarAxesRenderer:
    """车头三轴指示器：forward=绿, up=红, right=蓝"""
    def __init__(self, view):
        self.fwd = gl.GLLinePlotItem(antialias=True); view.addItem(self.fwd)
        self.up  = gl.GLLinePlotItem(antialias=True); view.addItem(self.up)
        self.rgt = gl.GLLinePlotItem(antialias=True); view.addItem(self.rgt)

    def update(self, pos_r, fwd_r, up_r, rgt_r, length=300.0):
        p = pos_r
        self.fwd.setData(pos=np.array([p, p + length * fwd_r]), color=(0,1,0,1), width=4)
        self.up.setData( pos=np.array([p, p + length * up_r]),  color=(1,0,0,1), width=4)
        self.rgt.setData(pos=np.array([p, p + 0.5*length * rgt_r]), color=(0,0.5,1,1), width=4)


class VelocityArrowRenderer:
    """初始速度方向与力度箭头（带 3D 箭头翼）：
    - 仅在接收到任何玩家/模型输入之前 (STATE_IDLE 或第 0 帧) 显示，一旦开始输入立即隐藏。
    - 车辆箭头长度至少等于车体侧向蓝轴长度 (min_len=150.0 uu)，哪怕初速度为 0 也能清晰看到初始指向；
      随初速度线性增长（最大约 1150 uu）。
    - 球体箭头当球有初速度时显示，长度随球速大小动态伸缩，直观展示球即将滚动的方向与力度。
    """
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
        # 构造水平面或垂直面正交向量以绘制箭头双翼
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


def setup_standard_camera(view, distance=7000):
    view.setCameraPosition(distance=distance, elevation=30, azimuth=45)