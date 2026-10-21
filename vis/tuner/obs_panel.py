"""obs 曲线面板 —— 复用 pyqtgraph 显示原始观测值。支持 delta 模式。"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from core.obs import OBS_GROUPS

FLAG_KEYS = [
    "on_ground", "is_jumping", "is_flipping", "auto_flip",
    "has_flipped", "has_djump", "flip_or_jump", "flip_reset",
    "world_cont",
]

_A_SCALES = [5120.0, 5120.0, 5120.0, 5120.0, 2000.0, 2000.0]

_F_DELTA_CLIP = 100.0
F_GROUP_NAME = "F.closing"


def _diff(ys):
    if not ys:
        return ys
    out = [0.0]
    for i in range(1, len(ys)):
        out.append(ys[i] - ys[i - 1])
    return out


class ObsPlotPanel:
    def __init__(self, widget: pg.PlotWidget, max_window: int = 600):
        self.widget = widget
        self.max_window = max(10, int(max_window))
        self.last_x_range = (0.0, float(self.max_window))
        self.widget.setBackground('k')
        self.widget.showGrid(x=True, y=True, alpha=0.3)
        self.widget.setLabel('left', 'obs')
        self.widget.setLabel('bottom', 'frame')
        self.widget.setMinimumHeight(260)
        self.widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.legend = self.widget.addLegend(offset=(-10, 10))
        self.curves = {}
        self.cursor_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
        self.widget.addItem(self.cursor_line, ignoreBounds=True)

    def set_max_window(self, max_window: int):
        self.max_window = max(10, int(max_window))

    def _apply_dynamic_x_range(self, xs, auto_y: bool = True):
        if not xs:
            return
        x_max = float(xs[-1])
        x_min = float(xs[0]) if len(xs) > 1 else max(0.0, x_max - 1.0)
        if x_max <= x_min:
            x_max = x_min + 1.0
        self.last_x_range = (x_min, x_max)
        self.cursor_line.setPos(float(xs[-1]))
        self.widget.setXRange(x_min, x_max, padding=0.0)
        if auto_y:
            self.widget.getViewBox().enableAutoRange(axis='y', enable=True)

    def clear(self):
        for c in self.curves.values():
            self.widget.removeItem(c)
        self.curves.clear()
        self.legend.clear()
        self.cursor_line.setPos(0)
        self.last_x_range = (0.0, float(self.max_window))
        self.widget.setXRange(0.0, float(self.max_window), padding=0.0)
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)

    def set_cursor(self, idx, frames=None):
        if frames and 0 <= idx < len(frames):
            self.cursor_line.setPos(frames[idx]["i"])
        else:
            self.cursor_line.setPos(idx)

    def update_group(self, group_name: str, frames, delta: bool = False, max_window: int = None):
        if max_window is not None:
            self.set_max_window(max_window)
        if not frames:
            return
        if len(frames) > self.max_window:
            frames = list(frames)[-self.max_window:]
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)

        for c in self.curves.values():
            self.widget.removeItem(c)
        self.curves.clear()
        self.legend.clear()

        palette = ['c', 'y', 'g', (255, 100, 100), (100, 255, 100),
                   (100, 100, 255), (255, 100, 255), (255, 200, 100)]
        xs = [f["i"] for f in frames]
        dtag = " Δ" if delta else ""

        if group_name == F_GROUP_NAME:
            lab = "closing_spd"
            ys = []
            for f in frames:
                parts = f.get("obs_parts")
                ys.append(0.0 if parts is None
                          else float(parts["F.closing"][0]))
            if delta:
                ys = _diff(ys)
                clip = _F_DELTA_CLIP
                ys = [max(-clip, min(clip, v)) for v in ys]
                self.curves[lab] = self.widget.plot(
                    xs, ys, pen=pg.mkPen('y', width=2),
                    name=f"{lab}{dtag} (clip±{clip:.0f})")
                c = _F_DELTA_CLIP * 1.1
                self.widget.setYRange(-c, c, padding=0)
                self._apply_dynamic_x_range(xs, auto_y=False)
            else:
                self.curves[lab] = self.widget.plot(
                    xs, ys, pen=pg.mkPen('y', width=2), name=lab)
                self._apply_dynamic_x_range(xs, auto_y=True)
            return

        if group_name not in OBS_GROUPS:
            return
        _, _, labels = OBS_GROUPS[group_name]

        if group_name == "A.face_dists":
            for k, lab in enumerate(labels):
                ys = []
                for f in frames:
                    parts = f.get("obs_parts")
                    v = 0.0 if parts is None else float(parts[group_name][k])
                    ys.append(v / _A_SCALES[k])
                if delta:
                    ys = _diff(ys)
                self.curves[lab] = self.widget.plot(
                    xs, ys, pen=pg.mkPen(palette[k], width=2),
                    name=f"{lab}{dtag}")
            self._apply_dynamic_x_range(xs, auto_y=True)
            return

        for k, lab in enumerate(labels):
            ys = []
            for f in frames:
                parts = f.get("obs_parts")
                arr_g = parts.get(group_name) if parts is not None else None
                ys.append(0.0 if (arr_g is None or k >= len(arr_g))
                          else float(arr_g[k]))
            if delta:
                ys = _diff(ys)
            self.curves[lab] = self.widget.plot(
                xs, ys, pen=pg.mkPen(palette[k % len(palette)], width=2),
                name=f"{lab}{dtag}")
        self._apply_dynamic_x_range(xs, auto_y=True)

    def update_flags(self, frames, max_window: int = None):
        if max_window is not None:
            self.set_max_window(max_window)
        if not frames:
            return
        if len(frames) > self.max_window:
            frames = list(frames)[-self.max_window:]
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)
        for c in self.curves.values():
            self.widget.removeItem(c)
        self.curves.clear()
        self.legend.clear()

        xs = [f["i"] for f in frames]
        palette = ['c', 'y', 'g', (255, 100, 100), (100, 255, 100),
                   (100, 100, 255), (255, 100, 255), (255, 200, 100),
                   (200, 200, 200)]
        for k, key in enumerate(FLAG_KEYS):
            ys = []
            for f in frames:
                fl = f.get("flags")
                v = 1.0 if (fl and fl.get(key)) else 0.0
                ys.append(float(k) + 0.7 * v)
            self.curves[key] = self.widget.plot(
                xs, ys, pen=pg.mkPen(palette[k % len(palette)], width=2),
                name=f"{k}:{key}")
        self._apply_dynamic_x_range(xs, auto_y=True)