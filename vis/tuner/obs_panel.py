"""obs 曲线面板 —— 复用 pyqtgraph 显示原始观测值。支持 delta 模式。"""
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from core.obs_v2 import OBS_V2_GROUPS

FLAG_KEYS = [
    "on_ground", "is_jumping", "is_flipping", "auto_flip",
    "has_flipped", "has_djump", "flip_or_jump", "flip_reset",
    "world_cont",
]

_A_SCALES = [5120.0, 5120.0, 5120.0, 5120.0, 2000.0, 2000.0]

_H_DELTA_CLIP = 100.0
H_GROUP_NAME = "H.closing"


def _diff(ys):
    if not ys:
        return ys
    out = [0.0]
    for i in range(1, len(ys)):
        out.append(ys[i] - ys[i - 1])
    return out


class ObsPlotPanel:
    def __init__(self, widget: pg.PlotWidget):
        self.widget = widget
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
        self.widget.addItem(self.cursor_line)

    def clear(self):
        for c in self.curves.values():
            self.widget.removeItem(c)
        self.curves.clear()
        self.legend.clear()
        self.cursor_line.setPos(0)
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)
        self.widget.getViewBox().autoRange()

    def set_cursor(self, idx):
        self.cursor_line.setPos(idx)

    def update_group(self, group_name: str, frames, delta: bool = False):
        if not frames:
            return
        # ★ 每次刷新前先解除 Y 轴锁定
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)

        for c in self.curves.values():
            self.widget.removeItem(c)
        self.curves.clear()
        self.legend.clear()

        palette = ['c', 'y', 'g', (255, 100, 100), (100, 255, 100),
                   (100, 100, 255), (255, 100, 255), (255, 200, 100)]
        xs = [f["i"] for f in frames]
        dtag = " Δ" if delta else ""

        # ---- H 组 ----
        if group_name == H_GROUP_NAME:
            lab = "closing_spd"
            ys = []
            for f in frames:
                parts = f.get("obs_parts")
                ys.append(0.0 if parts is None
                          else float(parts["H.closing"][0]))
            if delta:
                ys = _diff(ys)
                clip = _H_DELTA_CLIP
                ys = [max(-clip, min(clip, v)) for v in ys]
                self.curves[lab] = self.widget.plot(
                    xs, ys, pen=pg.mkPen('y', width=2),
                    name=f"{lab}{dtag} (clip±{clip:.0f})")
                c = _H_DELTA_CLIP * 1.1
                self.widget.setYRange(-c, c, padding=0)
            else:
                self.curves[lab] = self.widget.plot(
                    xs, ys, pen=pg.mkPen('y', width=2), name=lab)
                # ★ 恢复自动范围
                self.widget.getViewBox().enableAutoRange(axis='y', enable=True)
                self.widget.getViewBox().autoRange()
            return

        if group_name not in OBS_V2_GROUPS:
            return
        _, _, labels = OBS_V2_GROUPS[group_name]

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
            self.widget.getViewBox().enableAutoRange(axis='y', enable=True)
            self.widget.getViewBox().autoRange()
            return

        for k, lab in enumerate(labels):
            ys = []
            for f in frames:
                parts = f.get("obs_parts")
                ys.append(0.0 if parts is None
                          else float(parts[group_name][k]))
            if delta:
                ys = _diff(ys)
            self.curves[lab] = self.widget.plot(
                xs, ys, pen=pg.mkPen(palette[k % len(palette)], width=2),
                name=f"{lab}{dtag}")
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)
        self.widget.getViewBox().autoRange()

    def update_flags(self, frames):
        if not frames:
            return
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
        self.widget.getViewBox().autoRange()