import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

MODE_NAMES = {
    1: "r_total (每帧)",
    2: "r_cum (累计)",
    3: "dist / dist_h",
    4: "分项 r_dist / r_slide / r_time",
    5: "全部叠加",
    6: "car_speed / closing",
}


def _diff(ys):
    if not ys:
        return ys
    out = [0.0]
    for i in range(1, len(ys)):
        out.append(ys[i] - ys[i - 1])
    return out


class RewardPlotPanel:
    def __init__(self, widget: pg.PlotWidget, max_window: int = 600):
        self.widget = widget
        self.max_window = max(10, int(max_window))
        self.last_x_range = (0.0, float(self.max_window))
        self.widget.setBackground('k')
        self.widget.showGrid(x=True, y=True, alpha=0.3)
        self.widget.setLabel('left', 'value')
        self.widget.setLabel('bottom', 'frame')
        self.widget.setMinimumHeight(260)
        self.widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.widget.addLegend(offset=(-10, 10))

        self.curve_total = self.widget.plot([], [], pen=pg.mkPen('c', width=2),
                                            name='r_total / r_dist')
        self.curve_cum   = self.widget.plot([], [], pen=pg.mkPen('y', width=2),
                                            name='r_cum / r_time')
        self.curve_dist  = self.widget.plot([], [], pen=pg.mkPen('g', width=2),
                                            name='dist / r_slide')
        self.curve_dh    = self.widget.plot([], [], pen=pg.mkPen((100, 255, 100), width=1),
                                            name='dist_h')
        self.cursor_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
        self.widget.addItem(self.cursor_line, ignoreBounds=True)

    def set_max_window(self, max_window: int):
        self.max_window = max(10, int(max_window))

    def _apply_dynamic_x_range(self, xs):
        if not xs:
            return
        x_max = float(xs[-1])
        x_min = float(xs[0]) if len(xs) > 1 else max(0.0, x_max - 1.0)
        if x_max <= x_min:
            x_max = x_min + 1.0
        self.last_x_range = (x_min, x_max)
        self.cursor_line.setPos(float(xs[-1]))
        self.widget.setXRange(x_min, x_max, padding=0.0)
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)

    def clear(self):
        for c in (self.curve_total, self.curve_cum, self.curve_dist, self.curve_dh):
            c.setData([], [])
        self.cursor_line.setPos(0)
        self.last_x_range = (0.0, float(self.max_window))
        self.widget.setXRange(0.0, float(self.max_window), padding=0.0)
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)

    def set_cursor(self, idx, frames=None):
        if frames and 0 <= idx < len(frames):
            self.cursor_line.setPos(frames[idx]["i"])
        else:
            self.cursor_line.setPos(idx)

    def update_mode(self, mode: int, frames, delta: bool = False, max_window: int = None):
        if max_window is not None:
            self.set_max_window(max_window)
        if not frames:
            return
        if len(frames) > self.max_window:
            frames = list(frames)[-self.max_window:]
        xs = [f["i"] for f in frames]
        fn = _diff if delta else (lambda v: v)

        self.curve_total.setData([], [])
        self.curve_cum.setData([], [])
        self.curve_dist.setData([], [])
        self.curve_dh.setData([], [])

        if mode == 1:
            self.curve_total.setData(xs, fn([f["r_total"] for f in frames]))
        elif mode == 2:
            self.curve_cum.setData(xs, fn([f["r_cum"] for f in frames]))
        elif mode == 3:
            self.curve_dist.setData(xs, fn([f["dist"] for f in frames]))
            self.curve_dh.setData(xs, fn([f["dist_h"] for f in frames]))
        elif mode == 4:
            self.curve_total.setData(xs, fn([f["detail"].get("r_dist", 0) for f in frames]))
            self.curve_cum.setData(xs, fn([f["detail"].get("r_time", 0) for f in frames]))
            self.curve_dist.setData(xs, fn([f["detail"].get("r_slide", 0) for f in frames]))
        elif mode == 5:
            self.curve_total.setData(xs, fn([f["r_total"] for f in frames]))
            self.curve_cum.setData(xs, fn([f["r_cum"] for f in frames]))
            self.curve_dist.setData(xs, fn([f["dist"] for f in frames]))
            self.curve_dh.setData(xs, fn([f["dist_h"] for f in frames]))
        elif mode == 6:
            self.curve_total.setData(xs, fn([f.get("car_speed", 0.0) for f in frames]))
            self.curve_cum.setData(xs, fn([f.get("closing", 0.0) for f in frames]))

        self._apply_dynamic_x_range(xs)
