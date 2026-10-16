"""reward 曲线面板。"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

MODE_NAMES = {
    1: "r_total (每帧)",
    2: "r_cum (累计)",
    3: "dist / dist_h",
    4: "分项 r_dist / r_align",
    5: "全部叠加",
    6: "peak_closing / closing_accum",
}


class RewardPlotPanel:
    def __init__(self, widget: pg.PlotWidget):
        self.widget = widget
        self.widget.setBackground('k')
        self.widget.showGrid(x=True, y=True, alpha=0.3)
        self.widget.setLabel('left', 'value')
        self.widget.setLabel('bottom', 'frame')
        self.widget.setMinimumHeight(260)
        self.widget.setFocusPolicy(QtCore.Qt.NoFocus)
        self.widget.addLegend(offset=(-10, 10))

        self.curve_total = self.widget.plot([], [], pen=pg.mkPen('c', width=2),
                                            name='r_total (每帧)')
        self.curve_cum   = self.widget.plot([], [], pen=pg.mkPen('y', width=2),
                                            name='r_cum (累计)')
        self.curve_dist  = self.widget.plot([], [], pen=pg.mkPen('g', width=2),
                                            name='dist')
        self.curve_dh    = self.widget.plot([], [], pen=pg.mkPen((100, 255, 100), width=1),
                                            name='dist_h')
        self.cursor_line = pg.InfiniteLine(
            angle=90, movable=False,
            pen=pg.mkPen('w', style=QtCore.Qt.DashLine))
        self.widget.addItem(self.cursor_line)

    def clear(self):
        for c in (self.curve_total, self.curve_cum, self.curve_dist, self.curve_dh):
            c.setData([], [])
        self.cursor_line.setPos(0)
        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)
        self.widget.getViewBox().autoRange()

    def set_cursor(self, idx):
        self.cursor_line.setPos(idx)

    def update_mode(self, mode: int, frames):
        if not frames:
            return
        xs = [f["i"] for f in frames]

        self.curve_total.setData([], [])
        self.curve_cum.setData([], [])
        self.curve_dist.setData([], [])
        self.curve_dh.setData([], [])

        if mode == 1:
            self.curve_total.setData(xs, [f["r_total"] for f in frames])
        elif mode == 2:
            self.curve_cum.setData(xs, [f["r_cum"] for f in frames])
        elif mode == 3:
            self.curve_dist.setData(xs, [f["dist"] for f in frames])
            self.curve_dh.setData(xs, [f["dist_h"] for f in frames])
        elif mode == 4:
            self.curve_total.setData(xs, [f["detail"].get("r_dist", 0) for f in frames])
            self.curve_cum.setData(xs, [f["detail"].get("r_align", 0) for f in frames])
        elif mode == 5:
            self.curve_total.setData(xs, [f["r_total"] for f in frames])
            self.curve_cum.setData(xs, [f["r_cum"] for f in frames])
            self.curve_dist.setData(xs, [f["dist"] for f in frames])
            self.curve_dh.setData(xs, [f["dist_h"] for f in frames])
        elif mode == 6:
            # peak_closing：整局历史最高接近速度
            # closing_accum：累积超速分（触球前会全部结清）
            self.curve_total.setData(
                xs, [f["detail"].get("peak_closing", 0) for f in frames])
            self.curve_cum.setData(
                xs, [f["detail"].get("closing_accum", 0) for f in frames])

        self.widget.getViewBox().enableAutoRange(axis='y', enable=True)
        self.widget.getViewBox().autoRange()