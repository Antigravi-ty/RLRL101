import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

from vis import scene as sc
from .plot_panel import RewardPlotPanel
from .obs_panel  import ObsPlotPanel


class SpeedDividerBar(QtWidgets.QWidget):
    MAX_SPEED = 2300.0
    MARKS = (
        (1700.0, "1700 (Flip→SS)"),
        (1800.0, "1800 (Flip→Max)"),
        (2200.0, "2200 (SS)"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.speed = 0.0
        self.is_supersonic = False
        self.boost_wasted = False

    def update_speed(self, speed, is_supersonic=False, boost_wasted=False):
        spd = float(max(0.0, speed))
        ss = bool(is_supersonic or spd >= 2200.0)
        bw = bool(boost_wasted)
        if abs(spd - self.speed) < 0.5 and ss == self.is_supersonic and bw == self.boost_wasted:
            return
        self.speed = spd
        self.is_supersonic = ss
        self.boost_wasted = bw
        self.update()

    def paintEvent(self, event):
        from pyqtgraph.Qt import QtGui
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing, False)
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        qp.fillRect(0, 0, w, h, QtGui.QColor(15, 15, 20))

        ratio = min(1.0, max(0.0, self.speed / self.MAX_SPEED))
        fill_w = int(round(ratio * w))
        x_1700 = int(round((1700.0 / self.MAX_SPEED) * w))
        x_2200 = int(round((2200.0 / self.MAX_SPEED) * w))

        if fill_w > 0:
            if self.speed < 1700.0:
                bar_col = QtGui.QColor(0, 190, 240)
                qp.fillRect(0, 0, fill_w, h, bar_col)
            elif not self.is_supersonic and self.speed < 2200.0:
                qp.fillRect(0, 0, x_1700, h, QtGui.QColor(0, 160, 210))
                qp.fillRect(x_1700, 0, fill_w - x_1700, h, QtGui.QColor(40, 225, 120))
            else:
                qp.fillRect(0, 0, min(fill_w, x_2200), h, QtGui.QColor(255, 180, 0))
                if fill_w > x_2200:
                    qp.fillRect(x_2200, 0, fill_w - x_2200, h, QtGui.QColor(255, 50, 220))

        if self.boost_wasted:
            qp.fillRect(max(0, fill_w - 18), 0, min(w, 18), h, QtGui.QColor(255, 30, 30))

        font = QtGui.QFont("Consolas", 9, QtGui.QFont.Bold)
        qp.setFont(font)
        for mark_v, label in self.MARKS:
            mx = int(round((mark_v / self.MAX_SPEED) * w))
            covered = (fill_w >= mx)
            if mark_v >= 2200.0:
                line_col = QtGui.QColor(0, 0, 0) if covered else QtGui.QColor(255, 255, 255)
                qp.setPen(QtGui.QPen(line_col, 3))
                qp.drawLine(mx, 0, mx, h)
                if covered:
                    qp.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255), 1))
                    qp.drawLine(mx - 2, 0, mx - 2, h)
            else:
                line_col = QtGui.QColor(10, 10, 10) if covered else QtGui.QColor(255, 220, 80)
                qp.setPen(QtGui.QPen(line_col, 2))
                qp.drawLine(mx, 0, mx, h)

            txt_col = QtGui.QColor(15, 15, 15) if (covered and mx - 4 < fill_w) else QtGui.QColor(210, 210, 210)
            qp.setPen(txt_col)
            qp.drawText(mx + 4, h - 6, f"{int(mark_v)}")

        ss_tag = " [SUPERSONIC]" if self.is_supersonic else (" [FLIP->SS ZONE]" if self.speed >= 1700.0 else "")
        waste_tag = " !! BOOST WASTED !!" if self.boost_wasted else ""
        info_str = f" SPEED: {self.speed:6.1f} / 2300{ss_tag}{waste_tag}"
        left_covered = (fill_w > 90 and self.speed >= 1700.0)
        qp.setPen(QtGui.QColor(10, 10, 10) if left_covered else QtGui.QColor(255, 255, 255))
        qp.drawText(6, h - 6, info_str)
        qp.end()


class WrapAnywhereLabel(QtWidgets.QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        sp = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)

    def _extract_padding_and_border(self):
        import re
        ss = self.styleSheet() or ""
        pad = 4
        bdr = 0
        m_pad = re.search(r"padding\s*:\s*(\d+)px", ss)
        if m_pad:
            pad = int(m_pad.group(1))
        m_bdr = re.search(r"border\s*:\s*(\d+)px", ss)
        if m_bdr:
            bdr = int(m_bdr.group(1))
        return pad, bdr

    def _layout_height(self, width):
        from pyqtgraph.Qt import QtGui
        pad, bdr = self._extract_padding_and_border()
        avail_w = max(16, width - 2 * (pad + bdr))
        fm = QtGui.QFontMetrics(self.font())
        opt = QtGui.QTextOption()
        opt.setWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        total_h = 0.0
        lines = (self.text() or " ").split("\n")
        for paragraph in lines:
            layout = QtGui.QTextLayout(paragraph if paragraph else " ", self.font())
            layout.setTextOption(opt)
            layout.beginLayout()
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(avail_w)
                line.setPosition(QtCore.QPointF(0.0, total_h))
                total_h += line.height()
            layout.endLayout()
        return int(np.ceil(total_h)) + 2 * (pad + bdr) + max(2, fm.leading())

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout_height(width)

    def sizeHint(self):
        w = self.width() if self.width() > 20 else 420
        return QtCore.QSize(0, self._layout_height(w))

    def minimumSizeHint(self):
        w = self.width() if self.width() > 20 else 420
        return QtCore.QSize(0, self._layout_height(w))

    def setText(self, text):
        prev = self.text()
        super().setText(text)
        if prev != text:
            self.updateGeometry()
            self.update()

    def setStyleSheet(self, styleSheet):
        super().setStyleSheet(styleSheet)
        self.updateGeometry()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateGeometry()

    def paintEvent(self, event):
        from pyqtgraph.Qt import QtGui
        txt = self.text()
        opt_style = QtWidgets.QStyleOptionFrame()
        self.initStyleOption(opt_style)
        qp = QtGui.QPainter(self)
        self.style().drawPrimitive(
            QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt_style, qp, self
        )

        pad, bdr = self._extract_padding_and_border()
        margin = pad + bdr
        rect = self.rect().adjusted(margin, margin, -margin, -margin)
        if rect.width() <= 0 or rect.height() <= 0:
            qp.end()
            return

        qp.setFont(self.font())
        qp.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        flags = int(
            QtCore.Qt.AlignmentFlag.AlignLeft
            | QtCore.Qt.AlignmentFlag.AlignVCenter
            | QtCore.Qt.TextFlag.TextWrapAnywhere
        )
        qp.drawText(rect, flags, txt)
        qp.end()


class UIMixin:
    RIGHT_PANEL_WIDTH = 440

    def _build_ui(self):
        from pyqtgraph.Qt import QtGui
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)

        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, stretch=1)

        self.view = sc.TunerGLViewWidget()
        self.view.setFocusPolicy(QtCore.Qt.NoFocus)
        sc.setup_standard_camera(self.view, distance=5500)
        sc.add_floor(self.view)
        sc.add_field_border(self.view)
        sc.add_goals(self.view)
        self.big_pads, self.small_pads, self.big_idx, self.small_idx, self.pads_ref = \
            sc.add_pads(self.view, self.env.arena)

        self.car_scatter = gl.GLScatterPlotItem()
        self.car_scatter.setData(pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=18)
        self.view.addItem(self.car_scatter)

        self.ball_scatter = gl.GLScatterPlotItem()
        self.ball_scatter.setData(pos=np.zeros((1, 3)), color=(1.0, 0.2, 0.2, 0.9), size=6)
        self.view.addItem(self.ball_scatter)

        self.target_scatter = gl.GLScatterPlotItem()
        self.target_scatter.setData(pos=np.zeros((0, 3)), color=(0.1, 1.0, 0.4, 0.95), size=16)
        self.view.addItem(self.target_scatter)

        from core.constants import BALL_RADIUS
        ball_md = gl.MeshData.sphere(rows=12, cols=16, radius=BALL_RADIUS)
        self.ball_mesh = gl.GLMeshItem(
            meshdata=ball_md,
            smooth=True,
            color=(1.0, 0.2, 0.2, 0.30),
            glOptions="translucent",
        )
        self.view.addItem(self.ball_mesh)

        self.trail = sc.TrailRenderer(self.view, max_len=800)
        self.axes = sc.CarAxesRenderer(self.view)
        self.ball_guides    = sc.BallGuideLinesRenderer(self.view)
        self.car_vel_arrow  = sc.VelocityArrowRenderer(self.view, color=(0.0, 1.0, 0.9, 0.95), width=4)
        self.ball_vel_arrow = sc.VelocityArrowRenderer(self.view, color=(1.0, 0.75, 0.1, 0.95), width=4)
        self.ground_iso     = sc.GroundIsochroneRenderer(self.view)
        self.external_blob  = sc.IrregularSmoothBlobRenderer(self.view)
        left.addWidget(self.view, stretch=1)

        self.speed_bar = SpeedDividerBar()
        left.addWidget(self.speed_bar, stretch=0)

        plot_widget = pg.PlotWidget()
        left.addWidget(plot_widget, stretch=0)
        self.plot = RewardPlotPanel(plot_widget)
        self.obs_panel = ObsPlotPanel(plot_widget)

        self.right_panel = QtWidgets.QWidget()
        self.right_panel.setFixedWidth(self.RIGHT_PANEL_WIDTH)
        self.right_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        right = QtWidgets.QVBoxLayout(self.right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)
        root.addWidget(self.right_panel, stretch=0)

        self.mode_indicator = WrapAnywhereLabel("HUMAN")
        self.mode_indicator.setStyleSheet(
            "color:#0fc;font-size:16px;font-weight:bold;"
            "background:#000;padding:6px;border:2px solid #0fc;")
        right.addWidget(self.mode_indicator, stretch=0)

        self.reset_indicator = WrapAnywhereLabel("—")
        self.reset_indicator.setStyleSheet(
            "color:#888;font-size:14px;font-weight:bold;"
            "background:#000;padding:4px;border:2px solid #888;")
        right.addWidget(self.reset_indicator, stretch=0)

        self.status_label = WrapAnywhereLabel("IDLE")
        self.status_label.setStyleSheet(
            "color:#0fc;font-size:18px;font-weight:bold;"
            "background:rgba(0,0,0,180);padding:8px;border:2px solid #0fc;")
        right.addWidget(self.status_label, stretch=0)

        self.boost_label = WrapAnywhereLabel("BOOST: --")
        self.boost_label.setStyleSheet(
            "color:#ffcc00;font-size:20px;font-weight:bold;"
            "background:#111;padding:6px;border:2px solid #ffcc00;")
        right.addWidget(self.boost_label, stretch=0)

        from .car_state_panel import CarStatePanel
        self.car_state_panel = CarStatePanel(on_change_callback=self._on_manual_car_state_changed)
        right.addWidget(self.car_state_panel, stretch=0)

        self.hud = QtWidgets.QPlainTextEdit()
        self.hud.setReadOnly(True)
        self.hud.setFocusPolicy(QtCore.Qt.NoFocus)
        self.hud.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAnywhere)
        self.hud.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.hud.setStyleSheet(
            "color:#0fc;background:#111;font-family:Consolas;font-size:12px;")
        right.addWidget(self.hud, stretch=1)

        self.mode_label = WrapAnywhereLabel("曲线 1: r_total")
        self.mode_label.setStyleSheet(
            "color:#ffcc00;background:#111;padding:4px;font-weight:bold;")
        right.addWidget(self.mode_label, stretch=0)

        self.cfg_label = WrapAnywhereLabel(self._cfg_text())
        self.cfg_label.setStyleSheet("color:#aaa;background:#111;padding:4px;"
                                     "font-family:Consolas;font-size:11px;")
        right.addWidget(self.cfg_label, stretch=0)

        help_txt = (
            "镜头: 1 恢复球场中央视角  |  2 切换POV(Space切车头/球视角)  |  3 自由漫游FreeCam(WASD+QE/Space升降+鼠标, Tab锁输入, B对准场外3D体)\n"
            "可达集: 地面1s(青绿)/2s(金黄)/3s(洋红)等时线(Boost=0)  |  K 切换[神经网络极速模型(0.1ms) ↔ 物理实时推演]  |  F 无限时FREERUN(禁CSV防爆内存)\n"
            "录制: WASD  /(jump)  .(boost)  Shift(slide)  Q/E(roll)  |  X: 手动摆靶/摆车/发球  |  Z: 展开Boost/HasJump面板\n"
            "编辑(X下): Tab切 [车头方向 / init速度方向 / 手动发球(聚焦于球)]  WASD平移  /.高度  左键拖方向  [/]调速  T发射\n"
            "回放: ←→(±1)  [ ](±30)  I/K(±60)  ,( -10)  ;(+10)  J 或 ' 播放/暂停  |  R 逐步重放  N 新场景  M 切 HUMAN/MODEL"
        )
        self.help = WrapAnywhereLabel(help_txt)
        self.help.setStyleSheet("color:#ffcc00;background:#111;padding:6px;"
                                "font-family:Consolas;font-size:11px;")
        right.addWidget(self.help, stretch=0)

    def _cfg_text(self):
        c = self.session.cfg
        return (f"dist={c.distance_delta_scale}(surf={c.touch_surface_dist:.0f}) "
                f"slide=+{c.powerslide_bonus_per_tick}/t min_d=+{c.min_dist_ratio_bonus:.0f} "
                f"time=-{c.time_penalty_per_tick}/t boost=-{c.boost_penalty_per_unit}/u "
                f"jump=-{c.jump_hold_penalty_per_tick}/t(-{c.djump_trigger_penalty}/dj) "
                f"touch=+{c.touch_bonus:.0f}(+{c.touch_time_bonus_max:.0f}*t) to=-{c.timeout_penalty:.0f}")
