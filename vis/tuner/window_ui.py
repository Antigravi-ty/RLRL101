"""UI 布局：3D 视图 + 曲线面板 + 右侧状态栏。"""
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets

from vis import scene as sc
from .plot_panel import RewardPlotPanel
from .obs_panel  import ObsPlotPanel


class SpeedDividerBar(QtWidgets.QWidget):
    """3D 场景与底部图表之间的 120Hz 实时车速进度条分隔线：
    - 量程上限: 2300.0 uu/s (最右端)
    - 3 条关键刻度竖线 (按正翻/斜翻 +500 uu/s 速度增益计算)：
      1. 1700 (= 2200 - 500): 在此速度以上翻滚可直接突破超音速 (2200)
      2. 1800 (= 2300 - 500): 在此速度以上翻滚可直接打满极速 (2300)
      3. 2200 (Supersonic):   超音速临界线（带反色/高亮显示，突破 2200 或 is_supersonic 时变色）
    - 当发生 Boost Wasted (满速或无效喷气) 时，右侧额外闪烁红色 WASTED 警示。
    """
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

        # 1. 黑色背景底槽
        qp.fillRect(0, 0, w, h, QtGui.QColor(15, 15, 20))

        # 2. 计算当前速度填充宽度
        ratio = min(1.0, max(0.0, self.speed / self.MAX_SPEED))
        fill_w = int(round(ratio * w))
        x_1700 = int(round((1700.0 / self.MAX_SPEED) * w))
        x_2200 = int(round((2200.0 / self.MAX_SPEED) * w))

        if fill_w > 0:
            if self.speed < 1700.0:
                # 1700 以下：青蓝色常规加速区
                bar_col = QtGui.QColor(0, 190, 240)
                qp.fillRect(0, 0, fill_w, h, bar_col)
            elif not self.is_supersonic and self.speed < 2200.0:
                # 1700 ~ 2200：翻滚可直达超音速黄金区间（翠绿->亮金）
                qp.fillRect(0, 0, x_1700, h, QtGui.QColor(0, 160, 210))
                qp.fillRect(x_1700, 0, fill_w - x_1700, h, QtGui.QColor(40, 225, 120))
            else:
                # >= 2200 (超音速区间)：进度条主体变亮橙金，超过 2200 的部分采用醒目反色（洋红/亮白紫）！
                qp.fillRect(0, 0, min(fill_w, x_2200), h, QtGui.QColor(255, 180, 0))
                if fill_w > x_2200:
                    # 突破 2200 后的反色高亮区
                    qp.fillRect(x_2200, 0, fill_w - x_2200, h, QtGui.QColor(255, 50, 220))

        # 若当前帧判定为 Boost Wasted，在进度条顶部/右侧叠加红色警示框
        if self.boost_wasted:
            qp.fillRect(max(0, fill_w - 18), 0, min(w, 18), h, QtGui.QColor(255, 30, 30))

        # 3. 绘制三条关键刻度竖线 (1700, 1800, 2200)，采用底色反色逻辑确保在任何进度下始终清晰可见
        font = QtGui.QFont("Consolas", 9, QtGui.QFont.Bold)
        qp.setFont(font)
        for mark_v, label in self.MARKS:
            mx = int(round((mark_v / self.MAX_SPEED) * w))
            covered = (fill_w >= mx)
            if mark_v >= 2200.0:
                # 2200 超音速竖线：未覆盖时亮白色，被覆盖时反色为纯黑+亮青双线
                line_col = QtGui.QColor(0, 0, 0) if covered else QtGui.QColor(255, 255, 255)
                qp.setPen(QtGui.QPen(line_col, 3))
                qp.drawLine(mx, 0, mx, h)
                if covered:
                    qp.setPen(QtGui.QPen(QtGui.QColor(0, 255, 255), 1))
                    qp.drawLine(mx - 2, 0, mx - 2, h)
            else:
                # 1700 / 1800 翻滚区间竖线：未覆盖时亮黄/白，被覆盖时反色为深黑
                line_col = QtGui.QColor(10, 10, 10) if covered else QtGui.QColor(255, 220, 80)
                qp.setPen(QtGui.QPen(line_col, 2))
                qp.drawLine(mx, 0, mx, h)

            # 刻度文字标签
            txt_col = QtGui.QColor(15, 15, 15) if (covered and mx - 4 < fill_w) else QtGui.QColor(210, 210, 210)
            qp.setPen(txt_col)
            qp.drawText(mx + 4, h - 6, f"{int(mark_v)}")

        # 4. 左侧实时显示精确车速数值与状态标签
        ss_tag = " [SUPERSONIC]" if self.is_supersonic else (" [FLIP->SS ZONE]" if self.speed >= 1700.0 else "")
        waste_tag = " !! BOOST WASTED !!" if self.boost_wasted else ""
        info_str = f" SPEED: {self.speed:6.1f} / 2300{ss_tag}{waste_tag}"
        # 根据左侧文字所在位置是否被高亮度进度条覆盖，自动反色文字
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
        # 1. 先让 QWidget/QFrame 绘制 stylesheet 中的背景与边框（临时清空文本避免默认单行绘制重叠）
        txt = self.text()
        opt_style = QtWidgets.QStyleOptionFrame()
        self.initStyleOption(opt_style)
        qp = QtGui.QPainter(self)
        self.style().drawPrimitive(
            QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt_style, qp, self
        )

        # 2. 使用 WrapAnywhere 逐字符自动折行绘制文本
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

        self.view = gl.GLViewWidget()
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

        # Stage 1 目标点 (target_pos) 3D 亮绿色高亮信标
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
        # 初始速度方向与力度箭头（车：亮青色，最短 150 uu 与蓝轴等长；球：亮橙黄色，按球速伸缩）
        self.car_vel_arrow  = sc.VelocityArrowRenderer(self.view, color=(0.0, 1.0, 0.9, 0.95), width=4)
        self.ball_vel_arrow = sc.VelocityArrowRenderer(self.view, color=(1.0, 0.75, 0.1, 0.95), width=4)
        left.addWidget(self.view, stretch=1)

        # 上下区域之间的 120Hz 动态车速进度条分隔线（含 1700 / 1800 / 2200 三条反色竖线）
        self.speed_bar = SpeedDividerBar()
        left.addWidget(self.speed_bar, stretch=0)

        plot_widget = pg.PlotWidget()
        left.addWidget(plot_widget, stretch=0)
        self.plot = RewardPlotPanel(plot_widget)
        self.obs_panel = ObsPlotPanel(plot_widget)

        # 右侧面板：固定横向宽度 (440px)，纵向可自由拉伸，彻底杜绝因文字长度变化导致的左右忽宽忽窄抖动
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

        # 第 3 行状态栏：固定宽度 + 任意字符处自动断行 (WrapAnywhere) + 纵向自动拉伸
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
            "录制: WASD  Space(jump)  .(boost)  /(jump)  Shift(slide)  Q/E(roll)\n"
            "回放: ←→(±1)  [ ](±30)  I/K(±60)  ,( -10)  ;(+10)  Space 播放/暂停\n"
            "速度: - / =   R 重放同场景   N 新随机场景   M 切 HUMAN/MODEL(保场景)   L 切 auto-reset\n"
            "B 切 latest/best   G 或 Shift+N 循环 prefix (自动同步训练场景 mode)\n"
            "F: FREERUN\n"
            "视图: O 切换 REWARD→OBS→FLAGS      1-8 obs 组(A-H)\n"
            "REWARD 模式: 1 r_total  2 r_cum  3 dist  4 分项  5 全部  6 peak_cl/accum\n"
            "P 切换 DELTA 模式     C 复制 HUD\n"
            "Enter 导出CSV(full)   Shift+Enter 导出CSV(compact)   Esc 退出"
        )
        self.help = WrapAnywhereLabel(help_txt)
        self.help.setStyleSheet("color:#ffcc00;background:#111;padding:6px;"
                                "font-family:Consolas;font-size:11px;")
        right.addWidget(self.help, stretch=0)

    def _cfg_text(self):
        c = self.session.cfg
        return (f"close={c.close_bonus} away={c.away_penalty} "
                f"time={c.time_penalty} air={c.air_scale} "
                f"touch={c.touch_bonus} | cl_th={c.closing_threshold} "
                f"prog={c.closing_progress_scale} "
                f"acc={c.closing_accum_scale}")
