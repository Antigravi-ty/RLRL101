"""vis/tuner/car_state_panel.py —— 手动车辆状态控制面板 (快捷键 Z 展开/收起: Boost Amount 滑块+数值输入 & Has Jump 复选框)。"""
from pyqtgraph.Qt import QtCore, QtWidgets


class CarStatePanel(QtWidgets.QFrame):
    """Collapsible panel for manually setting Boost Amount (slider + numeric input)
    and Has Jump (checkbox), toggled via shortcut key `Z`.
    """

    def __init__(self, on_change_callback=None, parent=None):
        super().__init__(parent)
        self._on_change_callback = on_change_callback
        self._syncing = False

        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "QFrame {"
            "  background-color: #161822;"
            "  border: 2px solid #00ffcc;"
            "  border-radius: 4px;"
            "  padding: 4px;"
            "}"
            "QLabel { color: #e8e8e8; font-family: Consolas; font-size: 12px; border: none; }"
            "QCheckBox { color: #00ffcc; font-family: Consolas; font-size: 12px; font-weight: bold; border: none; }"
            "QDoubleSpinBox {"
            "  color: #ffcc00; background-color: #0b0c10; border: 1px solid #555;"
            "  font-family: Consolas; font-size: 13px; font-weight: bold; padding: 2px 4px;"
            "}"
            "QPushButton {"
            "  color: #e0e0e0; background-color: #252836; border: 1px solid #444;"
            "  font-family: Consolas; font-size: 11px; font-weight: bold; padding: 2px 6px;"
            "}"
            "QPushButton:hover { background-color: #32384d; border-color: #00ffcc; }"
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        # Header row
        header_row = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel("⚙ 手动状态控制面板 [按 Z 收起]")
        title_lbl.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 13px; border: none;")
        header_row.addWidget(title_lbl, stretch=1)
        layout.addLayout(header_row)

        # Row 1: Boost Slider + SpinBox
        boost_row = QtWidgets.QHBoxLayout()
        boost_lbl = QtWidgets.QLabel("Boost:")
        boost_lbl.setStyleSheet("color: #ffcc00; font-weight: bold; border: none;")
        boost_row.addWidget(boost_lbl)

        self.boost_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.boost_slider.setRange(0, 100)
        self.boost_slider.setValue(33)
        self.boost_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        boost_row.addWidget(self.boost_slider, stretch=1)

        self.boost_spin = QtWidgets.QDoubleSpinBox()
        self.boost_spin.setRange(0.0, 100.0)
        self.boost_spin.setDecimals(1)
        self.boost_spin.setSingleStep(1.0)
        self.boost_spin.setValue(33.0)
        self.boost_spin.setFixedWidth(72)
        self.boost_spin.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        boost_row.addWidget(self.boost_spin)
        layout.addLayout(boost_row)

        # Row 2: Quick Boost Preset Buttons
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(4)
        preset_lbl = QtWidgets.QLabel("快捷值:")
        preset_lbl.setStyleSheet("color: #aaa; font-size: 11px; border: none;")
        preset_row.addWidget(preset_lbl)
        for val in (0, 12, 33, 53, 100):
            btn = QtWidgets.QPushButton(str(val))
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda _=False, v=float(val): self.set_boost_value(v, trigger=True))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)

        # Row 3: Has Jump Checkbox + Lock Override Checkbox
        jump_row = QtWidgets.QHBoxLayout()
        self.has_jump_cb = QtWidgets.QCheckBox("Has Jump (可跳/可二段跳)")
        self.has_jump_cb.setChecked(True)
        self.has_jump_cb.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        jump_row.addWidget(self.has_jump_cb)

        self.lock_cb = QtWidgets.QCheckBox("锁定(N/R保持)")
        self.lock_cb.setChecked(False)
        self.lock_cb.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.lock_cb.setStyleSheet("color: #ffaa00; font-family: Consolas; font-size: 11px; font-weight: bold; border: none;")
        jump_row.addWidget(self.lock_cb)
        layout.addLayout(jump_row)

        # Signals
        self.boost_slider.valueChanged.connect(self._on_slider_changed)
        self.boost_spin.valueChanged.connect(self._on_spin_changed)
        self.has_jump_cb.toggled.connect(self._on_checkbox_changed)
        self.lock_cb.toggled.connect(self._on_checkbox_changed)

        self.setVisible(False)

    @property
    def boost_amount(self) -> float:
        return float(self.boost_spin.value())

    @property
    def has_jump(self) -> bool:
        return bool(self.has_jump_cb.isChecked())

    @property
    def lock_override(self) -> bool:
        return bool(self.lock_cb.isChecked())

    def sync_from_state(self, boost_amount: float, has_jump: bool, lock_override: bool = None):
        """Sync UI controls from environment/editor without firing change callback."""
        self._syncing = True
        try:
            b = float(max(0.0, min(100.0, boost_amount)))
            self.boost_slider.setValue(int(round(b)))
            self.boost_spin.setValue(b)
            self.has_jump_cb.setChecked(bool(has_jump))
            if lock_override is not None:
                self.lock_cb.setChecked(bool(lock_override))
        finally:
            self._syncing = False

    def set_boost_value(self, val: float, trigger: bool = True):
        b = float(max(0.0, min(100.0, val)))
        self._syncing = True
        try:
            self.boost_slider.setValue(int(round(b)))
            self.boost_spin.setValue(b)
        finally:
            self._syncing = False
        if trigger:
            self._emit_change()

    def set_has_jump(self, has_jump: bool, trigger: bool = True):
        self._syncing = True
        try:
            self.has_jump_cb.setChecked(bool(has_jump))
        finally:
            self._syncing = False
        if trigger:
            self._emit_change()

    def _on_slider_changed(self, int_val: int):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.boost_spin.setValue(float(int_val))
        finally:
            self._syncing = False
        self._emit_change()

    def _on_spin_changed(self, flt_val: float):
        if self._syncing:
            return
        self._syncing = True
        try:
            self.boost_slider.setValue(int(round(flt_val)))
        finally:
            self._syncing = False
        self._emit_change()

    def _on_checkbox_changed(self, _checked: bool):
        if self._syncing:
            return
        self._emit_change()

    def _emit_change(self):
        if callable(self._on_change_callback):
            self._on_change_callback(
                self.boost_amount,
                self.has_jump,
                self.lock_override,
            )
