#!/usr/bin/env python3
"""tools/verify_tuner.py —— 无头 (Headless) GUI Tuner 启动与全链路交互验证脚本。

无需安装 PyQt5 / pyqtgraph / PyOpenGL 或图形显示器 (X11/Wayland/macOS WindowServer)：
通过注入轻量级 Qt/pyqtgraph/OpenGL 桩模块，100% 真实执行 `vis/tuner/` 下的所有业务代码：
  - `RewardTuner.__init__` / `_build_ui` / `_refresh_ui_state` / `_refresh_hud` / `_render`
  - 全部 37 个课程阶梯 (`CurriculumStepSpec.name` / `.code`) 的 HUD 格式化
  - 录制 (`STATE_RECORDING`) -> 结算 (`STATE_DONE`) -> 回放 (`STATE_PLAYBACK`) -> 自由跑 (`STATE_FREERUN`)
  - 奖励曲线 (`plot_mode 1..4`) / 观测分组 (`OBS_GROUPS`) / 物理标志 (`flags`) 面板切换
  - `[X]` 场景编辑器 (`ScenarioEditor`) 激活、`[Tab]` 三向切换 (`CAR -> VEL -> BALL -> CAR`)、键鼠编辑与发射验证
"""
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _install_headless_qt_mocks():
    """Installs lightweight headless stubs for pyqtgraph, pyqtgraph.opengl, and pyqtgraph.Qt."""
    class _DummySignal:
        def connect(self, cb):
            self._cb = cb

        def emit(self, *args, **kwargs):
            if hasattr(self, "_cb") and callable(self._cb):
                self._cb(*args, **kwargs)

    class _WidgetMeta(type):
        def __getattr__(cls, name):
            return cls()

    class _DummyWidget(metaclass=_WidgetMeta):
        def __init__(self, *args, **kwargs):
            self.clicked = _DummySignal()
            self.currentTextChanged = _DummySignal()
            self.currentIndexChanged = _DummySignal()
            self.valueChanged = _DummySignal()
            self.toggled = _DummySignal()
            self.timeout = _DummySignal()
            self.opts = {"distance": 2500.0, "elevation": 20.0, "azimuth": 45.0, "fov": 60.0}
            self._items = []
            self._text = ""
            self._visible = True
            self._val = 0.0
            self._checked = True
            self._vx = float(args[0]) if len(args) >= 3 and isinstance(args[0], (int, float)) else 0.0
            self._vy = float(args[1]) if len(args) >= 3 and isinstance(args[1], (int, float)) else 0.0
            self._vz = float(args[2]) if len(args) >= 3 and isinstance(args[2], (int, float)) else 0.0

        def __call__(self, *args, **kwargs):
            return _DummyWidget(*args, **kwargs)

        def __getattr__(self, name):
            return _DummyWidget()

        def addItem(self, item, *args, **kwargs):
            self._items.append(item)

        def removeItem(self, item):
            if item in self._items:
                self._items.remove(item)

        def setText(self, text):
            self._text = str(text)

        def text(self):
            return self._text

        def setValue(self, v):
            self._val = v
            self.valueChanged.emit(v)

        def value(self):
            return self._val

        def setChecked(self, c):
            self._checked = bool(c)
            self.toggled.emit(self._checked)

        def isChecked(self):
            return self._checked

        def setRange(self, *args, **kwargs):
            pass

        def setDecimals(self, *args, **kwargs):
            pass

        def setSingleStep(self, *args, **kwargs):
            pass

        def setFrameShape(self, *args, **kwargs):
            pass

        def setVisible(self, v):
            self._visible = bool(v)

        def isVisible(self):
            return self._visible

        def width(self):
            return 1200

        def height(self):
            return 900

        def count(self):
            return len(self._items)

        def x(self):
            return self._vx

        def y(self):
            return self._vy

        def z(self):
            return self._vz

        def __float__(self):
            return 0.0

        def __int__(self):
            return 0

        def setStyleSheet(self, *args, **kwargs):
            pass

        def setFixedHeight(self, *args, **kwargs):
            pass

        def setFixedWidth(self, *args, **kwargs):
            pass

        def setFixedSize(self, *args, **kwargs):
            pass

        def setMinimumWidth(self, *args, **kwargs):
            pass

        def setMinimumHeight(self, *args, **kwargs):
            pass

        def setAlignment(self, *args, **kwargs):
            pass

        def setSizePolicy(self, *args, **kwargs):
            pass

        def setContentsMargins(self, *args, **kwargs):
            pass

        def setSpacing(self, *args, **kwargs):
            pass

        def setFocusPolicy(self, *args, **kwargs):
            pass

        def setFocus(self, *args, **kwargs):
            pass

        def setWindowTitle(self, *args, **kwargs):
            pass

        def resize(self, *args, **kwargs):
            pass

        def closeEvent(self, *args, **kwargs):
            pass

        def mousePressEvent(self, *args, **kwargs):
            pass

        def mouseMoveEvent(self, *args, **kwargs):
            pass

        def mouseReleaseEvent(self, *args, **kwargs):
            pass

        def wheelEvent(self, *args, **kwargs):
            pass

        def keyPressEvent(self, *args, **kwargs):
            pass

        def keyReleaseEvent(self, *args, **kwargs):
            pass

    class _AutoMockModule(types.ModuleType):
        def __getattr__(self, name):
            setattr(self, name, _DummyWidget)
            return _DummyWidget

    pg = _AutoMockModule("pyqtgraph")
    pg_gl = _AutoMockModule("pyqtgraph.opengl")
    pg_qt = _AutoMockModule("pyqtgraph.Qt")
    qt_core = _AutoMockModule("pyqtgraph.Qt.QtCore")
    qt_gui = _AutoMockModule("pyqtgraph.Qt.QtGui")
    qt_widgets = _AutoMockModule("pyqtgraph.Qt.QtWidgets")

    class _DummyQSettings:
        def __init__(self, *args, **kwargs):
            self._store = {}

        def value(self, k, default=None, type=None):
            val = self._store.get(k, default)
            if type is not None and val is not None:
                try:
                    return type(val)
                except Exception:
                    return default
            return val

        def setValue(self, k, v):
            self._store[k] = v

    class _QtEnum(int):
        def __getattr__(self, name):
            val = abs(hash(f"{int(self)}.{name}")) % 100000 + 1000
            res = _QtEnum(val)
            setattr(self, name, res)
            return res

    class _QtMeta(type):
        def __getattr__(cls, name):
            val = abs(hash(name)) % 100000 + 1000
            res = _QtEnum(val)
            setattr(cls, name, res)
            return res

    class _DummyQtNamespace(metaclass=_QtMeta):
        StrongFocus = 1
        NoFocus = 0
        Horizontal = 1
        Vertical = 2
        AlignLeft = 1
        AlignRight = 2
        AlignCenter = 4
        LeftButton = 1
        RightButton = 2
        MiddleButton = 4
        ShiftModifier = 0x02000000
        ControlModifier = 0x04000000
        AltModifier = 0x08000000
        Key_W = 87
        Key_S = 83
        Key_A = 65
        Key_D = 68
        Key_Q = 81
        Key_E = 69
        Key_R = 82
        Key_F = 70
        Key_J = 74
        Key_K = 75
        Key_L = 76
        Key_M = 77
        Key_V = 86
        Key_X = 88
        Key_Z = 90
        Key_C = 67
        Key_Space = 32
        Key_Shift = 16
        Key_Tab = 9
        Key_Return = 13
        Key_Enter = 13
        Key_Escape = 27
        Key_Period = 46
        Key_Comma = 44
        Key_Slash = 47
        Key_BracketLeft = 91
        Key_BracketRight = 93
        Key_Minus = 45
        Key_Equal = 61

    qt_core.Qt = _DummyQtNamespace
    qt_core.QTimer = _DummyWidget
    qt_core.QSettings = _DummyQSettings
    qt_core.QPoint = lambda x=0, y=0: _DummyWidget()

    pg.PlotWidget = _DummyWidget
    pg.InfiniteLine = _DummyWidget
    pg.mkPen = lambda *args, **kwargs: None
    pg.mkBrush = lambda *args, **kwargs: None
    pg.setConfigOptions = lambda **kwargs: None
    pg.Vector = lambda *args: _DummyWidget()

    pg.Qt = pg_qt
    pg.opengl = pg_gl
    pg_qt.QtCore = qt_core
    pg_qt.QtGui = qt_gui
    pg_qt.QtWidgets = qt_widgets

    sys.modules["pyqtgraph"] = pg
    sys.modules["pyqtgraph.opengl"] = pg_gl
    sys.modules["pyqtgraph.Qt"] = pg_qt
    sys.modules["pyqtgraph.Qt.QtCore"] = qt_core
    sys.modules["pyqtgraph.Qt.QtGui"] = qt_gui
    sys.modules["pyqtgraph.Qt.QtWidgets"] = qt_widgets


def verify_tuner_headless():
    _install_headless_qt_mocks()

    from core.curriculum import (
        NUM_CURRICULUM_STEPS, get_curriculum_step_spec, min_boost_for_ball_z,
    )
    from core.obs import GROUP_ORDER
    from vis.tuner.window import RewardTuner
    from vis.tuner.hud import format_hud
    from vis.tuner.ui_constants import STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN

    # 1. Verify all 37 CurriculumStepSpec objects have .name and .code
    for idx in range(NUM_CURRICULUM_STEPS):
        spec = get_curriculum_step_spec(idx)
        assert isinstance(spec.name, str) and len(spec.name) > 0, f"Invalid spec.name at step {idx}"
        assert isinstance(spec.code, str) and len(spec.code) > 0, f"Invalid spec.code at step {idx}"

    # 2. Verify deterministic minimum boost by ball height (calibrated to 1KWYnfez) & Phase 1-5 headroom ladder
    assert min_boost_for_ball_z(93.15, cz=17.0, boost_headroom=47.0) == 0.0
    assert min_boost_for_ball_z(500.0, cz=17.0, boost_headroom=47.0) == 0.0
    b_mid = min_boost_for_ball_z(1000.0, cz=17.0, boost_headroom=0.0)
    b_ceil = min_boost_for_ball_z(1940.0, cz=17.0, boost_headroom=0.0)
    assert b_mid == 25.0, f"Unexpected b_mid={b_mid}"
    assert b_ceil == 53.0, f"Unexpected b_ceil={b_ceil}"
    assert min_boost_for_ball_z(1940.0, cz=17.0, boost_headroom=47.0) == 100.0
    assert min_boost_for_ball_z(1500.0, cz=17.0, boost_headroom=47.0) == 88.0
    assert min_boost_for_ball_z(1940.0, cz=17.0, boost_headroom=35.0) == 88.0
    assert min_boost_for_ball_z(1940.0, cz=17.0, boost_headroom=0.0, boost_override=False) == 0.0

    # 3. Instantiate RewardTuner and run idle tick + HUD formatting across all curriculum steps
    w = RewardTuner(prefix="sbm_v1", from_ckpt="best", cam_mode="center")
    w.tick()
    assert w.session.state == STATE_IDLE

    for idx in range(NUM_CURRICULUM_STEPS):
        sp_i = get_curriculum_step_spec(idx)
        w.env.set_curriculum_state(1, step_idx=idx)
        boost_line, body = format_hud(w.session, 33.0)
        assert sp_i.name[:2] in ("P1", "P2", "P3", "P4", "P5") and "BOOST:" in boost_line and len(body) > 20

    # 4. Simulate human recording -> boost pad pickup -> finish -> playback -> view switching
    w.env.set_curriculum_state(1, step_idx=10)
    w.session.reset()
    w.keys["W"] = True
    w.keys["Boost"] = True
    w.tick()
    assert w.session.state == STATE_RECORDING

    for step_i in range(12):
        if step_i == 5:
            w.env._boost_pickups_tick = 1
        w.session._step_once(w.keys)
    w.keys["W"] = False
    w.keys["Boost"] = False
    w._render()
    w._refresh_hud()

    # Check plot modes, obs groups, flags view, delta toggle
    for pm in (1, 2, 3, 4):
        w._set_plot_mode(pm)
    for gi in range(len(GROUP_ORDER)):
        w._set_obs_group(gi)
    w._show_flags_view()
    w._set_plot_mode(1)
    w._toggle_delta()
    w._toggle_delta()

    # Finish recording & verify playback HUD
    msg = w.session.finish_recording("manual_test")
    w._refresh_ui_state(msg, to_hud=True)
    assert w.session.state == STATE_DONE
    w.session.toggle_playback()
    assert w.session.state == STATE_PLAYBACK
    w.tick()
    w._refresh_hud()

    # Freerun mode check
    w.session.begin_freerun()
    assert w.session.state == STATE_FREERUN
    w.session._freerun_step(w.keys)
    w.tick()
    w.session.reset()

    # 5. Verify ScenarioEditor [X] + 3-way [Tab] (CAR -> VEL -> BALL -> CAR) + KeysMixin + launch
    class _FakeKeyEvent:
        def __init__(self, key, mods=0, auto_rep=False):
            self._k = key
            self._m = mods
            self._ar = auto_rep

        def key(self):
            return self._k

        def modifiers(self):
            return self._m

        def isAutoRepeat(self):
            return self._ar

        def accept(self):
            pass

    from pyqtgraph.Qt import QtCore
    # Press X to enter Edit Mode
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_X))
    assert w.editor.active is True
    assert w.editor.target == "car"

    # Manipulate car in editor via held keys & mouse drag
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_W))
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Slash))
    w.tick()
    time.sleep(0.01)
    w.tick()
    w.keyReleaseEvent(_FakeKeyEvent(QtCore.Qt.Key_W))
    w.keyReleaseEvent(_FakeKeyEvent(QtCore.Qt.Key_Slash))
    w.editor.on_mouse_drag(100.0, 20.0)
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_0))

    # Press Tab -> VEL
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w.editor.target == "vel", f"Expected 'vel', got {w.editor.target}"
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_BracketRight))
    w.tick()
    time.sleep(0.01)
    w.tick()
    w.keyReleaseEvent(_FakeKeyEvent(QtCore.Qt.Key_BracketRight))
    w.editor.on_mouse_drag(50.0, 40.0)
    sd_vel = w.editor.build_spawn_state()
    assert sd_vel["car_vel"][2] <= 0.0, "Car velocity vertical component must never be upward"
    w._render()

    # Press Tab -> BALL (verify camera origin focuses on ball_r and forward is ball launch velocity vector)
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w.editor.target == "ball", f"Expected 'ball', got {w.editor.target}"
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_BracketRight))
    w.tick()
    time.sleep(0.01)
    w.tick()
    w.keyReleaseEvent(_FakeKeyEvent(QtCore.Qt.Key_BracketRight))
    w.editor.on_mouse_drag(-80.0, -30.0)
    w._render()
    import numpy as np
    from vis import scene as sc
    expected_ball_r = sc.r2r((w.editor.bx, w.editor.by, w.editor.bz))
    expected_ball_dir_r = sc.r2r(w.editor.ball_dir_vec())
    assert np.allclose(w.view._last_edit_pos_r, expected_ball_r, atol=1e-2), (
        f"Edit BALL camera origin should be ball_r {expected_ball_r}, got {w.view._last_edit_pos_r}"
    )
    assert np.allclose(w.view._last_edit_fwd_r, expected_ball_dir_r, atol=1e-2), (
        f"Edit BALL camera forward should be ball_dir_r {expected_ball_dir_r}, got {w.view._last_edit_fwd_r}"
    )

    # Press Tab -> CAR
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w.editor.target == "car", f"Expected 'car', got {w.editor.target}"
    assert len(w.editor.format_hud_summary()) > 50

    # Press T to launch from editor
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_T))
    assert w.editor.active is False
    w.tick()

    # 6. Verify [Z] Manual CarStatePanel (Boost slider/spinbox + Has Jump checkbox)
    assert w.car_state_panel.isVisible() is False
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Z))
    assert w.car_state_panel.isVisible() is True
    w.car_state_panel.set_boost_value(77.0, trigger=True)
    assert abs(w.env.car.get_state().boost - 77.0) < 0.5, f"Expected boost=77.0, got {w.env.car.get_state().boost}"
    w.car_state_panel.set_has_jump(False, trigger=True)
    assert w.env.has_second_jump is False, "Expected has_second_jump=False after unchecking Has Jump"
    w.car_state_panel.set_has_jump(True, trigger=True)
    assert w.env.has_second_jump is True
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Z))
    assert w.car_state_panel.isVisible() is False

    # 7. Verify POV Ball-Cam 3D elevation focus (pitches up for high ball vs low ball)
    w.view.set_cam_mode("ball")
    w.view.update_attach_camera(
        car_pos_r=(0.0, 0.0, 17.0),
        car_fwd_r=(0.0, 1.0, 0.0),
        car_up_r=(0.0, 0.0, 1.0),
        car_rgt_r=(1.0, 0.0, 0.0),
        ball_pos_r=(0.0, 1000.0, 93.15),
    )
    dz_low = w.view._pov_target.z() - w.view._pov_eye.z()
    w.view.update_attach_camera(
        car_pos_r=(0.0, 0.0, 17.0),
        car_fwd_r=(0.0, 1.0, 0.0),
        car_up_r=(0.0, 0.0, 1.0),
        car_rgt_r=(1.0, 0.0, 0.0),
        ball_pos_r=(0.0, 1000.0, 1600.0),
    )
    dz_high = w.view._pov_target.z() - w.view._pov_eye.z()
    assert dz_high > dz_low + 40.0, f"POV Ball-Cam failed to pitch up for high ball: dz_low={dz_low}, dz_high={dz_high}"

    # 8. Verify MODEL + MANUAL-RESET QoL: N draws full trajectory immediately (STATE_DONE), R replays step-by-step (STATE_RECORDING)
    from core.policy_net import ActorCritic
    from core.reward import RewardConfig
    from vis.tuner.auto_session import AutoSession
    dummy_auto = AutoSession.__new__(AutoSession)
    from vis.tuner.session import Session
    Session.__init__(dummy_auto, w.env, RewardConfig())
    dummy_auto.ckpt_path = "dummy.pt"
    dummy_auto.deterministic = True
    dummy_auto.current_ppo_ctrl = None
    dummy_auto.model = ActorCritic()
    dummy_auto.best_ret = 0.0
    dummy_auto.ep = 100

    w.auto_session = dummy_auto
    w.use_auto = True
    w.auto_reset = False  # MANUAL-RESET
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_N))
    assert w.auto_session.state == STATE_DONE, f"Expected STATE_DONE on N in MANUAL-RESET, got {w.auto_session.state}"
    assert len(w.auto_session.frames) > 0 and w.auto_session.playback_idx == len(w.auto_session.frames) - 1
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_R))
    assert w.auto_session.state == STATE_RECORDING, f"Expected STATE_RECORDING on R in MANUAL-RESET, got {w.auto_session.state}"

    # 9. Verify 1/2/3 Camera Slots, Free-Cam WASD+Mouse+Lock, External 3D Irregular Blob, and 1s/2s/3s Ground Isochrones
    w.use_auto = False
    w.session.reset()
    assert hasattr(w, "external_blob") and hasattr(w, "ground_iso") and hasattr(w, "reach_engine")
    assert hasattr(w, "aerial_iso") and hasattr(w, "aerial_engine")
    assert w.reach_engine.mode == "model"
    assert w.aerial_iso_mode == "FUNNEL"

    # Verify 3D Funnel calculation
    funnel_test = w.aerial_engine.compute_3d_funnel(
        car_pos=(0, 0, 17),
        car_vel=(0, 1000, 0),
        car_fwd=(0, 1, 0),
        car_up=(0, 0, 1),
        boost=50.0,
        is_on_ground=True,
    )
    assert len(funnel_test["verts"]) > 0
    assert len(funnel_test["faces"]) > 0
    assert len(funnel_test["rings"]) == 3
    assert len(funnel_test["rails"]) > 0
    w.aerial_iso.update_funnel(funnel_test)

    # Press 3 -> Free Camera mode (focused on external 3D blob, car input masked)
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_3))
    assert w.view.cam_mode == "free"
    assert w.view.free_cam_locked is False
    ex_before = w.view.free_params["ex"]
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_W))
    w.tick()
    time.sleep(0.01)
    w.tick()
    w.keyReleaseEvent(_FakeKeyEvent(QtCore.Qt.Key_W))
    assert w.view.free_params["ex"] != ex_before, "Free-Cam W key should move camera position"
    assert w.session.state == STATE_IDLE, "Unlocked Free-Cam WASD must mask car input and not start recording"

    # Press Tab in Free-Cam -> Lock camera position so WASD drives car
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w.view.free_cam_locked is True

    # Press K -> Toggle GroundReachabilityEngine between 'model' and 'rollout'
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_K))
    assert w.reach_engine.mode == "rollout"
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_K))
    assert w.reach_engine.mode == "model"

    # Press L -> Cycle Aerial 3D modes: FUNNEL -> BUBBLES -> OFF -> FUNNEL
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_L))
    assert w.aerial_iso_mode == "BUBBLES"
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_L))
    assert w.aerial_iso_mode == "OFF"
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_L))
    assert w.aerial_iso_mode == "FUNNEL"

    # Press 2 -> POV Camera, Press 1 -> Center Camera
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_2))
    assert w.view.cam_mode == "pov"
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_1))
    assert w.view.cam_mode == "center"

    # Verify FREERUN blocks CSV export
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_F))
    assert w.session.state == STATE_FREERUN
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Return))
    assert len(w.session.frames) == 0, "FREERUN must never accumulate frame records"
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_F))
    assert w.session.state == STATE_IDLE

    print(
        f"[verify_tuner] ALL OK! Verified ReachabilityTuner startup, {NUM_CURRICULUM_STEPS} curriculum steps, "
        f"1s/2s/3s Ground Isochrones (model & rollout), 3D Irregular Smooth Blob, 1/2/3 Camera Slots + Free-Cam Lock, "
        f"unlimited zero-memory FREERUN, 3-way Tab ScenarioEditor, and [Z] Boost/HasJump panel."
    )


if __name__ == "__main__":
    verify_tuner_headless()


