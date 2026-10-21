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

        def setPlainText(self, text):
            self._text = str(text)

        def toPlainText(self):
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
        KeypadModifier = 0x20000000
        Key_0 = 48
        Key_1 = 49
        Key_2 = 50
        Key_3 = 51
        Key_4 = 52
        Key_5 = 53
        Key_6 = 54
        Key_7 = 55
        Key_8 = 56
        Key_9 = 57
        Key_F1 = 0x01000030
        Key_F2 = 0x01000031
        Key_F3 = 0x01000032
        Key_F4 = 0x01000033
        Key_F5 = 0x01000034
        Key_F6 = 0x01000035
        Key_F7 = 0x01000036
        Key_W = 87
        Key_S = 83
        Key_A = 65
        Key_D = 68
        Key_Q = 81
        Key_E = 69
        Key_R = 82
        Key_N = 78
        Key_T = 84
        Key_O = 79
        Key_P = 80
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
        Key_Backtab = 0x01000002
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

    # 9. Verify Speed-Flip / Half-Flip / Double-Jump / custom u[...]d[...] spawn distributions
    from vis.tuner.window_views import validate_spawn_dist_spec
    for spec_str, expected_dir in (
        ("mode=speedflip", 0),
        ("mode=halfflip", 1),
        ("u[01000]d[010]", 1),
        ("mode=djump", None),
    ):
        ok, _ = validate_spawn_dist_spec(spec_str)
        assert ok, f"validate_spawn_dist_spec rejected valid spec {spec_str}"
        w._apply_spawn_dist_str(spec_str)
        if expected_dir is not None:
            assert w.env.current_dir5_idx == expected_dir, f"Expected dir5={expected_dir} for {spec_str}, got {w.env.current_dir5_idx}"
    w._apply_spawn_dist_str("auto")

    # 10. Verify TrailRenderer synchronized expiration (pts, jump_pts, djump_pts) and Nexto (--nexto 6 2 & 1 1)
    from nexto import parse_nexto_teams
    assert parse_nexto_teams(["6", "2"]) == (6, 2)
    assert parse_nexto_teams([]) == (3, 3)
    assert parse_nexto_teams(["2"]) == (2, 2)
    assert parse_nexto_teams(["1v1"]) == (1, 1)

    tr = sc.TrailRenderer(w.view, max_len=10, max_age_sec=100.0)
    tr.push((0.0, 0.0, 17.0), boosting=1.0, jumping=True, djump_or_flip=True)
    for step_k in range(1, 10):
        tr.push((float(step_k * 10), 0.0, 17.0), boosting=0.0, jumping=False, djump_or_flip=False)
    assert len(tr.pts) == 10 and len(tr.jump_pts) == 1 and len(tr.djump_pts) == 1
    # Pushing the 11th point evicts step 0 from pts AND simultaneously evicts step 0 from jump_pts & djump_pts
    tr.push((100.0, 0.0, 17.0), boosting=0.0, jumping=False, djump_or_flip=False)
    assert len(tr.pts) == 10, f"Expected len(tr.pts)==10, got {len(tr.pts)}"
    assert len(tr.jump_pts) == 0, f"Expected jump_pts to expire with trail pts, got {len(tr.jump_pts)}"
    assert len(tr.djump_pts) == 0, f"Expected djump_pts to expire with trail pts, got {len(tr.djump_pts)}"

    # Verify Nexto 6v2 match mode
    w_nexto = RewardTuner(nexto_teams=[6, 2])
    assert w_nexto.is_nexto_mode is True
    assert w_nexto.env.n_blue == 6 and w_nexto.env.n_orange == 2 and w_nexto.env.n_cars == 8
    assert w_nexto.view.cam_mode == "center", f"Nexto default camera must be 'center', got {w_nexto.view.cam_mode}"
    for _ in range(16):
        w_nexto.auto_session._step_once()
    w_nexto._render()
    assert len(w_nexto.auto_session.frames) == 16
    assert "NEXTO 6v2" in w_nexto.hud.toPlainText() or "NEXTO 6v2" in w_nexto.mode_indicator.text()

    # Verify keys 1..8 select cars 0..7, 9 cycles prev (-1), 0 cycles next (+1)
    w_nexto.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_8))
    assert w_nexto.env.focused_car_idx == 7
    w_nexto.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_9))
    assert w_nexto.env.focused_car_idx == 6
    w_nexto.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_0))
    assert w_nexto.env.focused_car_idx == 7
    w_nexto.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_0))
    assert w_nexto.env.focused_car_idx == 0

    # Verify Tab cycles sub-curves / obs groups without conflicting with 1..8 car selection
    prev_pm = w_nexto.plot_mode
    w_nexto.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w_nexto.plot_mode == (prev_pm % 6) + 1

    # Verify R triggers random standard Kickoff Spawn reset
    w_nexto.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_R))
    assert len(w_nexto.auto_session.frames) == 0
    bs_kickoff = w_nexto.env.arena.ball.get_state()
    assert abs(float(bs_kickoff.pos.x)) < 1.0 and abs(float(bs_kickoff.pos.y)) < 1.0

    # Verify Nexto 1v1 car selection clamping (pressing 7 in 1v1 selects the 2nd/last car, index 1)
    w_1v1 = RewardTuner(nexto_teams=[1, 1])
    assert w_1v1.env.n_cars == 2
    w_1v1.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_7, mods=QtCore.Qt.KeypadModifier))
    assert w_1v1.env.focused_car_idx == 1, f"Expected pressing 7 in 1v1 to clamp to car idx 1, got {w_1v1.env.focused_car_idx}"
    w_1v1.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_9))
    assert w_1v1.env.focused_car_idx == 0
    w_1v1.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_0))
    assert w_1v1.env.focused_car_idx == 1

    print(
        f"[verify_tuner] ALL OK! Verified RewardTuner startup, {NUM_CURRICULUM_STEPS} curriculum steps, "
        f"HUD formatting, recording/playback/freerun, plot/obs/flags panels, 3-way Tab ScenarioEditor, "
        f"[Z] Boost/HasJump panel, 3D POV Ball-Cam, SpeedFlip/HalfFlip/DJump presets, MANUAL-RESET instant trajectory on N + replay on R, "
        f"TrailRenderer synchronized jump/djump expiration, and Nexto (--nexto 6 2 & 1 1) multi-car combat."
    )


if __name__ == "__main__":
    verify_tuner_headless()


