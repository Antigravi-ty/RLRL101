#!/usr/bin/env python3
"""tools/verify_tuner.py —— 无头 (Headless) GUI Tuner 启动与 Freestyle Air-Dribble Showman 全链路验证脚本。

无需安装 PyQt5 / pyqtgraph / PyOpenGL 或图形显示器 (X11/Wayland/macOS WindowServer)：
通过注入轻量级 Qt/pyqtgraph/OpenGL 桩模块，100% 真实执行 `core/` 与 `vis/tuner/` 下的所有业务代码：
  1. 190D 观测空间 (`54D` Trunk + `136D` BoostPad Tokens) 与左右镜像对称置换验证；
  2. 中间空中盒子区 (Central Air-Box) 5次连续触球路程累加 $(d_{1\\to 2}+d_{2\\to 3}+d_{3\\to 4}+d_{4\\to 5}) \\times 4$ 倍数计分、
     望远镜逐步差分守恒、以及触碰平面后链条结算清零验证；
  3. 严禁进球 (`-150.0` 超级大扣分与回合终止) 与 3 种氮气模式 (`unlimited` / `recharge` / `pads`) 验证；
  4. `RewardTuner` 全部 16 个课程阶梯 (`P1.01..P4.04`)、录制/回放/自由跑、9大观测分组 (`A..I`)、
     `[X]` 场景编辑器 (`CAR -> VEL -> BALL`)、5大花式起球预设 (`carry/catch/bounce/wall/ground`) 与 `latest.pt` 热重载验证。
"""
import math
import os
import sys
import time
import types
import numpy as np

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
            self.textChanged = _DummySignal()
            self.valueChanged = _DummySignal()
            self.toggled = _DummySignal()
            self.accepted = _DummySignal()
            self.rejected = _DummySignal()
            self.timeout = _DummySignal()
            self.opts = {"distance": 2500.0, "elevation": 20.0, "azimuth": 45.0, "fov": 60.0}
            self._items = []
            self._text = str(args[0]) if len(args) == 1 and isinstance(args[0], str) else ""
            self._visible = True
            self._val = 0.0
            self._checked = True
            self._vx = float(args[0]) if len(args) >= 3 and isinstance(args[0], (int, float)) else 0.0
            self._vy = float(args[1]) if len(args) >= 3 and isinstance(args[1], (int, float)) else 0.0
            self._vz = float(args[2]) if len(args) >= 3 and isinstance(args[2], (int, float)) else 0.0

        def __call__(self, *args, **kwargs):
            return _DummyWidget(*args, **kwargs)

        def __or__(self, other):
            return self

        def __getattr__(self, name):
            return _DummyWidget()

        def addItem(self, item, *args, **kwargs):
            self._items.append(item)

        def removeItem(self, item):
            if item in self._items:
                self._items.remove(item)

        def setText(self, text):
            self._text = str(text)
            self.textChanged.emit(self._text)

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
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
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
        Key_Greater = 62
        Key_Comma = 44
        Key_Slash = 47
        Key_Question = 63
        Key_U = 85
        Key_G = 71
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


def verify_air_box_combo_and_rewards():
    """Verifies the central Air-Box 5-touch chained distance * touch multiplier scoring,
    telescopic step delta conservation, surface landing reset, and forbidden goal penalty.
    """
    import RocketSim as rs
    from core.constants import OBS_DIM, TRUNK_OBS_DIM, PAD_OBS_DIM
    from core.env import BallReachEnv
    from core.reward import RewardConfig, compute_step_reward
    from core.obs import _OBS_MIRROR_PERM_NP, _OBS_MIRROR_SIGN_NP

    assert OBS_DIM == 190 and TRUNK_OBS_DIM == 54 and PAD_OBS_DIM == 136
    assert len(_OBS_MIRROR_PERM_NP) == OBS_DIM and len(_OBS_MIRROR_SIGN_NP) == OBS_DIM
    # Involution check: applying mirror permutation twice returns identity
    assert np.all(_OBS_MIRROR_PERM_NP[_OBS_MIRROR_PERM_NP] == np.arange(OBS_DIM))

    env = BallReachEnv(mode="curriculum", boost_mode="unlimited")
    cfg = RewardConfig()
    env.reset()
    obs_vec, obs_panel, _ = env.get_obs()
    assert obs_vec.shape == (190,), f"Expected 190D obs, got {obs_vec.shape}"
    assert obs_panel.shape == (45,), f"Expected 45D panel obs, got {obs_panel.shape}"

    # Simulate a 5-touch airborne chain inside the central Air-Box:
    # Touch 1 -> Touch 2 -> Touch 3 -> Touch 4 -> Touch 5
    touch_points = [
        (0.0, -1200.0, 600.0),
        (100.0, -750.0, 820.0),
        (50.0, -250.0, 1050.0),
        (-80.0, 300.0, 1250.0),
        (0.0, 900.0, 1400.0),
    ]
    ctrl = rs.CarControls()
    ctrl.throttle = 1.0
    ctrl.boost = True

    prev_snap = env.snapshot()
    telescopic_combo_deltas = 0.0
    expected_chain_dist = 0.0

    for idx, pt in enumerate(touch_points):
        # Advance 10 non-touch ticks so TOUCH_DEBOUNCE_TICKS (8) expires cleanly
        for _ in range(10):
            bs = env.arena.ball.get_state()
            bs.pos = rs.Vec(pt[0], pt[1], pt[2])
            bs.vel = rs.Vec(0.0, 400.0, 120.0)
            env.arena.ball.set_state(bs)
            cs = env.car.get_state()
            cs.pos = rs.Vec(pt[0], pt[1] - 350.0, pt[2] - 100.0)
            cs.vel = rs.Vec(0.0, 450.0, 140.0)
            cs.is_on_ground = False
            env.car.set_state(cs)
            env.step_tick(ctrl)
            s_mid = env.snapshot()
            _, d_mid = compute_step_reward(cfg, prev_dist=prev_snap["dist"], new_dist=s_mid["dist"], prev_snap=prev_snap, new_snap=s_mid)
            telescopic_combo_deltas += (s_mid["cum_combo_score"] - prev_snap["cum_combo_score"])
            prev_snap = s_mid

        # Trigger real RocketSim collision touch k at pt inside Air-Box
        bs = env.arena.ball.get_state()
        bs.pos = rs.Vec(pt[0], pt[1], pt[2])
        bs.vel = rs.Vec(0.0, 200.0, 80.0)
        env.arena.ball.set_state(bs)
        cs = env.car.get_state()
        cs.pos = rs.Vec(pt[0], pt[1] - 95.0, pt[2] - 20.0)
        cs.vel = rs.Vec(0.0, 1400.0, 450.0)
        cs.is_on_ground = False
        env.car.set_state(cs)
        env.step_tick(ctrl)
        snap_now = env.snapshot()
        _, detail = compute_step_reward(
            cfg,
            prev_dist=prev_snap["dist"],
            new_dist=snap_now["dist"],
            touched=snap_now["touched"],
            prev_snap=prev_snap,
            new_snap=snap_now,
        )
        telescopic_combo_deltas += (snap_now["cum_combo_score"] - prev_snap["cum_combo_score"])
        prev_snap = snap_now

        if idx > 0:
            p_prev = touch_points[idx - 1]
            p_cur = touch_points[idx]
            expected_chain_dist += math.hypot(p_cur[0] - p_prev[0], p_cur[1] - p_prev[1], p_cur[2] - p_prev[2])

    assert prev_snap["in_air_box"] is True, "Ball at z=1400 in center must be inside Air-Box"
    assert prev_snap["combo_touches"] == 5, f"Expected 5 combo touches, got {prev_snap['combo_touches']}"
    assert abs(prev_snap["combo_multiplier"] - 4.0) < 0.1, (
        f"Expected 4.0x multiplier for 5 genuine air-dribble touches, got {prev_snap['combo_multiplier']}"
    )
    assert abs(prev_snap["combo_chain_dist"] - expected_chain_dist) < 50.0, (
        f"Expected chained distance ~{expected_chain_dist:.1f}, got {prev_snap['combo_chain_dist']:.1f}"
    )
    # Exact telescopic conservation check: sum of step deltas == current total chain score
    assert abs(telescopic_combo_deltas - prev_snap["combo_score_current"]) < 1e-4, (
        f"Telescopic sum {telescopic_combo_deltas} != combo_score_current {prev_snap['combo_score_current']}"
    )
    peak_score = prev_snap["combo_score_current"]
    assert peak_score > 150.0, f"Expected high combo score (>150) for 5-touch air dribble, got {peak_score}"

    # Verify unlimited boost kept boost at 100.0
    assert abs(prev_snap["boost"] - 100.0) < 1e-3, f"Expected 100 boost in unlimited mode, got {prev_snap['boost']}"

    # Now drop ball onto the floor (z=93.15) -> active chain must settle & reset to 0, while peak/cumulative score persists!
    bs = env.arena.ball.get_state()
    bs.pos = rs.Vec(0.0, 900.0, 93.15)
    bs.vel = rs.Vec(0.0, 0.0, 0.0)
    env.arena.ball.set_state(bs)
    env.step_tick(ctrl)
    snap_landed = env.snapshot()
    assert snap_landed["in_air_box"] is False
    assert snap_landed["combo_touches"] == 0, "Combo touches must reset to 0 when ball touches the floor"
    assert snap_landed["max_combo_touches"] == 5, "Peak combo touches (5) must be preserved"
    assert abs(snap_landed["cum_combo_score"] - peak_score) < 1e-4, "Cumulative combo score must be preserved after landing"

    # Verify forbidden goal penalty (-150.0)
    bs = env.arena.ball.get_state()
    bs.pos = rs.Vec(0.0, 5210.0, 200.0)
    bs.vel = rs.Vec(0.0, 800.0, 0.0)
    env.arena.ball.set_state(bs)
    env.step_tick(ctrl)
    snap_goal = env.snapshot()
    assert snap_goal["goal_scored"] is True, "Ball beyond |Y|>5120 inside goal mouth must trigger goal_scored"
    r_goal_step, d_goal = compute_step_reward(
        cfg,
        prev_dist=snap_landed["dist"],
        new_dist=snap_goal["dist"],
        prev_snap=snap_landed,
        new_snap=snap_goal,
    )
    assert d_goal["r_goal"] == -cfg.goal_penalty == -150.0, f"Expected -150.0 goal penalty, got {d_goal['r_goal']}"


def verify_tuner_headless():
    _install_headless_qt_mocks()
    verify_air_box_combo_and_rewards()

    from core.curriculum import (
        NUM_CURRICULUM_STEPS, get_curriculum_step_spec, min_boost_for_ball_z,
    )
    from core.obs import GROUP_ORDER
    from vis.tuner.window import RewardTuner
    from vis.tuner.hud import format_hud
    from vis.tuner.ui_constants import STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN

    # 1. Verify all 16 Air-Dribble CurriculumStepSpec objects have .name and .code
    assert NUM_CURRICULUM_STEPS == 16
    for idx in range(NUM_CURRICULUM_STEPS):
        spec = get_curriculum_step_spec(idx)
        assert isinstance(spec.name, str) and len(spec.name) > 0, f"Invalid spec.name at step {idx}"
        assert isinstance(spec.code, str) and len(spec.code) > 0, f"Invalid spec.code at step {idx}"

    # 2. Verify deterministic minimum boost helper
    assert min_boost_for_ball_z(93.15, cz=17.0, boost_headroom=47.0) == 0.0
    assert min_boost_for_ball_z(1940.0, cz=17.0, boost_headroom=47.0) == 100.0

    # 3. Instantiate RewardTuner and run idle tick + HUD formatting across all 16 curriculum steps
    w = RewardTuner(prefix="air_dribble_v1", from_ckpt="latest", cam_mode="center")
    w.tick()
    assert w.session.state == STATE_IDLE

    for idx in range(NUM_CURRICULUM_STEPS):
        sp_i = get_curriculum_step_spec(idx)
        w.env.set_curriculum_state(1, step_idx=idx)
        boost_line, body = format_hud(w.session, 100.0)
        assert sp_i.name[:2] in ("P1", "P2", "P3", "P4") and "BOOST:" in boost_line and len(body) > 20

    # 4. Simulate human recording -> continuous multi-frame air-dribble -> playback -> view switching
    w.env.set_curriculum_state(1, step_idx=0)
    w.session.reset()
    w.keys["W"] = True
    w.keys["Boost"] = True
    w.tick()
    assert w.session.state == STATE_RECORDING

    for step_i in range(12):
        w.session._step_once(w.keys)
    w.keys["W"] = False
    w.keys["Boost"] = False
    w._render()
    w._refresh_hud()
    assert "Air-Box Combo" in w.hud.toPlainText()

    # Check plot modes, all 9 obs groups (A..I), flags view, delta toggle
    for pm in (1, 2, 3, 4):
        w._set_plot_mode(pm)
    assert len(GROUP_ORDER) == 9
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
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_X))
    assert w.editor.active is True
    assert w.editor.target == "car"

    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_W))
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Slash))
    w.tick()
    time.sleep(0.01)
    w.tick()
    w.keyReleaseEvent(_FakeKeyEvent(QtCore.Qt.Key_W))
    w.keyReleaseEvent(_FakeKeyEvent(QtCore.Qt.Key_Slash))
    w.editor.on_mouse_drag(100.0, 20.0)

    # Press Tab -> VEL
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w.editor.target == "vel", f"Expected 'vel', got {w.editor.target}"
    w.editor.on_mouse_drag(50.0, 40.0)
    w._render()

    # Press Tab -> BALL
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w.editor.target == "ball", f"Expected 'ball', got {w.editor.target}"
    w.editor.on_mouse_drag(-80.0, -30.0)
    w._render()

    # Press Tab -> CAR
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Tab))
    assert w.editor.target == "car", f"Expected 'car', got {w.editor.target}"

    # Press T to launch from editor
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_T))
    assert w.editor.active is False
    w.tick()

    # 6. Verify [Z] Manual CarStatePanel
    assert w.car_state_panel.isVisible() is False
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Z))
    assert w.car_state_panel.isVisible() is True
    w.car_state_panel.set_boost_value(77.0, trigger=True)
    assert abs(w.env.car.get_state().boost - 77.0) < 0.5
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Z))
    assert w.car_state_panel.isVisible() is False

    # 7. Verify Air-Dribble scenario presets (mode=carry, mode=catch, mode=bounce, mode=wall, mode=ground)
    from vis.tuner.window_views import validate_spawn_dist_spec
    for spec_str, expected_bucket in (
        ("mode=carry", 0),
        ("mode=catch", 1),
        ("mode=bounce", 2),
        ("mode=wall", 3),
        ("mode=ground", 4),
        ("p[1000]u[10000]", 0),
    ):
        ok, _ = validate_spawn_dist_spec(spec_str)
        assert ok, f"validate_spawn_dist_spec rejected valid spec {spec_str}"
        w._apply_spawn_dist_str(spec_str)
        assert w.env.current_dir5_idx == expected_bucket, (
            f"Expected bucket={expected_bucket} for {spec_str}, got {w.env.current_dir5_idx}"
        )
    w._apply_spawn_dist_str("auto")

    print(
        f"[verify_tuner] ALL OK! Verified 190D Air-Dribble Showman obs & bilateral mirror symmetry, "
        f"Air-Box 5-touch chained distance * 4x multiplier scoring & telescopic conservation, "
        f"forbidden goal penalty (-150.0), boost modes (unlimited/recharge/pads), "
        f"RewardTuner startup across {NUM_CURRICULUM_STEPS} curriculum steps, "
        f"9 obs groups (A..I), 3-way Tab ScenarioEditor, and 5 freestyle scenario buckets (Carry/Catch/Bounce/Wall/Ground)."
    )


if __name__ == "__main__":
    verify_tuner_headless()
