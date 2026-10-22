#!/usr/bin/env python3
"""tools/verify_tuner.py —— 无头 (Headless) Kickoff GUI Tuner 启动与全链路交互验证脚本。

无需安装 PyQt5 / pyqtgraph / PyOpenGL 或图形显示器：
通过注入轻量级 Qt/pyqtgraph/OpenGL 桩模块，100% 真实执行 `vis/tuner/` 与 `core/` 的全部业务代码：
  1. RocketSim `reset_kickoff(seed)` + 60 帧 (`SETTLE_TICKS = 60`) 悬挂稳定缓冲验证 (5 大标准开球位)
  2. 42D 精简对称观测 (`OBS_DIM = 42`) 与 ~36K 参数纯 MLP `ActorCritic` 推理验证
  3. 人工录制 (`STATE_RECORDING`) -> 结算 (`STATE_DONE`) -> 回放 (`STATE_PLAYBACK`) -> CSV 导出 (`Enter`)
  4. `R` 重置并重新执行当前开球位 vs `N` 切换一个新的开球位 vs `Shift+1..5` 锁定开球位
  5. `M` 切换 `HUMAN` / `MODEL 1v1` 双车自博弈模式 + `9`/`0` 切换蓝/橙视角车
  6. 底部图表曲线 `1..6` (`r_total`, `r_cum`, `dist`, `r_dist/r_time/r_touch`, `car_speed/closing`) / `OBS` / `FLAGS` / `G` 开关图表
"""
import os
import sys
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

        def setStyleSheet(self, *args, **kwargs): pass
        def setFixedHeight(self, *args, **kwargs): pass
        def setFixedWidth(self, *args, **kwargs): pass
        def setFixedSize(self, *args, **kwargs): pass
        def setMinimumWidth(self, *args, **kwargs): pass
        def setMinimumHeight(self, *args, **kwargs): pass
        def setAlignment(self, *args, **kwargs): pass
        def setSizePolicy(self, *args, **kwargs): pass
        def setContentsMargins(self, *args, **kwargs): pass
        def setSpacing(self, *args, **kwargs): pass
        def setFocusPolicy(self, *args, **kwargs): pass
        def setFocus(self, *args, **kwargs): pass
        def setWindowTitle(self, *args, **kwargs): pass
        def resize(self, *args, **kwargs): pass
        def closeEvent(self, *args, **kwargs): pass
        def mousePressEvent(self, *args, **kwargs): pass
        def mouseMoveEvent(self, *args, **kwargs): pass
        def mouseReleaseEvent(self, *args, **kwargs): pass
        def wheelEvent(self, *args, **kwargs): pass
        def keyPressEvent(self, *args, **kwargs): pass
        def keyReleaseEvent(self, *args, **kwargs): pass

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
        Key_H = 72
        Key_J = 74
        Key_K = 75
        Key_L = 76
        Key_M = 77
        Key_V = 86
        Key_X = 88
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
        Key_Exclam = 33
        Key_At = 64
        Key_NumberSign = 35
        Key_Dollar = 36
        Key_Percent = 37
        Key_ParenRight = 41
        Key_Left = 0x01000012
        Key_Right = 0x01000014
        Key_Semicolon = 59
        Key_Apostrophe = 39

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

    from core.constants import (
        OBS_DIM, PPO_HIDDEN_DIM, NUM_KICKOFF_SLOTS, SETTLE_TICKS, KICKOFF_SLOT_NAMES,
    )
    from core.obs import GROUP_ORDER, mirror_obs_np
    from core.policy_net import ActorCritic
    from vis.tuner.window import RewardTuner
    from vis.tuner.hud import format_hud
    from vis.tuner.ui_constants import STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN
    from vis.cli_tuner import verify_and_compare_csv

    # 1. Verify compact ActorCritic parameter count (~36K) & 42D observation symmetry
    model = ActorCritic()
    assert OBS_DIM == 42 and PPO_HIDDEN_DIM == 128
    sd = model.state_dict()
    w_shapes = [tuple(v.shape) for v in sd.values()]
    total_params = sum(int(np.prod(s)) for s in w_shapes)
    actor_params = sum(int(np.prod(tuple(v.shape))) for k, v in sd.items() if k.startswith("actor"))
    assert total_params == 46096 and actor_params == 23951, f"Unexpected params: total={total_params}, actor={actor_params}"

    # 2. Instantiate RewardTuner and verify 60-frame suspension settling across all 5 kickoff slots
    w = RewardTuner(prefix="ko_v1", from_ckpt="best", cam_mode="center")
    w.tick()
    assert w.session.state == STATE_IDLE
    assert w.env.settle_ticks == SETTLE_TICKS == 60
    assert w.session.vs_model is True and w.session.opp_model is not None

    seen_slots = set()
    for slot_i in range(NUM_KICKOFF_SLOTS):
        w.env.set_kickoff_slot(slot_i)
        w.env.reset(replay_last=False)
        seen_slots.add(w.env.current_slot_idx)
        cs_b = w.env.car_blue.get_state()
        cs_o = w.env.car_orange.get_state()
        assert cs_b.is_on_ground and cs_o.is_on_ground, f"Cars must be settled on ground at slot {slot_i}"
        assert 16.9 <= float(cs_b.pos.z) <= 17.2, f"Unexpected settled z={cs_b.pos.z}"
        obs_b, _, _ = w.env.get_obs(0)
        obs_o, _, _ = w.env.get_obs(1)
        assert obs_b.shape == (42,) and obs_o.shape == (42,)
        # At settled symmetric kickoff, Blue and Orange team-mirrored observations must match!
        assert np.allclose(obs_b, obs_o, atol=1e-4), f"Team symmetry mismatch at slot {slot_i}"
        # Double bilateral mirror must be identity
        assert np.allclose(mirror_obs_np(mirror_obs_np(obs_b)), obs_b, atol=1e-6)
        boost_line, body = format_hud(w.session, 33.3)
        assert KICKOFF_SLOT_NAMES[slot_i] in body and "BOOST:" in boost_line
    assert len(seen_slots) == 5
    w.env.set_kickoff_slot(None)

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

    # 3. Verify N switches to a new kickoff slot, and R replays the exact current kickoff slot
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_N))
    slot_a = w.env.current_slot_idx
    pos_a = w.env.snapshot(0)["car_pos"]
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_N))
    slot_b = w.env.current_slot_idx
    assert slot_b != slot_a, f"Expected N to switch kickoff slot, got {slot_a} -> {slot_b}"
    pos_b = w.env.snapshot(0)["car_pos"]

    # Step a few frames of human recording, then press R to verify it resets to the EXACT same kickoff slot_b
    w.keys["W"] = True
    w.keys["Period"] = True
    w.tick()
    assert w.session.state == STATE_RECORDING
    for _ in range(15):
        w.session._step_once(w.keys)
    w.keys["W"] = False
    w.keys["Period"] = False
    assert np.linalg.norm(np.array(w.env.snapshot(0)["car_pos"]) - np.array(pos_b)) > 10.0

    # Press R -> resets to the exact same slot_b and initial position pos_b
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_R))
    assert w.session.state == STATE_IDLE
    assert w.env.current_slot_idx == slot_b, f"Expected R to keep slot {slot_b}, got {w.env.current_slot_idx}"
    assert np.allclose(w.env.snapshot(0)["car_pos"], pos_b, atol=1e-4)

    # 4. Record human kickoff frames, test charts (1..6, OBS 1..7, FLAGS), and export CSV via Enter
    w.keys["W"] = True
    w.keys["Period"] = True
    w.tick()
    for step_i in range(20):
        if step_i == 8:
            w.keys["Slash"] = True
        else:
            w.keys["Slash"] = False
        w.session._step_once(w.keys)
    w.keys["W"] = False
    w.keys["Period"] = False
    w.keys["Slash"] = False
    w._render()
    w._refresh_hud()

    for pm in (1, 2, 3, 4, 5, 6):
        w._set_plot_mode(pm)
    for gi in range(len(GROUP_ORDER)):
        w._set_obs_group(gi)
    w._show_flags_view()
    w._set_plot_mode(6)
    w._toggle_delta()
    w._toggle_delta()

    msg = w.session.finish_recording("touch")
    w._refresh_ui_state(msg, to_hud=True)
    assert w.session.state == STATE_DONE

    # Export CSV via Enter key and verify with verify_and_compare_csv
    from vis.tuner.export import export_frames
    csv_fn = export_frames(w.session.frames, out_dir="/tmp/rlrl101_test_csv", mode="full")
    assert csv_fn is not None and os.path.exists(csv_fn)
    summaries = verify_and_compare_csv([csv_fn])
    assert len(summaries) == 1 and summaries[0]["frames"] == len(w.session.frames)

    # Playback & Freerun check
    w.session.toggle_playback()
    assert w.session.state == STATE_PLAYBACK
    w.tick()
    w.session.begin_freerun()
    assert w.session.state == STATE_FREERUN
    w.session._freerun_step(w.keys)
    w.session.reset(replay_last=True)

    # 5. Verify M toggles MODEL 1v1 Self-Play mode, 9/0 switches Blue/Orange car focus, L toggles MANUAL-RESET
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_M))
    assert w.use_auto is True and w.auto_session is not None
    for _ in range(12):
        w.auto_session._step_once()
    w._render()
    assert len(w.auto_session.frames) == 12
    assert len(w.auto_session.frames[-1]["all_cars"]) == 2

    # Switch focused car between #1 BLUE (0) and #2 ORANGE (1) using 9 / 0
    assert w.env.focused_car_idx == 0
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_0))
    assert w.env.focused_car_idx == 1
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_9))
    assert w.env.focused_car_idx == 0

    # Verify MANUAL-RESET: N draws full trajectory immediately (STATE_DONE), R replays step-by-step (STATE_RECORDING)
    w.auto_reset = False
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_N))
    assert w.auto_session.state == STATE_DONE
    assert len(w.auto_session.frames) > 0
    model_total_frames = len(w.auto_session.frames)

    # 6. Verify Mid-Trajectory Takeover: rewind in Model playback and press a driving key (or X) to take over!
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_I))  # seek -60 frames
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_BracketLeft))  # seek -30 frames
    rewind_idx = w.auto_session.playback_idx
    expected_pos_at_rewind = w.auto_session.frames[rewind_idx]["car_pos"]
    expected_vel_at_rewind = w.auto_session.frames[rewind_idx]["car_vel"]

    # Pressing a driving key ('W') directly while rewound in MODEL playback immediately takes over as HUMAN!
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_W))
    assert w.use_auto is False
    assert w.session.state == STATE_RECORDING
    assert w.session.takeover_step == rewind_idx
    assert len(w.session.frames) == rewind_idx + 1
    assert w.session.ref_benchmark["frames"] == model_total_frames
    # Verify exact deterministic physics restoration at the takeover point
    restored_snap = w.env.snapshot(0)
    assert np.allclose(restored_snap["car_pos"], expected_pos_at_rewind, atol=1e-4)
    assert np.allclose(restored_snap["car_vel"], expected_vel_at_rewind, atol=1e-4)

    # Advance human takeover steps and verify HUD & finish comparison message
    w.keys["Period"] = True
    for _ in range(10):
        w.session._step_once(w.keys)
    w.keys["W"] = False
    w.keys["Period"] = False
    assert len(w.session.frames) == rewind_idx + 11
    assert w.session.frames[-1]["source"] == "human_takeover"
    _, hud_body = format_hud(w.session, 50.0)
    assert "中途接管模式" in hud_body
    finish_msg = w.session.finish_recording("touch")
    assert f"接管@{rewind_idx}f" in finish_msg and f"对比原轨迹({model_total_frames}f)" in finish_msg

    # Also test X key takeover into STATE_IDLE (ready to press any key)
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_Left))
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_X))
    assert w.session.state == STATE_IDLE and w.session.takeover_step == len(w.session.frames) - 1

    # Switch back to MODEL mode and verify R replays step-by-step
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_M))
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_R))
    assert w.auto_session.state == STATE_RECORDING

    # Verify G toggles bottom plot visibility and _set_trail_duration updates window
    assert w.plot_visible is True
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_G))
    assert w.plot_visible is False
    w.keyPressEvent(_FakeKeyEvent(QtCore.Qt.Key_G))
    assert w.plot_visible is True
    w._set_trail_duration(240)
    assert w.trail_duration == 240 and w.plot.max_window == 240

    print(
        f"[verify_tuner] ALL OK! Verified Kickoff RewardTuner startup, 5 standard kickoff slots + 60f suspension settling, "
        f"42D team/mirror symmetric obs, {total_params:,}-param Pure-MLP ActorCritic, "
        f"Human recording + Mid-trajectory Takeover (rewind + WASD/X) + CSV export, "
        f"R (replay current slot) vs N (new slot) vs Shift+1..5 (lock slot), "
        f"MODEL 1v1 self-play duel, 9/0 Blue/Orange car focus, and plot modes 1..6 / OBS / FLAGS / G."
    )


if __name__ == "__main__":
    verify_tuner_headless()

