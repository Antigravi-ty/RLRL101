"""键盘事件处理 (1v1 Fastest Kickoff Tuner)。"""
import os
from pyqtgraph.Qt import QtCore, QtWidgets

from core.constants import NUM_KICKOFF_SLOTS
from .ui_constants import (
    STATE_IDLE, STATE_PLAYBACK, STATE_DONE,
    KEY_MAP, SPEED_STEP,
)
from .export import export_frames


_SHIFT_SLOT_MAP = {}
try:
    _SHIFT_SLOT_MAP = {
        QtCore.Qt.Key_Exclam: 0,
        QtCore.Qt.Key_At: 1,
        QtCore.Qt.Key_NumberSign: 2,
        QtCore.Qt.Key_Dollar: 3,
        QtCore.Qt.Key_Percent: 4,
    }
except Exception:
    pass


class KeysMixin:
    def focusNextPrevChild(self, next_child):
        return False

    def keyPressEvent(self, e):
        k = e.key()
        sess = self._sess()
        mods = e.modifiers()

        if k == QtCore.Qt.Key_Escape:
            QtWidgets.QApplication.quit()
            return

        if k == QtCore.Qt.Key_B:
            if mods & QtCore.Qt.ShiftModifier:
                self._cycle_ckpt_prefix()
            else:
                self._toggle_ckpt_source()
            return

        # N: 重新切换一个新的开球位 (New Kickoff Slot)
        if k == QtCore.Qt.Key_N:
            self._reset_session(replay_last=False)
            return

        # R: 重置并重新执行当前开球位 (Replay Current Kickoff Slot)
        if k == QtCore.Qt.Key_R:
            self._reset_session(replay_last=True)
            return

        if k == QtCore.Qt.Key_V:
            self._toggle_camera_mode()
            return

        if k == QtCore.Qt.Key_Y:
            self._prompt_camera_params()
            return

        if k == QtCore.Qt.Key_Space:
            self._toggle_pov_submode()
            return

        if k == QtCore.Qt.Key_O:
            self._cycle_view_mode()
            return

        if k == QtCore.Qt.Key_P:
            self._toggle_delta()
            return

        if k == QtCore.Qt.Key_F:
            self._toggle_freerun()
            return

        if k == QtCore.Qt.Key_U:
            self._prompt_trail_duration()
            return

        if k == QtCore.Qt.Key_G:
            self._toggle_plot_visibility()
            return

        if k in (QtCore.Qt.Key_Tab, QtCore.Qt.Key_Backtab):
            delta = -1 if (k == QtCore.Qt.Key_Backtab or (mods & QtCore.Qt.ShiftModifier)) else 1
            self._cycle_sub_curve_or_obs_group(delta)
            return

        # Shift + 1..5: 锁定指定开球位 0..4; Shift + 0: 恢复自动轮换开球位
        if (
            (mods & QtCore.Qt.ShiftModifier)
            or (k in _SHIFT_SLOT_MAP)
            or (k == QtCore.Qt.Key_ParenRight)
        ):
            s_idx = None
            if QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_5:
                s_idx = k - QtCore.Qt.Key_1
            elif k in _SHIFT_SLOT_MAP:
                s_idx = _SHIFT_SLOT_MAP[k]
            if s_idx is not None and 0 <= s_idx < NUM_KICKOFF_SLOTS:
                self._set_kickoff_slot(s_idx)
                return
            if k in (QtCore.Qt.Key_0, QtCore.Qt.Key_ParenRight):
                self._set_kickoff_slot(None)
                return

        # 9 / 0: 切换 #1 BLUE / #2 ORANGE 视角车
        if k == QtCore.Qt.Key_9:
            self._cycle_focused_car(-1)
            return
        if k == QtCore.Qt.Key_0:
            self._cycle_focused_car(+1)
            return

        if self.view_mode == "obs":
            if QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_7:
                self._set_obs_group(k - QtCore.Qt.Key_1)
                return
        elif self.view_mode == "reward":
            if k == QtCore.Qt.Key_1: self._set_plot_mode(1); return
            if k == QtCore.Qt.Key_2: self._set_plot_mode(2); return
            if k == QtCore.Qt.Key_3: self._set_plot_mode(3); return
            if k == QtCore.Qt.Key_4: self._set_plot_mode(4); return
            if k == QtCore.Qt.Key_5: self._set_plot_mode(5); return
            if k == QtCore.Qt.Key_6: self._set_plot_mode(6); return

        if k == QtCore.Qt.Key_Minus: self._adjust_speed(-SPEED_STEP); return
        if k in (QtCore.Qt.Key_Equal, QtCore.Qt.Key_Plus): self._adjust_speed(+SPEED_STEP); return

        if k == QtCore.Qt.Key_M: self._toggle_auto(); return
        if k == QtCore.Qt.Key_H: self._toggle_human_vs_model(); return
        if k == QtCore.Qt.Key_T: self._toggle_deterministic(); return
        if k == QtCore.Qt.Key_L: self._toggle_auto_reset(); return
        if k == QtCore.Qt.Key_C: self._copy_hud(); return
        if k == QtCore.Qt.Key_X: self._takeover_from_current_frame(start_immediately=False); return

        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            mode = "compact" if (mods & QtCore.Qt.ShiftModifier) else "full"
            fn = export_frames(sess.frames, mode=mode)
            if fn:
                self._refresh_ui_state(f"已导出CSV: {os.path.basename(fn)}")
            return

        if sess.state in (STATE_PLAYBACK, STATE_DONE):
            if k in (
                QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_BracketRight,
                QtCore.Qt.Key_I, QtCore.Qt.Key_K,
                QtCore.Qt.Key_Comma, QtCore.Qt.Key_Semicolon,
                QtCore.Qt.Key_J, QtCore.Qt.Key_Apostrophe,
            ):
                self._pending_reset = None
            if k == QtCore.Qt.Key_Left:          self._seek(-1);  return
            if k == QtCore.Qt.Key_Right:         self._seek(+1);  return
            if k == QtCore.Qt.Key_BracketLeft:   self._seek(-30); return
            if k == QtCore.Qt.Key_BracketRight:  self._seek(+30); return
            if k == QtCore.Qt.Key_I:             self._seek(-60); return
            if k == QtCore.Qt.Key_K:             self._seek(+60); return
            if k == QtCore.Qt.Key_Comma:         self._seek(-10); return
            if k == QtCore.Qt.Key_Semicolon:     self._seek(+10); return
            if k in (QtCore.Qt.Key_J, QtCore.Qt.Key_Apostrophe):
                sess.toggle_playback()
                self._refresh_ui_state()
                return

        if k in KEY_MAP:
            self.keys[KEY_MAP[k]] = True
            # 若在回放/结算状态已经回溯到了中途位置，或处于 MODEL 模式回放中，按任一操作键即可直接从当前回溯帧接管继续跑!
            if sess.state in (STATE_PLAYBACK, STATE_DONE) and sess.frames:
                is_rewound = (sess.playback_idx < len(sess.frames) - 1)
                if is_rewound or self.use_auto:
                    self._takeover_from_current_frame(start_immediately=True)
                    e.accept()
                    return
            if not self.use_auto and sess.state == STATE_IDLE:
                sess.begin_recording()
                self._refresh_ui_state()
            e.accept()
            return

        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        k = e.key()
        if k in KEY_MAP:
            self.keys[KEY_MAP[k]] = False
        super().keyReleaseEvent(e)
