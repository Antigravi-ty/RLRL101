"""键盘事件处理。"""
from pyqtgraph.Qt import QtCore, QtWidgets

from .ui_constants import (
    STATE_IDLE, STATE_PLAYBACK, STATE_DONE, STATE_FREERUN,
    KEY_MAP, SPEED_STEP,
)
from .export import export_frames


class KeysMixin:
    def keyPressEvent(self, e):
        k = e.key()
        sess = self._sess()
        mods = e.modifiers()

        if k == QtCore.Qt.Key_Escape:
            QtWidgets.QApplication.quit(); return

        if k == QtCore.Qt.Key_B: self._toggle_ckpt_source(); return
        if k == QtCore.Qt.Key_G: self._cycle_ckpt_prefix(); return
        if k == QtCore.Qt.Key_N:
            if mods & QtCore.Qt.ShiftModifier:
                self._cycle_ckpt_prefix()
            else:
                self._reset_session(replay_last=False)
            return

        if k == QtCore.Qt.Key_O:
            self._cycle_view_mode(); return

        if k == QtCore.Qt.Key_P:
            self._toggle_delta(); return

        if k == QtCore.Qt.Key_F:
            self._toggle_freerun(); return

        if self.view_mode == "obs":
            if QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_8:
                self._set_obs_group(k - QtCore.Qt.Key_1); return
        elif self.view_mode == "reward":
            if k == QtCore.Qt.Key_1: self._set_plot_mode(1); return
            if k == QtCore.Qt.Key_2: self._set_plot_mode(2); return
            if k == QtCore.Qt.Key_3: self._set_plot_mode(3); return
            if k == QtCore.Qt.Key_4: self._set_plot_mode(4); return
            if k == QtCore.Qt.Key_5: self._set_plot_mode(5); return
            if k == QtCore.Qt.Key_6: self._set_plot_mode(6); return

        if k == QtCore.Qt.Key_Minus: self._adjust_speed(-SPEED_STEP); return
        if k == QtCore.Qt.Key_Equal: self._adjust_speed(+SPEED_STEP); return

        if k == QtCore.Qt.Key_M: self._toggle_auto(); return
        if k == QtCore.Qt.Key_L: self._toggle_auto_reset(); return
        if k == QtCore.Qt.Key_C: self._copy_hud(); return
        if k == QtCore.Qt.Key_R: self._reset_session(replay_last=True); return

        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            import os
            mode = "compact" if (mods & QtCore.Qt.ShiftModifier) else "full"
            fn = export_frames(sess.frames, mode=mode)
            if fn:
                self._refresh_ui_state(f"已保存: {os.path.basename(fn)}")
            return

        if sess.state in (STATE_PLAYBACK, STATE_DONE):
            if k == QtCore.Qt.Key_Left:          self._seek(-1);  return
            if k == QtCore.Qt.Key_Right:         self._seek(+1);  return
            if k == QtCore.Qt.Key_BracketLeft:   self._seek(-30); return
            if k == QtCore.Qt.Key_BracketRight:  self._seek(+30); return
            if k == QtCore.Qt.Key_I:             self._seek(-60); return
            if k == QtCore.Qt.Key_K:             self._seek(+60); return
            if k == QtCore.Qt.Key_Comma:         self._seek(-10); return
            if k == QtCore.Qt.Key_Semicolon:     self._seek(+10); return
            if k == QtCore.Qt.Key_Space:
                sess.toggle_playback()
                self._refresh_ui_state(); return

        if k in KEY_MAP:
            self.keys[KEY_MAP[k]] = True
            if not self.use_auto and sess.state == STATE_IDLE:
                sess.begin_recording()
                self._refresh_ui_state()
            e.accept(); return

        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        k = e.key()
        if k in KEY_MAP:
            self.keys[KEY_MAP[k]] = False
        super().keyReleaseEvent(e)
