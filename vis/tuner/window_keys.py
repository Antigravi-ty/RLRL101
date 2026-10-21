"""键盘事件处理。"""
from pyqtgraph.Qt import QtCore, QtWidgets

from core.constants import NUM_CURRICULUM_STAGES
from .ui_constants import (
    STATE_IDLE, STATE_PLAYBACK, STATE_DONE, STATE_FREERUN,
    KEY_MAP, SPEED_STEP,
)
from .export import export_frames


_EDIT_HOLD_MAP = {}
_SHIFT_STAGE_MAP = {}
try:
    _EDIT_HOLD_MAP = {
        QtCore.Qt.Key_W: "W",
        QtCore.Qt.Key_A: "A",
        QtCore.Qt.Key_S: "S",
        QtCore.Qt.Key_D: "D",
        QtCore.Qt.Key_Slash: "UP",
        QtCore.Qt.Key_Question: "UP",
        QtCore.Qt.Key_Up: "UP",
        QtCore.Qt.Key_Period: "DOWN",
        QtCore.Qt.Key_Greater: "DOWN",
        QtCore.Qt.Key_Down: "DOWN",
        QtCore.Qt.Key_Q: "Q",
        QtCore.Qt.Key_E: "E",
        QtCore.Qt.Key_BracketLeft: "LBRACKET",
        QtCore.Qt.Key_BracketRight: "RBRACKET",
        QtCore.Qt.Key_BraceLeft: "LBRACKET",
        QtCore.Qt.Key_BraceRight: "RBRACKET",
        QtCore.Qt.Key_9: "9",
        QtCore.Qt.Key_0: "0",
    }
    _SHIFT_STAGE_MAP = {
        QtCore.Qt.Key_Exclam: 0,
        QtCore.Qt.Key_At: 1,
        QtCore.Qt.Key_NumberSign: 2,
        QtCore.Qt.Key_Dollar: 3,
    }
except Exception:
    pass


_TARGET_CN = {
    "car": "CAR (车头方向)",
    "vel": "VEL (init速度方向)",
    "ball": "BALL (球的发射方向)",
}


class KeysMixin:
    def focusNextPrevChild(self, next_child):
        if getattr(self, "editor", None) is not None and self.editor.active:
            return False
        return super().focusNextPrevChild(next_child)

    def keyPressEvent(self, e):
        k = e.key()
        sess = self._sess()
        mods = e.modifiers()

        if k == QtCore.Qt.Key_Escape:
            if getattr(self, "editor", None) is not None and self.editor.active:
                self._toggle_edit_mode()
                return
            QtWidgets.QApplication.quit(); return

        if k == QtCore.Qt.Key_X:
            self._toggle_edit_mode()
            return

        if k == QtCore.Qt.Key_Z:
            self._toggle_car_state_panel()
            return

        if getattr(self, "editor", None) is not None and self.editor.active:
            if e.isAutoRepeat():
                e.accept(); return
            if k == QtCore.Qt.Key_Tab:
                t_now = self.editor.toggle_target()
                if hasattr(self.view, "_temp_pov_azim"):
                    self.view._temp_pov_azim = 0.0
                    self.view._temp_pov_elev = 0.0
                self._render()
                self._refresh_ui_state(f"编辑目标切换为: {_TARGET_CN.get(t_now, t_now.upper())}")
                e.accept(); return
            if k in (QtCore.Qt.Key_Y, QtCore.Qt.Key_H):
                self._prompt_camera_params(); e.accept(); return
            if k == QtCore.Qt.Key_0 and self.editor.target in ("car", "vel"):
                self.editor.has_second_jump = not self.editor.has_second_jump
                self.editor.sync_to_arena()
                if hasattr(self, "car_state_panel") and self.car_state_panel is not None:
                    self.car_state_panel.sync_from_state(self.editor.car_boost, self.editor.has_second_jump)
                self._refresh_ui_state(f"Has Jump = {'ON(可跳/可二段跳)' if self.editor.has_second_jump else 'OFF(已消耗)'}")
                e.accept(); return
            if k in (QtCore.Qt.Key_T, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self._launch_from_editor()
                e.accept(); return
            if k in _EDIT_HOLD_MAP:
                self.editor._held.add(_EDIT_HOLD_MAP[k])
                e.accept(); return
            e.accept(); return

        if k == QtCore.Qt.Key_B:
            if mods & QtCore.Qt.ShiftModifier:
                self._cycle_ckpt_prefix()
            else:
                self._toggle_ckpt_source()
            return
        if k == QtCore.Qt.Key_N:
            self._reset_session(replay_last=False)
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
            self._cycle_view_mode(); return

        if k == QtCore.Qt.Key_P:
            self._toggle_delta(); return

        if k == QtCore.Qt.Key_F:
            self._toggle_freerun(); return

        if k == QtCore.Qt.Key_U:
            self._prompt_tuning_params(); return

        if k == QtCore.Qt.Key_G:
            self._toggle_plot_visibility(); return

        # Shift + 1..4: Quick-lock Phase 1..4; Shift+5: Carry; Shift+6: Catch; Shift+7: Wall; Shift+8: Ground; Shift+0: restore auto
        if (
            (mods & QtCore.Qt.ShiftModifier)
            or (k in _SHIFT_STAGE_MAP)
            or (k in (QtCore.Qt.Key_ParenRight, QtCore.Qt.Key_Percent, QtCore.Qt.Key_AsciiCircum, QtCore.Qt.Key_Ampersand, QtCore.Qt.Key_Asterisk))
        ):
            if k in (QtCore.Qt.Key_5, QtCore.Qt.Key_Percent):
                self._apply_spawn_dist_str("mode=carry")
                return
            if k in (QtCore.Qt.Key_6, QtCore.Qt.Key_AsciiCircum):
                self._apply_spawn_dist_str("mode=catch")
                return
            if k in (QtCore.Qt.Key_7, QtCore.Qt.Key_Ampersand):
                self._apply_spawn_dist_str("mode=wall")
                return
            if k in (QtCore.Qt.Key_8, QtCore.Qt.Key_Asterisk):
                self._apply_spawn_dist_str("mode=ground")
                return
            s_idx = None
            if QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_4:
                s_idx = k - QtCore.Qt.Key_1
            elif k in _SHIFT_STAGE_MAP:
                s_idx = _SHIFT_STAGE_MAP[k]
            if s_idx is not None and 0 <= s_idx < NUM_CURRICULUM_STAGES:
                weights = ["0"] * NUM_CURRICULUM_STAGES
                weights[s_idx] = "1"
                spec = f"p[{''.join(weights)}]"
                self._apply_spawn_dist_str(spec)
                return
            if k in (QtCore.Qt.Key_0, QtCore.Qt.Key_ParenRight):
                self._apply_spawn_dist_str("")
                return

        if self.view_mode == "obs":
            if QtCore.Qt.Key_1 <= k <= QtCore.Qt.Key_9:
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
        if k == QtCore.Qt.Key_T: self._toggle_deterministic(); return
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
            if k in (QtCore.Qt.Key_J, QtCore.Qt.Key_Apostrophe):
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
        if getattr(self, "editor", None) is not None and self.editor.active:
            if not e.isAutoRepeat() and k in _EDIT_HOLD_MAP:
                self.editor._held.discard(_EDIT_HOLD_MAP[k])
            e.accept(); return
        if k in KEY_MAP:
            self.keys[KEY_MAP[k]] = False
        super().keyReleaseEvent(e)
