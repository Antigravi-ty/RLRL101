"""键盘事件处理 (支持 1/2/3 镜头快速切换、Free-Cam WASD+鼠标漫游与按键屏蔽锁、K 切换可达集推理引擎、FREERUN 防 CSV 爆内存)。"""
from pyqtgraph.Qt import QtCore, QtWidgets

from core.constants import NUM_CURRICULUM_STAGES
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_PLAYBACK, STATE_DONE, STATE_FREERUN,
    KEY_MAP, SPEED_STEP,
)
from .export import export_frames


_EDIT_HOLD_MAP = {}
_FREE_CAM_HOLD_MAP = {}
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
    _FREE_CAM_HOLD_MAP = {
        QtCore.Qt.Key_W: "W",
        QtCore.Qt.Key_A: "A",
        QtCore.Qt.Key_S: "S",
        QtCore.Qt.Key_D: "D",
        QtCore.Qt.Key_E: "UP",
        QtCore.Qt.Key_Space: "UP",
        QtCore.Qt.Key_Slash: "UP",
        QtCore.Qt.Key_Q: "DOWN",
        QtCore.Qt.Key_Shift: "DOWN",
        QtCore.Qt.Key_Period: "DOWN",
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
        if getattr(self.view, "cam_mode", "center") == "free":
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
            if getattr(self.view, "cam_mode", "center") == "free":
                self._select_camera_slot(1)
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

        # Camera Slots 1 / 2 / 3 (without Shift/Ctrl/Alt modifiers)
        no_mods = not (mods & (QtCore.Qt.ShiftModifier | QtCore.Qt.ControlModifier | QtCore.Qt.AltModifier))
        if no_mods and k in (QtCore.Qt.Key_1, QtCore.Qt.Key_2, QtCore.Qt.Key_3):
            slot = 1 if k == QtCore.Qt.Key_1 else (2 if k == QtCore.Qt.Key_2 else 3)
            self._select_camera_slot(slot)
            e.accept(); return

        # Free Camera special controls when in cam_mode == "free"
        if getattr(self.view, "cam_mode", "center") == "free":
            if k == QtCore.Qt.Key_Tab and not e.isAutoRepeat():
                self.view.free_cam_locked = not getattr(self.view, "free_cam_locked", False)
                self.view._free_held.clear()
                for ck in self.keys:
                    self.keys[ck] = False
                lock_str = (
                    "🔒 视角已固定 (WASD 恢复控车)"
                    if self.view.free_cam_locked
                    else "🕊 自由飞行中 (WASD+QE/Space 移动相机, 已屏蔽控车输入)"
                )
                self._refresh_ui_state(f"FREE-CAM: {lock_str}")
                e.accept(); return
            if k == QtCore.Qt.Key_B and not e.isAutoRepeat():
                if hasattr(self.view, "focus_free_cam_on_blob"):
                    self.view.focus_free_cam_on_blob()
                self._last_rendered_idx = None
                self._render()
                self._refresh_ui_state("FREE-CAM 已对准场外 3D 非规则平滑立体曲面 (拖拽鼠标+WASD多角度观察)")
                e.accept(); return
            if not getattr(self.view, "free_cam_locked", False):
                if k in _FREE_CAM_HOLD_MAP:
                    if not e.isAutoRepeat():
                        self.view._free_held.add(_FREE_CAM_HOLD_MAP[k])
                        self._update_timer_for_state(sess)
                    e.accept(); return

        if k == QtCore.Qt.Key_K and sess.state not in (STATE_PLAYBACK, STATE_DONE):
            if hasattr(self, "reach_engine"):
                new_m = self.reach_engine.toggle_mode()
                self._last_rendered_idx = None
                self._render()
                m_desc = (
                    f"MODEL (双层神经网络+LUT混合, 耗时 {self.reach_engine.last_compute_ms:.2f}ms, 120Hz零阻塞)"
                    if new_m == "model"
                    else f"ROLLOUT (RocketSim 实时确定性物理推演, 耗时 {self.reach_engine.last_compute_ms:.1f}ms)"
                )
                self._refresh_ui_state(f"地面可达集引擎 = {m_desc}")
            return

        if k == QtCore.Qt.Key_L:
            modes = ["FUNNEL", "BUBBLES", "OFF"]
            cur_m = getattr(self, "aerial_iso_mode", "FUNNEL")
            next_idx = (modes.index(cur_m) + 1) % len(modes) if cur_m in modes else 0
            self.aerial_iso_mode = modes[next_idx]
            self._last_rendered_idx = None
            self._render()
            m_cn = {
                "FUNNEL": "FUNNEL (三维时空喇叭锥体: 连续无断层拦截走廊 + 1s/2s/3s 环线 + 纵向导轨)",
                "BUBBLES": "BUBBLES (1s/2s/3s 离散空间等时胞囊 + 经纬肋线)",
                "OFF": "OFF (关闭空中 3D 锥体/胞囊, 仅保留地面/表面等时线)",
            }.get(self.aerial_iso_mode, self.aerial_iso_mode)
            self._refresh_ui_state(f"空中 3D 可达性模式 = {m_cn}")
            return

        if k == QtCore.Qt.Key_B: self._toggle_ckpt_source(); return
        if k == QtCore.Qt.Key_G: self._cycle_ckpt_prefix(); return
        if k == QtCore.Qt.Key_N:
            self._reset_session(replay_last=False)
            return

        if k == QtCore.Qt.Key_V:
            self._toggle_camera_mode()
            return

        if k in (QtCore.Qt.Key_Y, QtCore.Qt.Key_H):
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
            self._prompt_spawn_dist(); return

        # Shift + 1..4: Quick-lock spawn distribution to Phase 1..4 (Shift+0: restore auto)
        if (mods & QtCore.Qt.ShiftModifier) or (k in _SHIFT_STAGE_MAP) or (k == QtCore.Qt.Key_ParenRight):
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
            if QtCore.Qt.Key_4 <= k <= QtCore.Qt.Key_7:
                self._set_obs_group(k - QtCore.Qt.Key_1); return
        elif self.view_mode == "reward":
            if k == QtCore.Qt.Key_4: self._set_plot_mode(4); return
            if k == QtCore.Qt.Key_5: self._set_plot_mode(5); return
            if k == QtCore.Qt.Key_6: self._set_plot_mode(6); return

        if k == QtCore.Qt.Key_Minus: self._adjust_speed(-SPEED_STEP); return
        if k == QtCore.Qt.Key_Equal: self._adjust_speed(+SPEED_STEP); return

        if k == QtCore.Qt.Key_M: self._toggle_auto(); return
        if k == QtCore.Qt.Key_T: self._toggle_deterministic(); return
        if k == QtCore.Qt.Key_C: self._copy_hud(); return
        if k == QtCore.Qt.Key_R: self._reset_session(replay_last=True); return

        if k in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if sess.state == STATE_FREERUN:
                self._refresh_ui_state("⚠ FREERUN 无限时模式下禁止记录/导出 CSV (防止内存与存储爆满)")
                return
            import os
            mode = "compact" if (mods & QtCore.Qt.ShiftModifier) else "full"
            fn = export_frames(sess.frames, mode=mode)
            if fn:
                self._refresh_ui_state(f"已保存: {os.path.basename(fn)}")
            return

        if k == QtCore.Qt.Key_Backspace and not e.isAutoRepeat():
            if self.use_auto and sess.state == STATE_RECORDING and sess.frames and hasattr(sess, "complete_remaining"):
                cur_idx = max(0, len(sess.frames) - 1)
                sess.complete_remaining()
                sess.playback_idx = min(cur_idx, max(0, len(sess.frames) - 2))
                self._takeover_from_playback(start_recording=True, step_first_tick=True)
                return
            if sess.state in (STATE_PLAYBACK, STATE_DONE) and sess.frames:
                if sess.playback_idx < len(sess.frames) - 1:
                    self._takeover_from_playback(start_recording=True, step_first_tick=True)
                else:
                    self._retry_last_takeover()
                return

        if (
            self.use_auto
            and sess.state == STATE_RECORDING
            and sess.frames
            and hasattr(sess, "complete_remaining")
            and k in (
                QtCore.Qt.Key_Left, QtCore.Qt.Key_Right,
                QtCore.Qt.Key_BracketLeft, QtCore.Qt.Key_BracketRight,
                QtCore.Qt.Key_I, QtCore.Qt.Key_Comma, QtCore.Qt.Key_Semicolon,
                QtCore.Qt.Key_J, QtCore.Qt.Key_Apostrophe,
            )
        ):
            cur_idx = max(0, len(sess.frames) - 1)
            sess.complete_remaining()
            sess.playback_idx = cur_idx
            self._pending_reset = None
            self._update_plots_from_frames(sess)
            if k in (QtCore.Qt.Key_J, QtCore.Qt.Key_Apostrophe):
                sess.state = STATE_PLAYBACK
                sess.playback_playing = False
                self._last_rendered_idx = None
                self._render_frame(sess.frames[sess.playback_idx])
                self._refresh_hud()
                self._refresh_ui_state(
                    f"已暂停 Model @f={sess.playback_idx}/{len(sess.frames)-1} [直接按 WASD/./ / 键接管继续跑!]"
                )
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
                self._pending_reset = None
                sess.toggle_playback()
                self._refresh_ui_state(); return

        if k in KEY_MAP:
            free_cam_mask = (
                getattr(self.view, "cam_mode", "center") == "free"
                and not getattr(self.view, "free_cam_locked", False)
            )
            if free_cam_mask:
                e.accept(); return
            self.keys[KEY_MAP[k]] = True
            if sess.state in (STATE_PLAYBACK, STATE_DONE):
                if not e.isAutoRepeat():
                    if sess.frames and sess.playback_idx < len(sess.frames) - 1:
                        self._takeover_from_playback(start_recording=True, step_first_tick=True)
                    elif self.use_auto:
                        self._refresh_ui_state("💡 请先按 ← / [ / , / I 回溯到中途某一帧，再按控制键直接接管继续跑!")
                e.accept(); return
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
        if getattr(self.view, "cam_mode", "center") == "free" and not getattr(self.view, "free_cam_locked", False):
            if not e.isAutoRepeat() and k in _FREE_CAM_HOLD_MAP:
                self.view._free_held.discard(_FREE_CAM_HOLD_MAP[k])
            e.accept(); return
        if k in KEY_MAP:
            self.keys[KEY_MAP[k]] = False
        super().keyReleaseEvent(e)
