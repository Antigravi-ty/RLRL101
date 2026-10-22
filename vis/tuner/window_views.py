import os
import time as _time
try:
    from pyqtgraph.Qt import QtCore, QtWidgets
    _QT_BASE_DIALOG = QtWidgets.QDialog
except ImportError:
    QtCore = None
    QtWidgets = None
    _QT_BASE_DIALOG = object

from core.constants import KICKOFF_SLOT_NAMES
from core.obs import GROUP_ORDER
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN,
)
try:
    from .plot_panel import MODE_NAMES
except ImportError:
    MODE_NAMES = {}


def parse_trail_duration_spec(spec: str):
    """解析轨迹与下方图表持续时间输入 (支持直接输入帧数如 '100', '720'，或秒数如 '0.8s', '2.5s')。"""
    s = str(spec or "").strip().lower()
    if not s:
        return False, 720, "❌ 请输入正整数帧数 (如 100, 300, 720) 或秒数 (如 1.5s, 6s)"
    try:
        if s.endswith("s"):
            sec = float(s[:-1].strip())
            if sec <= 0.0:
                return False, 720, "❌ 持续秒数必须大于 0"
            frames = int(round(sec * 120.0))
        elif s.endswith("f"):
            frames = int(round(float(s[:-1].strip())))
        else:
            frames = int(round(float(s)))
    except Exception:
        return False, 720, f"❌ 无法解析 '{spec}'，请输入帧数 (如 100, 300, 720) 或秒数 (如 2.0s)"

    if frames < 10 or frames > 7200:
        return False, max(10, min(7200, frames)), "❌ 持续帧数范围需在 [10, 7200] 之间 (对应 0.08s ~ 60.0s @ 120Hz)"
    sec_equiv = frames / 120.0
    return True, frames, f"✓ 合法 → 轨迹与下方图表窗口 = {frames} 帧 ({sec_equiv:.2f} 秒 @ 120Hz)"


class TrailDurationDialog(_QT_BASE_DIALOG):
    """弹窗动态修改 3D 轨迹持续时间与下方图表 X 轴滚动窗口宽度。"""

    def __init__(self, parent, current_frames: int = 720):
        super().__init__(parent)
        self.setWindowTitle("调整轨迹与图表持续时间 (Trail & Graph Duration)")
        self.setMinimumWidth(620)
        self.setStyleSheet("QDialog { background-color: #f5f5f5; color: #000000; }")
        layout = QtWidgets.QVBoxLayout(self)

        help_lbl = QtWidgets.QLabel(
            "修改 3D 车辆轨迹 (含 Jump/DoubleJump 触发点) 与下方图表 X 轴动态滚动窗口的持续时长:\n"
            "• 直接输入帧数 (120Hz 物理步): 例如输入 240 为 2.00s 开球冲刺窗口，输入 720 为完整 6.00s 开球上限\n"
            "• 也支持直接输入带 's' 的秒数: 例如输入 2.5s 自动换算为 300 帧"
        )
        help_lbl.setStyleSheet("font-family:Consolas;font-size:12px;color:#000000;background:transparent;")
        layout.addWidget(help_lbl)

        self.edit = QtWidgets.QLineEdit(str(int(current_frames)))
        self.edit.setStyleSheet(
            "font-family:Consolas;font-size:14px;font-weight:bold;"
            "color:#000000;background-color:#ffffff;border:1px solid #888888;padding:6px;"
        )
        layout.addWidget(self.edit)

        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setStyleSheet("font-family:Consolas;font-size:12px;font-weight:bold;padding:4px;")
        layout.addWidget(self.status_lbl)

        preset_row = QtWidgets.QHBoxLayout()
        presets = [
            ("120帧 (1.0s)", "120"),
            ("240帧 (2.0s 极速开球)", "240"),
            ("300帧 (2.5s 标准开球)", "300"),
            ("480帧 (4.0s)", "480"),
            ("720帧 (6.0s 完整上限)", "720"),
        ]
        for title, val in presets:
            btn = QtWidgets.QPushButton(title)
            btn.setStyleSheet("color:#000000;font-weight:bold;padding:4px 8px;")
            btn.clicked.connect(lambda _=False, v=val: self.edit.setText(v))
            preset_row.addWidget(btn)
        layout.addLayout(preset_row)

        self.btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

        self.edit.textChanged.connect(self._on_text_changed)
        self._on_text_changed(self.edit.text())
        self.edit.selectAll()

    def _on_text_changed(self, txt: str):
        ok, _, msg = parse_trail_duration_spec(txt)
        self.status_lbl.setText(msg)
        if ok:
            self.status_lbl.setStyleSheet("color:#007a1c;font-family:Consolas;font-size:12px;font-weight:bold;padding:4px;")
        else:
            self.status_lbl.setStyleSheet("color:#d92626;font-family:Consolas;font-size:12px;font-weight:bold;padding:4px;")
        ok_btn = self.btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(ok)

    def frames_value(self) -> int:
        ok, frames, _ = parse_trail_duration_spec(self.edit.text())
        return frames if ok else 720


class ViewsMixin:
    def _save_prefs(self):
        if not hasattr(self, "_qsettings"):
            return
        self._qsettings.setValue("view_mode", self.view_mode)
        self._qsettings.setValue("obs_group_idx", self.obs_group_idx)
        self._qsettings.setValue("delta_mode", self.delta_mode)
        self._qsettings.setValue("plot_mode", getattr(self, "plot_mode", self._sess().plot_mode))
        self._qsettings.setValue("trail_duration", int(getattr(self, "trail_duration", 720)))
        self._qsettings.setValue("plot_visible", bool(getattr(self, "plot_visible", True)))
        if hasattr(self, "view") and hasattr(self.view, "active_mode_key"):
            self._qsettings.setValue("cam_mode", self.view.active_mode_key())
            self._qsettings.setValue("cam_pov_params_v2", self.view.format_current_params("pov"))
            self._qsettings.setValue("cam_center_params", self.view.format_current_params("center"))

    def _set_trail_duration(self, frames_count: int):
        """同步修改 3D 轨迹持续帧数、会话缓冲区与下方图表 X 轴滚动窗口宽度。"""
        dur = max(10, min(7200, int(frames_count)))
        self.trail_duration = dur
        age_s = max(0.05, float(dur) / 120.0)

        for s_obj in (getattr(self, "session", None), getattr(self, "auto_session", None)):
            if s_obj is not None and hasattr(s_obj, "set_buffer_duration"):
                s_obj.set_buffer_duration(dur)

        sess = self._sess()
        if getattr(self, "multi_car_renderer", None) is not None:
            self.multi_car_renderer.set_max_trail_len(dur, frames=sess.frames if sess else None)
        if getattr(self, "trail", None) is not None and hasattr(self.trail, "set_max_len"):
            self.trail.set_max_len(dur, max_age_s=age_s)

        if hasattr(self, "plot") and self.plot is not None:
            self.plot.set_max_window(dur)
        if hasattr(self, "obs_panel") and self.obs_panel is not None:
            self.obs_panel.set_max_window(dur)

        if sess is not None:
            self._update_plots_from_frames(sess)
        self._last_rendered_idx = None
        self._render()
        self._save_prefs()
        self._refresh_ui_state(f"⏱ 轨迹与图表时长 → {dur} 帧 ({dur / 120.0:.2f}s @ 120Hz)")

    def _prompt_trail_duration(self):
        cur_dur = int(getattr(self, "trail_duration", 720))
        dlg = TrailDurationDialog(self, current_frames=cur_dur)
        if dlg.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            self._set_trail_duration(dlg.frames_value())

    def _set_kickoff_slot(self, slot_idx):
        self.env.set_kickoff_slot(slot_idx)
        self._reset_session(replay_last=False)
        if slot_idx is None:
            self._refresh_ui_state("开球位 = 自动轮换 5 个标准开球位 (Shift+0)")
        else:
            s_name = KICKOFF_SLOT_NAMES[slot_idx]
            self._refresh_ui_state(f"锁定开球位 → KO{slot_idx}:{s_name} (Shift+{slot_idx+1})")

    def _cycle_focused_car(self, delta: int):
        new_idx = self.env.cycle_car(delta)
        self._last_rendered_idx = None
        self._render()
        lbl = self.env.car_label(new_idx)
        self._refresh_ui_state(f"视角车切换 → {lbl}")

    def _toggle_plot_visibility(self):
        """一键关闭/展开下方图表 (plot_widget)，将垂直空间释放给 3D 球场视野。"""
        new_vis = not bool(getattr(self, "plot_visible", True))
        self.plot_visible = new_vis
        if hasattr(self, "plot_widget") and self.plot_widget is not None:
            self.plot_widget.setVisible(new_vis)
        if new_vis:
            self._update_plots_from_frames(self._sess())
        self.setFocus()
        self._save_prefs()
        msg = (
            "📊 下方图表已展开 (按 G 关闭图表以扩大球场视野)"
            if new_vis else "📊 下方图表已关闭，3D 球场视野已最大化 (按 G 恢复图表)"
        )
        self._refresh_ui_state(msg)

    def _toggle_camera_mode(self):
        if not hasattr(self.view, "toggle_cam_mode"):
            return
        new_key = self.view.toggle_cam_mode()
        self._last_rendered_idx = None
        self._render()
        self._save_prefs()
        p_str = self.view.format_current_params(new_key)
        label = "THIRD-PARTY(CENTER)" if new_key == "center" else f"POV:{new_key.upper()}"
        self._refresh_ui_state(f"相机={label} ({p_str})")

    def _toggle_pov_submode(self):
        if not hasattr(self.view, "toggle_pov_submode"):
            return
        new_sub = self.view.toggle_pov_submode()
        self._last_rendered_idx = None
        self._render()
        self._save_prefs()
        p_str = self.view.format_current_params(new_sub)
        sub_name = "HARD-ATTACH(车头3D硬挂载)" if new_sub == "attach" else "BALL-CAM(球视角)"
        self._refresh_ui_state(f"POV相机={sub_name} ({p_str})")

    def _prompt_camera_params(self):
        if not hasattr(self.view, "format_current_params"):
            return
        cur_key = self.view.active_mode_key()
        cur_params = self.view.format_current_params(cur_key)
        if cur_key in ("attach", "ball", "pov"):
            sub_desc = "HARD-ATTACH (前向=绿轴车头, 纵向=红轴车顶)" if cur_key == "attach" else "BALL-CAM (对准球心, 纵向竖直朝上)"
            hint = (
                f"当前模式: POV -> {sub_desc}\n"
                "• back : 后退偏移 (uu, 默认 1500.0)\n"
                "• up   : 向上高度偏移 (uu, 默认 250.0)\n"
                "• elev : 俯仰角偏移 (度, [-85, 85], 默认 20.0)\n"
                "• azim : 水平角偏移 (度, [-180, 180], 默认 0.0)\n"
                "• fov  : 视野角度 FOV (度, [15, 140], 默认 130.0)"
            )
        else:
            hint = (
                "当前模式: THIRD-PARTY (CENTER 球场环绕视角，按 V 可切换为 POV 模式)\n"
                "• distance  : 距离中心点距离 (uu, 默认 5500.0)\n"
                "• elevation : 俯仰角 (度, [-89, 89], 默认 30.0)\n"
                "• azimuth   : 水平方位角 (度, 默认 45.0)\n"
                "• fov       : 视野角度 (度, [15, 140], 默认 60.0)\n"
                "• cx, cy, cz: 观察中心点坐标 (uu)"
            )
        dlg = QtWidgets.QInputDialog(self)
        dlg.setWindowTitle(f"修改相机参数 ({cur_key.upper()} Mode)")
        dlg.setLabelText(hint)
        dlg.setTextValue(cur_params)
        dlg.setStyleSheet(
            "QInputDialog, QDialog { background-color: #f5f5f5; color: #000000; }"
            "QLabel { color: #000000; font-family: Consolas; font-size: 12px; }"
            "QLineEdit { color: #000000; background-color: #ffffff; font-family: Consolas; font-size: 14px; font-weight: bold; padding: 6px; }"
        )
        if dlg.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            text = dlg.textValue().strip()
            if text:
                self.view.apply_params_str(text, mode=cur_key)
                self._last_rendered_idx = None
                self._render()
                self._save_prefs()
                self._refresh_ui_state(f"相机参数已更新: {self.view.format_current_params(cur_key)}")

    def _set_plot_mode(self, m):
        self.plot_mode = m
        self.session.plot_mode = m
        if self.auto_session is not None:
            self.auto_session.plot_mode = m
        tag = "Δ " if self.delta_mode else ""
        self.mode_label.setText(f"曲线 {m}: {tag}{MODE_NAMES[m]}")
        self.plot.update_mode(
            m, self._sess().frames,
            delta=self.delta_mode,
            max_window=getattr(self, "trail_duration", 720),
        )
        self._save_prefs()

    def _cycle_view_mode(self):
        if self.view_mode == "reward":
            self.view_mode = "obs"
        elif self.view_mode == "obs":
            self.view_mode = "flags"
        else:
            self.view_mode = "reward"
        self.plot.clear()
        self.obs_panel.clear()
        if self.view_mode == "reward":
            self._set_plot_mode(getattr(self, "plot_mode", self._sess().plot_mode))
        elif self.view_mode == "obs":
            self._set_obs_group(self.obs_group_idx)
        else:
            self._show_flags_view()
        self._refresh_ui_state()
        self._save_prefs()

    def _show_flags_view(self):
        self.mode_label.setText("flags 视图（0/1 方波，堆叠显示）")
        self.obs_panel.update_flags(
            self._sess().frames,
            max_window=getattr(self, "trail_duration", 720),
        )
        self._refresh_hud()

    def _set_obs_group(self, idx):
        self.obs_group_idx = max(0, min(idx, len(GROUP_ORDER) - 1))
        name = GROUP_ORDER[self.obs_group_idx]
        tag = "Δ " if self.delta_mode else ""
        self.mode_label.setText(
            f"obs [{self.obs_group_idx+1}/{len(GROUP_ORDER)}] {tag}{name}")
        self.obs_panel.update_group(
            name, self._sess().frames,
            delta=self.delta_mode,
            max_window=getattr(self, "trail_duration", 720),
        )
        self._refresh_hud()
        self._save_prefs()

    def _toggle_delta(self):
        self.delta_mode = not self.delta_mode
        if self.view_mode == "obs":
            self._set_obs_group(self.obs_group_idx)
        elif self.view_mode == "reward":
            self._set_plot_mode(getattr(self, "plot_mode", self._sess().plot_mode))
        self._refresh_ui_state()
        self._save_prefs()

    def _toggle_freerun(self):
        sess = self._sess()
        if sess.state == STATE_FREERUN:
            report = sess.format_freerun_report()
            print("\n" + report + "\n")
            os.makedirs("data", exist_ok=True)
            fn = f"data/obs_range_{int(_time.time())}.txt"
            with open(fn, "w") as f:
                f.write(report + "\n")
            print(f"[tuner] freerun report → {fn}")
            sess.end_freerun()
            self._refresh_ui_state(f"FREERUN end → {fn}")
        else:
            sess.begin_freerun()
            self.trail.clear()
            if getattr(self, "multi_car_renderer", None) is not None:
                self.multi_car_renderer.clear_trails()
            self.plot.clear()
            self.obs_panel.clear()
            self._refresh_ui_state("FREERUN 中，F 结束并导出")

    def _seek(self, delta):
        sess = self._sess()
        sess.seek(delta)
        if self.view_mode == "reward":
            self.plot.set_cursor(sess.playback_idx, sess.frames)
        else:
            self.obs_panel.set_cursor(sess.playback_idx, sess.frames)
        if sess.frames:
            self._render_frame(sess.frames[sess.playback_idx])
        self._refresh_hud()

    def _update_plots_from_frames(self, sess):
        if not sess or not sess.frames:
            return
        win = getattr(self, "trail_duration", 720)
        if self.view_mode == "reward":
            self.plot.update_mode(sess.plot_mode, sess.frames, delta=self.delta_mode, max_window=win)
            self.plot.set_cursor(sess.playback_idx, sess.frames)
        elif self.view_mode == "obs":
            self.obs_panel.update_group(
                GROUP_ORDER[self.obs_group_idx], sess.frames,
                delta=self.delta_mode, max_window=win,
            )
            self.obs_panel.set_cursor(sess.playback_idx, sess.frames)
        else:
            self.obs_panel.update_flags(sess.frames, max_window=win)
            self.obs_panel.set_cursor(sess.playback_idx, sess.frames)

    def _cycle_sub_curve_or_obs_group(self, delta: int = 1):
        if self.view_mode == "reward":
            cur = getattr(self, "plot_mode", 1)
            nxt = ((cur - 1 + delta) % 6) + 1
            self._set_plot_mode(nxt)
            self._refresh_ui_state(f"子曲线切换 → 曲线 {nxt}: {MODE_NAMES.get(nxt, '')}")
        elif self.view_mode == "obs":
            nxt = (self.obs_group_idx + delta) % len(GROUP_ORDER)
            self._set_obs_group(nxt)
            self._refresh_ui_state(f"Obs 分组切换 → [{nxt + 1}/{len(GROUP_ORDER)}] {GROUP_ORDER[nxt]}")
        else:
            self._show_flags_view()

    def _reset_session(self, replay_last=True, force_instant=False):
        self._pending_reset = None
        cur_plot_mode = getattr(self, "plot_mode", int(self._qsettings.value("plot_mode", 1)))
        self.plot_mode = cur_plot_mode
        self.session.plot_mode = cur_plot_mode
        if self.auto_session is not None:
            self.auto_session.plot_mode = cur_plot_mode

        self.trail.clear()
        if getattr(self, "multi_car_renderer", None) is not None:
            self.multi_car_renderer.clear_trails()
        self.plot.clear()
        self._last_rendered_idx = None
        self.obs_panel.clear()
        tag = "Δ " if self.delta_mode else ""
        if self.view_mode == "reward":
            self.mode_label.setText(f"曲线 {cur_plot_mode}: {tag}{MODE_NAMES[cur_plot_mode]}")
        elif self.view_mode == "obs":
            self.mode_label.setText(f"obs [{self.obs_group_idx+1}/{len(GROUP_ORDER)}] {tag}{GROUP_ORDER[self.obs_group_idx]}")
        else:
            self.mode_label.setText("flags 视图（0/1 方波，堆叠显示）")

        instant_manual_new = (
            self.use_auto
            and self.auto_session is not None
            and (force_instant or ((not self.auto_reset) and (not replay_last)))
        )

        if instant_manual_new:
            if not replay_last:
                self.env.reset(replay_last=False)
            msg = self.auto_session.instant_simulate(replay_last=True)
            self._update_plots_from_frames(self.auto_session)
            self._last_rendered_idx = None
            self._render()
            self._refresh_ui_state(msg, to_hud=True)
            return

        if self.use_auto and self.auto_session is not None:
            if not replay_last:
                self.env.reset(replay_last=False)
                self.auto_session.begin_recording(replay_last=True)
            else:
                self.auto_session.begin_recording(replay_last=True)
        else:
            self.session.reset(replay_last=replay_last)

        slot = int(getattr(self.env, "current_slot_idx", 0))
        slot_name = KICKOFF_SLOT_NAMES[max(0, min(len(KICKOFF_SLOT_NAMES) - 1, slot))]
        msg = f"重置并重新执行当前开球位 [KO{slot}:{slot_name}] (R)" if replay_last else f"切换新开球位 [KO{slot}:{slot_name}] (N)"
        self._refresh_ui_state(msg)

    def _adjust_speed(self, delta):
        self._sess().adjust_speed(delta)
        self._refresh_ui_state()

    def _copy_hud(self):
        text = self.hud.toPlainText()
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(text)
        self._refresh_ui_state("HUD → 剪贴板 ✓")

    def _refresh_ui_state(self, extra="", to_hud=False):
        sess = self._sess()

        if to_hud:
            self._hud_extra = str(extra) if extra else ""
            extra_short = ""
        else:
            extra_short = extra

        cam_tag = getattr(self.view, "active_mode_key", lambda: "center")().upper() if hasattr(self, "view") else "CENTER"
        slot = int(getattr(self.env, "current_slot_idx", 0))
        slot_name = KICKOFF_SLOT_NAMES[max(0, min(len(KICKOFF_SLOT_NAMES) - 1, slot))]
        car_lbl = self.env.car_label() if hasattr(self.env, "car_label") else "#1 BLUE"

        if self.use_auto:
            det_tag = "DET" if getattr(self.auto_session, "deterministic", True) else "STOCH"
            self.mode_indicator.setText(f"MODEL 1v1({det_tag}) [KO{slot}:{slot_name} | {car_lbl} | CAM:{cam_tag}]")
            self.mode_indicator.setStyleSheet(
                "color:#f0f;font-size:16px;font-weight:bold;"
                "background:#000;padding:6px;border:2px solid #f0f;")
            if self.auto_reset:
                self.reset_indicator.setText("AUTO-RESET")
                self.reset_indicator.setStyleSheet(
                    "color:#0f0;font-size:14px;font-weight:bold;"
                    "background:#000;padding:4px;border:2px solid #0f0;")
            else:
                self.reset_indicator.setText("MANUAL-RESET")
                self.reset_indicator.setStyleSheet(
                    "color:#fa0;font-size:14px;font-weight:bold;"
                    "background:#000;padding:4px;border:2px solid #fa0;")
        else:
            vs_tag = "vs MODEL 1v1" if getattr(self.session, "vs_model", True) else "SOLO"
            tk_step = getattr(self.session, "takeover_step", None)
            tk_tag = f" | ⚡接管@{tk_step}f" if tk_step is not None else ""
            self.mode_indicator.setText(f"HUMAN RECORD ({vs_tag}{tk_tag}) [KO{slot}:{slot_name} | {car_lbl} | CAM:{cam_tag}]")
            self.mode_indicator.setStyleSheet(
                "color:#0fc;font-size:16px;font-weight:bold;"
                "background:#000;padding:6px;border:2px solid #0fc;")
            self.reset_indicator.setText("←→回溯后按操作键/X中途接管 | R: 重置 | N: 新开球位 | H: 切对战 | Enter: 导出CSV")
            self.reset_indicator.setStyleSheet(
                "color:#0fc;font-size:13px;font-weight:bold;"
                "background:#000;padding:4px;border:2px solid #0fc;")

        color = {
            STATE_IDLE: "#888", STATE_RECORDING: "#0f0",
            STATE_DONE: "#fa0", STATE_PLAYBACK: "#0ff",
            STATE_FREERUN: "#f0f"
        }.get(sess.state, "#888")
        speed_pct = f"{int(sess.speed * 100)}%"
        play_pct = " ▶" if sess.playback_playing else ""
        view_tag = {"reward": "RWD", "obs": "OBS", "flags": "FLG"}[self.view_mode]
        delta_tag = " Δ" if self.delta_mode else ""
        if self.use_auto:
            pfx = getattr(self, "ckpt_prefix", "")
            ckpt_tag = f" ckpt={pfx}_{self.ckpt_source}" if pfx else f" ckpt={self.ckpt_source}"
        else:
            ckpt_tag = ""
        self.status_label.setText(
            f"[{view_tag}{delta_tag}]{ckpt_tag} {sess.state}{play_pct}  "
            f"速度 {speed_pct}  {extra_short}")
        self.status_label.setStyleSheet(
            f"color:{color};font-size:18px;font-weight:bold;"
            f"background:rgba(0,0,0,180);padding:8px;border:2px solid {color};")
