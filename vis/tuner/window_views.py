import os
import re
import time as _time
import numpy as np
try:
    from pyqtgraph.Qt import QtCore, QtWidgets
    _QT_BASE_DIALOG = QtWidgets.QDialog
except ImportError:
    QtCore = None
    QtWidgets = None
    _QT_BASE_DIALOG = object

from core.constants import NUM_CURRICULUM_STAGES, DIR5_LABELS
from core.obs import GROUP_ORDER
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN,
)
try:
    from .plot_panel import MODE_NAMES
except ImportError:
    MODE_NAMES = {}


def validate_spawn_dist_spec(spec: str):
    """Validate p/s[4 digits] u[5 digits] d[3 digits] or step=N."""
    s = (spec or "").strip().lower()
    if not s or s in ("auto", "off", "none", "default"):
        return True, "✓ AUTO 模式: 跟随当前课程/Checkpoint 默认随机分布"

    cleaned = re.sub(r"([spud]\[[^\]]*\]|step\s*=\s*\d+)", "", s).strip()
    if cleaned:
        return False, f"❌ 存在未闭合括号或未知参数 '{cleaned}' (合法格式: p[4位] u[5位] d[3位] step=N)"

    matches = re.findall(r"([spud])\[([^\]]*)\]", s)
    m_step = re.search(r"step\s*=\s*(\d+)", s)
    if not matches and not m_step:
        return False, "❌ 未检测到 p[...] / u[...] / d[...] / step=N 参数块"

    parts_summary = []
    if m_step:
        parts_summary.append(f"Step={int(m_step.group(1))}")

    for key, val in matches:
        if not val.isdigit():
            return False, f"❌ {key}[{val}] 包含非数字字符 (仅允许数字 0-9)"
        n = len(val)
        if key in ("p", "s"):
            if n != NUM_CURRICULUM_STAGES:
                return False, f"❌ {key}[{val}] 必须为 {NUM_CURRICULUM_STAGES} 位数字 (对应 P1..P4)"
            if all(c == "0" for c in val):
                return False, f"❌ {key}[...] 4个Phase权重不能全为 0"
            active_p = [f"P{i+1}" for i, c in enumerate(val) if c != "0"]
            parts_summary.append(f"Phase={','.join(active_p)}")
        elif key == "u":
            if n != 5:
                return False, f"❌ u[{val}] 必须为 5 位数字 (对应 F/B/R/L/U 五方向)"
            if all(c == "0" for c in val):
                return False, "❌ u[...] 五方向权重不能全为 0"
            active_u = [DIR5_LABELS[i] for i, c in enumerate(val) if c != "0"]
            parts_summary.append(f"Dir5={','.join(active_u)}")
        elif key == "d":
            if n != 3:
                return False, f"❌ d[{val}] 必须为 3 位数字 (对应 Near/Mid/Far 距离段)"
            if all(c == "0" for c in val):
                return False, "❌ d[...] 距离段权重不能全为 0"
            parts_summary.append(f"DistTier=d[{val}]")

    return True, "✓ 格式合法 → " + " | ".join(parts_summary)


class SpawnDistDialog(_QT_BASE_DIALOG):
    """Dialog for editing p[...]u[...]d[...] spawn distribution with live regex validation."""

    def __init__(self, parent, current_spec: str, default_template: str):
        super().__init__(parent)
        self.setWindowTitle("动态修改静态靶生成分布 (Spawn Distribution)")
        self.setMinimumWidth(640)
        self.setStyleSheet("QDialog { background-color: #f5f5f5; color: #000000; }")
        layout = QtWidgets.QVBoxLayout(self)

        help_lbl = QtWidgets.QLabel(
            "直接在下方修改当前分布参数 (留空或输入 auto 恢复默认课程分布):\n"
            "• p[w1..w4] (4位): P1(5向扩角->方盒子), P2(方盒子加距离至1500), P3(全向初速+离地+跳标志), P4(远距拉满至6000)\n"
            "• u[u1..u5] (5位): 五方向权重 F(前), B(后), R(右), L(左), U(上)\n"
            "• d[d1..d3] (3位): Near(160-750), Mid(750-2100), Far(2100-6000)\n"
            "• step=N         : 直接指定课程 Step 索引 (0..36)"
        )
        help_lbl.setStyleSheet("font-family:Consolas;font-size:12px;color:#000000;background:transparent;")
        layout.addWidget(help_lbl)

        init_text = current_spec if current_spec else default_template
        self.edit = QtWidgets.QLineEdit(init_text)
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
            ("AUTO(默认)", "auto"),
            ("P1:5向方盒子", "p[1000]u[11111]"),
            ("P2:静车1500", "p[0100]u[11111]"),
            ("P3:全向初速+漂移", "p[0010]u[11111]"),
            ("P4:全距离拉满", "p[0001]u[11111]"),
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
        ok, msg = validate_spawn_dist_spec(txt)
        self.status_lbl.setText(msg)
        if ok:
            self.status_lbl.setStyleSheet("color:#007a1c;font-family:Consolas;font-size:12px;font-weight:bold;padding:4px;")
        else:
            self.status_lbl.setStyleSheet("color:#d92626;font-family:Consolas;font-size:12px;font-weight:bold;padding:4px;")
        ok_btn = self.btn_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setEnabled(ok)

    def value(self) -> str:
        return self.edit.text().strip()


class ViewsMixin:
    def _save_prefs(self):
        if not hasattr(self, "_qsettings"):
            return
        self._qsettings.setValue("view_mode", self.view_mode)
        self._qsettings.setValue("obs_group_idx", self.obs_group_idx)
        self._qsettings.setValue("delta_mode", self.delta_mode)
        self._qsettings.setValue("plot_mode", getattr(self, "plot_mode", self._sess().plot_mode))
        if hasattr(self, "view") and hasattr(self.view, "active_mode_key"):
            self._qsettings.setValue("cam_mode", self.view.active_mode_key())
            self._qsettings.setValue("cam_pov_params_v2", self.view.format_current_params("pov"))
            self._qsettings.setValue("cam_center_params", self.view.format_current_params("center"))

    def _select_camera_slot(self, slot: int):
        for ck in self.keys:
            self.keys[ck] = False
        if hasattr(self.view, "_free_held"):
            self.view._free_held.clear()

        if slot == 1:
            if hasattr(self.view, "reset_center_default"):
                self.view.set_cam_mode("center")
                self.view.reset_center_default()
            else:
                self.view.set_cam_mode("center")
            self._last_rendered_idx = None
            self._render()
            self._save_prefs()
            self._refresh_ui_state("镜头 [1]: 已恢复球场中央环绕视角 (CENTER)")
        elif slot == 2:
            if self.view.cam_mode == "pov":
                sub = self.view.toggle_pov_submode()
            else:
                self.view.set_cam_mode("pov")
                sub = self.view.pov_submode
            self._last_rendered_idx = None
            self._render()
            self._save_prefs()
            sub_desc = "车头硬挂载(HARD-ATTACH)" if sub == "attach" else "球视角(BALL-CAM)"
            self._refresh_ui_state(f"镜头 [2]: POV -> {sub_desc} (按 2 或 Space 切换车/球视角)")
        elif slot == 3:
            if self.view.cam_mode != "free":
                self.view.free_cam_locked = False
                self.view.set_cam_mode("free")
                self.view.focus_free_cam_on_blob()
            else:
                self.view.focus_free_cam_on_blob()
            self._last_rendered_idx = None
            self._render()
            self._save_prefs()
            lock_tag = "🔒已锁视角(WASD控车)" if self.view.free_cam_locked else "🕊自由漫游(WASD+QE/Space飞, 鼠标转向, Tab锁视角)"
            self._refresh_ui_state(f"镜头 [3]: FREE-CAM [{lock_tag}] (已对准场外3D非规则立体曲面)")

    def _toggle_camera_mode(self):
        if not hasattr(self.view, "toggle_cam_mode"):
            return
        new_key = self.view.toggle_cam_mode()
        self._last_rendered_idx = None
        self._render()
        self._save_prefs()
        p_str = self.view.format_current_params(new_key)
        if new_key == "center":
            label = "1:CENTER(球场中央)"
        elif new_key == "free":
            label = "3:FREE-CAM(自由漫游)"
        else:
            label = f"2:POV({new_key.upper()})"
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
        is_edit = getattr(self, "editor", None) is not None and self.editor.active
        cur_key = "pov" if is_edit else self.view.active_mode_key()
        cur_params = self.view.format_current_params(cur_key)
        if cur_key == "free":
            hint = (
                "当前模式: 3 -> FREE-CAM (WASD + Q/E/Space + 鼠标自由飞行视角)\n"
                "• ex, ey, ez : 摄像机 3D 坐标 (uu)\n"
                "• yaw, pitch : 水平偏航角与俯仰角 (度)\n"
                "• speed      : 漫游飞行速度 (uu/s, 滚轮也可直接调节)\n"
                "• fov        : 视野角度 FOV (度, [15, 140])"
            )
        elif cur_key in ("attach", "ball", "pov"):
            sub_desc = (
                f"EDIT-POV (挂载于当前编辑对象: {self.editor.target.upper()})"
                if is_edit
                else ("HARD-ATTACH (前向=绿轴车头, 纵向=红轴车顶)" if cur_key == "attach" else "BALL-CAM (对准球心, 纵向竖直朝上)")
            )
            hint = (
                f"当前模式: POV -> {sub_desc}\n"
                "HARD-ATTACH、BALL-CAM 与编辑模式共享同一组 POV 参数 (按住鼠标左键可临时环顾，松手立刻弹回):\n"
                "• back : 后退偏移 (uu, 默认 1500.0)\n"
                "• up   : 向上高度偏移 (uu, 默认 250.0)\n"
                "• elev : 俯仰角偏移 (度, [-85, 85], 默认 20.0)\n"
                "• azim : 水平角偏移 (度, [-180, 180], 默认 0.0)\n"
                "• fov  : 视野角度 FOV (度, [15, 140], 默认 130.0)"
            )
        else:
            hint = (
                "当前模式: THIRD-PARTY (CENTER 球场环绕视角，按 1/2/3 或 V 可切换模式)\n"
                "直接在下方当前原始参数基础上修改:\n"
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
        self.plot.update_mode(m, self._sess().frames, delta=self.delta_mode)
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
        self.obs_panel.update_flags(self._sess().frames)
        self._refresh_hud()

    def _set_obs_group(self, idx):
        self.obs_group_idx = max(0, min(idx, len(GROUP_ORDER) - 1))
        name = GROUP_ORDER[self.obs_group_idx]
        tag = "Δ " if self.delta_mode else ""
        self.mode_label.setText(
            f"obs [{self.obs_group_idx+1}/{len(GROUP_ORDER)}] {tag}{name}")
        self.obs_panel.update_group(name, self._sess().frames,
                                     delta=self.delta_mode)
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
            steps = getattr(sess, "_freerun_steps", 0)
            sess.end_freerun()
            self._refresh_ui_state(f"FREERUN 已结束 (共运行 {steps} 步, 未写盘/零内存累积)")
        else:
            sess.frames = []
            sess.begin_freerun()
            self.trail.clear()
            self.plot.clear()
            self.obs_panel.clear()
            self._refresh_ui_state("FREERUN 无限时运行中 (已禁用CSV帧记录防爆内存, 按 F 结束)")

    def _seek(self, delta):
        sess = self._sess()
        sess.seek(delta)
        if self.view_mode == "reward":
            self.plot.set_cursor(sess.playback_idx)
        else:
            self.obs_panel.set_cursor(sess.playback_idx)
        if sess.frames:
            self._render_frame(sess.frames[sess.playback_idx])
        self._refresh_hud()

    def _update_plots_from_frames(self, sess):
        if not sess or not sess.frames:
            return
        if self.view_mode == "reward":
            self.plot.update_mode(sess.plot_mode, sess.frames, delta=self.delta_mode)
            self.plot.set_cursor(sess.playback_idx)
        elif self.view_mode == "obs":
            self.obs_panel.update_group(
                GROUP_ORDER[self.obs_group_idx], sess.frames,
                delta=self.delta_mode,
            )
            self.obs_panel.set_cursor(sess.playback_idx)
        else:
            self.obs_panel.update_flags(sess.frames)
            self.obs_panel.set_cursor(sess.playback_idx)

    def _sync_car_state_panel_from_env(self):
        if not hasattr(self, "car_state_panel") or self.car_state_panel is None:
            return
        if getattr(self, "editor", None) is not None and self.editor.manual_override_locked:
            self.editor.apply_manual_car_state(
                boost_amount=self.car_state_panel.boost_amount,
                has_jump=self.car_state_panel.has_jump,
                lock_override=True,
            )
            return
        sd = self.env.last_spawn_state or {}
        cs = self.env.car.get_state()
        b_val = float(sd.get("car_boost", cs.boost))
        hj_val = bool(sd.get("has_second_jump", getattr(self.env, "has_second_jump", True)))
        if getattr(self, "editor", None) is not None:
            self.editor.car_boost = b_val
            self.editor.has_second_jump = hj_val
        self.car_state_panel.sync_from_state(b_val, hj_val)

    def _toggle_car_state_panel(self):
        if not hasattr(self, "car_state_panel") or self.car_state_panel is None:
            return
        new_vis = not self.car_state_panel.isVisible()
        if new_vis:
            if getattr(self, "editor", None) is not None and self.editor.active:
                self.car_state_panel.sync_from_state(
                    self.editor.car_boost,
                    self.editor.has_second_jump,
                    self.editor.manual_override_locked,
                )
            else:
                self._sync_car_state_panel_from_env()
        self.car_state_panel.setVisible(new_vis)
        self.setFocus()
        state_msg = (
            f"⚙ 手动面板已展开 [Boost={self.car_state_panel.boost_amount:.0f} | HasJump={'ON' if self.car_state_panel.has_jump else 'OFF'}]"
            if new_vis else "⚙ 手动面板已收起 (按 Z 重新展开)"
        )
        self._refresh_ui_state(state_msg)

    def _on_manual_car_state_changed(self, boost_amount: float, has_jump: bool, lock_override: bool):
        if not hasattr(self, "editor") or self.editor is None:
            return
        self.editor.apply_manual_car_state(
            boost_amount=boost_amount,
            has_jump=has_jump,
            lock_override=lock_override,
        )
        self.setFocus()
        summary = f"手动设置: Boost={boost_amount:.0f} | HasJump={'ON' if has_jump else 'OFF'}"
        if self.editor.active:
            self._last_rendered_idx = None
            self._render()
            self._refresh_ui_state(summary)
            return

        if self.use_auto and self.auto_session is not None and (not self.auto_reset):
            self._reset_session(replay_last=True, force_instant=True)
            self._refresh_ui_state(f"{summary} (已即时重绘轨迹, 按 R 逐步回放)")
        else:
            self._reset_session(replay_last=True)
            self._refresh_ui_state(summary)

    def _reset_session(self, replay_last=True, force_instant=False):
        self._pending_reset = None
        cur_plot_mode = getattr(self, "plot_mode", int(self._qsettings.value("plot_mode", 1)))
        self.plot_mode = cur_plot_mode
        self.session.plot_mode = cur_plot_mode
        if self.auto_session is not None:
            self.auto_session.plot_mode = cur_plot_mode

        self.trail.clear()
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
            self._sync_car_state_panel_from_env()
            msg = self.auto_session.instant_simulate(replay_last=True)
            self._update_plots_from_frames(self.auto_session)
            self._last_rendered_idx = None
            self._render()
            self._refresh_ui_state(msg, to_hud=True)
            return

        if self.use_auto and self.auto_session is not None:
            if not replay_last:
                self.env.reset(replay_last=False)
                self._sync_car_state_panel_from_env()
                self.auto_session.begin_recording(replay_last=True)
            else:
                self._sync_car_state_panel_from_env()
                self.auto_session.begin_recording(replay_last=True)
        else:
            self.session.reset(replay_last=replay_last)
            self._sync_car_state_panel_from_env()
            if getattr(self, "editor", None) is not None and self.editor.manual_override_locked:
                self.session.reset(replay_last=True)

        msg = "逐步重放当前场景 (R)" if replay_last else "新随机场景 (N)"
        self._refresh_ui_state(msg)

    def _adjust_speed(self, delta):
        self._sess().adjust_speed(delta)
        self._refresh_ui_state()

    def _apply_spawn_dist_str(self, spec: str):
        clean = (spec or "").strip()
        if clean.lower() in ("off", "none", "auto", "default"):
            clean = ""
        self.env.set_spawn_dist(clean)
        self._reset_session(replay_last=False)
        tag = f"分布={clean}" if clean else "分布=AUTO(跟随课程)"
        self._refresh_ui_state(tag)

    def _prompt_spawn_dist(self):
        cur = self.env.spawn_dist_cfg.raw if self.env.spawn_dist_cfg.is_active else ""
        slv = max(1, min(NUM_CURRICULUM_STAGES, int(getattr(self.env, "stage_level", NUM_CURRICULUM_STAGES))))
        p_bits = "".join("1" if i < slv else "0" for i in range(NUM_CURRICULUM_STAGES))
        default_tmpl = f"p[{p_bits}]u[11111]"
        dlg = SpawnDistDialog(self, cur, default_tmpl)
        if dlg.exec_() == QtWidgets.QDialog.DialogCode.Accepted:
            self._apply_spawn_dist_str(dlg.value())

    def _toggle_edit_mode(self):
        if not hasattr(self, "editor") or self.editor is None:
            return
        self._pending_reset = None
        if self.editor.active:
            self.editor.active = False
            self.editor._held.clear()
            self.editor._last_t = None
            if hasattr(self.view, "edit_drag_callback"):
                self.view.edit_drag_callback = None
            if hasattr(self.view, "edit_pov_active"):
                self.view.edit_pov_active = False
                self.view._temp_pov_azim = 0.0
                self.view._temp_pov_elev = 0.0
                if self.view.cam_mode == "center":
                    self.view.apply_center_params()
                else:
                    self.view.update_attach_camera()
            use_inst = bool(self.use_auto and (not self.auto_reset))
            self._reset_session(replay_last=True, force_instant=use_inst)
            self._refresh_ui_state("已退出编辑模式 (按 T 发射或按 R 逐步重放)")
        else:
            if hasattr(self.view, "sync_center_params_from_opts"):
                self.view.sync_center_params_from_opts()
            self.editor.active = True
            self.editor._held.clear()
            self.editor._last_t = None
            self.editor.load_from_current_env()
            if hasattr(self, "car_state_panel") and self.car_state_panel is not None:
                self.car_state_panel.sync_from_state(
                    self.editor.car_boost,
                    self.editor.has_second_jump,
                    self.editor.manual_override_locked,
                )
            if hasattr(self.view, "edit_drag_callback"):
                self.view.edit_drag_callback = self.editor.on_mouse_drag
            if hasattr(self.view, "edit_pov_active"):
                self.view.edit_pov_active = True
                self.view._temp_pov_azim = 0.0
                self.view._temp_pov_elev = 0.0
            self.trail.clear()
            self.plot.clear()
            self.obs_panel.clear()
            self._last_rendered_idx = None
            self._render()
            self._refresh_ui_state(f"进入手动编辑模式 [当前: {self.editor.target.upper()}] (Tab切车头/初速/球向, 左键拖方向, Z面板, T发射)")

    def _launch_from_editor(self):
        if not hasattr(self, "editor") or self.editor is None:
            return
        self.editor.sync_to_arena()
        if hasattr(self, "car_state_panel") and self.car_state_panel is not None:
            self.car_state_panel.sync_from_state(
                self.editor.car_boost,
                self.editor.has_second_jump,
                self.editor.manual_override_locked,
            )
        self.editor.active = False
        self.editor._held.clear()
        self.editor._last_t = None
        if hasattr(self.view, "edit_drag_callback"):
            self.view.edit_drag_callback = None
        if hasattr(self.view, "edit_pov_active"):
            self.view.edit_pov_active = False
            self.view._temp_pov_azim = 0.0
            self.view._temp_pov_elev = 0.0
            if self.view.cam_mode == "center":
                self.view.apply_center_params()
            else:
                self.view.update_attach_camera()
        use_inst = bool(self.use_auto and (not self.auto_reset))
        self._reset_session(replay_last=True, force_instant=use_inst)
        sess = self._sess()
        if not self.use_auto and sess.state == STATE_IDLE:
            sess.begin_recording()
        self._refresh_ui_state("🚀 已发射自定义场景! (按 R 逐步重放 / 按 M 切换模型对比 / 按 X 继续微调)")

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

        s_idx = getattr(self.env, "current_stage_idx", 0) + 1
        c_step = getattr(self.env, "curriculum_step_idx", 0)
        dist_raw = self.env.spawn_dist_cfg.raw if getattr(self.env, "spawn_dist_cfg", None) and self.env.spawn_dist_cfg.is_active else f"step={c_step}"
        cam_tag = getattr(self.view, "active_mode_key", lambda: "center")().upper() if hasattr(self, "view") else "CENTER"
        if getattr(self, "editor", None) is not None and self.editor.active:
            self.mode_indicator.setText(f"EDIT MODE [{self.editor.target.upper()} | Tab切车头/初速/球向 | T发射]")
            self.mode_indicator.setStyleSheet(
                "color:#ff0;font-size:16px;font-weight:bold;"
                "background:#000;padding:6px;border:2px solid #ff0;")
        elif self.use_auto:
            env_mode = getattr(self.env, "mode", "all").upper()
            det_tag = "DET" if getattr(self.auto_session, "deterministic", True) else "STOCH"
            self.mode_indicator.setText(f"MODEL({det_tag}) [{env_mode} | P{s_idx} | {dist_raw} | CAM:{cam_tag}]")
            self.mode_indicator.setStyleSheet(
                "color:#f0f;font-size:16px;font-weight:bold;"
                "background:#000;padding:6px;border:2px solid #f0f;")
        else:
            self.mode_indicator.setText(f"HUMAN [P{s_idx} | {dist_raw} | CAM:{cam_tag}]")
            self.mode_indicator.setStyleSheet(
                "color:#0fc;font-size:16px;font-weight:bold;"
                "background:#000;padding:6px;border:2px solid #0fc;")

        if self.use_auto:
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
            self.reset_indicator.setText("—")
            self.reset_indicator.setStyleSheet(
                "color:#555;font-size:14px;font-weight:bold;"
                "background:#000;padding:4px;border:2px solid #555;")

        is_edit = getattr(self, "editor", None) is not None and self.editor.active
        color = "#ff0" if is_edit else {
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
