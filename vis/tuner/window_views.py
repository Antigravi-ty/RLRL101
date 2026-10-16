"""视图切换 / 曲线模式 / 回放 seek / FREERUN / 状态文本刷新。"""
import os
import time as _time
import numpy as np
from pyqtgraph.Qt import QtWidgets

from core.obs_v2 import GROUP_ORDER
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN,
)
from .plot_panel import MODE_NAMES


class ViewsMixin:
    def _save_prefs(self):
        if not hasattr(self, "_qsettings"):
            return
        self._qsettings.setValue("view_mode", self.view_mode)
        self._qsettings.setValue("obs_group_idx", self.obs_group_idx)
        self._qsettings.setValue("delta_mode", self.delta_mode)
        self._qsettings.setValue("plot_mode", getattr(self, "plot_mode", self._sess().plot_mode))

    def _set_plot_mode(self, m):
        self.plot_mode = m
        self.session.plot_mode = m
        if self.auto_session is not None:
            self.auto_session.plot_mode = m
        self.mode_label.setText(f"曲线 {m}: {MODE_NAMES[m]}")
        self.plot.update_mode(m, self._sess().frames)
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

    # ---------- FREERUN ----------
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
            self.plot.clear()
            self.obs_panel.clear()
            self._refresh_ui_state("FREERUN 中，F 结束并导出")

    # ---------- 回放 / 重置 / 调速 / 复制 ----------
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

    def _reset_session(self, replay_last=True):
        self._pending_reset = None
        cur_plot_mode = getattr(self, "plot_mode", int(self._qsettings.value("plot_mode", 1)))
        self.plot_mode = cur_plot_mode
        self.session.plot_mode = cur_plot_mode
        if self.auto_session is not None:
            self.auto_session.plot_mode = cur_plot_mode

        if self.use_auto and self.auto_session is not None:
            self.auto_session.begin_recording(replay_last=replay_last)
        else:
            self.session.reset(replay_last=replay_last)

        self.trail.clear()
        self.plot.clear()
        self._last_rendered_idx = None
        self.obs_panel.clear()
        if self.view_mode == "reward":
            self.mode_label.setText(f"曲线 {cur_plot_mode}: {MODE_NAMES[cur_plot_mode]}")
        elif self.view_mode == "obs":
            tag = "Δ " if self.delta_mode else ""
            self.mode_label.setText(f"obs [{self.obs_group_idx+1}/{len(GROUP_ORDER)}] {tag}{GROUP_ORDER[self.obs_group_idx]}")
        else:
            self.mode_label.setText("flags 视图（0/1 方波，堆叠显示）")
        msg = "重放当前场景 (R)" if replay_last else "新随机场景 (N)"
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

        if self.use_auto:
            env_mode = getattr(self.env, "mode", "ground").upper()
            self.mode_indicator.setText(f"MODEL [{env_mode}]")
            self.mode_indicator.setStyleSheet(
                "color:#f0f;font-size:16px;font-weight:bold;"
                "background:#000;padding:6px;border:2px solid #f0f;")
        else:
            self.mode_indicator.setText("HUMAN")
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

        color = {STATE_IDLE: "#888", STATE_RECORDING: "#0f0",
                 STATE_DONE: "#fa0", STATE_PLAYBACK: "#0ff",
                 STATE_FREERUN: "#f0f"}.get(sess.state, "#888")
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
