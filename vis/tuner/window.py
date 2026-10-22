import sys
import os
import time
import pyqtgraph as pg  # noqa: F401
from pyqtgraph.Qt import QtCore, QtWidgets

from core.env import KickoffEnv
from core.reward import RewardConfig
from core.obs import GROUP_ORDER
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN,
    KEY_NAMES,
)
from .session import Session
from .window_ui     import UIMixin
from .window_keys   import KeysMixin
from .window_views  import ViewsMixin
from .window_render import RenderMixin
from .window_auto   import AutoMixin, AUTO_RESET_DELAY


TICK_MS_ACTIVE = 8
TICK_MS_IDLE   = 33
PLOT_REFRESH_S = 0.05


class RewardTuner(UIMixin, KeysMixin, ViewsMixin, RenderMixin, AutoMixin,
                  QtWidgets.QMainWindow):
    def __init__(self, prefix=None, from_ckpt=None, spawn_dist=None, stage_level=None,
                 cam_mode=None, cam_params=None):
        super().__init__()

        self._qsettings = QtCore.QSettings("RLRL101", "KickoffRewardTuner")

        from train.checkpoint import discover_prefixes, ckpt_paths
        prefixes = discover_prefixes()
        if prefix is None:
            newest, newest_mt = "kickoff_v1", -1
            for p in prefixes:
                lp, bp = ckpt_paths(p)
                for path in (lp, bp):
                    if os.path.exists(path):
                        mt = os.path.getmtime(path)
                        if mt > newest_mt:
                            newest, newest_mt = p, mt
            prefix = newest
        self.ckpt_prefix = prefix
        self._all_prefixes = prefixes
        self.ckpt_source = from_ckpt if from_ckpt in ("best", "latest") else "best"
        self.auto_session = None
        self.use_auto = False
        self.auto_reset = True
        self._last_ckpt_mtime = None
        self._pending_reset = None
        self._hud_extra = ""

        self.view_mode = self._qsettings.value("view_mode", "reward", type=str)
        if self.view_mode not in ("reward", "obs", "flags"):
            self.view_mode = "reward"
        self.obs_group_idx = int(self._qsettings.value("obs_group_idx", 0))
        self.delta_mode = self._qsettings.value("delta_mode", False, type=bool)
        self.plot_mode = int(self._qsettings.value("plot_mode", 1))
        self.trail_duration = max(10, int(self._qsettings.value("trail_duration", 720)))
        self.plot_visible = self._qsettings.value("plot_visible", True, type=bool)

        self._last_plot_t = 0.0
        self._last_rendered_idx = None
        self._timer_ms = TICK_MS_IDLE

        self.setWindowTitle(
            "Kickoff Tuner (1v1 Fastest Kickoff — R: 重置当前开球位 | N: 新开球位 | M: 切HUMAN/MODEL | Enter: 导出CSV)"
        )
        self.resize(1600, 900)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.env = KickoffEnv(spawn_dist=spawn_dist)
        self.session = Session(self.env, RewardConfig(), vs_model=True)
        self.session.plot_mode = self.plot_mode
        self._sync_human_opp_model()
        self.keys = {k: False for k in KEY_NAMES}

        self._build_ui()

        saved_pov = self._qsettings.value("cam_pov_params_v2", "", type=str)
        if saved_pov:
            self.view.apply_params_str(saved_pov, mode="pov")
        saved_center = self._qsettings.value("cam_center_params", "", type=str)
        if saved_center:
            self.view.apply_params_str(saved_center, mode="center")

        init_cam_mode = cam_mode or self._qsettings.value("cam_mode", "center", type=str)
        self.view.set_cam_mode(init_cam_mode)
        if cam_params:
            self.view.apply_params_str(cam_params, mode=self.view.active_mode_key())

        if self.view_mode == "reward":
            self._set_plot_mode(self.plot_mode)
        elif self.view_mode == "obs":
            self._set_obs_group(self.obs_group_idx)
        else:
            self._show_flags_view()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(self._timer_ms)

        self._refresh_ui_state()
        self.setFocus()

    def _sess(self):
        if self.use_auto and self.auto_session is not None:
            return self.auto_session
        return self.session

    def tick(self):
        sess = self._sess()
        if sess is None:
            return

        if (not self.use_auto
                and sess.state == STATE_IDLE
                and any(self.keys.values())):
            sess.begin_recording()
            self._refresh_ui_state()

        if sess.state == STATE_RECORDING:
            reason = sess.advance_recording(self.keys, time.time())
            now = time.time()
            if getattr(self, "plot_visible", True) and (now - self._last_plot_t) >= PLOT_REFRESH_S:
                self._last_plot_t = now
                if self.view_mode == "reward":
                    self.plot.update_mode(
                        sess.plot_mode, sess.frames,
                        delta=self.delta_mode, max_window=self.trail_duration,
                    )
                elif self.view_mode == "obs":
                    self.obs_panel.update_group(
                        GROUP_ORDER[self.obs_group_idx], sess.frames,
                        delta=self.delta_mode, max_window=self.trail_duration,
                    )
                else:
                    self.obs_panel.update_flags(sess.frames, max_window=self.trail_duration)

            if reason:
                msg = sess.finish_recording(reason)
                self._refresh_ui_state(msg, to_hud=True)
                if self.use_auto and self.auto_reset:
                    self._pending_reset = time.time() + AUTO_RESET_DELAY

        elif sess.state == STATE_DONE:
            if (self._pending_reset is not None
                    and time.time() >= self._pending_reset):
                self._pending_reset = None
                self._auto_reset_now()

        elif sess.state == STATE_PLAYBACK:
            before = sess.playback_idx
            ended = sess.tick_playback()
            if sess.playback_idx != before:
                if self.view_mode == "reward":
                    self.plot.set_cursor(sess.playback_idx, sess.frames)
                else:
                    self.obs_panel.set_cursor(sess.playback_idx, sess.frames)
            if ended:
                self._refresh_ui_state()

        elif sess.state == STATE_FREERUN:
            sess.advance_freerun(self.keys, time.time())

        self._render()
        self._update_timer_for_state(sess)

    def _update_timer_for_state(self, sess):
        if sess.state in (STATE_IDLE, STATE_DONE):
            target = TICK_MS_IDLE
        else:
            target = TICK_MS_ACTIVE
        if target != self._timer_ms:
            self._timer_ms = target
            self.timer.setInterval(target)

    def closeEvent(self, e):
        self._save_prefs()
        super().closeEvent(e)


def main(prefix=None, from_ckpt=None, spawn_dist=None, stage_level=None,
         cam_mode=None, cam_params=None):
    if (
        prefix is None and from_ckpt is None and spawn_dist is None
        and stage_level is None and cam_mode is None and cam_params is None
    ):
        import argparse
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--prefix", default=None)
        ap.add_argument("--from", "--from-ckpt", dest="from_ckpt", default=None)
        ap.add_argument("--dist", "--spawn-dist", dest="spawn_dist", default=None)
        ap.add_argument("--stage-level", type=int, default=None)
        ap.add_argument("--cam", "--cam-mode", dest="cam_mode", default=None)
        ap.add_argument("--cam-params", dest="cam_params", default=None)
        args, _ = ap.parse_known_args()
        prefix = args.prefix
        from_ckpt = args.from_ckpt
        spawn_dist = args.spawn_dist
        stage_level = args.stage_level
        cam_mode = args.cam_mode
        cam_params = args.cam_params
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = RewardTuner(
        prefix=prefix, from_ckpt=from_ckpt, spawn_dist=spawn_dist,
        stage_level=stage_level, cam_mode=cam_mode, cam_params=cam_params,
    )
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
