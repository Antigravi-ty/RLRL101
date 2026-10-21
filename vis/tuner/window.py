import sys
import os
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from core.constants import NUM_CURRICULUM_STAGES
from core.env import BallReachEnv
from core.reward import RewardConfig
from core.obs import GROUP_ORDER
from .ui_constants import (
    STATE_IDLE, STATE_RECORDING, STATE_DONE, STATE_PLAYBACK, STATE_FREERUN,
    KEY_NAMES,
)
from .session import Session
from .editor import ScenarioEditor
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
                 cam_mode=None, cam_params=None, nexto_teams=None):
        super().__init__()

        self._qsettings = QtCore.QSettings("RLRL101", "RewardTuner")
        from nexto import parse_nexto_teams
        self.nexto_teams = parse_nexto_teams(nexto_teams)
        self.is_nexto_mode = (self.nexto_teams is not None)

        from train.checkpoint import discover_prefixes, ckpt_paths
        prefixes = discover_prefixes()
        if prefix is None:
            newest, newest_mt = "static_v0", -1
            for p in prefixes:
                lp, bp = ckpt_paths(p)
                for path in (lp, bp):
                    if os.path.exists(path):
                        mt = os.path.getmtime(path)
                        if mt > newest_mt:
                            newest, newest_mt = p, mt
            prefix = newest
        elif (not self.is_nexto_mode) and prefix not in prefixes:
            print(f"[tuner] warn: prefix '{prefix}' has no checkpoint, "
                  f"available={prefixes}")
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
        self.trail_duration = max(10, int(self._qsettings.value("trail_duration", 600)))
        self.plot_visible = self._qsettings.value("plot_visible", True, type=bool)

        self._last_plot_t = 0.0
        self._last_rendered_idx = None
        self._timer_ms = TICK_MS_ACTIVE if self.is_nexto_mode else TICK_MS_IDLE

        if self.is_nexto_mode:
            nb, no = self.nexto_teams
            self.setWindowTitle(
                f"Reward Tuner v10 — Nexto Live Match ({nb}v{no}) "
                f"[R: 重置 | 1-8/9/0: 选车 | V: 视角 | U: 参数调整 | G: 开关图表]"
            )
        else:
            self.setWindowTitle("Reward Tuner v10 (Ground-Centric Static Ball Master + 3-Way Tab Editor)")
        self.resize(1600, 900)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        if self.is_nexto_mode:
            from nexto import NextoMatchEnv
            from .nexto_session import NextoSession
            nb, no = self.nexto_teams
            init_sp = spawn_dist or self._qsettings.value("nexto_spawn_spec", "kickoff", type=str)
            self.env = NextoMatchEnv(n_blue=nb, n_orange=no, spawn_spec=init_sp)
            self.session = NextoSession(self.env, RewardConfig(), max_buffer_frames=self.trail_duration)
            self.session.plot_mode = self.plot_mode
            self.auto_session = self.session
            self.use_auto = True
            self.editor = None
        else:
            self.env = BallReachEnv(mode="curriculum", stage_level=stage_level or NUM_CURRICULUM_STAGES, spawn_dist=spawn_dist)
            self._sync_env_curriculum_from_ckpt(stage_level_override=stage_level)
            self.session = Session(self.env, RewardConfig())
            self.session.plot_mode = self.plot_mode
            self.editor = ScenarioEditor(self.env)
        self.keys = {k: False for k in KEY_NAMES}

        self._build_ui()

        saved_pov = self._qsettings.value("cam_pov_params_v2", "", type=str)
        if saved_pov:
            self.view.apply_params_str(saved_pov, mode="pov")
        saved_center = self._qsettings.value("cam_center_params", "", type=str)
        if saved_center:
            self.view.apply_params_str(saved_center, mode="center")

        if self.is_nexto_mode:
            # Nexto 模式默认聚焦球场中心 (0, 0, 0) 的全局视角，而非跟随某辆车
            self.view.center_params["cx"] = 0.0
            self.view.center_params["cy"] = 0.0
            self.view.center_params["cz"] = 0.0
            init_cam_mode = cam_mode or "center"
        else:
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
        if getattr(self, "editor", None) is not None and self.editor.active:
            cam_fwd = self.view.camera_forward_world_xy() if hasattr(self.view, "camera_forward_world_xy") else (0.0, 1.0)
            self.editor.step_held_keys(time.time(), cam_forward_world=cam_fwd)
            self._render()
            if self._timer_ms != TICK_MS_ACTIVE:
                self._timer_ms = TICK_MS_ACTIVE
                self.timer.setInterval(TICK_MS_ACTIVE)
            return

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
         cam_mode=None, cam_params=None, nexto_teams=None):
    if (
        prefix is None and from_ckpt is None and spawn_dist is None
        and stage_level is None and cam_mode is None and cam_params is None
        and nexto_teams is None
    ):
        import argparse
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--prefix", default=None)
        ap.add_argument("--from", "--from-ckpt", dest="from_ckpt", default=None)
        ap.add_argument("--dist", "--spawn-dist", dest="spawn_dist", default=None)
        ap.add_argument("--stage-level", type=int, default=None)
        ap.add_argument("--cam", "--cam-mode", dest="cam_mode", default=None)
        ap.add_argument("--cam-params", dest="cam_params", default=None)
        ap.add_argument("--nexto", "--what", dest="nexto_teams", nargs="*", default=None)
        args, _ = ap.parse_known_args()
        prefix = args.prefix
        from_ckpt = args.from_ckpt
        spawn_dist = args.spawn_dist
        stage_level = args.stage_level
        cam_mode = args.cam_mode
        cam_params = args.cam_params
        nexto_teams = args.nexto_teams
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = RewardTuner(
        prefix=prefix, from_ckpt=from_ckpt, spawn_dist=spawn_dist,
        stage_level=stage_level, cam_mode=cam_mode, cam_params=cam_params,
        nexto_teams=nexto_teams,
    )
    w.show()
    w.raise_()
    w.activateWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
