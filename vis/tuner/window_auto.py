import os

from core.reward import RewardConfig
from train.checkpoint import ckpt_paths, discover_prefixes, normalize_prefix

AUTO_RESET_DELAY = 0.5


class AutoMixin:
    def _current_ckpt(self):
        prefix = getattr(self, "ckpt_prefix", "ground_v0")
        latest, best = ckpt_paths(prefix)
        target = best if getattr(self, "ckpt_source", "best") == "best" else latest
        if not os.path.exists(target):
            alt = latest if getattr(self, "ckpt_source", "best") == "best" else best
            if os.path.exists(alt):
                return alt
        return target

    def _toggle_auto(self):
        if self.use_auto:
            self.use_auto = False
            self._pending_reset = None
            cur_plot_mode = getattr(self, "plot_mode", 1)
            self.session.reset(replay_last=True)
            self.session.plot_mode = cur_plot_mode
            self.trail.clear(); self.plot.clear(); self.obs_panel.clear()
            self._last_rendered_idx = None
            self._refresh_ui_state("HUMAN (同场景重放)")
            return

        ckpt_path = self._current_ckpt()
        if not os.path.exists(ckpt_path):
            self._refresh_ui_state(f"找不到 {ckpt_path}")
            return
        cur_plot_mode = getattr(self, "plot_mode", self.session.plot_mode)
        if self.auto_session is None:
            from vis.tuner.auto_session import AutoSession
            self.auto_session = AutoSession(self.env, RewardConfig(), ckpt_path)
            self.auto_session.plot_mode = cur_plot_mode
        else:
            self.auto_session.set_ckpt(ckpt_path)
            self.auto_session.plot_mode = cur_plot_mode
        self._last_ckpt_mtime = os.path.getmtime(ckpt_path)

        self.use_auto = True
        self._pending_reset = None
        self.auto_session.plot_mode = cur_plot_mode
        sp_m = (self.env.last_spawn_state or {}).get("mode")
        env_m = self.env.mode
        norm_sp = "ground" if sp_m == "stage0" else ("target" if sp_m == "stage1" else sp_m)
        norm_env = "ground" if env_m == "stage0" else ("target" if env_m == "stage1" else env_m)
        has_same_mode = (self.env.last_spawn_state is not None and norm_sp == norm_env)
        self.auto_session.begin_recording(replay_last=has_same_mode)
        self.trail.clear(); self.plot.clear(); self.obs_panel.clear()
        self._last_rendered_idx = None
        self._refresh_ui_state("MODEL (同场景对比)" if has_same_mode else "MODEL (新场景)")

    def _toggle_ckpt_source(self):
        old_source = self.ckpt_source
        self.ckpt_source = "best" if self.ckpt_source == "latest" else "latest"
        if self.use_auto and self.auto_session is not None:
            ckpt_path = self._current_ckpt()
            if not os.path.exists(ckpt_path):
                self.ckpt_source = old_source
                self._refresh_ui_state(f"找不到 {ckpt_path}")
                return
            try:
                self._toggle_auto()
                self._toggle_auto()
            except Exception as e:
                print(f"[tuner] ckpt source '{self.ckpt_source}' load failed: {e}")
                self.ckpt_source = old_source
                if not self.use_auto:
                    try:
                        self._toggle_auto()
                    except Exception:
                        pass
                self._refresh_ui_state(f"ckpt 加载失败: {e}")
                return
        self._refresh_ui_state(f"ckpt = {self.ckpt_source}")

    def _cycle_ckpt_prefix(self):
        self._all_prefixes = discover_prefixes()
        if not self._all_prefixes:
            return
        cur = getattr(self, "ckpt_prefix", "ground_v0")
        if cur in self._all_prefixes:
            i = (self._all_prefixes.index(cur) + 1) % len(self._all_prefixes)
        else:
            i = 0
        new_prefix = self._all_prefixes[i]
        self.ckpt_prefix = new_prefix
        self._last_ckpt_mtime = None
        if self.use_auto and self.auto_session is not None:
            ckpt_path = self._current_ckpt()
            if not os.path.exists(ckpt_path):
                self.ckpt_prefix = cur
                self._refresh_ui_state(f"找不到 {ckpt_path}")
                return
            try:
                self._toggle_auto()
                self._toggle_auto()
            except Exception as e:
                print(f"[tuner] prefix '{self.ckpt_prefix}' load failed: {e}")
                self.ckpt_prefix = cur
                try:
                    if not self.use_auto:
                        self._toggle_auto()
                except Exception:
                    pass
                self._refresh_ui_state(f"prefix 加载失败: {e}")
                return
        self._refresh_ui_state(f"ckpt prefix = {self.ckpt_prefix or 'ground_v0'}")

    def _toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        self._refresh_ui_state()

    def _auto_reset_now(self):
        if self.auto_session is None:
            return
        ckpt_path = self._current_ckpt()
        if os.path.exists(ckpt_path):
            mtime = os.path.getmtime(ckpt_path)
            if self._last_ckpt_mtime is None or mtime > self._last_ckpt_mtime:
                try:
                    self.auto_session.reload()
                    self._last_ckpt_mtime = mtime
                    print(f"[tuner] auto-reloaded {ckpt_path} (mtime={mtime:.0f})")
                except Exception as e:
                    print(f"[tuner] reload failed: {e}")
        cur_plot_mode = getattr(self, "plot_mode", self.auto_session.plot_mode)
        self.auto_session.plot_mode = cur_plot_mode
        self.auto_session.begin_recording(replay_last=False)
        self.trail.clear(); self.plot.clear(); self.obs_panel.clear()
        self._last_rendered_idx = None
        self._refresh_ui_state("auto-reset (新场景)")
