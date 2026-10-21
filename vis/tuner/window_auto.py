import os

from core.constants import NUM_CURRICULUM_STAGES, NUM_CURRICULUM_STEPS
from core.reward import RewardConfig
from train.checkpoint import ckpt_paths, discover_prefixes

AUTO_RESET_DELAY = 0.5


class AutoMixin:
    def _current_ckpt(self):
        prefix = getattr(self, "ckpt_prefix", "static_v0")
        latest, best = ckpt_paths(prefix)
        target = best if getattr(self, "ckpt_source", "best") == "best" else latest
        if not os.path.exists(target):
            alt = latest if getattr(self, "ckpt_source", "best") == "best" else best
            if os.path.exists(alt):
                return alt
        return target

    def _sync_env_curriculum_from_ckpt(self, stage_level_override=None):
        ckpt_path = self._current_ckpt()
        if os.path.exists(ckpt_path):
            try:
                from train.checkpoint import _load_ckpt_file_raw
                raw = _load_ckpt_file_raw(ckpt_path, map_location="cpu")
                if isinstance(raw, dict):
                    ckpt_mode = raw.get("mode", "curriculum")
                    self.env.mode = ckpt_mode
                    slv = stage_level_override if stage_level_override is not None else int(raw.get("stage_level", NUM_CURRICULUM_STAGES))
                    c_step = int(raw.get("curriculum_step_idx", raw.get("step_idx", NUM_CURRICULUM_STEPS - 1)))
                    self.env.set_curriculum_state(slv, step_idx=c_step)
                    return
            except Exception as e:
                print(f"[tuner] warn: could not read ckpt metadata from {ckpt_path}: {e}")
        if stage_level_override is not None:
            self.env.stage_level = int(stage_level_override)

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
        has_same_mode = (self.env.last_spawn_state is not None)
        if not self.auto_reset:
            self._reset_session(replay_last=has_same_mode, force_instant=True)
            return
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
        else:
            self._sync_env_curriculum_from_ckpt()
        self._refresh_ui_state(f"ckpt = {self.ckpt_source} (step={self.env.curriculum_step_idx})")

    def _cycle_ckpt_prefix(self):
        self._all_prefixes = discover_prefixes()
        if not self._all_prefixes:
            return
        cur = getattr(self, "ckpt_prefix", "static_v0")
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
        else:
            self._sync_env_curriculum_from_ckpt()
        self._refresh_ui_state(f"ckpt prefix = {self.ckpt_prefix or 'static_v0'} (step={self.env.curriculum_step_idx})")

    def _toggle_auto_reset(self):
        self.auto_reset = not self.auto_reset
        self._pending_reset = None
        if self.use_auto and (not self.auto_reset) and self.auto_session is not None:
            from .ui_constants import STATE_RECORDING
            if self.auto_session.state == STATE_RECORDING:
                self._reset_session(replay_last=True, force_instant=True)
                return
        self._refresh_ui_state(
            "MANUAL-RESET (按 N 直接画出完整轨迹, 按 R 逐步回放)" if not self.auto_reset else "AUTO-RESET (自动连播)"
        )

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

    def _toggle_deterministic(self):
        if self.auto_session is None:
            self._refresh_ui_state("请先按 M 切换到 MODEL 模式")
            return
        self.auto_session.deterministic = not getattr(self.auto_session, "deterministic", True)
        mode_str = "DET(argmax)" if self.auto_session.deterministic else "STOCH(采样)"
        use_inst = bool(self.use_auto and (not self.auto_reset))
        self._reset_session(replay_last=True, force_instant=use_inst)
        self._refresh_ui_state(f"策略推理 = {mode_str}")
