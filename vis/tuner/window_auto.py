import os

from core.policy_net import ActorCritic
from core.reward import RewardConfig
from train.checkpoint import ckpt_paths, discover_prefixes, load_ppo_model

AUTO_RESET_DELAY = 0.5


class AutoMixin:
    def _current_ckpt(self):
        prefix = getattr(self, "ckpt_prefix", "kickoff_v1")
        latest, best = ckpt_paths(prefix)
        target = best if getattr(self, "ckpt_source", "best") == "best" else latest
        if not os.path.exists(target):
            alt = latest if getattr(self, "ckpt_source", "best") == "best" else best
            if os.path.exists(alt):
                return alt
        return target

    def _sync_human_opp_model(self):
        if getattr(self, "session", None) is None:
            return
        if self.auto_session is not None and self.auto_session.model is not None:
            self.session.set_opp_model(self.auto_session.model)
            return
        ckpt_path = self._current_ckpt()
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                m, _ = load_ppo_model(ckpt_path, device="cpu")
                self.session.set_opp_model(m)
                return
            except Exception:
                pass
        if self.session.opp_model is None:
            self.session.set_opp_model(ActorCritic())

    def _toggle_human_vs_model(self):
        if getattr(self, "session", None) is None:
            return
        self.session.vs_model = not bool(getattr(self.session, "vs_model", True))
        if self.session.vs_model:
            self._sync_human_opp_model()
        if not self.use_auto:
            self._reset_session(replay_last=True)
        mode_tag = "HUMAN vs MODEL 1v1 (对手车由Model实时对抗开球)" if self.session.vs_model else "HUMAN SOLO (对手车静止)"
        self._refresh_ui_state(f"人工模式 → {mode_tag} (按 H 切换)")

    def _toggle_auto(self):
        if self.use_auto:
            self.use_auto = False
            self._pending_reset = None
            cur_plot_mode = getattr(self, "plot_mode", 1)
            self._sync_human_opp_model()
            self.session.reset(replay_last=True)
            self.session.plot_mode = cur_plot_mode
            self.trail.clear()
            if getattr(self, "multi_car_renderer", None) is not None:
                self.multi_car_renderer.clear_trails()
            self.plot.clear()
            self.obs_panel.clear()
            self._last_rendered_idx = None
            vs_tag = "vs MODEL 1v1" if getattr(self.session, "vs_model", True) else "SOLO"
            self._refresh_ui_state(f"HUMAN 人工录制 ({vs_tag} | 同开球位重放)")
            return

        ckpt_path = self._current_ckpt()
        cur_plot_mode = getattr(self, "plot_mode", self.session.plot_mode)
        if self.auto_session is None:
            from vis.tuner.auto_session import AutoSession
            self.auto_session = AutoSession(self.env, RewardConfig(), ckpt_path)
            self.auto_session.plot_mode = cur_plot_mode
        else:
            self.auto_session.set_ckpt(ckpt_path)
            self.auto_session.plot_mode = cur_plot_mode
        self._sync_human_opp_model()
        if os.path.exists(ckpt_path):
            self._last_ckpt_mtime = os.path.getmtime(ckpt_path)

        self.use_auto = True
        self._pending_reset = None
        self.auto_session.plot_mode = cur_plot_mode
        has_same_mode = (self.env.last_spawn_state is not None)
        if not self.auto_reset:
            self._reset_session(replay_last=has_same_mode, force_instant=True)
            return
        self.auto_session.begin_recording(replay_last=has_same_mode)
        self.trail.clear()
        if getattr(self, "multi_car_renderer", None) is not None:
            self.multi_car_renderer.clear_trails()
        self.plot.clear()
        self.obs_panel.clear()
        self._last_rendered_idx = None
        ckpt_note = os.path.basename(ckpt_path) if os.path.exists(ckpt_path) else "untrained-init"
        self._refresh_ui_state(f"MODEL 1v1 ({ckpt_note} | 同开球位对比)")

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
                self._refresh_ui_state(f"ckpt 加载失败: {e}")
                return
        else:
            self._sync_human_opp_model()
        self._refresh_ui_state(f"ckpt = {self.ckpt_source}")

    def _cycle_ckpt_prefix(self):
        self._all_prefixes = discover_prefixes()
        if not self._all_prefixes:
            return
        cur = getattr(self, "ckpt_prefix", "kickoff_v1")
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
                self._refresh_ui_state(f"prefix 加载失败: {e}")
                return
        else:
            self._sync_human_opp_model()
        self._refresh_ui_state(f"ckpt prefix = {self.ckpt_prefix or 'kickoff_v1'}")

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
        self.trail.clear()
        if getattr(self, "multi_car_renderer", None) is not None:
            self.multi_car_renderer.clear_trails()
        self.plot.clear()
        self.obs_panel.clear()
        self._last_rendered_idx = None
        self._refresh_ui_state("auto-reset (新开球位)")

    def _toggle_deterministic(self):
        if self.auto_session is None:
            self._refresh_ui_state("请先按 M 切换到 MODEL 模式")
            return
        self.auto_session.deterministic = not getattr(self.auto_session, "deterministic", True)
        mode_str = "DET(argmax)" if self.auto_session.deterministic else "STOCH(采样)"
        use_inst = bool(self.use_auto and (not self.auto_reset))
        self._reset_session(replay_last=True, force_instant=use_inst)
        self._refresh_ui_state(f"策略推理 = {mode_str}")
