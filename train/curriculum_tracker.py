"""train/curriculum_tracker.py —— 静态球大王 4-Phase 课程阶梯追踪器 (5向独立 100 局滑动窗口 + 70/30 双池监控 + 70% 门控)。"""
from collections import deque
from core.constants import (
    NUM_CURRICULUM_STAGES, NUM_DIR5,
    POOL_FRONTIER,
)
from core.curriculum import (
    CurriculumState, NUM_CURRICULUM_STEPS,
    get_curriculum_step_spec, stage_level_to_step_idx,
    format_phase_coord_code, format_dir5_code, format_pool_code,
    format_milestone_tag,
)
from train.checkpoint import format_6stage_code


class CurriculumTracker:
    def __init__(
        self,
        init_stage_level: int = 1,
        init_step_idx: int = None,
        unlock_threshold: float = 0.70,
        dir_window_size: int = 100,
        min_samples_per_dir: int = 30,
        pool_window_size: int = 200,
        curriculum_state: CurriculumState = None,
        **kwargs,
    ):
        if init_step_idx is None and "start_step_idx" in kwargs:
            init_step_idx = kwargs["start_step_idx"]
        if init_step_idx is not None:
            self.step_idx = max(0, min(NUM_CURRICULUM_STEPS - 1, int(init_step_idx)))
        else:
            self.step_idx = stage_level_to_step_idx(init_stage_level)

        spec = get_curriculum_step_spec(self.step_idx)
        self.stage_level = max(1, min(NUM_CURRICULUM_STAGES, spec.phase))
        self.unlock_threshold = float(unlock_threshold)
        self.dir_window_size = int(dir_window_size)
        self.min_samples_per_dir = int(min_samples_per_dir)

        # 5 independent sliding windows (capacity=100) for Frontier samples in [+Fwd, -Fwd, +Rgt, -Rgt, +Up]
        self.dir5_windows = [deque(maxlen=self.dir_window_size) for _ in range(NUM_DIR5)]

        # 2 independent sliding windows for Frontier (F) and Memory (M) pools
        self.pool_windows = [deque(maxlen=int(pool_window_size)) for _ in range(2)]

        if curriculum_state is None:
            curriculum_state = kwargs.get("aerial_state")
        self.curriculum_state = curriculum_state if curriculum_state is not None else CurriculumState()
        self.aerial_state = self.curriculum_state
        self._sync_state()

    @property
    def current_spec(self):
        return get_curriculum_step_spec(self.step_idx)

    def _sync_state(self):
        spec = self.current_spec
        self.stage_level = max(1, min(NUM_CURRICULUM_STAGES, spec.phase))
        self.curriculum_state.step_idx = self.step_idx
        self.curriculum_state.phase = spec.phase
        self.curriculum_state.dir5_rates = tuple(self.get_dir5_rates())
        self.curriculum_state.pool_rates = tuple(self.get_pool_rates())
        self.curriculum_state.pool_counts = tuple(len(w) for w in self.pool_windows)

    def record_episode(
        self,
        stage_idx: int = 0,
        touched: bool = False,
        pool_id: int = POOL_FRONTIER,
        dir5_idx: int = None,
        dir6_idx: int = 0,
        ep_step_idx: int = None,
        **kwargs,
    ):
        if ep_step_idx is not None and int(ep_step_idx) != int(self.step_idx):
            return
        val = 1 if touched else 0
        p_id = 0 if int(pool_id) == POOL_FRONTIER else 1
        d_raw = dir5_idx if dir5_idx is not None else dir6_idx
        d_id = max(0, min(NUM_DIR5 - 1, int(d_raw)))

        self.pool_windows[p_id].append(val)
        if p_id == POOL_FRONTIER:
            self.dir5_windows[d_id].append(val)

        self._sync_state()

    def record_episodes(self, ep_list):
        for item in ep_list:
            if isinstance(item, dict):
                self.record_episode(**item)
            elif len(item) == 3:
                pool_id, dir_idx, touched = item
                self.record_episode(pool_id=pool_id, dir5_idx=dir_idx, touched=touched)

    def get_dir5_rates(self):
        return [
            (sum(w) / len(w)) if len(w) > 0 else 0.0
            for w in self.dir5_windows
        ]

    get_dir6_rates = get_dir5_rates
    get_stage_rates = get_dir5_rates

    def get_dir5_counts(self):
        return [len(w) for w in self.dir5_windows]

    def get_pool_rates(self):
        return [
            (sum(w) / len(w)) if len(w) > 0 else 0.0
            for w in self.pool_windows
        ]

    def get_5dir_code(self, unlocked_count=None) -> str:
        return format_6stage_code(self.get_dir5_rates(), unlocked_stages=NUM_DIR5)

    get_6stage_code = get_5dir_code
    get_8stage_code = get_5dir_code

    def format_status_code(self) -> str:
        spec = self.current_spec
        c1 = format_phase_coord_code(self.step_idx)
        c2 = format_dir5_code(self.get_dir5_rates(), gate=spec.gate)
        c3 = format_pool_code(self.get_pool_rates(), [len(w) for w in self.pool_windows])
        return f"{c1} {c2} {c3}"

    format_compact_status = format_status_code

    def get_milestone_tag(self, ep_num: int) -> str:
        return format_milestone_tag(self.step_idx, ep_num)

    def check_eval_gate(self, min_touch=None, warmup_eps=60, total_eps=0) -> bool:
        if total_eps < warmup_eps:
            return False
        spec = self.current_spec
        gate = float(spec.gate) if min_touch is None else float(min_touch)
        rates = self.get_dir5_rates()
        return all(r >= gate - 1e-6 for r in rates)

    def maybe_advance(self, auto_unlock=True):
        if not auto_unlock:
            return None
        spec = self.current_spec
        if spec.is_done or self.step_idx >= NUM_CURRICULUM_STEPS - 1:
            return None

        gate = float(spec.gate)
        rates = self.get_dir5_rates()
        for d_i in range(NUM_DIR5):
            if len(self.dir5_windows[d_i]) < self.min_samples_per_dir:
                return None
            if rates[d_i] < gate - 1e-6:
                return None

        prev_coord = format_phase_coord_code(self.step_idx)
        prev_dir5  = format_dir5_code(rates, gate=gate)

        self.step_idx += 1
        for w in self.dir5_windows:
            w.clear()
        self.pool_windows[POOL_FRONTIER].clear()
        self._sync_state()

        next_coord = format_phase_coord_code(self.step_idx)
        return f"[curriculum] Mastered {prev_coord} {prev_dir5} -> Advanced to {next_coord}!"
