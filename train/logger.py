"""train/logger.py —— 双轨训练日志模块 (人类可读 .log + Agent 友好型分表精简 CSV: _train.csv 与 _events.csv)。"""
import os
import re
import csv
import time
from core.curriculum import get_curriculum_step_spec

TRAIN_CSV_HEADER = (
    "ep", "upd", "step_idx", "phase", "substep",
    "dir_f", "dir_b", "dir_r", "dir_l", "dir_u", "min_dir_pct",
    "pool_f_pct", "pool_m_pct",
    "touch_rate", "mean_step", "mean_spd", "mean_rew", "fps",
    "kl", "ent", "pad_ent", "pad_max", "ev", "lr", "tk", "to",
)

EVENTS_CSV_HEADER = (
    "ep", "event", "step_idx", "phase", "substep",
    "cand_touch", "best_touch", "cand_time_s", "best_time_s",
    "is_new_best", "detail",
)


class TrainingLogger:
    """Dual-format logger:
    1. `logs/<prefix>.log`        : Human-readable console mirror with timestamps.
    2. `logs/<prefix>_train.csv`  : Compact numeric table of periodic rollout & curriculum metrics.
    3. `logs/<prefix>_events.csv` : Sparse milestone table of curriculum advances and best-checkpoint duels.
    """

    def __init__(self, log_txt_path: str, save_prefix: str = None):
        self.log_txt_path = os.path.abspath(log_txt_path)
        log_dir = os.path.dirname(self.log_txt_path)
        os.makedirs(log_dir, exist_ok=True)

        if save_prefix is None:
            base = os.path.basename(self.log_txt_path)
            save_prefix = os.path.splitext(base)[0]
        self.save_prefix = save_prefix

        self.train_csv_path = os.path.join(log_dir, f"{save_prefix}_train.csv")
        self.events_csv_path = os.path.join(log_dir, f"{save_prefix}_events.csv")

        self._ensure_csv_header(self.train_csv_path, TRAIN_CSV_HEADER)
        self._ensure_csv_header(self.events_csv_path, EVENTS_CSV_HEADER)

    @staticmethod
    def _ensure_csv_header(path: str, header: tuple):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def log(self, msg: str, metrics_dict: dict = None):
        """Print to stdout, append human-readable line to `.log`, and route structured row to `_train.csv` or `_events.csv`."""
        print(msg, flush=True)
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_txt_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{ts}] {msg}\n")
            if metrics_dict is not None:
                ev_type = metrics_dict.get("event", "")
                if ev_type == "train":
                    self._write_train_row(metrics_dict)
                elif ev_type in ("advance", "eval_done", "eval_sync"):
                    self._write_event_row(metrics_dict)
        except Exception:
            pass

    def _write_train_row(self, m: dict):
        step_idx = int(m.get("step_idx", 0))
        spec = get_curriculum_step_spec(step_idx)
        d5 = str(m.get("dir5_code", "00000")).ljust(5, "0")[:5]
        dir_rates = m.get("dir5_rates")
        if dir_rates and len(dir_rates) >= 5:
            df, db, dr, dl, du = [round(float(x) * 100.0, 1) for x in dir_rates[:5]]
            min_dir_pct = round(min(dir_rates[:5]) * 100.0, 1)
        else:
            df, db, dr, dl, du = [int(c) * 10 for c in d5]
            min_dir_pct = float(min(df, db, dr, dl, du))

        pr = m.get("pool_rates", (0.0, 0.0))
        pf = round(float(pr[0]) * 100.0, 1) if len(pr) > 0 else 0.0
        pm = round(float(pr[1]) * 100.0, 1) if len(pr) > 1 else 0.0

        row = (
            int(m.get("ep", 0)),
            int(m.get("upd", 0)),
            step_idx,
            int(spec.phase),
            int(spec.substep),
            df, db, dr, dl, du,
            min_dir_pct,
            pf, pm,
            round(float(m.get("touch_rate", 0.0)), 4),
            round(float(m.get("mean_step", 0.0)), 2),
            round(float(m.get("mean_spd", 0.0)), 1),
            round(float(m.get("mean_rew", 0.0)), 2),
            int(round(float(m.get("fps", 0.0)))),
            round(float(m.get("kl", 0.0)), 5),
            round(float(m.get("ent", 0.0)), 3),
            round(float(m.get("pad_ent", 0.0)), 4),
            round(float(m.get("pad_max", 0.0)), 4),
            round(float(m.get("ev", 0.0)), 3),
            f"{float(m.get('lr', 0.0)):.2e}",
            int(m.get("tk", 0)),
            int(m.get("to", 0)),
        )
        with open(self.train_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def _write_event_row(self, m: dict):
        ev = str(m.get("event", "info"))
        ep = int(m.get("ep", 0))
        step_idx = int(m.get("new_step_idx", m.get("step_idx", 0)))
        spec = get_curriculum_step_spec(step_idx)
        raw_msg = str(m.get("msg", "")).strip()

        cand_tc = best_tc = cand_t = best_t = ""
        is_best = 0
        detail = raw_msg

        if ev == "advance":
            prev_idx = int(m.get("prev_step_idx", max(0, step_idx - 1)))
            prev_sp = get_curriculum_step_spec(prev_idx)
            detail = f"P{prev_sp.phase}.{prev_sp.substep:02d}->P{spec.phase}.{spec.substep:02d} ({m.get('milestone_ckpt', '')})"
        elif "eval(" in raw_msg or "duel(" in raw_msg:
            is_best = 1 if ("*BEST*" in raw_msg or "NEW BEST" in raw_msg) else 0
            ev = "eval_best" if is_best else "eval_duel"
            m_vs = re.search(r"\[(\d{5})\]\s*(\d+/\d+)\(([\d.]+)s\)\s*vs\s*\[(\d{5})\]\s*(\d+/\d+)\(([\d.]+)s\)", raw_msg)
            if m_vs:
                cand_tc, cand_t = m_vs.group(2), m_vs.group(3)
                best_tc, best_t = m_vs.group(5), m_vs.group(6)
                detail = f"cand=[{m_vs.group(1)}] vs best=[{m_vs.group(4)}]"
            else:
                m_init = re.search(r"\[(\d{5})\]\s*(\d+/\d+)\s*([\d.]+)s\(all:([\d.]+)s\)", raw_msg)
                if m_init:
                    cand_tc, cand_t = m_init.group(2), m_init.group(4)
                    detail = f"init=[{m_init.group(1)}] touch_t={m_init.group(3)}s"

        row = (
            ep, ev, step_idx, int(spec.phase), int(spec.substep),
            cand_tc, best_tc, cand_t, best_t, is_best, detail,
        )
        with open(self.events_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
