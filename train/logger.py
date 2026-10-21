"""train/logger.py —— 1v1 最快开球双轨训练日志模块 (人类可读 .log + Agent 友好型分表精简 CSV: _train.csv 与 _events.csv)。"""
import os
import csv
import time

TRAIN_CSV_HEADER = (
    "ep", "upd",
    "touch_rate", "mean_frames", "mean_sec", "min_frames", "min_sec",
    "mean_spd", "mean_rew", "fps",
    "kl", "ent", "ev", "lr", "tk", "to",
)

EVENTS_CSV_HEADER = (
    "ep", "event",
    "cand_touch", "cand_avg_frames", "cand_avg_sec",
    "best_avg_frames", "is_new_best",
    "dr_f", "dl_f", "br_f", "bl_f", "ct_f",
    "detail",
)


class TrainingLogger:
    """Dual-format logger:
    1. `logs/<prefix>.log`        : Human-readable console mirror with timestamps.
    2. `logs/<prefix>_train.csv`  : Compact numeric table of periodic rollout & kickoff speed metrics.
    3. `logs/<prefix>_events.csv` : Sparse milestone table of 5-Slot kickoff evaluations and new fastest records.
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
        print(msg, flush=True)
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_txt_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{ts}] {msg}\n")
            if metrics_dict is not None:
                ev_type = metrics_dict.get("event", "")
                if ev_type == "train":
                    self._write_train_row(metrics_dict)
                elif ev_type in ("eval_done", "eval_sync", "eval_best"):
                    self._write_event_row(metrics_dict)
        except Exception:
            pass

    def _write_train_row(self, m: dict):
        mean_f = float(m.get("mean_frames", m.get("mean_step", 0.0) * 4.0))
        min_f = float(m.get("min_frames", mean_f))
        row = (
            int(m.get("ep", 0)),
            int(m.get("upd", 0)),
            round(float(m.get("touch_rate", 0.0)), 4),
            round(mean_f, 1),
            round(mean_f / 120.0, 3),
            round(min_f, 1),
            round(min_f / 120.0, 3),
            round(float(m.get("mean_spd", 0.0)), 1),
            round(float(m.get("mean_rew", 0.0)), 2),
            int(round(float(m.get("fps", 0.0)))),
            round(float(m.get("kl", 0.0)), 5),
            round(float(m.get("ent", 0.0)), 3),
            round(float(m.get("ev", 0.0)), 3),
            f"{float(m.get('lr', 0.0)):.2e}",
            int(m.get("tk", 0)),
            int(m.get("to", 0)),
        )
        with open(self.train_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def _write_event_row(self, m: dict):
        ep = int(m.get("ep", 0))
        is_best = int(bool(m.get("is_new_best", False)))
        ev = "eval_best" if is_best else "eval_duel"
        sf = m.get("slot_frames") or [720, 720, 720, 720, 720]
        row = (
            ep,
            ev,
            f"{m.get('cand_touch', 0)}/5",
            round(float(m.get("cand_avg_frames", 720.0)), 1),
            round(float(m.get("cand_avg_frames", 720.0)) / 120.0, 3),
            round(float(m.get("best_avg_frames", 720.0)), 1),
            is_best,
            sf[0], sf[1], sf[2], sf[3], sf[4],
            str(m.get("msg", "")),
        )
        with open(self.events_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
