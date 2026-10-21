#!/usr/bin/env python3
"""tools/log_to_csv.py —— 将 1v1 Fastest Kickoff 训练日志 (.log) 转换为精简 CSV (_train.csv 与 _events.csv)。"""
import os
import re
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.logger import TRAIN_CSV_HEADER, EVENTS_CSV_HEADER

_TRAIN_RE = re.compile(
    r"Ep\s+(\d+)\s+upd=\s*(\d+)\s+"
    r"Slot\[(\d{5})\]\s+"
    r"touch=\s*([\d.]+)%\s+"
    r"time=\s*([\d.]+)s\(\s*([\d.]+)f\)\s+"
    r"spd=\s*([\d.]+)\s+"
    r"r=\s*([+-]?[\d.]+)\s+"
    r"fps=\s*([\d.]+)\s*\|\s*"
    r"kl=([\d.]+)\s+ent=([\d.]+)\s+ev=([+-]?[\d.]+)\s+lr=([\d.eE+-]+)\s*\|\s*"
    r"tk=(\d+)\s+to=(\d+)"
)

_EVAL_LINE_RE = re.compile(
    r"Ep\s+(\d+)\s*\|\s*eval\(5-Slot\):\s*(.*)"
)


def convert_log_file(log_path: str, out_dir: str = None, prefix: str = None):
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(out_dir, exist_ok=True)

    if prefix is None:
        prefix = os.path.splitext(os.path.basename(log_path))[0]

    train_csv = os.path.join(out_dir, f"{prefix}_train.csv")
    events_csv = os.path.join(out_dir, f"{prefix}_events.csv")

    train_rows = []
    event_rows = []

    with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
        for line in lf:
            m_ev = _EVAL_LINE_RE.search(line)
            if m_ev:
                ep_num = int(m_ev.group(1))
                body = m_ev.group(2).strip()
                is_best = 1 if ("*BEST*" in body or "NEW BEST" in body) else 0
                ev_name = "eval_best" if is_best else "eval_duel"
                event_rows.append((
                    ep_num, ev_name, "", "", "", "", is_best, body,
                ))
                continue

            m_tr = _TRAIN_RE.search(line)
            if m_tr:
                ep = int(m_tr.group(1))
                upd = int(m_tr.group(2))
                s5 = m_tr.group(3)
                tr = round(float(m_tr.group(4)) / 100.0, 4)
                touch_s = float(m_tr.group(5))
                touch_f = float(m_tr.group(6))
                spd = float(m_tr.group(7))
                rew = float(m_tr.group(8))
                fps = int(round(float(m_tr.group(9))))
                kl = float(m_tr.group(10))
                ent = float(m_tr.group(11))
                ev = float(m_tr.group(12))
                lr = m_tr.group(13)
                tk = int(m_tr.group(14))
                to = int(m_tr.group(15))
                s0, s1, s2, s3, s4 = [int(c) * 10 for c in s5]
                train_rows.append((
                    ep, upd,
                    s0, s1, s2, s3, s4,
                    tr, touch_s, touch_f, spd, rew, fps,
                    kl, ent, ev, lr, tk, to,
                ))

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TRAIN_CSV_HEADER)
        w.writerows(train_rows)

    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EVENTS_CSV_HEADER)
        w.writerows(event_rows)

    print(
        f"[log_to_csv] Converted {log_path} -> {train_csv} ({len(train_rows)} rows), {events_csv} ({len(event_rows)} rows)"
    )
    return train_csv, events_csv


def main():
    ap = argparse.ArgumentParser(description="Convert Kickoff training .log into _train.csv and _events.csv")
    ap.add_argument("log_path", nargs="?", default="logs/kickoff_v1.log")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    convert_log_file(args.log_path, out_dir=args.out_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
