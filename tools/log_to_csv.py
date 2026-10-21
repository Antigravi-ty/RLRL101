#!/usr/bin/env python3
"""tools/log_to_csv.py —— 将人类可读训练日志 (.log) 转换为 Agent 友好型分表精简 CSV (_train.csv 与 _events.csv)。"""
import os
import re
import sys
import csv
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.curriculum import get_curriculum_step_spec
from train.logger import TRAIN_CSV_HEADER, EVENTS_CSV_HEADER

_TRAIN_RE = re.compile(
    r"Ep\s+(\d+)\s+upd=\s*(\d+)\s+"
    r"P(\d)\[([^\]]+)\](?:\(DONE\))?\s+"
    r"Dir5\[(\d{5})\]\([A-Z]:(\d+)%/\d+%\)\s+"
    r"Pool\[F:(\d+)%\s+M:(\d+)%\]\s+"
    r"touch=\s*([\d.]+)%\s+"
    r"step=\s*([\d.]+)\(\s*\d+f\)\s+"
    r"spd=\s*([\d.]+)\s+"
    r"r=\s*([+-]?[\d.]+)\s+"
    r"fps=\s*([\d.]+)\s*\|\s*"
    r"kl=([\d.]+)\s+ent=([\d.]+)\s+pad=([\d.]+)/(\d+)%\s+ev=([+-]?[\d.]+)\s+lr=([\d.eE+-]+)\s*\|\s*"
    r"tk=(\d+)\s+to=(\d+)"
)

_ADV_RE = re.compile(
    r"\[curriculum\]\s+Mastered\s+(P\d\[[^\]]+\])\s+Dir5\[(\d{5})\]\(([^)]+)\)\s*->\s*Advanced to\s+(P\d\[[^\]]+\](?:\(DONE\))?)!.*saved milestone ->\s*(\S+)"
)

_EVAL_LINE_RE = re.compile(
    r"Ep\s+(\d+)\s*\|\s*eval\(S(\d+)\):\s*(.*)"
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
    cur_step_idx = 0
    cur_spec = get_curriculum_step_spec(0)

    with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
        for line in lf:
            m_load = re.search(r"step=(\d+)/\d+\s+P(\d)", line)
            if m_load and "[ppo]" in line:
                cur_step_idx = max(0, int(m_load.group(1)) - 1)
                cur_spec = get_curriculum_step_spec(cur_step_idx)

            m_adv = _ADV_RE.search(line)
            if m_adv:
                d5_str = m_adv.group(2)
                gate_info = m_adv.group(3)
                ms_file = m_adv.group(5)
                m_ep = re.search(r"_ep(\d+)\.pt", ms_file)
                ep_num = int(m_ep.group(1)) if m_ep else (train_rows[-1][0] if train_rows else 0)
                prev_idx = cur_step_idx
                prev_sp = get_curriculum_step_spec(prev_idx)
                cur_step_idx = min(40, cur_step_idx + 1)
                cur_spec = get_curriculum_step_spec(cur_step_idx)
                event_rows.append((
                    ep_num, "advance", cur_step_idx, int(cur_spec.phase), int(cur_spec.substep),
                    "", "", "", "", 0,
                    f"P{prev_sp.phase}.{prev_sp.substep:02d}->P{cur_spec.phase}.{cur_spec.substep:02d} Dir5[{d5_str}]({gate_info}) {ms_file}",
                ))
                continue

            m_ev = _EVAL_LINE_RE.search(line)
            if m_ev:
                ep_num = int(m_ev.group(1))
                s_num = max(0, int(m_ev.group(2)) - 1)
                ev_spec = get_curriculum_step_spec(s_num)
                body = m_ev.group(3).strip()
                is_best = 1 if ("*BEST*" in body or "NEW BEST" in body) else 0
                ev_name = "eval_best" if is_best else "eval_duel"
                cand_tc = best_tc = cand_t = best_t = ""
                detail = body
                m_vs = re.search(r"\[(\d{5})\]\s*(\d+/\d+)\(([\d.]+)s\)\s*vs\s*\[(\d{5})\]\s*(\d+/\d+)\(([\d.]+)s\)", body)
                if m_vs:
                    cand_tc, cand_t = m_vs.group(2), m_vs.group(3)
                    best_tc, best_t = m_vs.group(5), m_vs.group(6)
                    detail = f"cand=[{m_vs.group(1)}] vs best=[{m_vs.group(4)}]"
                else:
                    m_init = re.search(r"\[(\d{5})\]\s*(\d+/\d+)\s*([\d.]+)s\(all:([\d.]+)s\)", body)
                    if m_init:
                        cand_tc, cand_t = m_init.group(2), m_init.group(4)
                        detail = f"init=[{m_init.group(1)}] touch_t={m_init.group(3)}s"
                event_rows.append((
                    ep_num, ev_name, s_num, int(ev_spec.phase), int(ev_spec.substep),
                    cand_tc, best_tc, cand_t, best_t, is_best, detail,
                ))
                continue

            m_tr = _TRAIN_RE.search(line)
            if m_tr:
                ep = int(m_tr.group(1))
                upd = int(m_tr.group(2))
                phase = int(m_tr.group(3))
                d5 = m_tr.group(5)
                min_dir = float(m_tr.group(6))
                pf = float(m_tr.group(7))
                pm = float(m_tr.group(8))
                tr = round(float(m_tr.group(9)) / 100.0, 4)
                step_m = float(m_tr.group(10))
                spd = float(m_tr.group(11))
                rew = float(m_tr.group(12))
                fps = int(round(float(m_tr.group(13))))
                kl = float(m_tr.group(14))
                ent = float(m_tr.group(15))
                pad_ent = float(m_tr.group(16))
                pad_max = round(float(m_tr.group(17)) / 100.0, 4)
                ev = float(m_tr.group(18))
                lr = m_tr.group(19)
                tk = int(m_tr.group(20))
                to = int(m_tr.group(21))
                df, db, dr, dl, du = [int(c) * 10 for c in d5]
                train_rows.append((
                    ep, upd, cur_step_idx, phase, int(cur_spec.substep),
                    df, db, dr, dl, du, min_dir,
                    pf, pm,
                    tr, step_m, spd, rew, fps,
                    kl, ent, pad_ent, pad_max, ev, lr, tk, to,
                ))

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TRAIN_CSV_HEADER)
        w.writerows(train_rows)

    with open(events_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EVENTS_CSV_HEADER)
        w.writerows(event_rows)

    orig_sz = os.path.getsize(log_path)
    tr_sz = os.path.getsize(train_csv)
    ev_sz = os.path.getsize(events_csv)
    tot_csv_sz = tr_sz + ev_sz
    ratio = (1.0 - tot_csv_sz / max(1, orig_sz)) * 100.0
    print(
        f"[log_to_csv] Converted {log_path} ({orig_sz/1024:.1f} KB)\n"
        f"  -> {train_csv} ({len(train_rows)} rows, {tr_sz/1024:.1f} KB)\n"
        f"  -> {events_csv} ({len(event_rows)} rows, {ev_sz/1024:.1f} KB)\n"
        f"  -> Total CSV size: {tot_csv_sz/1024:.1f} KB ({ratio:.1f}% size reduction)"
    )
    return train_csv, events_csv


def main():
    ap = argparse.ArgumentParser(description="Convert training .log into compact _train.csv and _events.csv")
    ap.add_argument("log_path", nargs="?", default="logs/sbm_v3.log")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()
    convert_log_file(args.log_path, out_dir=args.out_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
