#!/usr/bin/env python3
"""
P1 Generation Watchdog + Live Progress Monitor

Shows real-time progress for all 6 models.
Auto-restarts any dead process with checkpoint-resume.

Usage:
    cd /path/to/project
    source .venv/bin/activate
    python scripts/watchdog.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR      = PROJECT_ROOT / "logs"
QA_ROOT      = PROJECT_ROOT / "outputs/qa/main"
VENV_PY      = PROJECT_ROOT / ".venv/bin/python3"
PYTHON       = str(VENV_PY) if VENV_PY.exists() else sys.executable

TOTAL_PER_MODEL = 4500   # 1800 MCQ + 900 TF + 900 Fill + 900 SA (× 3 seeds each)
POLL_SECONDS    = 15

MODELS = [
    {
        "label":    "GPT-4o",
        "model_id": "gpt-4o-2024-11-20",
        "backend":  "openai",
        "log":      "p1_gpt4o.log",
        "tag":      "gpt-4o-2024-11-20",
    },
    {
        "label":    "Claude-Sonnet",
        "model_id": "claude-sonnet-4-5-20250929",
        "backend":  "openrouter",
        "log":      "p1_claude.log",
        "tag":      "claude-sonnet-4-5-20250929",
    },
    {
        "label":    "Gemini-2.5-Pro",
        "model_id": "gemini-2.5-pro",
        "backend":  "openrouter",
        "log":      "p1_gemini.log",
        "tag":      "gemini-2_5-pro",
    },
    {
        "label":    "Llama-3.1-70B",
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "backend":  "openrouter",
        "log":      "p1_llama70b.log",
        "tag":      "meta-llama_Llama-3_1-70B-Instruct",
    },
    {
        "label":    "Qwen2.5-72B",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "backend":  "openrouter",
        "log":      "p1_qwen72b.log",
        "tag":      "Qwen_Qwen2_5-72B-Instruct",
    },
    {
        "label":    "DeepSeek-V3",
        "model_id": "deepseek-ai/DeepSeek-V3",
        "backend":  "openrouter",
        "log":      "p1_deepseek.log",
        "tag":      "deepseek-ai_DeepSeek-V3",
    },
]

# ANSI colours
RED    = "\033[91m"
GRN    = "\033[92m"
YLW    = "\033[93m"
BLU    = "\033[94m"
MGN    = "\033[95m"
CYN    = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RST    = "\033[0m"

MODEL_COLORS = [GRN, CYN, BLU, YLW, MGN, RED]

# ─── State ───────────────────────────────────────────────────────────────────
_pids: dict[str, int]            = {}   # tag → pid
_history: dict[str, list[int]]   = {}   # tag → [count_at_t-1, count_at_t]
_start_time: float               = time.time()


# ─── Helpers ─────────────────────────────────────────────────────────────────
def count_files(tag: str) -> int:
    d = QA_ROOT / tag
    if not d.exists():
        return 0
    return sum(1 for _ in d.rglob("*.json"))


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def launch(m: dict) -> int:
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    log_path = LOG_DIR / m["log"]
    LOG_DIR.mkdir(exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(f"\n\n[WATCHDOG RESTART @ {time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
    proc = subprocess.Popen(
        [PYTHON, "src/generation/qa_runner.py",
         "--model",    m["model_id"],
         "--backend",  m["backend"],
         "--qtypes",   "MCQ", "TF", "Fill", "SA",
         "--seeds",    "17", "42", "91",
         "--phase",    "main",
         "--max-tokens", "500",
         "--temperature", "0.7"],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
    )
    return proc.pid


def bar(pct: float, width: int = 20, color: str = GRN) -> str:
    filled = int(width * pct / 100)
    return f"{color}{'█' * filled}{'░' * (width - filled)}{RST}"


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "  ∞"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h:2d}h{m:02d}m"
    return f"{m:2d}m{s:02d}s"


def last_log_line(log: str) -> str:
    log_path = LOG_DIR / log
    if not log_path.exists():
        return ""
    try:
        with log_path.open() as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if line and "%" in line:
                m = re.search(r'\|\s*(\d+)/(\d+)\s*\[', line)
                if m:
                    return f"{m.group(1)}/{m.group(2)}"
        return ""
    except Exception:
        return ""


# ─── Display ─────────────────────────────────────────────────────────────────
def render(total_now: dict[str, int]) -> None:
    term_w = shutil.get_terminal_size((100, 24)).columns
    os.system("clear")

    elapsed = time.time() - _start_time
    h, r = divmod(int(elapsed), 3600)
    m, s = divmod(r, 60)

    print(f"{BOLD}{'═'*min(term_w, 78)}{RST}")
    print(f"{BOLD}  P1 Generation Watchdog — IEEE TLT Naive-Student Study{RST}")
    print(f"  Elapsed: {h:02d}h{m:02d}m{s:02d}s   |   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*min(term_w, 78)}")
    print()

    grand_total = 0
    grand_target = 0

    for i, m_info in enumerate(MODELS):
        tag   = m_info["tag"]
        color = MODEL_COLORS[i]
        cnt   = total_now.get(tag, 0)
        pct   = min(cnt / TOTAL_PER_MODEL * 100, 100)
        pid   = _pids.get(tag, 0)
        alive = is_alive(pid) if pid else False
        status_icon = f"{GRN}●{RST}" if alive else f"{RED}✗{RST}"

        # Rate (items/min)
        hist = _history.get(tag, [cnt])
        rate_per_min = 0.0
        if len(hist) >= 2:
            delta_n = hist[-1] - hist[-2]
            rate_per_min = delta_n * (60 / POLL_SECONDS)

        remaining = TOTAL_PER_MODEL - cnt
        eta_sec   = (remaining / rate_per_min * 60) if rate_per_min > 0 else -1

        log_hint = last_log_line(m_info["log"])

        label_str = f"{color}{BOLD}{m_info['label']:<16}{RST}"
        bar_str   = bar(pct, width=22, color=color)
        pct_str   = f"{pct:5.1f}%"
        cnt_str   = f"{cnt:>5}/{TOTAL_PER_MODEL}"
        rate_str  = f"{rate_per_min:4.1f}/min" if rate_per_min > 0 else "   ---  "
        eta_str   = fmt_eta(eta_sec)

        print(f"  {status_icon} {label_str} {bar_str} {pct_str}  {cnt_str}  {rate_str}  ETA {eta_str}")
        if log_hint:
            print(f"      {DIM}{log_hint}{RST}")

        grand_total  += cnt
        grand_target += TOTAL_PER_MODEL

    grand_pct = grand_total / grand_target * 100 if grand_target else 0
    print()
    print(f"{'─'*min(term_w, 78)}")
    grand_bar = bar(grand_pct, width=40, color=CYN)
    print(f"  {BOLD}TOTAL{RST}  {grand_bar} {grand_pct:5.1f}%  {grand_total}/{grand_target}")

    # Per-qtype breakdown
    qtype_counts: dict[str, int] = {}
    for tag in [m["tag"] for m in MODELS]:
        d = QA_ROOT / tag
        if not d.exists():
            continue
        for qtype_dir in d.iterdir():
            if qtype_dir.is_dir():
                n = sum(1 for _ in qtype_dir.rglob("*.json"))
                qtype_counts[qtype_dir.name] = qtype_counts.get(qtype_dir.name, 0) + n

    if qtype_counts:
        print()
        print(f"  {DIM}QType breakdown:{RST}")
        for qt, n in sorted(qtype_counts.items()):
            expected = 1800 if qt == "MCQ" else 900
            expected_total = expected * 6
            pct_qt = n / expected_total * 100 if expected_total else 0
            print(f"    {qt:<6} {bar(pct_qt, width=20, color=DIM+GRN)}"
                  f" {pct_qt:5.1f}%  {n}/{expected_total}")

    print()
    print(f"  {DIM}Watchdog polling every {POLL_SECONDS}s · Ctrl-C to exit{RST}")
    print(f"{'═'*min(term_w, 78)}")


# ─── Main loop ───────────────────────────────────────────────────────────────
def main() -> None:
    global _pids

    # Detect already-running processes
    result = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True
    )
    for m_info in MODELS:
        for line in result.stdout.splitlines():
            if "qa_runner" in line and m_info["model_id"] in line:
                pid = int(line.split()[1])
                _pids[m_info["tag"]] = pid
                break

    print(f"Detected PIDs: { {m['label']: _pids.get(m['tag'], 'none') for m in MODELS} }")
    print("Starting watchdog… (Ctrl-C to stop)")
    time.sleep(2)

    try:
        while True:
            total_now: dict[str, int] = {}
            for m_info in MODELS:
                tag = m_info["tag"]
                cnt = count_files(tag)
                total_now[tag] = cnt

                # Update history
                hist = _history.setdefault(tag, [])
                hist.append(cnt)
                if len(hist) > 10:
                    hist.pop(0)

                # Watchdog: restart if dead and not done
                pid = _pids.get(tag, 0)
                if cnt < TOTAL_PER_MODEL and (not pid or not is_alive(pid)):
                    print(f"\n  [WATCHDOG] {m_info['label']} dead (pid={pid}). Restarting…")
                    time.sleep(2)
                    new_pid = launch(m_info)
                    _pids[tag] = new_pid
                    print(f"  [WATCHDOG] {m_info['label']} restarted → PID {new_pid}")
                    time.sleep(3)

            render(total_now)

            # Exit if all done
            all_done = all(total_now.get(m["tag"], 0) >= TOTAL_PER_MODEL for m in MODELS)
            if all_done:
                print(f"\n{GRN}{BOLD}All models complete! Run aggregation + packetization.{RST}")
                break

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print(f"\n\n  Watchdog stopped. Generation processes continue in background.")
        print(f"  Re-run: python scripts/watchdog.py")


if __name__ == "__main__":
    main()
