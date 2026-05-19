"""
Monitor P1 generation progress and auto-trigger packetization when ready.

Exits when coder packets have been generated.

Usage:
    python scripts/wait_and_packetize.py [--min-per-stratum 20] [--n-packets 480]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
QTYPES = ["MCQ", "TF", "Fill", "SA"]


def count_by_stratum() -> dict:
    """Return dict {(model_tag, qtype): count} from raw JSON files."""
    qa_dir = PROJECT_ROOT / "outputs/qa/main"
    counts: dict[tuple, int] = {}
    if not qa_dir.exists():
        return counts
    for model_dir in qa_dir.iterdir():
        if not model_dir.is_dir():
            continue
        mtag = model_dir.name
        for qtype_dir in model_dir.iterdir():
            if not qtype_dir.is_dir():
                continue
            qtype = qtype_dir.name
            n = sum(1 for _ in qtype_dir.rglob("*.json"))
            counts[(mtag, qtype)] = n
    return counts


def run_aggregation() -> None:
    import os
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    subprocess.run(
        [sys.executable, "src/analysis/aggregate_responses.py"],
        cwd=str(PROJECT_ROOT), env=env, check=False
    )


def run_packetize(n: int) -> None:
    import os, glob
    env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
    csv_files = glob.glob(str(PROJECT_ROOT / "outputs/qa_responses_*.csv"))
    if not csv_files:
        print("  [ERROR] No aggregate CSVs found; run aggregation first.")
        return
    subprocess.run(
        [sys.executable, "src/coding/qa_packetize.py",
         "--responses", *csv_files, "--n", str(n),
         "--stratify-by", "model_tag", "qtype"],
        cwd=str(PROJECT_ROOT), env=env, check=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-per-stratum", type=int, default=15,
                        help="Minimum responses per (model × qtype) cell before packetizing")
    parser.add_argument("--n-packets", type=int, default=480)
    parser.add_argument("--poll-interval", type=int, default=300,
                        help="Seconds between progress checks (default: 300)")
    parser.add_argument("--once", action="store_true",
                        help="Check once and exit (don't loop)")
    args = parser.parse_args()

    print(f"Monitoring P1 generation (min={args.min_per_stratum}/stratum, "
          f"poll={args.poll_interval}s)…")

    models = {
        "gpt-4o-2024-11-20", "claude-sonnet-4-5-20250929", "gemini-2_5-pro",
        "meta-llama_Llama-3_1-70B-Instruct", "Qwen_Qwen2_5-72B-Instruct",
        "deepseek-ai_DeepSeek-V3",
    }

    while True:
        counts = count_by_stratum()
        missing_strata = []
        for mtag in models:
            for qtype in QTYPES:
                n = counts.get((mtag, qtype), 0)
                if n < args.min_per_stratum:
                    missing_strata.append((mtag, qtype, n))

        total = sum(counts.values())
        print(f"\n[{time.strftime('%H:%M:%S')}] Total responses: {total}")
        print(f"  Missing strata (< {args.min_per_stratum}): {len(missing_strata)}")
        for mtag, qtype, n in missing_strata[:10]:
            print(f"    {mtag}/{qtype}: {n}")

        if not missing_strata:
            print("\nAll strata ready! Running aggregation + packetization…")
            run_aggregation()
            run_packetize(args.n_packets)
            print("\nDone — coder packets ready for distribution.")
            break

        if args.once:
            print("(--once mode: exiting)")
            break

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
