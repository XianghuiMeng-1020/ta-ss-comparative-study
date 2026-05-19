"""
Aggregate all individual JSON response files under outputs/qa/ into
per-model CSV files and a master combined CSV.

Run at any point (incremental-safe — just re-aggregates from scratch):
    python src/analysis/aggregate_responses.py [--phase main]
    python src/analysis/aggregate_responses.py --status  # count only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

QA_ROOT = Path("outputs/qa")
OUT_ROOT = Path("outputs")

KEEP_COLS = [
    "item_id", "qtype", "domain", "concept", "difficulty",
    "persona", "model_id", "model_tag", "seed",
    "demographic_context", "generated_at",
    "response", "correct", "grade_method",
    "input_tokens", "output_tokens", "latency_ms", "cost_usd", "error",
]


def collect_rows(phase_dir: Path) -> list[dict]:
    rows = []
    for json_path in sorted(phase_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
            row = {k: data.get(k) for k in KEEP_COLS}
            rows.append(row)
        except Exception:
            continue
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="main")
    parser.add_argument("--status", action="store_true",
                        help="Print counts only, no CSV output")
    args = parser.parse_args()

    phase_dir = QA_ROOT / args.phase
    if not phase_dir.exists():
        print(f"No outputs at {phase_dir}")
        return

    print(f"Scanning {phase_dir}…")
    rows = collect_rows(phase_dir)
    if not rows:
        print("No responses found.")
        return

    df = pd.DataFrame(rows)
    total = len(df)
    print(f"Total responses: {total}")

    if args.status:
        print("\nBy model:")
        print(df.groupby("model_tag").size().to_string())
        print("\nBy model × qtype:")
        print(df.groupby(["model_tag", "qtype"]).size().unstack(fill_value=0).to_string())
        return

    # Per-model CSVs
    for mtag, sub in df.groupby("model_tag"):
        out_path = OUT_ROOT / f"qa_responses_{mtag}.csv"
        sub.to_csv(out_path, index=False)
        acc = sub[sub["correct"].notna()]["correct"].mean()
        cost = sub["cost_usd"].sum()
        print(f"  {mtag}: {len(sub)} rows | acc={acc:.3f} | cost=${cost:.3f} → {out_path}")

    # Combined CSV
    combined_path = OUT_ROOT / "qa_responses_all.csv"
    df.to_csv(combined_path, index=False)
    print(f"\nCombined: {combined_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
