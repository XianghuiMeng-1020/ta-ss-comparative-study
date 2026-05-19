"""
Fast-path difficulty injection (no API IRT needed).

Sources:
  Python MCQ:     GPT-4o-mini self-labelled difficulty from generation
  Math MCQ/Fill/SA: difficulty_raw from MathDial scenarios.csv column
  Stubs / missing: default "Medium"

Outputs:
  data/item_bank/{qtype}/items_with_difficulty.jsonl
  outputs/difficulty_summary.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BANK_ROOT = Path("data/item_bank")
OUTPUT_DIR = Path("outputs")

VALID_LEVELS = {"easy", "medium", "hard"}


def normalise(raw) -> str:
    if raw and str(raw).strip().lower() in VALID_LEVELS:
        return str(raw).strip().capitalize()
    return "Medium"


def inject_qtype(qtype: str) -> dict:
    src = BANK_ROOT / qtype.lower() / "items.jsonl"
    dst = BANK_ROOT / qtype.lower() / "items_with_difficulty.jsonl"

    items = [json.loads(l) for l in src.open()]
    stats = {"Easy": 0, "Medium": 0, "Hard": 0, "total": len(items)}

    enriched = []
    for item in items:
        diff = normalise(item.get("difficulty_raw"))
        item["difficulty"] = diff
        # IRT proxy placeholder (will be filled if IRT proxy is later run)
        item.setdefault("irt_b", None)
        item.setdefault("p_correct_irt_proxy", None)
        stats[diff] += 1
        enriched.append(item)

    with dst.open("w") as f:
        for it in enriched:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"  [{qtype}] {stats['total']} items → {dst.name}")
    print(f"    Easy={stats['Easy']}  Medium={stats['Medium']}  Hard={stats['Hard']}")
    return stats


def main() -> None:
    rows = []
    for qtype in ["MCQ", "TF", "Fill", "SA"]:
        print(f"\n=== {qtype} ===")
        s = inject_qtype(qtype)
        rows.append({"qtype": qtype, **s})

    df = pd.DataFrame(rows)
    report_path = OUTPUT_DIR / "difficulty_summary.csv"
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(report_path, index=False)
    print(f"\nDifficulty summary → {report_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
