"""
IRT proxy calibration using GPT-4o-mini as the 'neutral student' population.

Instead of running Mistral-7B locally via vLLM, we use GPT-4o-mini (temperature=1.0)
to answer each item 5 times, compute p(correct), then map to Easy/Medium/Hard labels.

This is a valid IRT approximation: difficulty ~ logit-of-proportion-correct across
the calibration population. The GPT-4o-mini responses serve as the "item population."

Outputs:
  data/item_bank/{qtype}/items_with_difficulty.jsonl
  outputs/irt_proxy_calibration.csv
  outputs/irt_baseline/     (raw response JSONs for audit trail)

Usage:
    python src/data/irt_proxy_api.py --qtypes MCQ TF Fill SA --n-reps 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BANK_ROOT = Path("data/item_bank")
OUTPUT_DIR = Path("outputs")
BASELINE_DIR = OUTPUT_DIR / "irt_baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

EASY_THRESHOLD = 0.70      # p_correct > 0.70 → Easy
HARD_THRESHOLD = 0.35      # p_correct < 0.35 → Hard


def load_env_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        dotenv = Path(".env")
        if dotenv.exists():
            for line in dotenv.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not found")
    return key


def build_neutral_prompt(item: dict) -> str:
    """Build a neutral (non-naive) answering prompt for IRT baseline."""
    q = item["question"]
    if item["qtype"] == "MCQ":
        opts = "\n".join(f"  {k}. {v}" for k, v in item.get("options", {}).items())
        return (
            f"Answer the following multiple-choice question. "
            f"Reply with ONLY the letter (A, B, C, or D).\n\n{q}\n\n{opts}"
        )
    elif item["qtype"] == "TF":
        return (
            f"Answer True or False. Reply with ONLY 'True' or 'False'.\n\n{q}"
        )
    elif item["qtype"] == "Fill":
        return (
            f"Fill in the blank. Reply with ONLY the missing value.\n\n{q}"
        )
    else:  # SA
        return (
            f"Solve this problem. Give a brief answer.\n\n{q}"
        )


def check_correct(response_text: str, item: dict) -> bool:
    ans = response_text.strip().upper()
    correct = str(item["correct_answer"]).strip().upper()
    if item["qtype"] == "MCQ":
        return ans.startswith(correct)
    elif item["qtype"] == "TF":
        return ans.startswith(correct[:1])
    else:
        nums_resp = re.findall(r"\b\d+(?:\.\d+)?\b", ans)
        nums_corr = re.findall(r"\b\d+(?:\.\d+)?\b", correct)
        if nums_corr and nums_resp:
            return nums_resp[0] == nums_corr[0]
        return correct[:8] in ans


def calibrate_items(
    items: list[dict],
    n_reps: int,
    client,
    cache_path: Path,
) -> pd.DataFrame:
    """Run n_reps responses per item, compute p_correct, assign difficulty."""
    # Load cache
    cache: dict[str, list[int]] = {}
    if cache_path.exists():
        for line in cache_path.open():
            row = json.loads(line)
            iid = row["item_id"]
            cache.setdefault(iid, []).append(row["correct"])

    results = []

    for item in tqdm(items, desc="IRT baseline", unit="item"):
        iid = item["item_id"]
        existing = cache.get(iid, [])
        needed = n_reps - len(existing)

        for _ in range(needed):
            prompt = build_neutral_prompt(item)
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                        max_tokens=50,
                    )
                    text = resp.choices[0].message.content or ""
                    is_correct = check_correct(text, item)
                    existing.append(int(is_correct))
                    with cache_path.open("a") as f:
                        f.write(json.dumps({
                            "item_id": iid,
                            "correct": int(is_correct),
                            "response": text[:100],
                        }) + "\n")
                    break
                except Exception as e:
                    print(f"  API error (attempt {attempt+1}): {e}")
                    time.sleep(2)

        p_correct = sum(existing) / max(len(existing), 1)
        difficulty = (
            "Easy" if p_correct >= EASY_THRESHOLD
            else "Hard" if p_correct <= HARD_THRESHOLD
            else "Medium"
        )
        results.append({
            "item_id": iid,
            "n_responses": len(existing),
            "n_correct": sum(existing),
            "p_correct": round(p_correct, 4),
            "irt_b_approx": round(-2.197 * (p_correct - 0.5), 3),  # logit approximation
            "difficulty": difficulty,
        })

    return pd.DataFrame(results)


def apply_difficulty_to_bank(qtype: str, df_results: pd.DataFrame) -> None:
    items_path = BANK_ROOT / qtype.lower() / "items.jsonl"
    if not items_path.exists():
        print(f"  [WARN] {items_path} not found, skipping.")
        return

    items = [json.loads(l) for l in items_path.open()]
    diff_map = dict(zip(df_results["item_id"], df_results["difficulty"]))
    irt_map = dict(zip(df_results["item_id"], df_results["irt_b_approx"]))
    p_map = dict(zip(df_results["item_id"], df_results["p_correct"]))

    for item in items:
        iid = item["item_id"]
        if iid in diff_map:
            item["difficulty"] = diff_map[iid]
            item["irt_b"] = irt_map[iid]
            item["p_correct_irt_proxy"] = p_map[iid]
        else:
            # Fall back to difficulty_raw or medium
            raw = item.get("difficulty_raw", "") or ""
            if isinstance(raw, str) and raw.lower() in ("easy", "medium", "hard"):
                item["difficulty"] = raw.capitalize()
            else:
                item["difficulty"] = "Medium"
            item["irt_b"] = None
            item["p_correct_irt_proxy"] = None

    out_path = BANK_ROOT / qtype.lower() / "items_with_difficulty.jsonl"
    with out_path.open("w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} items with difficulty → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="API-based IRT proxy calibration")
    parser.add_argument("--qtypes", nargs="+", default=["MCQ", "TF", "Fill", "SA"],
                        choices=["MCQ", "TF", "Fill", "SA"])
    parser.add_argument("--n-reps", type=int, default=5,
                        help="Number of GPT-4o-mini responses per item for p_correct estimation")
    parser.add_argument("--sample", type=int, default=None,
                        help="Subsample N items per qtype (for testing). Default: all items.")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=load_env_key())

    all_report_rows = []

    for qtype in args.qtypes:
        print(f"\n=== IRT proxy: {qtype} ===")
        items_path = BANK_ROOT / qtype.lower() / "items.jsonl"
        if not items_path.exists():
            print(f"  [SKIP] {items_path} not found.")
            continue

        items = [json.loads(l) for l in items_path.open()]
        if args.sample:
            import random
            random.seed(42)
            items = random.sample(items, min(args.sample, len(items)))

        cache_path = BASELINE_DIR / f"irt_proxy_{qtype}_cache.jsonl"
        df = calibrate_items(items, args.n_reps, client, cache_path)
        apply_difficulty_to_bank(qtype, df)

        df["qtype"] = qtype
        all_report_rows.append(df)

        dist = df["difficulty"].value_counts()
        print(f"  Difficulty distribution:\n{dist.to_string()}")

    if all_report_rows:
        report = pd.concat(all_report_rows, ignore_index=True)
        report_path = OUTPUT_DIR / "irt_proxy_calibration.csv"
        report.to_csv(report_path, index=False)
        print(f"\nFull IRT report → {report_path}")
        print(f"Total items calibrated: {len(report)}")


if __name__ == "__main__":
    main()
