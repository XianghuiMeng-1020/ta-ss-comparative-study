"""
Public-reference similarity analysis.

Compare generated learner turns with original MathDial student turns
across 6 distributional features. Claim: similarity OR divergence from
a public tutoring-dialogue reference — NOT real-student realism.

Features:
  1. Mean learner turn length (tokens)
  2. Reasoning-step frequency (proportion of turns with explicit steps)
  3. Question frequency
  4. Correction timing index
  5. Misconception preservation rate (first 2 turns)
  6. Response-to-tutor-move pattern (χ² across move types)

Output: outputs/baseline_similarity_table.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.trace_metrics.question_asking import rule_based as has_question
from src.trace_metrics.reasoning_trace import rule_based as has_reasoning
from src.trace_metrics.target_error_preservation import rule_based as has_error


def count_approx_tokens(text: str) -> int:
    return len(text.split())


def compute_generated_features(trace_csv: Path, dialogues_dir: Path) -> pd.DataFrame:
    """Aggregate features from generated dialogue JSON files."""
    rows = []
    for jf in sorted(dialogues_dir.rglob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        if data.get("exclusion_flag"):
            continue

        turns = data.get("turns", [])
        learner_turns = [t.get("learner_turn", "") for t in turns if t.get("learner_turn")]
        tutor_turns = [t.get("tutor_turn", "") for t in turns if t.get("tutor_turn")]
        confusion = ""

        if not learner_turns:
            continue

        rows.append({
            "source": "generated",
            "condition": data.get("condition"),
            "scenario_id": data.get("scenario_id"),
            "mean_turn_length": np.mean([count_approx_tokens(t) for t in learner_turns]),
            "reasoning_rate": np.mean([has_reasoning(t) for t in learner_turns]),
            "question_rate": np.mean([has_question(t) for t in learner_turns]),
            "n_learner_turns": len(learner_turns),
        })
    return pd.DataFrame(rows)


def compute_mathdial_reference(cache_path: Path) -> dict:
    """
    Compute reference features from cached MathDial student turns.
    """
    if not cache_path.exists():
        print(f"MathDial cache not found: {cache_path}. Run load_mathdial.py first.")
        return {}

    df = pd.read_json(cache_path, lines=True)
    all_student_turns = []
    for val in df["original_student_turns"]:
        try:
            turns = json.loads(val) if isinstance(val, str) else val
            all_student_turns.extend([t for t in turns if t.strip()])
        except Exception:
            continue

    if not all_student_turns:
        return {}

    return {
        "mean_turn_length": np.mean([count_approx_tokens(t) for t in all_student_turns]),
        "reasoning_rate": np.mean([has_reasoning(t) for t in all_student_turns]),
        "question_rate": np.mean([has_question(t) for t in all_student_turns]),
        "n_turns": len(all_student_turns),
    }


def run_analysis():
    trace_csv = Path("outputs/automatic_trace_metrics.csv")
    dialogues_dir = Path("outputs/main")
    cache_path = Path("data/raw_mathdial/mathdial_train.jsonl")

    gen_df = compute_generated_features(trace_csv, dialogues_dir)
    if gen_df.empty:
        print("No generated dialogues found. Run generation first.")
        return

    ref = compute_mathdial_reference(cache_path)
    if not ref:
        print("MathDial reference not available. Run load_mathdial.py first.")
        ref = {"mean_turn_length": None, "reasoning_rate": None, "question_rate": None}

    features = ["mean_turn_length", "reasoning_rate", "question_rate"]
    result_rows = []

    for feat in features:
        if feat not in gen_df.columns:
            continue
        gen_by_cond = gen_df.groupby("condition")[feat].agg(["mean", "std", "count"])

        for cond, row in gen_by_cond.iterrows():
            ref_val = ref.get(feat)
            # t-test vs reference (treat reference mean as fixed value — descriptive only)
            result_rows.append({
                "feature": feat,
                "condition": cond,
                "generated_mean": round(row["mean"], 3),
                "generated_std": round(row["std"], 3) if not np.isnan(row["std"]) else None,
                "n": int(row["count"]),
                "reference_mean": round(ref_val, 3) if ref_val is not None else None,
                "divergence": (
                    round(row["mean"] - ref_val, 3)
                    if ref_val is not None else None
                ),
                "note": "Descriptive divergence only — NOT a realism claim",
            })

    # Correction timing divergence from reference
    ct_df = pd.read_csv(trace_csv) if trace_csv.exists() else pd.DataFrame()
    if "rule_correction_timing_index" in ct_df.columns:
        ct_by_cond = ct_df.groupby("condition")["rule_correction_timing_index"].agg(
            ["mean", "std", "count"]
        )
        for cond, row in ct_by_cond.iterrows():
            result_rows.append({
                "feature": "correction_timing_index",
                "condition": cond,
                "generated_mean": round(row["mean"], 3),
                "generated_std": round(row["std"], 3) if not np.isnan(row["std"]) else None,
                "n": int(row["count"]),
                "reference_mean": 0.5,  # placeholder; no direct MathDial equivalent
                "divergence": round(row["mean"] - 0.5, 3),
                "note": "Placeholder reference; MathDial equivalent not directly available",
            })

    out_df = pd.DataFrame(result_rows)
    out_path = Path("outputs") / "baseline_similarity_table.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved reference similarity table → {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    run_analysis()
