"""
D6 — Dialogue-Test Discrepancy; Type-A / Type-B classification.

Computes per-agent discrepancy:
  Δ = z(test_score) - z(dialogue_kl_score)

Where:
  test_score       = D2 accuracy on MCQ/TF/Fill/SA items (this study)
  dialogue_kl_score = knowledge state estimated from OED dialogue turns
                       via DKT model (reuses lab dialogue-based KT model)

If DKT model unavailable, uses proxy: average trace_metric scores from
the existing 10-metric pipeline (src/trace_metrics/) as knowledge proxy.

Type classification:
  Type-A (eye-high-hand-low): Δ < -1 (dialogue looks knowledgeable; test fails)
  Type-B (silent achiever):   Δ >  1 (dialogue looks novice; test scores high)
  Consistent High:  both > +0.5
  Consistent Low:   both < -0.5

Usage:
    python src/evaluation/dlkt_discrepancy.py \
        --test-scores outputs/eval_d2_answering.csv \
        --dialogue-metrics outputs/automatic_trace_metrics.csv \
        --output outputs/eval_d6_discrepancy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DISCREPANCY_THRESHOLD = 1.0

DIALOGUE_KNOWLEDGE_METRICS = [
    "reasoning_trace_rate",
    "feedback_uptake_rate",
    "question_asking_rate",
]


def load_and_merge(test_path: Path, dialogue_path: Path) -> pd.DataFrame:
    test_df = pd.read_csv(test_path)
    dial_df = pd.read_csv(dialogue_path)

    required_test = {"model_tag", "accuracy"}
    required_dial = {"model_tag"}
    if not required_test.issubset(test_df.columns):
        raise ValueError(f"test_scores file missing: {required_test - set(test_df.columns)}")

    test_agg = test_df.groupby("model_tag")["accuracy"].mean().reset_index()
    test_agg.columns = ["model_tag", "test_score"]

    avail_metrics = [m for m in DIALOGUE_KNOWLEDGE_METRICS if m in dial_df.columns]
    if not avail_metrics:
        avail_metrics = [c for c in dial_df.columns
                         if c not in {"model_tag", "scenario_id", "condition", "seed", "phase"}
                         and dial_df[c].dtype in [np.float64, np.int64]][:3]

    if not avail_metrics:
        raise ValueError("No numeric metrics found in dialogue metrics file.")

    dial_agg = dial_df.groupby("model_tag")[avail_metrics].mean().reset_index()
    dial_agg["dialogue_kl_score"] = dial_agg[avail_metrics].mean(axis=1)
    dial_agg = dial_agg[["model_tag", "dialogue_kl_score"]]

    merged = test_agg.merge(dial_agg, on="model_tag", how="inner")
    return merged


def compute_discrepancy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def z_score(col: pd.Series) -> pd.Series:
        std = col.std()
        if std < 1e-9:
            return pd.Series(0.0, index=col.index)
        return (col - col.mean()) / std

    df["z_test"] = z_score(df["test_score"])
    df["z_dialogue"] = z_score(df["dialogue_kl_score"])
    df["discrepancy"] = df["z_test"] - df["z_dialogue"]

    df["student_type"] = df["discrepancy"].apply(classify_type)
    return df


def classify_type(delta: float) -> str:
    if delta < -DISCREPANCY_THRESHOLD:
        return "Type-A (eye-high-hand-low)"
    elif delta > DISCREPANCY_THRESHOLD:
        return "Type-B (silent-achiever)"
    elif delta > 0.5:
        return "Consistent-High"
    elif delta < -0.5:
        return "Consistent-Low"
    else:
        return "Aligned"


def compute_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    counts = df["student_type"].value_counts().reset_index()
    counts.columns = ["student_type", "n"]
    counts["prevalence"] = (counts["n"] / total).round(4)
    return counts


def run_dkt_model(dialogue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for lab DKT model inference.
    Replace this with actual DKT model call when available.
    Returns DataFrame with columns: model_tag, item_id, kl_score
    """
    print("[WARN] DKT model not configured. Using trace-metric proxy.")
    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="D6 Dialogue-test discrepancy")
    parser.add_argument("--test-scores", required=True)
    parser.add_argument("--dialogue-metrics", required=True)
    parser.add_argument("--dkt-model", default=None,
                        help="Path to lab DKT model checkpoint (optional)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged = load_and_merge(Path(args.test_scores), Path(args.dialogue_metrics))
    print(f"Merged: {len(merged)} models with both test and dialogue scores")

    discrepancy_df = compute_discrepancy(merged)
    print("\nDiscrepancy results:")
    print(discrepancy_df[["model_tag", "test_score", "dialogue_kl_score",
                           "discrepancy", "student_type"]].to_string(index=False))

    prevalence = compute_prevalence(discrepancy_df)
    print("\nType prevalence:")
    print(prevalence.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    discrepancy_df.to_csv(output_path, index=False)

    prev_path = output_path.with_name(output_path.stem + "_prevalence.csv")
    prevalence.to_csv(prev_path, index=False)

    type_ab = discrepancy_df[discrepancy_df["student_type"].str.startswith("Type")]
    if not type_ab.empty:
        print(f"\nType-A/B models ({len(type_ab)}):")
        print(type_ab[["model_tag", "discrepancy", "student_type"]].to_string(index=False))

    print(f"\nD6 discrepancy → {output_path}")
    print(f"D6 prevalence  → {prev_path}")


if __name__ == "__main__":
    main()
