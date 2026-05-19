"""
D2 — Answering performance evaluation.

Metrics:
  - Exact-match accuracy (MCQ, TF)
  - Fill/SA: partial-credit rubric (2=correct, 1=plausible error, 0=nonsense)
  - Explanation quality 1–5 (LLM judge for SA/OED; requires calibration)
  - Expected error profile: higher error rate at Hard difficulty (validity check)

Usage:
    python src/evaluation/answering.py \
        --responses outputs/qa_responses_*.csv \
        --output outputs/eval_d2_answering.csv

    # With LLM judge for SA/OED:
    python src/evaluation/answering.py \
        --responses outputs/qa_responses_gpt-4o.csv \
        --judge-model gpt-4o-2024-11-20 \
        --output outputs/eval_d2_with_judge.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PARTIAL_CREDIT_JUDGE_PROMPT = """\
You are grading a student's answer on a short-answer or fill-in-blank question.

Question: {question}
Correct answer: {correct_answer}
Student's response: {student_response}

Grade the student's response on this rubric:
  2 — Correct or essentially correct
  1 — Plausible student error (wrong but shows partial understanding / realistic misconception)
  0 — Nonsensical, completely off-topic, or refuses to answer

Respond with ONLY the number (0, 1, or 2) followed by a brief justification (1 sentence).
Format: "GRADE: X — justification"
"""

EXPLANATION_QUALITY_PROMPT = """\
You are evaluating the quality of a student's explanation on a 1–5 scale.

Question: {question}
Student's explanation: {student_response}

Rate explanation quality:
  5 — Clear, complete, shows genuine understanding (expert-level)
  4 — Mostly clear with minor gaps
  3 — Partial understanding, key ideas present but confused
  2 — Minimal understanding, mostly guessing
  1 — Incoherent or no real explanation

Respond with ONLY: "QUALITY: X — justification"
"""


def load_all_responses(paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        if not Path(p).exists():
            continue
        dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError(f"No response CSVs found in {paths}")
    return pd.concat(dfs, ignore_index=True)


def compute_accuracy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accuracy by model × qtype × difficulty."""
    graded = df[df["correct"].notna()].copy()
    graded["correct"] = graded["correct"].astype(float)

    agg = graded.groupby(["model_tag", "qtype", "difficulty"]).agg(
        n_responses=("correct", "count"),
        accuracy=("correct", "mean"),
        accuracy_sd=("correct", "std"),
    ).reset_index()

    agg["accuracy"] = agg["accuracy"].round(4)
    agg["accuracy_sd"] = agg["accuracy_sd"].round(4)
    return agg


def compute_partial_credit(
    df: pd.DataFrame,
    judge_model: str,
    judge_backend: str = "openai",
) -> pd.DataFrame:
    """Score Fill/SA responses with partial-credit rubric via LLM judge."""
    from src.generation.clients import LLMConfig, call_llm

    config = LLMConfig(
        model_id=judge_model,
        temperature=0.0,
        max_tokens=100,
        api_backend=judge_backend,
    )

    target = df[df["qtype"].isin(["Fill", "SA"]) & df["correct"].isna()].copy()
    grades = []

    for _, row in target.iterrows():
        prompt = PARTIAL_CREDIT_JUDGE_PROMPT.format(
            question=row.get("question", ""),
            correct_answer=row.get("correct_answer", ""),
            student_response=row.get("response", ""),
        )
        messages = [
            {"role": "system", "content": "You are a strict but fair grader."},
            {"role": "user", "content": prompt},
        ]
        try:
            result = call_llm(messages, config, seed=42)
            text = result.content.strip()
            grade = int(text.split("GRADE:")[1].strip()[0]) if "GRADE:" in text else None
        except Exception:
            grade = None

        grades.append({
            "item_id": row["item_id"],
            "model_tag": row.get("model_tag"),
            "partial_credit": grade,
        })

    return pd.DataFrame(grades)


def compute_explanation_quality(
    df: pd.DataFrame,
    judge_model: str,
    judge_backend: str = "openai",
    sample_n: int = 200,
) -> pd.DataFrame:
    """Rate explanation quality for SA/OED responses (calibration subset)."""
    from src.generation.clients import LLMConfig, call_llm

    config = LLMConfig(
        model_id=judge_model,
        temperature=0.0,
        max_tokens=80,
        api_backend=judge_backend,
    )

    target = df[df["qtype"].isin(["SA", "OED"])].sample(
        min(sample_n, len(df)), random_state=42
    )
    results = []

    for _, row in target.iterrows():
        prompt = EXPLANATION_QUALITY_PROMPT.format(
            question=row.get("question", ""),
            student_response=row.get("response", ""),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            result = call_llm(messages, config, seed=42)
            text = result.content.strip()
            quality = int(text.split("QUALITY:")[1].strip()[0]) if "QUALITY:" in text else None
        except Exception:
            quality = None

        results.append({
            "item_id": row["item_id"],
            "model_tag": row.get("model_tag"),
            "explanation_quality": quality,
        })

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="D2 Answering performance evaluation")
    parser.add_argument("--responses", nargs="+", required=True,
                        help="qa_responses CSV files (supports glob patterns)")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-backend", default="openai")
    parser.add_argument("--partial-credit", action="store_true")
    parser.add_argument("--explanation-quality", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import glob
    paths = []
    for pattern in args.responses:
        paths.extend(glob.glob(pattern))

    df = load_all_responses(paths)
    print(f"Loaded {len(df)} responses from {len(paths)} files")

    acc_df = compute_accuracy_stats(df)
    print("\nAccuracy summary (MCQ/TF auto-graded):")
    print(acc_df.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    acc_df.to_csv(output_path, index=False)

    if args.judge_model and args.partial_credit:
        pc_df = compute_partial_credit(df, args.judge_model, args.judge_backend)
        pc_path = output_path.with_name(output_path.stem + "_partial_credit.csv")
        pc_df.to_csv(pc_path, index=False)
        print(f"\nPartial-credit scores → {pc_path}")

    if args.judge_model and args.explanation_quality:
        eq_df = compute_explanation_quality(df, args.judge_model, args.judge_backend)
        eq_path = output_path.with_name(output_path.stem + "_explanation_quality.csv")
        eq_df.to_csv(eq_path, index=False)
        print(f"Explanation quality → {eq_path}")

    print(f"\nD2 main output → {output_path}")


if __name__ == "__main__":
    main()
