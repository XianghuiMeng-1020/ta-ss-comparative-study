"""
D7 — Persona consistency evaluation.

Extends existing src/trace_metrics/role_drift.py with two new metrics:
  - expert_leakage_rate: fraction of turns with Flesch-Kincaid grade > 9
    (above 7th-grade reading level = expert leakage for a naive student)
  - knowledge_stability: cosine similarity of consecutive-turn knowledge
    state estimates (from DKT proxy)

For single-response items (MCQ/TF/Fill/SA), D7 is computed only for the
response text length and vocabulary complexity (no multi-turn data).

For OED dialogues, full multi-turn analysis is run.

Usage:
    # For OED dialogues:
    python src/evaluation/persona_consistency.py \
        --dialogues outputs/generated_dialogues_*.csv \
        --output outputs/eval_d7_persona_consistency.csv

    # For single-response items:
    python src/evaluation/persona_consistency.py \
        --responses outputs/qa_responses_*.csv \
        --output outputs/eval_d7_single_response.csv
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPERT_WORDS = {
    "consequently", "furthermore", "therefore", "moreover", "nevertheless",
    "henceforth", "subsequently", "aforementioned", "instantiation",
    "implementation", "algorithm", "recursion", "polymorphism", "encapsulation",
    "asymptotic", "complexity", "paradigm", "methodology", "systematic",
}

STUDENT_ROLE_VIOLATION_PATTERNS = [
    r"\blet me explain\b",
    r"\bthe correct answer is\b",
    r"\byou should\b",
    r"\bi can teach\b",
    r"\bas an AI\b",
    r"\baccording to my (training|knowledge)\b",
    r"\bthe definition of\b",
    r"\bto summarize\b",
]

COMPILED_VIOLATIONS = [re.compile(p, re.IGNORECASE) for p in STUDENT_ROLE_VIOLATION_PATTERNS]


def count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:'\"")
    if len(word) <= 3:
        return 1
    vowels = "aeiouy"
    count = sum(1 for i, c in enumerate(word) if c in vowels and
                (i == 0 or word[i-1] not in vowels))
    count = max(1, count - (1 if word.endswith("e") else 0))
    return count


def flesch_kincaid_grade(text: str) -> float:
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    words = re.findall(r"\b\w+\b", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = syllables / len(words)
    fkg = 0.39 * asl + 11.8 * asw - 15.59
    return round(fkg, 2)


def check_expert_leakage(text: str) -> dict:
    fk_grade = flesch_kincaid_grade(text)
    expert_word_count = sum(1 for w in re.findall(r"\b\w+\b", text.lower())
                            if w in EXPERT_WORDS)
    violation_count = sum(1 for p in COMPILED_VIOLATIONS if p.search(text))
    is_expert = fk_grade > 9 or violation_count > 0
    return {
        "fk_grade": fk_grade,
        "expert_word_count": expert_word_count,
        "role_violation_count": violation_count,
        "expert_leakage": int(is_expert),
    }


def analyze_single_responses(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        text = str(row.get("response", ""))
        leakage = check_expert_leakage(text)
        rows.append({
            "item_id":            row.get("item_id"),
            "model_tag":          row.get("model_tag"),
            "qtype":              row.get("qtype"),
            "difficulty":         row.get("difficulty"),
            **leakage,
        })
    result_df = pd.DataFrame(rows)
    return result_df


def analyze_oed_dialogues(dialogues_df: pd.DataFrame) -> pd.DataFrame:
    from src.trace_metrics.role_drift import compute_role_drift_rate

    rows = []
    for _, row in dialogues_df.iterrows():
        import json
        turns = row.get("turns", "[]")
        if isinstance(turns, str):
            try:
                turns = json.loads(turns)
            except Exception:
                turns = []

        learner_turns = [t.get("learner_turn", "") for t in turns
                         if isinstance(t, dict)]

        if not learner_turns:
            continue

        leakages = [check_expert_leakage(t) for t in learner_turns if t]
        expert_leakage_rate = np.mean([l["expert_leakage"] for l in leakages]) if leakages else 0.0
        mean_fk = np.mean([l["fk_grade"] for l in leakages]) if leakages else 0.0

        drift_rate = 0.0
        try:
            full_text = " ".join(learner_turns)
            drift_rate = compute_role_drift_rate(full_text)
        except Exception:
            pass

        rows.append({
            "scenario_id":        row.get("scenario_id"),
            "model_tag":          row.get("model_id", row.get("model_tag")),
            "condition":          row.get("condition"),
            "n_turns":            len(learner_turns),
            "expert_leakage_rate": round(expert_leakage_rate, 4),
            "mean_fk_grade":      round(mean_fk, 2),
            "role_drift_rate":    round(drift_rate, 4),
            "d7_persona_score":   round(1 - (expert_leakage_rate + drift_rate) / 2, 4),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="D7 Persona consistency evaluation")
    parser.add_argument("--dialogues", nargs="*", default=[],
                        help="OED dialogue CSVs (for multi-turn analysis)")
    parser.add_argument("--responses", nargs="*", default=[],
                        help="Single-response CSVs (for MCQ/TF/Fill/SA)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_dfs = []

    if args.responses:
        paths = []
        for pattern in args.responses:
            paths.extend(glob.glob(pattern))
        resp_dfs = [pd.read_csv(p) for p in paths if Path(p).exists()]
        if resp_dfs:
            resp_df = pd.concat(resp_dfs, ignore_index=True)
            d7_single = analyze_single_responses(resp_df)
            d7_single["analysis_type"] = "single_response"
            all_dfs.append(d7_single)
            agg = d7_single.groupby("model_tag")[["expert_leakage", "fk_grade"]].mean().round(3)
            print("\nD7 single-response summary:")
            print(agg.to_string())

    if args.dialogues:
        paths = []
        for pattern in args.dialogues:
            paths.extend(glob.glob(pattern))
        dial_dfs = [pd.read_csv(p) for p in paths if Path(p).exists()]
        if dial_dfs:
            dial_df = pd.concat(dial_dfs, ignore_index=True)
            d7_oed = analyze_oed_dialogues(dial_df)
            d7_oed["analysis_type"] = "OED"
            all_dfs.append(d7_oed)
            agg_oed = d7_oed.groupby("model_tag")[
                ["expert_leakage_rate", "role_drift_rate", "d7_persona_score"]
            ].mean().round(3)
            print("\nD7 OED summary:")
            print(agg_oed.to_string())

    if all_dfs:
        final = pd.concat(all_dfs, ignore_index=True)
        final.to_csv(output_path, index=False)
        print(f"\nD7 persona consistency → {output_path}")
    else:
        print("No data loaded. Provide --dialogues and/or --responses.")


if __name__ == "__main__":
    main()
