"""
Build Bridge cross-dataset scenario bank from raw Bridge corpus.

Bridge corpus: rose-e-wang/bridge (HuggingFace, CC-BY-NC-4.0)
  - Expert-novice tutoring conversations with annotated error types and teacher strategies.
  - We use EXPERT tutor conversations only (matching MathDial's quality-controlled design).

Output: data/scenarios_bridge.csv (same schema as data/scenarios.csv)

Decision Log: D8 (field mapping, expert-only filter, sampling rules)

Usage:
  python src/data/build_bridge_scenarios.py
  python src/data/build_bridge_scenarios.py --n 80 --seed 2026
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Field mapping: Bridge → MathDial schema (Decision Log D8) ─────────────────

ERROR_TYPE_MAP = {
    "conceptual":   "conceptual_misconception",
    "procedural":   "procedural_slip",
    "arithmetic":   "arithmetic_error",
    "setup":        "setup_error",
    "unit":         "unit_error",
    "other":        "miscellaneous",
    "miscellaneous": "miscellaneous",
    # Catch-all for unmapped values
}

DIFFICULTY_MAP = {
    "easy":   "easy",
    "medium": "medium",
    "hard":   "hard",
    "low":    "easy",
    "high":   "hard",
}

# Bridge topic → MathDial-compatible topic (best-effort)
TOPIC_MAP = {
    "fractions":    "fractions",
    "algebra":      "algebra_words",
    "arithmetic":   "arithmetic",
    "geometry":     "geometry",
    "percentages":  "percentages",
    "multi_step":   "multi_step",
    "word_problem": "multi_step",
}

RARE_STRATA_THRESHOLD = 5


def load_bridge_raw(data_dir: Path) -> pd.DataFrame:
    """Load all Bridge splits into a single DataFrame."""
    dfs = []
    for fname in ["train.csv", "validation.csv", "test.csv"]:
        fpath = data_dir / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            df["_source_split"] = fname.replace(".csv", "")
            dfs.append(df)
    if not dfs:
        # Try JSONL
        jsonl_path = data_dir / "bridge_all.jsonl"
        if jsonl_path.exists():
            records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
            return pd.DataFrame(records)
        raise FileNotFoundError(f"No Bridge data found in {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def detect_expert_filter_column(df: pd.DataFrame) -> str | None:
    """Find the column that identifies expert tutors."""
    for col in ["expertise_level", "tutor_expertise", "tutor_type", "expertise", "tutor_level"]:
        if col in df.columns:
            return col
    return None


def extract_turns(row: pd.Series) -> list[str]:
    """Extract tutor turns as a list of strings from a Bridge row."""
    turns_col_candidates = ["tutor_turns", "turns", "conversation", "dialogue", "utterances"]
    for col in turns_col_candidates:
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        # Extract text from each turn
                        result = []
                        for t in parsed:
                            if isinstance(t, dict):
                                text = t.get("text", t.get("content", t.get("utterance", "")))
                                role = t.get("role", t.get("speaker", ""))
                                if "tutor" in str(role).lower() or "teacher" in str(role).lower():
                                    result.append(str(text))
                            elif isinstance(t, str):
                                result.append(t)
                        return result
                except (json.JSONDecodeError, TypeError):
                    return [str(val)]
    return []


def map_error_type(raw_error: str | None) -> str:
    if pd.isna(raw_error) or raw_error is None:
        return "miscellaneous"
    raw = str(raw_error).lower().strip()
    for key, mapped in ERROR_TYPE_MAP.items():
        if key in raw:
            return mapped
    return "miscellaneous"


def map_topic(raw_topic: str | None) -> str:
    if pd.isna(raw_topic) or raw_topic is None:
        return "arithmetic"
    raw = str(raw_topic).lower().strip()
    for key, mapped in TOPIC_MAP.items():
        if key in raw:
            return mapped
    return "arithmetic"


def map_difficulty(raw_diff: str | None) -> str:
    if pd.isna(raw_diff) or raw_diff is None:
        return "medium"
    raw = str(raw_diff).lower().strip()
    return DIFFICULTY_MAP.get(raw, "medium")


def build_scenario_from_row(row: pd.Series, scenario_id: str) -> dict:
    """Convert a Bridge row to MathDial-compatible scenario dict."""
    # Problem text — try multiple column names
    problem = ""
    for col in ["problem", "question", "math_problem", "problem_text", "task"]:
        if col in row.index and pd.notna(row[col]):
            problem = str(row[col])
            break

    # Incorrect solution / first student utterance
    incorrect = ""
    for col in ["student_answer", "initial_answer", "incorrect_solution",
                "student_response", "first_student_turn"]:
        if col in row.index and pd.notna(row[col]):
            incorrect = str(row[col])
            break

    # Misconception label (teacher-described confusion equivalent)
    confusion = ""
    for col in ["e", "error_type", "misconception", "teacher_described_confusion",
                "student_error", "error_description", "z_what"]:
        if col in row.index and pd.notna(row[col]):
            confusion = str(row[col])
            break

    # Tutor turns for replay
    tutor_turns = extract_turns(row)
    if not tutor_turns:
        # Fallback: try direct column
        for col in ["tutor_utterances", "teacher_turns"]:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                try:
                    tutor_turns = json.loads(str(val))
                except Exception:
                    tutor_turns = [str(val)]
                break

    raw_error = row.get("e", row.get("error_type", None))
    raw_topic = row.get("topic", row.get("subject", None))
    raw_diff = row.get("difficulty", None)

    return {
        "scenario_id": scenario_id,
        "problem": problem,
        "original_incorrect_solution": incorrect,
        "teacher_described_confusion": confusion,
        "student_profile": f"A 7th-grade student with a {map_error_type(raw_error)} error pattern.",
        "error_type": map_error_type(raw_error),
        "topic": map_topic(raw_topic),
        "difficulty": map_difficulty(raw_diff),
        "source_dataset": "bridge",
        "source_split": row.get("_source_split", "unknown"),
        "tutor_turns_json": json.dumps(tutor_turns),
        "n_tutor_turns": len(tutor_turns),
    }


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample by error_type; merge rare strata (< threshold)."""
    rng = random.Random(seed)

    strata_counts = df["error_type"].value_counts()
    rare = strata_counts[strata_counts < RARE_STRATA_THRESHOLD].index.tolist()
    df = df.copy()
    if rare:
        print(f"Merging {len(rare)} rare strata into 'miscellaneous': {rare}")
        df.loc[df["error_type"].isin(rare), "error_type"] = "miscellaneous"

    strata = df.groupby("error_type")
    n_strata = df["error_type"].nunique()
    per_stratum = max(1, n // n_strata)

    selected = []
    for stratum, group in strata:
        take = min(per_stratum, len(group))
        selected.append(group.sample(n=take, random_state=seed))

    # Fill remaining slots randomly
    selected_df = pd.concat(selected, ignore_index=True)
    remaining = n - len(selected_df)
    if remaining > 0:
        pool = df[~df.index.isin(selected_df.index)]
        if len(pool) >= remaining:
            extras = pool.sample(n=remaining, random_state=seed + 1)
            selected_df = pd.concat([selected_df, extras], ignore_index=True)

    return selected_df.head(n).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Bridge cross-dataset scenario bank")
    parser.add_argument("--data-dir", default="data/raw_bridge")
    parser.add_argument("--output", default="data/scenarios_bridge.csv")
    parser.add_argument("--n", type=int, default=80, help="Number of scenarios to sample")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--expert-only", action="store_true", default=True,
        help="Use only expert-tutor conversations (Decision Log D8)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Loading Bridge data from {data_dir} ...")
    df = load_bridge_raw(data_dir)
    print(f"Raw records: {len(df)} | Columns: {list(df.columns[:10])}")

    # Expert-only filter (Decision Log D8)
    expert_col = detect_expert_filter_column(df)
    if expert_col and args.expert_only:
        expert_vals = {"expert", "experienced", "1", "true", "yes", "high"}
        mask = df[expert_col].astype(str).str.lower().isin(expert_vals)
        n_before = len(df)
        df = df[mask].reset_index(drop=True)
        print(f"Expert-only filter on '{expert_col}': {n_before} → {len(df)} rows")
    else:
        print(
            f"No expertise column found (checked: expertise_level, tutor_expertise, etc.). "
            f"Using all {len(df)} rows."
        )

    # Require minimum viable fields
    viable_mask = pd.Series([True] * len(df))
    for col in ["e", "error_type", "problem", "question"]:
        if col in df.columns:
            break
    else:
        print("WARNING: No problem or error_type column found; proceeding with defaults.")

    print(f"Viable rows after filtering: {len(df)}")

    # Build scenario dicts
    scenarios = []
    for i, (_, row) in enumerate(df.iterrows()):
        scenario_id = f"bridge_{i:04d}"
        s = build_scenario_from_row(row, scenario_id)
        if s["problem"] or s["original_incorrect_solution"]:
            scenarios.append(s)

    scenario_df = pd.DataFrame(scenarios)
    print(f"Built {len(scenario_df)} valid scenarios")

    # Stratified sample
    if len(scenario_df) > args.n:
        scenario_df = stratified_sample(scenario_df, args.n, args.seed)
    print(f"Sampled {len(scenario_df)} scenarios")

    # Print distribution
    print("\nError type distribution:")
    print(scenario_df["error_type"].value_counts())
    print("\nTopic distribution:")
    print(scenario_df["topic"].value_counts())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_df.to_csv(output_path, index=False)
    print(f"\nSaved Bridge scenarios → {output_path}")
    print(f"Schema columns: {list(scenario_df.columns)}")


if __name__ == "__main__":
    main()
