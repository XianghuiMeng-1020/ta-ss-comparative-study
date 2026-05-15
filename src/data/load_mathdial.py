"""
Load and cache the MathDial dataset from HuggingFace Hub.
Dataset: eth-nlped/mathdial (CC BY-4.0)
Reference: Macina et al. (2023). MathDial. EMNLP Findings.
"""

import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw_mathdial")
CACHE_FILE = RAW_DIR / "mathdial_train.jsonl"


def parse_conversation(conv_str: str) -> tuple[list[str], list[str]]:
    """
    Parse MathDial conversation string (|EOM| delimited) into
    alternating tutor / student turn lists.

    MathDial conversations begin with a teacher turn.
    Pattern: Teacher: ... |EOM| Student: ... |EOM| Teacher: ...
    """
    turns = [t.strip() for t in conv_str.split("|EOM|") if t.strip()]
    tutor_turns: list[str] = []
    student_turns: list[str] = []
    for turn in turns:
        if turn.startswith("Teacher:"):
            tutor_turns.append(turn[len("Teacher:"):].strip())
        elif turn.startswith("Student:"):
            student_turns.append(turn[len("Student:"):].strip())
    return tutor_turns, student_turns


def load_and_cache() -> pd.DataFrame:
    """Download (or read from cache) the MathDial train split."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists():
        print(f"Loading from cache: {CACHE_FILE}")
        records = [json.loads(l) for l in CACHE_FILE.read_text().splitlines() if l.strip()]
        return pd.DataFrame(records)

    print("Downloading MathDial from HuggingFace Hub…")
    ds = load_dataset("eth-nlped/mathdial", split="train")

    records = []
    for row in tqdm(ds, desc="Parsing conversations"):
        tutor_turns, student_turns = parse_conversation(row.get("conversation", ""))
        records.append({
            "mathdial_qid": row.get("qid", ""),
            "mathdial_scenario_idx": row.get("scenario", ""),
            "problem": row.get("question", ""),
            "correct_solution": row.get("ground_truth", ""),
            "original_incorrect_solution": row.get("student_incorrect_solution", ""),
            "student_profile": row.get("student_profile", ""),
            "teacher_described_confusion": row.get("teacher_described_confusion", ""),
            "original_conversation": row.get("conversation", ""),
            "original_tutor_turns": json.dumps(tutor_turns),
            "original_student_turns": json.dumps(student_turns),
            "n_tutor_turns": len(tutor_turns),
            "n_student_turns": len(student_turns),
            "self_correctness": row.get("self-correctness", ""),
            "self_typical_confusion": row.get("self-typical-confusion", ""),
            "self_typical_interactions": row.get("self-typical-interactions", ""),
        })

    df = pd.DataFrame(records)
    with CACHE_FILE.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Cached {len(df)} rows to {CACHE_FILE}")
    return df


if __name__ == "__main__":
    df = load_and_cache()
    print(f"Loaded {len(df)} MathDial train rows.")
    print(df.dtypes)
    print(df.head(2).to_string())
