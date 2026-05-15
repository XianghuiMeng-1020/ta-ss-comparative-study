"""
Add stratification labels to MathDial rows:
  - topic       : 6-class problem topic (keyword-based, then LLM fallback)
  - error_type  : 6-class error category (from student_profile + confusion text)
  - difficulty  : easy / medium / hard (based on solution chain length)
  - self_correction_flag : bool
  - teacher_move_mix     : dict of Focus/Probing/Telling/Generic proportions

Reference: MathDial teacher moves taxonomy (Macina et al., 2023, Table 2)
"""

import json
import re
from pathlib import Path

import pandas as pd

# ── Topic classification ──────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "fractions": [
        "fraction", "numerator", "denominator", "half", "third", "quarter",
        "⅓", "½", "¼", "simplify", "improper", "mixed number",
    ],
    "percentages": [
        "percent", "%", "discount", "tax", "rate", "ratio per", "interest",
        "markup", "sale price",
    ],
    "geometry": [
        "area", "perimeter", "volume", "angle", "triangle", "rectangle",
        "circle", "square", "radius", "diameter", "circumference", "shape",
    ],
    "algebra_words": [
        "variable", "equation", "solve for", "expression", "unknown",
        "linear", "coefficient", "inequality",
    ],
    "multi_step": [
        "total", "altogether", "combined", "first", "then", "after that",
        "remaining", "left over", "how many more", "how much more",
        "several steps",
    ],
    "arithmetic": [],  # fallback
}

TOPIC_PRIORITY = [
    "fractions", "percentages", "geometry", "algebra_words", "multi_step", "arithmetic"
]


def classify_topic(text: str) -> str:
    text_lower = text.lower()
    for topic in TOPIC_PRIORITY[:-1]:
        for kw in TOPIC_KEYWORDS[topic]:
            if kw in text_lower:
                return topic
    return "arithmetic"


# ── Error type classification ─────────────────────────────────────────────────

ERROR_KEYWORDS: dict[str, list[str]] = {
    "procedural_slip": [
        "forgot", "missed", "skipped", "forgot to", "should have divided",
        "forgot to multiply", "forgot to subtract", "operation", "sign error",
        "wrong operation",
    ],
    "conceptual_misconception": [
        "doesn't understand", "confuses", "thinks", "believes",
        "misconception", "misunderstands", "wrong concept", "concept",
        "doesn't know", "misapplied",
    ],
    "arithmetic_error": [
        "calculation", "arithmetic", "computed", "added incorrectly",
        "subtracted incorrectly", "multiplication error", "division error",
        "computed wrong", "arithmetic mistake",
    ],
    "setup_error": [
        "set up", "setup", "wrong equation", "wrong formula", "incorrect model",
        "wrong approach", "wrong strategy", "incorrect representation",
    ],
    "unit_error": [
        "unit", "units", "dimension", "convert", "conversion", "cm", "km",
        "kg", "lb", "litre", "gallon",
    ],
}

ERROR_PRIORITY = [
    "unit_error", "setup_error", "procedural_slip",
    "arithmetic_error", "conceptual_misconception", "miscellaneous"
]


def classify_error_type(profile: str, confusion: str) -> str:
    combined = (str(profile) + " " + str(confusion)).lower()
    for etype in ERROR_PRIORITY[:-1]:
        for kw in ERROR_KEYWORDS.get(etype, []):
            if kw in combined:
                return etype
    return "miscellaneous"


# ── Difficulty classification ─────────────────────────────────────────────────

def estimate_difficulty(solution_text: str) -> str:
    """
    Approximate difficulty by counting reasoning steps in the solution.
    Looks for sentence-ending punctuation and numeric expressions.
    Three-quantile split: easy / medium / hard.
    """
    if not solution_text or str(solution_text).strip() == "":
        return "medium"
    sentences = re.split(r"[.!?\n]+", str(solution_text))
    steps = [s for s in sentences if re.search(r"\d", s) and len(s.strip()) > 5]
    return len(steps)


# ── Teacher move mix ──────────────────────────────────────────────────────────

MOVE_KEYWORDS: dict[str, list[str]] = {
    "Focus": ["can you calculate", "what is", "how many", "how much", "find the"],
    "Probing": [
        "what would happen", "how would", "why", "can you explain",
        "walk me through", "what do you think",
    ],
    "Telling": [
        "you need to", "the answer is", "you should", "remember that",
        "no,", "actually,", "let me show",
    ],
    "Generic": ["good job", "great", "okay", "alright", "sure", "correct!"],
}


def classify_teacher_moves(tutor_turns_json: str) -> dict[str, float]:
    try:
        turns = json.loads(tutor_turns_json)
    except (json.JSONDecodeError, TypeError):
        return {"Focus": 0.0, "Probing": 0.0, "Telling": 0.0, "Generic": 0.0}

    counts: dict[str, int] = {"Focus": 0, "Probing": 0, "Telling": 0, "Generic": 0}
    for turn in turns:
        turn_lower = turn.lower()
        matched = False
        for move in ["Telling", "Probing", "Focus", "Generic"]:
            for kw in MOVE_KEYWORDS[move]:
                if kw in turn_lower:
                    counts[move] += 1
                    matched = True
                    break
            if matched:
                break
        if not matched:
            counts["Generic"] += 1

    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in counts.items()}


# ── Main entry ────────────────────────────────────────────────────────────────

def add_stratification_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Topic
    df["topic"] = df["problem"].apply(classify_topic)

    # Error type
    df["error_type"] = df.apply(
        lambda r: classify_error_type(r["student_profile"], r["teacher_described_confusion"]),
        axis=1,
    )

    # Difficulty — first compute raw step count, then three-quantile bins
    step_counts = df["correct_solution"].apply(estimate_difficulty)
    q33 = step_counts.quantile(0.33)
    q67 = step_counts.quantile(0.67)

    def bin_difficulty(n: int) -> str:
        if n <= q33:
            return "easy"
        elif n <= q67:
            return "medium"
        return "hard"

    df["difficulty"] = step_counts.apply(bin_difficulty)

    # Self-correction flag
    df["self_correction_flag"] = df["self_correctness"].apply(
        lambda v: str(v).strip() == "Yes"
    )

    # Teacher move mix
    df["teacher_move_mix"] = df["original_tutor_turns"].apply(classify_teacher_moves)

    return df


if __name__ == "__main__":
    from src.data.load_mathdial import load_and_cache

    df = load_and_cache()
    df = add_stratification_labels(df)
    print("Topic distribution:\n", df["topic"].value_counts())
    print("\nError type distribution:\n", df["error_type"].value_counts())
    print("\nDifficulty distribution:\n", df["difficulty"].value_counts())
    print("\nSelf-correction rate:", df["self_correction_flag"].mean().round(3))
