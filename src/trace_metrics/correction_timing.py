"""
Trace metric: correction-timing index.

Definition: the turn index at which the learner first produces a response
that no longer contains the target misconception. Lower values = faster
(potentially premature) correction; higher values = gradual (more plausible).

Computed as: correction_turn / total_turns (normalised 0–1).
High score (near 1) = misconception persists long → more plausible for C2.
Low score (near 0) = corrects immediately → potential premature correctness.
"""

import re
from difflib import SequenceMatcher


def misconception_still_present(teacher_described_confusion: str, learner_turn: str) -> bool:
    """Heuristic presence check using keyword overlap."""
    conf_words = set(re.findall(r"\b[a-z]{3,}\b", teacher_described_confusion.lower()))
    turn_words = set(re.findall(r"\b[a-z]{3,}\b", learner_turn.lower()))
    if not conf_words:
        return False
    overlap = len(conf_words & turn_words) / len(conf_words)
    return overlap >= 0.25


def compute(
    teacher_described_confusion: str,
    learner_turns: list[str],
) -> dict:
    if not learner_turns:
        return {
            "correction_turn_idx": None,
            "correction_timing_index": None,
            "misconception_persists_to_end": None,
        }

    correction_turn = None
    for i, turn in enumerate(learner_turns):
        if not misconception_still_present(teacher_described_confusion, turn):
            correction_turn = i
            break

    n = len(learner_turns)
    timing_index = correction_turn / n if correction_turn is not None else 1.0
    persists = correction_turn is None

    return {
        "correction_turn_idx": correction_turn,
        "correction_timing_index": timing_index,
        "misconception_persists_to_end": persists,
    }
