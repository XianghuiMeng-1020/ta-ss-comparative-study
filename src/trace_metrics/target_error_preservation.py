"""
Trace metric: target-error preservation rate.

Definition: whether the learner's first 1–2 turns contain an error that
is semantically similar to the teacher-described confusion (target misconception).

Rule-based: keyword overlap between teacher_described_confusion and learner turn.
LLM-judge: ask whether the target misconception is visible in learner's first response.

Reference: Koedinger et al. (2015) error-model alignment (SS2).
"""

import re
from difflib import SequenceMatcher


def keyword_overlap(target_confusion: str, learner_turn: str) -> float:
    """Normalised keyword overlap between confusion label and learner turn."""
    if not target_confusion or not learner_turn:
        return 0.0
    conf_words = set(re.findall(r"\b[a-z]{3,}\b", target_confusion.lower()))
    turn_words = set(re.findall(r"\b[a-z]{3,}\b", learner_turn.lower()))
    if not conf_words:
        return 0.0
    return len(conf_words & turn_words) / len(conf_words)


def sequence_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def rule_based(teacher_described_confusion: str, learner_turn: str) -> bool:
    """Return True if target error is likely preserved in learner_turn."""
    kw_score = keyword_overlap(teacher_described_confusion, learner_turn)
    if kw_score >= 0.3:
        return True
    seq_score = sequence_similarity(teacher_described_confusion, learner_turn)
    return seq_score >= 0.25


def llm_judge_prompt(teacher_described_confusion: str, learner_turn: str) -> str:
    return (
        "The student has a specific mathematical misconception: "
        f"'{teacher_described_confusion}'\n\n"
        "Does the following student response reflect or exhibit this misconception?\n\n"
        f"Student response: {learner_turn}\n\n"
        "Answer with exactly 'yes' or 'no'."
    )


def compute(teacher_described_confusion: str, learner_first_turns: list[str]) -> dict:
    """Check first 1-2 learner turns for target error presence."""
    check_turns = learner_first_turns[:2]
    rule_result = any(rule_based(teacher_described_confusion, t) for t in check_turns)
    return {
        "rule": rule_result,
        "n_turns_checked": len(check_turns),
    }
