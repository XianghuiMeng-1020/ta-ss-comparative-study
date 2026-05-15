"""
Trace metric: role-drift rate (LLM validity failure flag).

Definition: proportion of learner turns that contain tutor-style language,
suggesting the model has drifted from learner to teacher/assistant role.

LLM validity threat: role instability (Yuan et al., 2026; Mannekote et al., 2025).
"""

import re


TUTOR_STYLE_PATTERNS = [
    r"\blet me explain\b",
    r"\bremember that\b",
    r"\bthe key (concept|idea|point|thing) is\b",
    r"\byou (should|need to|must|have to) (think|consider|remember|note)\b",
    r"\bi'?ll (show|explain|walk you through|demonstrate)\b",
    r"\bhere'?s (how|why|what)\b",
    r"\bto solve this(,| you)\b",
    r"\bthe (formula|equation|rule|method) (is|for|to)\b",
    r"\bfirst[,]? (you|we) (need|should|must)\b",
    r"\bthink of it (as|like)\b",
    r"\bthat'?s (correct|right|exactly)!?\s*[\.,]",
    r"\bgood (job|work|thinking|question)",
]


def rule_based(learner_turn: str) -> bool:
    lower = learner_turn.lower()
    return any(re.search(pat, lower) for pat in TUTOR_STYLE_PATTERNS)


def compute_dialogue_drift_rate(learner_turns: list[str]) -> float:
    """Returns proportion of turns with role drift."""
    if not learner_turns:
        return 0.0
    drifted = sum(rule_based(t) for t in learner_turns)
    return drifted / len(learner_turns)


def llm_judge_prompt(learner_turn: str) -> str:
    return (
        "The following is supposed to be a student's response to a math tutor. "
        "Does the student adopt a teacher or assistant role in this turn — "
        "for example, by explaining concepts, evaluating the tutor, or instructing?\n\n"
        f"Student turn: {learner_turn}\n\n"
        "Answer with exactly 'yes' (role drift detected) or 'no'."
    )


def compute(learner_turns: list[str]) -> dict:
    drift_rate = compute_dialogue_drift_rate(learner_turns)
    any_drift = drift_rate > 0
    return {
        "rule_drift_rate": drift_rate,
        "rule_any_drift": any_drift,
    }
