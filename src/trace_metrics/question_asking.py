"""
Trace metric: question-asking rate.

Definition: proportion of learner turns that contain a genuine question
(not rhetorical, not tutor-directed affirmation).

Rule-based: presence of "?" not at end of a formulaic phrase.
LLM-judge channel: prompt asks whether the turn contains a learner-directed question.
"""

import re


RHETORICAL_PATTERNS = [
    r"right\?$",
    r"ok\?$",
    r"okay\?$",
    r"correct\?$",
    r"isn'?t it\?$",
    r"don'?t (you|we)\?$",
]


def rule_based(learner_turn: str) -> bool:
    text = learner_turn.strip()
    if "?" not in text:
        return False
    text_lower = text.lower()
    for pat in RHETORICAL_PATTERNS:
        if re.search(pat, text_lower):
            return False
    return True


def llm_judge_prompt(learner_turn: str) -> str:
    return (
        "Does the following student turn contain a genuine clarifying or understanding-seeking "
        "question (not a rhetorical tag question)?\n\n"
        f"Student turn: {learner_turn}\n\n"
        "Answer with exactly 'yes' or 'no'."
    )


def compute(learner_turn: str) -> dict:
    return {"rule": rule_based(learner_turn)}
