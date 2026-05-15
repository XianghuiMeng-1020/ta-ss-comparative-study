"""
Trace metric: unsupported-reasoning rate (LLM validity failure flag).

Definition: learner asserts a numerical answer without any visible
mathematical reasoning (bare answer), or contradicts themselves logically
between turns without acknowledging the contradiction.

LLM validity threat: logical inconsistency (Yuan et al., 2026).
"""

import re


BARE_ANSWER_PATTERN = re.compile(
    r"^\s*(the answer is\s*)?(\d+(?:\.\d+)?)\s*[.!]?\s*$", re.IGNORECASE
)

REASONING_INDICATORS = [
    r"\d+\s*[+\-×÷*/]\s*\d+",
    r"=\s*\d",
    r"\b(because|so|since|therefore|that means|step|i calculated|i added|i multiplied)\b",
]


def is_bare_answer(learner_turn: str) -> bool:
    """True if the turn is just a number or 'the answer is N' with no reasoning."""
    if BARE_ANSWER_PATTERN.match(learner_turn.strip()):
        return True
    if len(learner_turn.strip()) < 30:
        for pat in REASONING_INDICATORS:
            if re.search(pat, learner_turn.lower()):
                return False
        return True
    return False


def detect_logical_contradiction(turn_a: str, turn_b: str) -> bool:
    """
    Heuristic: if learner gives different numerical answers in consecutive turns
    without any revision marker, flag as potential logical inconsistency.
    """
    REVISION_MARKERS = ["i was wrong", "actually", "let me redo", "sorry", "i see", "revising"]
    nums_a = set(re.findall(r"\b\d+(?:\.\d+)?\b", turn_a))
    nums_b = set(re.findall(r"\b\d+(?:\.\d+)?\b", turn_b))
    if not nums_a or not nums_b:
        return False
    if nums_a != nums_b:
        # Different answers — check for revision marker
        lower_b = turn_b.lower()
        if not any(m in lower_b for m in REVISION_MARKERS):
            return True
    return False


def compute(learner_turns: list[str]) -> dict:
    if not learner_turns:
        return {
            "rule_bare_answer_rate": 0.0,
            "rule_contradiction_rate": 0.0,
            "rule_any_unsupported": False,
        }
    bare = [is_bare_answer(t) for t in learner_turns]
    bare_rate = sum(bare) / len(learner_turns)

    contradictions = []
    for i in range(len(learner_turns) - 1):
        contradictions.append(
            detect_logical_contradiction(learner_turns[i], learner_turns[i + 1])
        )
    contradiction_rate = sum(contradictions) / max(len(contradictions), 1)

    return {
        "rule_bare_answer_rate": bare_rate,
        "rule_contradiction_rate": contradiction_rate,
        "rule_any_unsupported": bare_rate > 0 or contradiction_rate > 0,
    }


def llm_judge_prompt(learner_turn: str) -> str:
    return (
        "Does the following student turn contain a bare assertion or answer "
        "with no mathematical reasoning to support it?\n\n"
        f"Student turn: {learner_turn}\n\n"
        "Answer with exactly 'yes' or 'no'."
    )
