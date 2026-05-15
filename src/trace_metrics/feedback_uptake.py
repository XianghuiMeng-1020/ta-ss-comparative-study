"""
Trace metric: feedback-uptake rate.

Definition: after a corrective tutor turn (k), the learner's turn (k+1)
constitutes a non-trivial modification of their previous answer.

Rule-based: token-level diff ≥ 30% between learner turn k and k+1,
AND a positive feedback acknowledgement word is present.
"""

import re


ACKNOWLEDGEMENT_WORDS = [
    "i see", "oh", "ah", "got it", "i understand", "you're right",
    "okay", "ok", "so", "let me try", "let me redo", "i think i",
    "i was wrong", "i made a mistake", "you said", "so i should",
    "revising", "actually",
]

CORRECTIVE_TUTOR_WORDS = [
    "not quite", "that's not", "incorrect", "let's", "think about",
    "what if", "try again", "remember", "actually", "no,", "hint",
    "consider", "look at", "check", "re-read",
]


def is_corrective_turn(tutor_turn: str) -> bool:
    lower = tutor_turn.lower()
    return any(w in lower for w in CORRECTIVE_TUTOR_WORDS)


def token_diff_ratio(prev: str, curr: str) -> float:
    prev_tokens = set(re.findall(r"\b\w+\b", prev.lower()))
    curr_tokens = set(re.findall(r"\b\w+\b", curr.lower()))
    if not prev_tokens:
        return 1.0
    changed = len(prev_tokens.symmetric_difference(curr_tokens))
    return changed / len(prev_tokens | curr_tokens)


def rule_based(
    tutor_turns: list[str],
    learner_turns: list[str],
) -> float:
    """
    Returns proportion of corrective tutor turns that elicit uptake.
    Requires aligned lists (tutor[k] → learner[k]).
    """
    if len(tutor_turns) < 2 or len(learner_turns) < 2:
        return 0.0

    n_corrective = 0
    n_uptake = 0

    for k in range(len(tutor_turns) - 1):
        if not is_corrective_turn(tutor_turns[k]):
            continue
        n_corrective += 1
        if k + 1 >= len(learner_turns) or k >= len(learner_turns):
            continue
        prev_learner = learner_turns[k] if k < len(learner_turns) else ""
        next_learner = learner_turns[k + 1] if k + 1 < len(learner_turns) else ""
        diff = token_diff_ratio(prev_learner, next_learner)
        ack = any(w in next_learner.lower() for w in ACKNOWLEDGEMENT_WORDS)
        if diff >= 0.30 or ack:
            n_uptake += 1

    if n_corrective == 0:
        return float("nan")
    return n_uptake / n_corrective


def llm_judge_prompt(tutor_turn: str, prev_learner: str, next_learner: str) -> str:
    return (
        "The tutor gave corrective feedback:\n"
        f"Tutor: {tutor_turn}\n\n"
        f"Previous student response: {prev_learner}\n\n"
        f"Next student response: {next_learner}\n\n"
        "Did the student meaningfully incorporate the tutor's feedback into their next response "
        "(not just parroting, but showing genuine revision)?\n"
        "Answer with exactly 'yes' or 'no'."
    )


def compute(tutor_turns: list[str], learner_turns: list[str]) -> dict:
    return {"rule": rule_based(tutor_turns, learner_turns)}
