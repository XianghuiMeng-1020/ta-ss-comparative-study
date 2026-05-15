"""
Trace metric: near-transfer attempt rate.

Definition: whether the learner's final turn (response to transfer probe)
constitutes a genuine independent attempt — shows reasoning steps and
produces a numerical answer without simply saying "I don't know".

Key for C1 (TA1: independent performance), expected absent in C3/C4.
"""

import re


NO_ATTEMPT_PATTERNS = [
    r"^(i don'?t know|i'?m not sure|no idea|i can'?t|i cannot|i give up)\.?$",
    r"^(help|hint|i need help)\.?$",
]

ATTEMPT_INDICATORS = [
    r"\d+\s*[+\-×÷*/]\s*\d+",
    r"=\s*\d",
    r"\b(step|first|so|therefore|that means|i think|my answer|i get)\b",
]

MIN_ATTEMPT_LENGTH = 25


def rule_based(transfer_turn: str) -> bool:
    text = transfer_turn.strip()
    if len(text) < MIN_ATTEMPT_LENGTH:
        return False
    lower = text.lower()
    for pat in NO_ATTEMPT_PATTERNS:
        if re.match(pat, lower):
            return False
    for pat in ATTEMPT_INDICATORS:
        if re.search(pat, lower):
            return True
    return len(text) >= MIN_ATTEMPT_LENGTH * 2


def llm_judge_prompt(transfer_probe: str, learner_turn: str) -> str:
    return (
        f"Transfer problem given to student: {transfer_probe}\n\n"
        f"Student response: {learner_turn}\n\n"
        "Did the student make a genuine independent attempt to solve the transfer problem "
        "(showing reasoning steps), rather than refusing, asking for help, or giving only "
        "a bare number with no reasoning?\n"
        "Answer with exactly 'yes' or 'no'."
    )


def compute(transfer_turn: str) -> dict:
    return {"rule": rule_based(transfer_turn)}
