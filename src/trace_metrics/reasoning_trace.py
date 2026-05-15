"""
Trace metric: reasoning-trace rate.

Definition: proportion of learner turns that contain explicit mathematical
reasoning steps (not just a bare numerical answer or acknowledgement).

Rule-based: presence of numeric expressions with operations, or connectives
like "because", "so", "therefore", "that means", "I calculated".
"""

import re


REASONING_INDICATORS = [
    r"\d+\s*[+\-×÷*/]\s*\d+",           # arithmetic expression
    r"=\s*\d",                            # equals something
    r"\b(because|so|therefore|thus|since|that means|i calculated|i computed|"
    r"i added|i subtracted|i multiplied|i divided|i got|step|first|then|next|"
    r"finally|my work|working out)\b",
]

MIN_REASONING_LENGTH = 20


def rule_based(learner_turn: str) -> bool:
    if len(learner_turn.strip()) < MIN_REASONING_LENGTH:
        return False
    text_lower = learner_turn.lower()
    for pat in REASONING_INDICATORS:
        if re.search(pat, text_lower):
            return True
    return False


def llm_judge_prompt(learner_turn: str) -> str:
    return (
        "Does the following student turn show explicit mathematical reasoning steps "
        "(not just a bare answer or a simple acknowledgement)?\n\n"
        f"Student turn: {learner_turn}\n\n"
        "Answer with exactly 'yes' or 'no'."
    )


def compute(learner_turn: str) -> dict:
    return {"rule": rule_based(learner_turn)}
