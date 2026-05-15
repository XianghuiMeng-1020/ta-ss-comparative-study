"""
Trace metric: premature-correctness rate (LLM validity failure flag).

Definition: learner achieves correct final answer before receiving ≥3
substantive tutor turns. Operationalises the 'competence paradox'
(Yuan et al., 2026) — the model knows the answer but is suppressing it.

Substantive tutor turn: any Focus, Probing, or Telling move
(not Generic/acknowledgement-only).

Reference: Yuan, Z., et al. (2026). Towards valid student simulation with LLMs.
"""

import re


GENERIC_ONLY_PATTERNS = [
    r"^(good job|great|okay|alright|sure|correct!|yes!?|right!?)[\s.!]*$",
    r"^(thank you|thanks|well done|nice work)[\s.!]*$",
]

# Patterns suggesting the learner has given a correct complete answer
CORRECT_ANSWER_PATTERNS = [
    r"\b(the answer is|so the answer is|therefore|my final answer is)\b",
    r"=\s*\d+\s*$",
    r"\b(correct|right|exactly|i got it|solved)\b",
]


def is_substantive_tutor_turn(tutor_turn: str) -> bool:
    lower = tutor_turn.strip().lower()
    for pat in GENERIC_ONLY_PATTERNS:
        if re.match(pat, lower):
            return False
    return len(lower) > 15


def learner_appears_correct(learner_turn: str, correct_solution: str) -> bool:
    """Heuristic: learner claims or shows a correct answer."""
    lower = learner_turn.lower()
    for pat in CORRECT_ANSWER_PATTERNS:
        if re.search(pat, lower):
            return True
    # Check if correct solution number appears verbatim
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", str(correct_solution))
    for n in nums:
        if n in learner_turn:
            return True
    return False


def rule_based(
    tutor_turns: list[str],
    learner_turns: list[str],
    correct_solution: str,
    threshold_substantive: int = 2,
) -> bool:
    """
    Returns True (premature correctness detected) if the learner appears
    correct before receiving threshold_substantive substantive tutor turns.
    """
    substantive_count = 0
    for k, (tt, lt) in enumerate(zip(tutor_turns, learner_turns)):
        if is_substantive_tutor_turn(tt):
            substantive_count += 1
        if substantive_count <= threshold_substantive:
            if learner_appears_correct(lt, correct_solution):
                return True
    return False


def llm_judge_prompt(
    learner_turns: list[str],
    correct_solution: str,
    n_substantive_before: int,
) -> str:
    early_turns = "\n".join(
        [f"Turn {i+1}: {t}" for i, t in enumerate(learner_turns[:3])]
    )
    return (
        f"Correct answer to the problem: {correct_solution}\n\n"
        f"Student's early responses (before receiving {n_substantive_before} "
        f"substantive tutor turns):\n{early_turns}\n\n"
        "Did the student provide the correct complete answer prematurely, "
        "before the tutor had a chance to adequately scaffold them?\n"
        "Answer with exactly 'yes' or 'no'."
    )


def compute(
    tutor_turns: list[str],
    learner_turns: list[str],
    correct_solution: str,
) -> dict:
    return {
        "rule": rule_based(tutor_turns, learner_turns, correct_solution),
    }
