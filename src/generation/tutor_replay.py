"""
D2 default tutor implementation: replay MathDial human tutor turns verbatim.

Each original tutor turn becomes the 'user' message to the learner LLM.
After all tutor turns, a near-transfer probe is appended as the final user message.
Turn budget = len(original_tutor_turns) + 1 (transfer probe).

This design eliminates tutor variation as a confound across conditions.
"""

import json
import re
from pathlib import Path


def load_transfer_probe_template() -> str:
    """Load the frozen transfer probe template."""
    path = Path("prompts/transfer_probe.txt")
    if path.exists():
        return path.read_text().strip()
    return (
        "Now try this similar problem on your own without any hints. "
        "Show your reasoning step by step:\n\n{transfer_problem}"
    )


def generate_transfer_problem(problem: str, correct_solution: str) -> str:
    """
    Generate a near-transfer problem by replacing numbers in the original.
    Keeps structure identical; replaces all integers and decimals with new values.
    """
    import random
    rng = random.Random(hash(problem) % 2**32)

    def replace_number(match: re.Match) -> str:
        original = float(match.group())
        # Perturb by ±20–50%, keep same sign, round sensibly
        factor = rng.uniform(1.2, 1.5) * rng.choice([-1, 1])
        new_val = abs(original * factor)
        if new_val == int(new_val):
            return str(int(new_val))
        return f"{new_val:.1f}"

    return re.sub(r"\b\d+(?:\.\d+)?\b", replace_number, problem)


def build_tutor_turns(scenario: dict) -> list[str]:
    """
    Return the list of tutor turn strings to replay, plus the transfer probe.

    scenario must have:
      - original_tutor_turns  : JSON list of strings
      - problem               : str
      - correct_solution      : str
    """
    raw = scenario.get("original_tutor_turns", "[]")
    try:
        turns = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        turns = []

    transfer_template = load_transfer_probe_template()
    transfer_problem = generate_transfer_problem(
        scenario.get("problem", ""),
        scenario.get("correct_solution", ""),
    )
    transfer_text = transfer_template.format(transfer_problem=transfer_problem)

    return turns + [transfer_text]


def build_conversation_messages(
    system_prompt: str,
    scenario: dict,
    prior_learner_turns: list[str],
    next_tutor_turn: str,
) -> list[dict[str, str]]:
    """
    Construct the full message list for the learner LLM at turn k.

    system_prompt       : condition-specific frozen system prompt
    scenario            : scenario row dict
    prior_learner_turns : learner responses so far (list of str)
    next_tutor_turn     : the current tutor message to respond to
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Interleave prior tutor turns and learner turns
    # prior_tutor_turns = all tutor turns up to (but not including) current
    raw = scenario.get("original_tutor_turns", "[]")
    try:
        all_tutor = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        all_tutor = []

    transfer_template = load_transfer_probe_template()
    transfer_problem = generate_transfer_problem(
        scenario.get("problem", ""), scenario.get("correct_solution", "")
    )
    all_tutor_with_transfer = all_tutor + [
        transfer_template.format(transfer_problem=transfer_problem)
    ]

    # Build alternating history
    for i, learner_resp in enumerate(prior_learner_turns):
        if i < len(all_tutor_with_transfer):
            messages.append({"role": "user", "content": all_tutor_with_transfer[i]})
        messages.append({"role": "assistant", "content": learner_resp})

    messages.append({"role": "user", "content": next_tutor_turn})
    return messages
