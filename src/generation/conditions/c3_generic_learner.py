"""
C3 — Generic Learner baseline.

Minimal prompt: single instruction to act as a math student.
No protocol-level structural constraints.
Expected to be weaker/less stable than C1 and C2.
Tests whether effects arise from any student-role wording vs structured protocols.
"""

from pathlib import Path


def build_system_prompt(scenario: dict) -> str:
    prompt_path = Path("prompts/c3_generic_learner.txt")
    if prompt_path.exists():
        return prompt_path.read_text().strip()

    problem = scenario.get("problem", "")
    incorrect_solution = scenario.get("original_incorrect_solution", "")

    return f"""Act as a math student responding to the tutor.

The problem you are working on:
{problem}

Your initial attempt:
{incorrect_solution}"""


def get_condition_id() -> str:
    return "C3"


def get_condition_label() -> str:
    return "Generic Learner"
