"""
C4 — No-role Assistant control.

Standard helpful-assistant system prompt; no learner role is assigned.
Expected to show low learner-role evidence, high expert behaviour,
and premature solution-revealing.
Controls for ordinary LLM answer behaviour vs role-prompted behaviour.
"""

from pathlib import Path


def build_system_prompt(scenario: dict) -> str:
    prompt_path = Path("prompts/c4_no_role_assistant.txt")
    if prompt_path.exists():
        return prompt_path.read_text().strip()

    return "You are a helpful assistant. Answer the user's questions clearly and accurately."


def get_condition_id() -> str:
    return "C4"


def get_condition_label() -> str:
    return "No-role Assistant"
