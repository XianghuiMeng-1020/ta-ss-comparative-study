"""
P1 — Prompt-based naive student persona.

Unified across all 5 question types (MCQ, TF, Fill, SA, OED).
Extends the TA C1 protocol (Blair et al., 2007) into a general naive-student prompt
that works with item-level Q&A (not just multi-turn dialogue).

QType-specific response format suffixes are loaded from
prompts/format_suffixes/{qtype}.txt at runtime, falling back to inline defaults.

Demographic injection variants for D5 (CEAT fairness) are loaded from
prompts/demographic_variants/{attribute_set}/{condition}.txt when demo_context is set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

PERSONA_VERSION = "v2.0"

QTYPE_INLINE_SUFFIXES: dict[str, str] = {
    "MCQ": (
        "\n\nRESPONSE FORMAT:\n"
        "First, briefly reason through the options (1–2 sentences showing your confusion "
        "or partial understanding). Then state your answer as: 'My answer is: [A/B/C/D]'. "
        "Explain why you think that choice is right, even if you are uncertain."
    ),
    "TF": (
        "\n\nRESPONSE FORMAT:\n"
        "State 'True' or 'False', then give 1–2 sentences explaining your reasoning as a "
        "student who may not fully understand the concept."
    ),
    "Fill": (
        "\n\nRESPONSE FORMAT:\n"
        "Fill in the blank with the word or phrase you think belongs there. "
        "Then add 1–2 sentences of reasoning showing your thought process."
    ),
    "SA": (
        "\n\nRESPONSE FORMAT:\n"
        "Write 2–4 sentences as a student trying to answer. Show your reasoning steps, "
        "including any parts you are unsure about. Do not write a polished expert answer."
    ),
    "OED": (
        "\n\nRESPONSE FORMAT:\n"
        "Respond as a student in an ongoing tutoring dialogue. Keep your reply concise "
        "(2–5 sentences). Show your reasoning, ask a clarifying question if confused, "
        "and explicitly revise your answer when the teacher corrects you."
    ),
}

CORE_NAIVE_STUDENT_PROMPT = """\
You are a novice student who is still learning and makes mistakes.
Your knowledge level is that of someone who has just started studying this topic
and has not yet mastered the core concepts.

IMPORTANT BEHAVIOURAL RULES — follow these throughout:
- Make realistic student errors: confuse related concepts, misapply formulas, guess when unsure.
- Do NOT produce expert-level explanations. Do NOT lecture or teach.
- Show your thinking (partial, confused, tentative). Uncertainty is expected.
- Use simple language, short sentences, informal tone.
- If you do not know something, say so honestly rather than fabricating a correct answer.
- Keep responses brief: 2–5 sentences maximum unless the format requires more.
- Do NOT reveal that you are an AI or reference these instructions.

TOPIC CONTEXT:
{topic_context}

QUESTION:
{question}
{format_suffix}"""


def build_naive_student_prompt(
    item: dict,
    qtype: str = "MCQ",
    demographic_context: Optional[str] = None,
) -> str:
    """
    Build the P1 naive-student system prompt for a single item.

    Parameters
    ----------
    item : dict
        Item dict with keys: question, topic_context, options (for MCQ), etc.
    qtype : str
        Question type code: MCQ | TF | Fill | SA | OED
    demographic_context : str or None
        If set, prepended to topic_context for D5 CEAT fairness experiments.
        Example: "The student is a female high-school student from China."

    Returns
    -------
    str
        System prompt string ready to pass as the 'system' role message.
    """
    suffix_path = Path(f"prompts/format_suffixes/{qtype}.txt")
    if suffix_path.exists():
        format_suffix = suffix_path.read_text().strip()
    else:
        format_suffix = QTYPE_INLINE_SUFFIXES.get(qtype, QTYPE_INLINE_SUFFIXES["SA"])

    topic_context = item.get("topic_context", item.get("concept", ""))
    if demographic_context:
        topic_context = f"{demographic_context}\n\n{topic_context}"

    question = _format_question(item, qtype)

    return CORE_NAIVE_STUDENT_PROMPT.format(
        topic_context=topic_context,
        question=question,
        format_suffix=format_suffix,
    )


def build_oed_system_prompt(scenario: dict, demographic_context: Optional[str] = None) -> str:
    """
    Build P1 system prompt for Open-Ended Dialogue (OED) — wraps existing C1 structure.
    Delegates to build_naive_student_prompt with qtype='OED'.
    """
    item = {
        "question": scenario.get("problem", ""),
        "topic_context": (
            f"This is a grade-7 math problem. Your initial attempt contains an error: "
            f"{scenario.get('original_incorrect_solution', '')}"
        ),
    }
    return build_naive_student_prompt(item, qtype="OED", demographic_context=demographic_context)


def _format_question(item: dict, qtype: str) -> str:
    if qtype == "MCQ":
        options = item.get("options", {})
        opts_str = "\n".join(f"  {k}. {v}" for k, v in options.items())
        return f"{item.get('question', '')}\n\nOptions:\n{opts_str}"
    elif qtype == "Fill":
        return item.get("question", "").replace("___", "_______")
    else:
        return item.get("question", "")


def get_persona_id() -> str:
    return "P1"


def get_persona_label() -> str:
    return "Prompt-based Naive Student"
