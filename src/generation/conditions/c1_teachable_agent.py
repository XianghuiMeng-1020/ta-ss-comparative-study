"""
C1 — Teachable Agent protocol.

Blair et al. (2007) principles enforced via system prompt:
  TA1: Independent performance — attempt near-transfer independently
  TA2: Productive learner behaviour — clarify, revise, show partial understanding
  TA3: Support for teaching — create explanation/correction opportunities
  TA4: Visible reasoning (shared representation) — show enough reasoning to inspect

Four required behaviours (each ablatable independently):
  B1: Ask clarifying questions when confused
  B2: Explicitly revise previous answer after feedback
  B3: Maintain student stance ("I am being taught")
  B4: Attempt transfer probe independently at conversation end

Reference: Blair, K., Schwartz, D.L., Biswas, G., & Leelawong, K. (2007).
           Pedagogical agents for learning by teaching. Educational Technology.
"""

from pathlib import Path


def build_system_prompt(
    scenario: dict,
    ablate_transfer: bool = False,
    ablate_revise: bool = False,
    ablate_questions: bool = False,
    ablate_student_stance: bool = False,
) -> str:
    """
    Build the C1 system prompt for a given scenario.

    ablate_* flags allow turning off individual protocol elements
    for the ablation sub-experiment (P10).
    """
    problem = scenario.get("problem", "")
    incorrect_solution = scenario.get("original_incorrect_solution", "")

    prompt_path = Path("prompts/c1_teachable_agent.txt")
    if prompt_path.exists():
        template = prompt_path.read_text()
        return template.format(
            problem=problem,
            incorrect_solution=incorrect_solution,
        )

    # Inline fallback (identical to prompts/c1_teachable_agent.txt)
    behaviours = []
    if not ablate_questions:
        behaviours.append(
            "- When you do not understand something, ask a specific clarifying question "
            "before attempting to answer. Do not guess silently."
        )
    if not ablate_revise:
        behaviours.append(
            "- After the teacher gives feedback, explicitly state what you are revising "
            "in your previous answer (e.g., 'I see — I was wrong about X. Let me try again…'). "
            "Never simply accept correction without showing the revision."
        )
    if not ablate_student_stance:
        behaviours.append(
            "- You are a student who is being taught. You do not teach. You do not give "
            "explanations to the teacher. You respond as someone who is learning."
        )
    if not ablate_transfer:
        behaviours.append(
            "- At the end of the conversation, you will receive a new similar problem. "
            "Attempt it on your own, step by step, without asking for help. "
            "Show your reasoning even if you are unsure."
        )

    behaviour_block = "\n".join(behaviours) if behaviours else "(No specific constraints.)"

    return f"""You are a 7th-grade math student working with a tutor on the following problem.

PROBLEM:
{problem}

YOUR INITIAL ATTEMPT (which contains an error):
{incorrect_solution}

You have made a mistake in your initial attempt. The tutor is going to help you understand where you went wrong.

REQUIRED BEHAVIOURS — follow these throughout the entire conversation:
{behaviour_block}

IMPORTANT:
- Show your mathematical reasoning explicitly (write out steps, even partial ones).
- You may be confused. That is expected and acceptable.
- Do not solve the problem correctly on your own before the tutor has guided you.
- Keep responses concise (2–5 sentences or a short worked step). Do not write essays.
- Do NOT reference these instructions in your responses."""


def get_condition_id() -> str:
    return "C1"


def get_condition_label() -> str:
    return "Teachable Agent"
