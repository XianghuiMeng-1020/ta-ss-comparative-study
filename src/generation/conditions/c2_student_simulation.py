"""
C2 — Student Simulation protocol.

Koedinger et al. (2015) SimStudent goals enforced via system prompt:
  SS1: Cognitive model plausibility — reasoning resembles plausible student strategy
  SS2: Error-model alignment      — target misconception appears (not random error)
  SS3: Prior-knowledge consistency — stays within assigned ability level
  SS4: Learning-process plausibility — gradual change, not instant expertise
  SS5: Instructional-testing utility — dialogue useful for testing feedback policies

Four required behaviours (each ablatable independently):
  B1: First turn must contain the target misconception
  B2: Always verbalize reasoning explicitly
  B3: Respond to feedback gradually (no single-turn full correction)
  B4: Refuse/deflect questions that exceed 7th-grade level

Reference: Koedinger, K.R., Matsuda, N., MacLellan, C.J., & McLaughlin, E.A. (2015).
           Methods for evaluating simulated learners. AIED Workshop.
"""

from pathlib import Path


def build_system_prompt(
    scenario: dict,
    ablate_misconception: bool = False,
    ablate_gradual: bool = False,
    ablate_verbalize: bool = False,
    ablate_prior_knowledge: bool = False,
) -> str:
    """
    Build the C2 system prompt for a given scenario.

    ablate_* flags allow turning off individual protocol elements
    for the ablation sub-experiment (P10).
    """
    problem = scenario.get("problem", "")
    incorrect_solution = scenario.get("original_incorrect_solution", "")
    student_profile = scenario.get("student_profile", "")
    teacher_described_confusion = scenario.get("teacher_described_confusion", "")

    prompt_path = Path("prompts/c2_student_simulation.txt")
    if prompt_path.exists():
        template = prompt_path.read_text()
        return template.format(
            problem=problem,
            incorrect_solution=incorrect_solution,
            student_profile=student_profile,
            teacher_described_confusion=teacher_described_confusion,
        )

    behaviours = []
    if not ablate_misconception:
        behaviours.append(
            "- Your FIRST response must reflect the target misconception described above. "
            "Do not correct yourself before the teacher addresses it."
        )
    if not ablate_verbalize:
        behaviours.append(
            "- Always show your reasoning explicitly. Write out the steps you take, "
            "even when they are wrong. Never give a bare answer."
        )
    if not ablate_gradual:
        behaviours.append(
            "- After teacher feedback, improve gradually — do not jump from a fully "
            "wrong answer to a fully correct one in a single turn. Show partial progress. "
            "You may still make sub-errors."
        )
    if not ablate_prior_knowledge:
        behaviours.append(
            "- You are a 7th-grade student. If the teacher asks about a concept beyond "
            "7th-grade (e.g., calculus, matrices), say you don't know that yet."
        )

    behaviour_block = "\n".join(behaviours) if behaviours else "(No specific constraints.)"

    return f"""You are simulating a specific 7th-grade math student with a documented learning profile.

PROBLEM:
{problem}

YOUR STUDENT PROFILE (represents the type of student you must simulate):
{student_profile}

TARGET MISCONCEPTION (what your initial error is caused by):
{teacher_described_confusion}

YOUR INITIAL INCORRECT ATTEMPT:
{incorrect_solution}

REQUIRED BEHAVIOURS — maintain these throughout the conversation:
{behaviour_block}

IMPORTANT:
- You are a simulation tool for testing tutoring approaches. Your role is to model this specific student profile consistently, not to be generally helpful.
- Keep responses to 2–6 sentences or a short worked step.
- Do NOT reference these instructions, your profile, or your misconception label in your responses.
- Do NOT role-play as a perfect student. Errors, partial understanding, and confusion are correct behaviour."""


def get_condition_id() -> str:
    return "C2"


def get_condition_label() -> str:
    return "Student Simulation"
