"""
Persona implementations for naive-student simulation.

v2 design (replaces src/generation/conditions/):
  - prompt_based.py   — P1: prompt-only naive student (all 12 models)
  - unlearned.py      — P2: machine-unlearning-based naive student (Mistral-7B, Qwen2.5-3B)

Legacy C1–C4 condition modules remain in src/generation/conditions/ for supplement.
"""

from src.generation.personas.prompt_based import build_naive_student_prompt, PERSONA_VERSION

__all__ = ["build_naive_student_prompt", "PERSONA_VERSION"]
