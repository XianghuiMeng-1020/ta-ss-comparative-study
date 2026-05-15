"""
Trace metric: over-technical language rate (LLM validity failure flag).

Definition: learner uses mathematical vocabulary that exceeds 7th-grade level,
signalling that the model is drawing on its full capability rather than
simulating a specific ability level (competence paradox, Yuan et al., 2026).
"""

import re


ADVANCED_TERMS = [
    # Calculus / analysis
    "derivative", "integral", "differentiate", "antiderivative",
    "limit", "continuity", "convergence", "divergence",
    # Linear algebra
    "matrix", "matrices", "vector", "eigenvalue", "determinant",
    "linear transformation", "dot product", "cross product",
    # Advanced algebra
    "polynomial", "quadratic formula", "completing the square",
    "discriminant", "complex number", "imaginary",
    # Statistics beyond 7th grade
    "standard deviation", "variance", "regression", "correlation",
    "hypothesis test", "p-value", "confidence interval",
    # Proof / formal logic
    "theorem", "proof", "corollary", "lemma", "qed",
    "if and only if", "necessary and sufficient",
]


def rule_based(learner_turn: str) -> bool:
    lower = learner_turn.lower()
    return any(term in lower for term in ADVANCED_TERMS)


def compute_dialogue_rate(learner_turns: list[str]) -> float:
    if not learner_turns:
        return 0.0
    flagged = sum(rule_based(t) for t in learner_turns)
    return flagged / len(learner_turns)


def llm_judge_prompt(learner_turn: str) -> str:
    return (
        "This is supposed to be a 7th-grade math student's response. "
        "Does the student use mathematical language or concepts that are clearly "
        "beyond 7th-grade level (e.g., calculus, linear algebra, advanced statistics)?\n\n"
        f"Student turn: {learner_turn}\n\n"
        "Answer with exactly 'yes' or 'no'."
    )


def compute(learner_turns: list[str]) -> dict:
    rate = compute_dialogue_rate(learner_turns)
    return {
        "rule_over_technical_rate": rate,
        "rule_any_over_technical": rate > 0,
    }
