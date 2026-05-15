"""
Pre-registered exclusion rules (frozen before main generation in prompt_protocols.md).

Rules:
  E1: Empty learner turn (<5 chars) — drop turn; ≥2 such turns → exclude dialogue
  E2: Non-math content (off-topic)
  E3: Unsafe / refusal response
  E4: Token overflow (>800 tokens in single turn)
  E5: Missing required turn (e.g., transfer probe response absent)
  E6: Condition leakage (learner mentions role label from prompt)

Reference: prompt_protocols.md §Exclusion Rules
"""

import re
import tiktoken

# Role label strings whose appearance in learner output signals leakage
ROLE_LABELS = [
    "teachable agent",
    "student simulation",
    "generic learner",
    "no-role assistant",
    "c1", "c2", "c3", "c4",
    "required behaviours",
    "required behavior",
    "do not reference",
]

REFUSAL_PATTERNS = [
    r"i (can'?t|cannot|will not|won'?t) (help|assist|respond|answer)",
    r"i'?m (not able|unable) to",
    r"this (request|question|content) (violates|is inappropriate)",
    r"i must decline",
]

NON_MATH_MARKERS = [
    "recipe", "sports", "celebrity", "movie", "song",
    "politics", "religion", "war", "violence",
]

try:
    _enc = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    _enc = None


def count_tokens(text: str) -> int:
    if _enc:
        return len(_enc.encode(text))
    return len(text.split())


class ExclusionResult:
    def __init__(self, excluded: bool, code: str, reason: str):
        self.excluded = excluded
        self.code = code
        self.reason = reason

    def __bool__(self):
        return self.excluded

    def __repr__(self):
        return f"ExclusionResult(excluded={self.excluded}, code={self.code!r}, reason={self.reason!r})"


def check_turn(turn_text: str, turn_index: int) -> ExclusionResult:
    """Check a single learner turn for exclusion. Returns first triggered rule."""
    if not turn_text or len(turn_text.strip()) < 5:
        return ExclusionResult(True, "E1", f"Turn {turn_index}: empty or <5 chars")

    tokens = count_tokens(turn_text)
    if tokens > 800:
        return ExclusionResult(True, "E4", f"Turn {turn_index}: {tokens} tokens > 800 limit")

    for pat in REFUSAL_PATTERNS:
        if re.search(pat, turn_text.lower()):
            return ExclusionResult(True, "E3", f"Turn {turn_index}: refusal pattern matched")

    for marker in NON_MATH_MARKERS:
        if marker in turn_text.lower():
            return ExclusionResult(True, "E2", f"Turn {turn_index}: non-math content ({marker})")

    lower = turn_text.lower()
    for label in ROLE_LABELS:
        if label in lower:
            return ExclusionResult(
                True, "E6", f"Turn {turn_index}: condition leakage — '{label}' found"
            )

    return ExclusionResult(False, "", "")


def check_dialogue(
    learner_turns: list[str],
    requires_transfer_attempt: bool = False,
) -> ExclusionResult:
    """
    Check a full dialogue (list of learner turn strings) for exclusion.

    requires_transfer_attempt: if True (C1), the last turn must be non-empty
    and constitute a genuine attempt (not just "I don't know").
    """
    empty_count = 0
    for i, turn in enumerate(learner_turns):
        result = check_turn(turn, i)
        if result.excluded:
            if result.code == "E1":
                empty_count += 1
                if empty_count >= 2:
                    return ExclusionResult(
                        True, "E1", "≥2 empty/short turns in dialogue"
                    )
                continue  # single empty turn: drop turn, don't exclude yet
            return result  # any other rule → immediate exclusion

    if requires_transfer_attempt:
        if not learner_turns:
            return ExclusionResult(True, "E5", "No turns at all (transfer probe missing)")
        last = learner_turns[-1].strip()
        if len(last) < 20 or re.match(r"^(i don'?t know|i'?m not sure|no idea)\.?$", last.lower()):
            return ExclusionResult(True, "E5", "Transfer probe response absent or non-attempt")

    return ExclusionResult(False, "", "")
