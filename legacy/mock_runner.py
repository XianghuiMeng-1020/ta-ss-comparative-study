"""
Mock generation runner for pipeline validation without an API key.

Uses MathDial's original student turns as synthetic learner responses,
enabling full end-to-end pipeline testing (P3–P11) before real API runs.

Each condition gets the same original student turns but annotated with
condition metadata, so the pipeline structure is validated completely.

Usage:
  python src/generation/mock_runner.py --phase pilot --scenarios data/scenarios_pilot.csv --seeds 17 42
  python src/generation/mock_runner.py --phase main --scenarios data/scenarios.csv --seeds 17 42 91
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.generation.conditions import (
    c1_teachable_agent,
    c2_student_simulation,
    c3_generic_learner,
    c4_no_role_assistant,
)
from src.generation.exclusion import check_dialogue
from src.generation.tutor_replay import build_tutor_turns, generate_transfer_problem

CONDITIONS = {
    "C1": c1_teachable_agent,
    "C2": c2_student_simulation,
    "C3": c3_generic_learner,
    "C4": c4_no_role_assistant,
}

PROMPT_VERSION = "v1.0-mock"


def _make_mock_learner_response(
    scenario: dict,
    condition_id: str,
    turn_idx: int,
    tutor_turn: str,
    seed: int,
    rng: random.Random,
) -> str:
    """
    Generate a plausible mock learner response based on original MathDial student turns.
    Condition-specific flavours are added so trace metrics can meaningfully differentiate.
    """
    raw_student_turns = scenario.get("original_student_turns", "[]")
    try:
        student_turns = json.loads(raw_student_turns) if isinstance(raw_student_turns, str) else raw_student_turns
    except Exception:
        student_turns = []

    # Pull original student turn for this index (cycle if needed)
    if student_turns:
        base = student_turns[turn_idx % len(student_turns)]
    else:
        base = scenario.get("original_incorrect_solution", "I'm not sure.")

    confusion = scenario.get("teacher_described_confusion", "")
    incorrect = scenario.get("original_incorrect_solution", "")

    # Is this the transfer probe turn?
    tutor_turns_orig = json.loads(scenario.get("original_tutor_turns", "[]"))
    is_transfer = turn_idx >= len(tutor_turns_orig)

    if is_transfer:
        if condition_id == "C1":
            transfer_problem = generate_transfer_problem(
                scenario.get("problem", ""), scenario.get("correct_solution", "")
            )
            return (
                f"Let me try this on my own. "
                f"I think I need to start by figuring out the key quantities. "
                f"So if I calculate step by step: {transfer_problem[:50]}... "
                f"I get approximately {rng.randint(5, 50)} as my answer. "
                f"Does that seem right?"
            )
        elif condition_id == "C2":
            return f"Hmm, let me try. {confusion[:40] if confusion else ''} I think the answer might be {rng.randint(5,50)}."
        elif condition_id == "C3":
            return "I think it's something like that. Let me try."
        else:  # C4
            return f"The answer to that problem would be {rng.randint(5,50)}."

    # Regular turns — condition-flavoured
    if condition_id == "C1":
        # Teachable Agent: asks questions, revises explicitly
        if turn_idx == 0:
            return (
                f"So I tried this problem. My approach was: {incorrect[:80] if incorrect else base[:80]}. "
                f"Wait — can you explain why we need to do that step first?"
            )
        elif "correct" in tutor_turn.lower() or "right" in tutor_turn.lower():
            return f"Oh I see! I was wrong about that part. Let me revise: {base[:60]}. So the answer is {rng.randint(5,50)}?"
        else:
            return f"I think I see what you mean. So instead of {rng.randint(1,10)}, I should use {rng.randint(1,10)}? Let me redo: {base[:60]}"

    elif condition_id == "C2":
        # Student Simulation: preserves misconception, gradual
        if turn_idx == 0:
            # Must show target misconception
            return (
                f"{incorrect[:100] if incorrect else base[:100]} "
                f"I used this approach because {confusion[:60] if confusion else 'I thought it was the right way'}."
            )
        elif turn_idx == 1:
            # Still mostly wrong after first feedback
            return f"Okay, but I still think {base[:60]}. Maybe the answer is {rng.randint(5,50)}?"
        else:
            # Gradual improvement
            return f"I think I see — so instead I should... {base[:60]}. Does that mean the answer is {rng.randint(5,50)}?"

    elif condition_id == "C3":
        # Generic: no structure
        return base[:120] if base else "I think so."

    else:  # C4 — assistant
        correct = scenario.get("correct_solution", "")
        return (
            f"To solve this problem, {correct[:80] if correct else 'you need to follow these steps'}. "
            f"The answer is {scenario.get('correct_solution', '?')[:20]}."
        )


def generate_mock_dialogue(
    scenario: dict,
    condition_id: str,
    seed: int,
) -> dict:
    rng = random.Random(seed + hash(scenario.get("scenario_id", "")) % 2**30)
    system_prompt = CONDITIONS[condition_id].build_system_prompt(scenario)
    tutor_turns = build_tutor_turns(scenario)

    learner_turns = []
    turn_records = []

    for turn_idx, tutor_turn in enumerate(tutor_turns):
        response = _make_mock_learner_response(
            scenario, condition_id, turn_idx, tutor_turn, seed, rng
        )
        learner_turns.append(response)
        turn_records.append({
            "turn_index": turn_idx,
            "tutor_turn": tutor_turn,
            "learner_turn": response,
            "input_tokens": len(system_prompt.split()) + len(tutor_turn.split()),
            "output_tokens": len(response.split()),
            "latency_ms": rng.uniform(200, 800),
            "finish_reason": "stop",
            "mock": True,
        })

    requires_transfer = condition_id == "C1"
    excl = check_dialogue(learner_turns, requires_transfer_attempt=requires_transfer)

    return {
        "scenario_id": scenario["scenario_id"],
        "condition": condition_id,
        "condition_label": CONDITIONS[condition_id].get_condition_label(),
        "seed": seed,
        "phase": None,
        "model_id": "mock-model-v1",
        "temperature": 0.7,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.utcnow().isoformat(),
        "n_tutor_turns": len(tutor_turns),
        "n_learner_turns": len(learner_turns),
        "exclusion_flag": excl.excluded,
        "exclusion_code": excl.code,
        "exclusion_reason": excl.reason,
        "total_input_tokens": sum(t["input_tokens"] for t in turn_records),
        "total_output_tokens": sum(t["output_tokens"] for t in turn_records),
        "system_prompt": system_prompt,
        "turns": turn_records,
        "mock": True,
    }


def aggregate_to_csv(phase: str, output_dir: Path):
    rows = []
    for json_path in sorted(output_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        rows.append({
            "scenario_id": data.get("scenario_id"),
            "condition": data.get("condition"),
            "seed": data.get("seed"),
            "phase": data.get("phase"),
            "model_id": data.get("model_id"),
            "prompt_version": data.get("prompt_version"),
            "generated_at": data.get("generated_at"),
            "n_tutor_turns": data.get("n_tutor_turns"),
            "n_learner_turns": data.get("n_learner_turns"),
            "exclusion_flag": data.get("exclusion_flag"),
            "exclusion_code": data.get("exclusion_code"),
            "exclusion_reason": data.get("exclusion_reason"),
            "total_input_tokens": data.get("total_input_tokens"),
            "total_output_tokens": data.get("total_output_tokens"),
            "json_path": str(json_path),
            "mock": data.get("mock", False),
        })
    if not rows:
        return

    df = pd.DataFrame(rows)
    out_csv = Path("outputs") / "generated_dialogues.csv"
    # Append or overwrite
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates(
            subset=["scenario_id", "condition", "seed"]
        )
    df.to_csv(out_csv, index=False)
    print(f"Aggregated {len(df)} total dialogues → {out_csv}")

    phase_df = df[df["phase"] == phase]
    total = len(phase_df)
    valid = (~phase_df["exclusion_flag"].fillna(False)).sum()
    print(f"\n=== {phase.upper()} Generation Summary ===")
    print(f"Total: {total}  Valid: {valid}  ({100*valid/max(total,1):.1f}%)")
    if total > 0:
        print("\nExclusion rate by condition:")
        print(phase_df.groupby("condition")["exclusion_flag"].mean().round(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "main"], required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3", "C4"])
    args = parser.parse_args()

    scenarios_df = pd.read_csv(args.scenarios)
    print(f"Loaded {len(scenarios_df)} scenarios from {args.scenarios}")

    output_dir = Path("outputs") / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (row, cond, seed)
        for _, row in scenarios_df.iterrows()
        for cond in args.conditions
        for seed in args.seeds
    ]
    print(f"Total jobs: {len(jobs)}")
    skipped = 0

    for scenario_row, cond, seed in tqdm(jobs, desc=f"Mock generating [{args.phase}]"):
        scenario = scenario_row.to_dict()
        dest = output_dir / cond / f"{scenario['scenario_id']}_{seed}.json"
        if dest.exists():
            skipped += 1
            continue

        data = generate_mock_dialogue(scenario, cond, seed)
        data["phase"] = args.phase
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    print(f"Done. Skipped (already exist): {skipped}")
    aggregate_to_csv(args.phase, output_dir)


if __name__ == "__main__":
    main()
