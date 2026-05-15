"""
Dialogue generation runner.

Schedules: scenario × condition × seed
- Reads scenario CSV (scenarios.csv or scenarios_pilot.csv)
- Applies frozen prompt templates per condition
- Replays MathDial tutor turns verbatim (D2 default)
- Saves each dialogue as JSON in outputs/{phase}/{condition}/{scenario_id}_{seed}.json
- Aggregates to outputs/generated_dialogues.csv
- Supports checkpoint-resume (skips existing files)

Usage:
  python src/generation/runner.py --phase pilot --scenarios data/scenarios_pilot.csv --seeds 17 42
  python src/generation/runner.py --phase main  --scenarios data/scenarios.csv       --seeds 17 42 91
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from src.generation.client import LLMConfig, call_llm
from src.generation.tutor_replay import build_tutor_turns, build_conversation_messages
from src.generation.exclusion import check_dialogue
from src.generation.conditions import c1_teachable_agent, c2_student_simulation
from src.generation.conditions import c3_generic_learner, c4_no_role_assistant

# ── Constants ──────────────────────────────────────────────────────────────────

CONDITIONS = {
    "C1": c1_teachable_agent,
    "C2": c2_student_simulation,
    "C3": c3_generic_learner,
    "C4": c4_no_role_assistant,
}

PROMPT_VERSION = "v1.0"  # Bump on any prompt change; triggers re-run warning


def get_system_prompt(condition_id: str, scenario: dict) -> str:
    mod = CONDITIONS[condition_id]
    return mod.build_system_prompt(scenario)


def generate_dialogue(
    scenario: dict,
    condition_id: str,
    seed: int,
    config: LLMConfig,
) -> dict:
    """
    Run the full dialogue for one (scenario, condition, seed) triple.
    Returns a result dict ready for JSON serialisation.
    """
    system_prompt = get_system_prompt(condition_id, scenario)
    tutor_turns = build_tutor_turns(scenario)

    learner_turns: list[str] = []
    turn_records: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0

    for turn_idx, tutor_turn in enumerate(tutor_turns):
        messages = build_conversation_messages(
            system_prompt=system_prompt,
            scenario=scenario,
            prior_learner_turns=learner_turns,
            next_tutor_turn=tutor_turn,
        )
        try:
            result = call_llm(messages, config, seed=seed)
        except Exception as e:
            turn_records.append({
                "turn_index": turn_idx,
                "tutor_turn": tutor_turn,
                "learner_turn": "",
                "error": str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": 0,
            })
            learner_turns.append("")
            continue

        learner_turns.append(result.content)
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        turn_records.append({
            "turn_index": turn_idx,
            "tutor_turn": tutor_turn,
            "learner_turn": result.content,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "finish_reason": result.finish_reason,
        })

    requires_transfer = condition_id == "C1"
    excl = check_dialogue(learner_turns, requires_transfer_attempt=requires_transfer)

    return {
        "scenario_id": scenario["scenario_id"],
        "condition": condition_id,
        "condition_label": CONDITIONS[condition_id].get_condition_label(),
        "seed": seed,
        "phase": None,  # set by caller
        "model_id": config.model_id,
        "temperature": config.temperature,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.utcnow().isoformat(),
        "n_tutor_turns": len(tutor_turns),
        "n_learner_turns": len(learner_turns),
        "exclusion_flag": excl.excluded,
        "exclusion_code": excl.code,
        "exclusion_reason": excl.reason,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "system_prompt": system_prompt,
        "turns": turn_records,
    }


def save_dialogue(data: dict, phase: str, output_dir: Path):
    cond = data["condition"]
    sid = data["scenario_id"]
    seed = data["seed"]
    dest = output_dir / cond / f"{sid}_{seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def aggregate_to_csv(phase: str, output_dir: Path):
    """Walk all JSON files and aggregate flat summary rows to CSV."""
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
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_csv = Path("outputs") / "generated_dialogues.csv"
    df.to_csv(out_csv, index=False)
    print(f"Aggregated {len(df)} dialogues → {out_csv}")

    # Go criterion check
    total = len(df)
    valid = df[~df["exclusion_flag"]].shape[0]
    print(f"\n=== Generation summary ===")
    print(f"Total: {total}  Valid: {valid}  ({valid/total*100:.1f}%)")
    print(df.groupby("condition")["exclusion_flag"].mean().round(3).rename("exclusion_rate"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "main"], required=True)
    parser.add_argument("--scenarios", required=True, help="Path to scenarios CSV")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--backend", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3", "C4"])
    args = parser.parse_args()

    config = LLMConfig(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_backend=args.backend,
    )

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

    for scenario_row, cond, seed in tqdm(jobs, desc=f"Generating [{args.phase}]"):
        scenario = scenario_row.to_dict()
        dest = output_dir / cond / f"{scenario['scenario_id']}_{seed}.json"

        if dest.exists():
            skipped += 1
            continue

        data = generate_dialogue(scenario, cond, seed, config)
        data["phase"] = args.phase
        save_dialogue(data, args.phase, output_dir)

    print(f"Done. Skipped (already exist): {skipped}")
    aggregate_to_csv(args.phase, output_dir)


if __name__ == "__main__":
    main()
