"""
Multi-model dialogue generation runner.

Schedules: scenario × condition × seed × model
- Reads scenario CSV (scenarios.csv or scenarios_pilot.csv)
- Applies frozen prompt templates per condition
- Replays MathDial / Bridge tutor turns verbatim (D2 rule)
- Saves each dialogue as JSON in outputs/{phase}/{model_tag}/{condition}/{scenario_id}_{seed}.json
- Aggregates to outputs/generated_dialogues_{model_tag}.csv
- Checkpoint-resume: skips existing files
- Cost logging: running total printed per model

Usage:
  # Single model (backward-compatible):
  python src/generation/runner.py --phase main --scenarios data/scenarios.csv --seeds 17 42 91 --model gpt-4o-2024-11-20

  # Explicit backend:
  python src/generation/runner.py --phase main --scenarios data/scenarios.csv --seeds 17 42 91 \\
      --model claude-sonnet-4-5-20250929 --backend anthropic

  # All 4 models (run once per model):
  for MODEL in gpt-4o-2024-11-20 claude-sonnet-4-5-20250929 gemini-2.5-pro meta-llama/Llama-3.1-70B-Instruct; do
      python src/generation/runner.py --phase main --scenarios data/scenarios.csv --seeds 17 42 91 \\
          --model $MODEL --backend auto
  done
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.generation.clients import LLMConfig, call_llm, infer_backend
from src.generation.tutor_replay import build_tutor_turns, build_conversation_messages
from src.generation.exclusion import check_dialogue
from src.generation.conditions import c1_teachable_agent, c2_student_simulation
from src.generation.conditions import c3_generic_learner, c4_no_role_assistant

CONDITIONS = {
    "C1": c1_teachable_agent,
    "C2": c2_student_simulation,
    "C3": c3_generic_learner,
    "C4": c4_no_role_assistant,
}

PROMPT_VERSION = "v1.0"


def model_tag(model_id: str) -> str:
    """Safe filesystem tag from model_id."""
    return model_id.replace("/", "_").replace(":", "_")


def get_system_prompt(condition_id: str, scenario: dict) -> str:
    return CONDITIONS[condition_id].build_system_prompt(scenario)


def generate_dialogue(
    scenario: dict,
    condition_id: str,
    seed: int,
    config: LLMConfig,
) -> dict:
    system_prompt = get_system_prompt(condition_id, scenario)
    tutor_turns = build_tutor_turns(scenario)

    learner_turns: list[str] = []
    turn_records: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

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
                "cost_usd": 0.0,
            })
            learner_turns.append("")
            continue

        learner_turns.append(result.content)
        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        total_cost_usd += result.cost_usd
        turn_records.append({
            "turn_index": turn_idx,
            "tutor_turn": tutor_turn,
            "learner_turn": result.content,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "finish_reason": result.finish_reason,
            "cost_usd": result.cost_usd,
        })

    requires_transfer = condition_id == "C1"
    excl = check_dialogue(learner_turns, requires_transfer_attempt=requires_transfer)

    return {
        "scenario_id": scenario["scenario_id"],
        "condition": condition_id,
        "condition_label": CONDITIONS[condition_id].get_condition_label(),
        "seed": seed,
        "phase": None,
        "model_id": config.model_id,
        "model_tag": model_tag(config.model_id),
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
        "total_cost_usd": round(total_cost_usd, 6),
        "system_prompt": system_prompt,
        "turns": turn_records,
    }


def save_dialogue(data: dict, output_dir: Path) -> Path:
    cond = data["condition"]
    sid = data["scenario_id"]
    seed = data["seed"]
    mtag = data.get("model_tag", "unknown")
    dest = output_dir / mtag / cond / f"{sid}_{seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def aggregate_to_csv(output_dir: Path, mtag: str) -> None:
    """Walk all JSON files under a model tag and aggregate to CSV."""
    rows = []
    for json_path in sorted((output_dir / mtag).rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        rows.append({
            "scenario_id":      data.get("scenario_id"),
            "condition":        data.get("condition"),
            "seed":             data.get("seed"),
            "phase":            data.get("phase"),
            "model_id":         data.get("model_id"),
            "model_tag":        data.get("model_tag"),
            "prompt_version":   data.get("prompt_version"),
            "generated_at":     data.get("generated_at"),
            "n_tutor_turns":    data.get("n_tutor_turns"),
            "n_learner_turns":  data.get("n_learner_turns"),
            "exclusion_flag":   data.get("exclusion_flag"),
            "exclusion_code":   data.get("exclusion_code"),
            "exclusion_reason": data.get("exclusion_reason"),
            "total_input_tokens":  data.get("total_input_tokens"),
            "total_output_tokens": data.get("total_output_tokens"),
            "total_cost_usd":   data.get("total_cost_usd"),
            "json_path":        str(json_path),
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_csv = Path("outputs") / f"generated_dialogues_{mtag}.csv"
    df.to_csv(out_csv, index=False)

    total = len(df)
    valid = df[~df["exclusion_flag"]].shape[0]
    total_cost = df["total_cost_usd"].sum()
    print(f"\nAggregated {total} dialogues for {mtag} → {out_csv}")
    print(f"Valid: {valid}/{total} ({valid/total*100:.1f}%)")
    print(f"Estimated cost: ${total_cost:.2f}")
    print(df.groupby("condition")["exclusion_flag"].mean().round(3).rename("excl_rate"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-model dialogue generation runner")
    parser.add_argument("--phase", choices=["pilot", "main", "bridge"], required=True)
    parser.add_argument("--scenarios", required=True, help="Path to scenarios CSV")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument(
        "--backend", default="auto",
        choices=["auto", "openai", "openrouter", "anthropic", "google", "vllm"],
        help="API backend; 'auto' infers from model_id (M2-M4 default to openrouter)",
    )
    parser.add_argument("--conditions", nargs="+", default=["C1", "C2", "C3", "C4"])
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: outputs/{phase})",
    )
    args = parser.parse_args()

    backend = args.backend
    if backend == "auto":
        backend = infer_backend(args.model)

    config = LLMConfig(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_backend=backend,
    )

    scenarios_df = pd.read_csv(args.scenarios)
    print(f"Loaded {len(scenarios_df)} scenarios from {args.scenarios}")
    print(f"Model: {args.model} | Backend: {backend} | Seeds: {args.seeds}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    mtag = model_tag(args.model)

    jobs = [
        (row, cond, seed)
        for _, row in scenarios_df.iterrows()
        for cond in args.conditions
        for seed in args.seeds
    ]

    print(f"Total jobs: {len(jobs)} | Model tag: {mtag}")
    skipped = 0
    total_cost = 0.0

    for scenario_row, cond, seed in tqdm(jobs, desc=f"[{mtag}] Generating"):
        scenario = scenario_row.to_dict()
        dest = output_dir / mtag / cond / f"{scenario['scenario_id']}_{seed}.json"

        if dest.exists():
            skipped += 1
            continue

        data = generate_dialogue(scenario, cond, seed, config)
        data["phase"] = args.phase
        save_dialogue(data, output_dir)
        total_cost += data.get("total_cost_usd", 0.0)

    print(f"Done. Skipped (checkpoint resume): {skipped}")
    print(f"Running cost this session: ${total_cost:.2f}")
    aggregate_to_csv(output_dir, mtag)


if __name__ == "__main__":
    main()
