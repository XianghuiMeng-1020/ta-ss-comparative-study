"""
Protocol-element ablation analysis.

Runs 4 ablation variants on a 25-scenario subset (3 seeds each):
  C1-ablate-transfer   : remove near-transfer requirement → expect TA1 to drop
  C1-ablate-revise     : remove must-revise → expect TA2 to drop
  C2-ablate-misconception: remove explicit misconception injection → expect SS2 to drop
  C2-ablate-gradual    : remove gradual-learning constraint → expect SS4 to drop

Generation is done inline; analysis compares ablation vs full-protocol condition.

Usage:
  python src/analysis/ablations.py --n-scenarios 25 --seeds 17 42 91
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from src.generation.client import LLMConfig, call_llm
from src.generation.tutor_replay import build_tutor_turns, build_conversation_messages
from src.generation.exclusion import check_dialogue
from src.generation.conditions.c1_teachable_agent import build_system_prompt as c1_prompt
from src.generation.conditions.c2_student_simulation import build_system_prompt as c2_prompt

ABLATION_CONFIGS = {
    "C1_ablate_transfer": {
        "base_condition": "C1",
        "prompt_fn": lambda s: c1_prompt(s, ablate_transfer=True),
        "expected_drop_dimension": "TA1",
    },
    "C1_ablate_revise": {
        "base_condition": "C1",
        "prompt_fn": lambda s: c1_prompt(s, ablate_revise=True),
        "expected_drop_dimension": "TA2",
    },
    "C2_ablate_misconception": {
        "base_condition": "C2",
        "prompt_fn": lambda s: c2_prompt(s, ablate_misconception=True),
        "expected_drop_dimension": "SS2",
    },
    "C2_ablate_gradual": {
        "base_condition": "C2",
        "prompt_fn": lambda s: c2_prompt(s, ablate_gradual=True),
        "expected_drop_dimension": "SS4",
    },
}


def select_ablation_subset(scenarios_path: Path, n: int, seed: int = 2026) -> pd.DataFrame:
    df = pd.read_csv(scenarios_path)
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), min(n, len(df)))
    return df.iloc[idx].reset_index(drop=True)


def generate_ablation_dialogues(
    scenario_df: pd.DataFrame,
    ablation_name: str,
    config: LLMConfig,
    seeds: list[int],
    output_dir: Path,
):
    cfg = ABLATION_CONFIGS[ablation_name]
    prompt_fn = cfg["prompt_fn"]

    dialogues = []
    for _, row in scenario_df.iterrows():
        scenario = row.to_dict()
        for seed in seeds:
            out_file = output_dir / ablation_name / f"{scenario['scenario_id']}_{seed}.json"
            if out_file.exists():
                dialogues.append(json.loads(out_file.read_text()))
                continue

            system_prompt = prompt_fn(scenario)
            tutor_turns = build_tutor_turns(scenario)
            learner_turns = []
            turn_records = []

            for turn_idx, tutor_turn in enumerate(tutor_turns):
                messages = build_conversation_messages(
                    system_prompt=system_prompt,
                    scenario=scenario,
                    prior_learner_turns=learner_turns,
                    next_tutor_turn=tutor_turn,
                )
                try:
                    result = call_llm(messages, config, seed=seed)
                    learner_turns.append(result.content)
                    turn_records.append({
                        "turn_index": turn_idx,
                        "tutor_turn": tutor_turn,
                        "learner_turn": result.content,
                    })
                except Exception as e:
                    learner_turns.append("")
                    turn_records.append({
                        "turn_index": turn_idx, "tutor_turn": tutor_turn,
                        "learner_turn": "", "error": str(e),
                    })

            excl = check_dialogue(learner_turns, requires_transfer_attempt=False)
            data = {
                "scenario_id": scenario["scenario_id"],
                "condition": ablation_name,
                "base_condition": cfg["base_condition"],
                "seed": seed,
                "exclusion_flag": excl.excluded,
                "exclusion_code": excl.code,
                "turns": turn_records,
            }

            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            dialogues.append(data)

    return dialogues


def summarise_ablations(output_dir: Path):
    """Compute trace metrics on ablation outputs and compare to full protocol."""
    from src.trace_metrics.compute_all import compute_metrics_for_dialogue

    summary_rows = []
    for ablation_name in ABLATION_CONFIGS:
        abl_dir = output_dir / ablation_name
        if not abl_dir.exists():
            continue
        for jf in sorted(abl_dir.rglob("*.json")):
            try:
                data = json.loads(jf.read_text())
            except Exception:
                continue
            if data.get("exclusion_flag"):
                continue
            metrics = compute_metrics_for_dialogue(data)
            metrics["ablation"] = ablation_name
            metrics["expected_drop"] = ABLATION_CONFIGS[ablation_name]["expected_drop_dimension"]
            summary_rows.append(metrics)

    if not summary_rows:
        print("No ablation dialogues found.")
        return

    df = pd.DataFrame(summary_rows)
    out_path = output_dir / "ablation_trace_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved ablation trace metrics → {out_path}")

    print("\n=== ABLATION SUMMARY ===")
    print("Expected: removing protocol element drops corresponding metric.\n")
    for ablation_name, cfg in ABLATION_CONFIGS.items():
        subset = df[df["ablation"] == ablation_name]
        if subset.empty:
            continue
        expected_col = cfg["expected_drop_dimension"]
        # Map dimension label to trace metric
        dim_to_metric = {
            "TA1": "rule_near_transfer_attempt",
            "TA2": "rule_feedback_uptake_rate",
            "SS2": "rule_target_error_preservation",
            "SS4": "rule_correction_timing_index",
        }
        metric_col = dim_to_metric.get(expected_col, "")
        if metric_col and metric_col in subset.columns:
            mean_val = subset[metric_col].mean()
            print(f"  {ablation_name} → {metric_col}: {mean_val:.3f}")
        else:
            print(f"  {ablation_name}: metric {metric_col!r} not available")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenarios", type=int, default=25)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 91])
    parser.add_argument("--scenarios", default="data/scenarios.csv")
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--backend", default="openai")
    args = parser.parse_args()

    config = LLMConfig(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=600,
        api_backend=args.backend,
    )

    scenario_df = select_ablation_subset(Path(args.scenarios), args.n_scenarios)
    print(f"Ablation subset: {len(scenario_df)} scenarios")

    output_dir = Path("outputs") / "ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for ablation_name in ABLATION_CONFIGS:
        print(f"\nRunning {ablation_name}…")
        generate_ablation_dialogues(scenario_df, ablation_name, config, args.seeds, output_dir)

    summarise_ablations(output_dir)


if __name__ == "__main__":
    main()
