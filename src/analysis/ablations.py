"""
Protocol-element ablation analysis — 8 ablation variants (Decision Log D12).

Removes one structural element at a time and measures the expected drop in the
corresponding automatic trace metric. All 8 ablation hypotheses are pre-registered (H5).

Ablation variants:
  C1_ablate_transfer      : remove near-transfer requirement       → expect TA1 drop
  C1_ablate_revise        : remove explicit revision requirement   → expect TA2 drop
  C1_ablate_questions     : remove clarifying-question requirement → expect TA2/TA3 drop
  C1_ablate_student_stance: remove student-stance maintenance      → expect TA3 drop
  C2_ablate_misconception : remove target misconception injection  → expect SS2 drop
  C2_ablate_gradual       : remove gradual-learning constraint     → expect SS4 drop
  C2_ablate_verbalize     : remove verbalize-reasoning requirement → expect SS1/TA4 drop
  C2_ablate_prior_knowledge: remove prior-knowledge boundary      → expect SS3 drop

Execution: 25 scenarios × M1 × 2 seeds = 50 dialogues × 8 ablations = 400 total.

Usage:
  python src/analysis/ablations.py --n-scenarios 25 --seeds 17 42
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
    # C1 ablations (Decision Log D12)
    "C1_ablate_transfer": {
        "base_condition": "C1",
        "prompt_fn": lambda s: c1_prompt(s, ablate_transfer=True),
        "expected_drop_dimension": "TA1",
        "expected_drop_metric": "rule_near_transfer_attempt",
    },
    "C1_ablate_revise": {
        "base_condition": "C1",
        "prompt_fn": lambda s: c1_prompt(s, ablate_revise=True),
        "expected_drop_dimension": "TA2",
        "expected_drop_metric": "rule_feedback_uptake_rate",
    },
    "C1_ablate_questions": {
        "base_condition": "C1",
        "prompt_fn": lambda s: c1_prompt(s, ablate_questions=True),
        "expected_drop_dimension": "TA2",
        "expected_drop_metric": "rule_question_asking_rate",
    },
    "C1_ablate_student_stance": {
        "base_condition": "C1",
        "prompt_fn": lambda s: c1_prompt(s, ablate_student_stance=True),
        "expected_drop_dimension": "TA3",
        "expected_drop_metric": "rule_role_drift_rate",  # stance removal → role drift increases
    },
    # C2 ablations (Decision Log D12)
    "C2_ablate_misconception": {
        "base_condition": "C2",
        "prompt_fn": lambda s: c2_prompt(s, ablate_misconception=True),
        "expected_drop_dimension": "SS2",
        "expected_drop_metric": "rule_target_error_preservation",
    },
    "C2_ablate_gradual": {
        "base_condition": "C2",
        "prompt_fn": lambda s: c2_prompt(s, ablate_gradual=True),
        "expected_drop_dimension": "SS4",
        "expected_drop_metric": "rule_correction_timing_index",
    },
    "C2_ablate_verbalize": {
        "base_condition": "C2",
        "prompt_fn": lambda s: c2_prompt(s, ablate_verbalize=True),
        "expected_drop_dimension": "SS1",
        "expected_drop_metric": "rule_reasoning_trace_rate",
    },
    "C2_ablate_prior_knowledge": {
        "base_condition": "C2",
        "prompt_fn": lambda s: c2_prompt(s, ablate_prior_knowledge=True),
        "expected_drop_dimension": "SS3",
        "expected_drop_metric": "rule_over_technical_language_rate",  # prior-knowledge removal → more technical
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
        metric_col = ABLATION_CONFIGS[ablation_name].get("expected_drop_metric", "")
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
