"""
Compute all 10 automatic trace metrics for every generated dialogue.

Output: outputs/automatic_trace_metrics.csv
  One row per dialogue (scenario_id × condition × seed).
  Each metric has a 'rule_*' column; LLM-judge columns are added when
  ENABLE_LLM_JUDGE=1 env var is set (requires API key).

Usage:
  python src/trace_metrics/compute_all.py --phase main
  ENABLE_LLM_JUDGE=1 python src/trace_metrics/compute_all.py --phase main
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.trace_metrics import (
    question_asking,
    reasoning_trace,
    target_error_preservation,
    feedback_uptake,
    near_transfer_attempt,
    premature_correctness,
    role_drift,
    over_technical_language,
    unsupported_reasoning,
    correction_timing,
)
from src.generation.client import LLMConfig, call_llm

ENABLE_LLM_JUDGE = os.environ.get("ENABLE_LLM_JUDGE", "0") == "1"
LLM_CONFIG = LLMConfig(model_id="gpt-4o-2024-11-20", temperature=0.0, max_tokens=10)


def call_judge(prompt: str) -> str:
    """Call the LLM judge and return 'yes' or 'no'."""
    try:
        result = call_llm(
            [{"role": "user", "content": prompt}],
            LLM_CONFIG,
            seed=0,
        )
        return result.content.strip().lower()
    except Exception as e:
        return f"error:{e}"


def compute_metrics_for_dialogue(data: dict) -> dict:
    """
    Compute all 10 metrics for a single dialogue JSON dict.
    Returns a flat dict of metric columns.
    """
    turns = data.get("turns", [])
    tutor_turns = [t["tutor_turn"] for t in turns]
    learner_turns = [t.get("learner_turn", "") for t in turns]
    scenario_id = data.get("scenario_id", "")
    condition = data.get("condition", "")
    seed = data.get("seed", 0)
    model_id = data.get("model_id", "")

    # Retrieve scenario context from embedded system_prompt fallback
    teacher_described_confusion = ""
    correct_solution = ""
    # Try to extract from the system_prompt field
    sys_prompt = data.get("system_prompt", "")
    for line in sys_prompt.splitlines():
        if line.strip().upper().startswith("TARGET MISCONCEPTION"):
            # next non-empty line
            pass
        if "confusion" in line.lower() and ":" in line:
            teacher_described_confusion = line.split(":", 1)[-1].strip()
        if "correct" in line.lower() and "answer" in line.lower() and ":" in line:
            correct_solution = line.split(":", 1)[-1].strip()

    # 1. Question-asking rate
    q_rates = [question_asking.rule_based(t) for t in learner_turns]
    qa_rate = sum(q_rates) / len(learner_turns) if learner_turns else 0.0

    # 2. Reasoning-trace rate
    rt_rates = [reasoning_trace.rule_based(t) for t in learner_turns]
    rt_rate = sum(rt_rates) / len(learner_turns) if learner_turns else 0.0

    # 3. Target-error preservation
    tep = target_error_preservation.compute(teacher_described_confusion, learner_turns)

    # 4. Feedback uptake
    fu = feedback_uptake.compute(tutor_turns, learner_turns)

    # 5. Near-transfer attempt (last learner turn = transfer probe response)
    transfer_turn = learner_turns[-1] if learner_turns else ""
    nta = near_transfer_attempt.compute(transfer_turn)

    # 6. Premature correctness
    pc = premature_correctness.compute(tutor_turns, learner_turns, correct_solution)

    # 7. Role drift
    rd = role_drift.compute(learner_turns)

    # 8. Over-technical language
    otl = over_technical_language.compute(learner_turns)

    # 9. Unsupported reasoning
    ur = unsupported_reasoning.compute(learner_turns)

    # 10. Correction timing
    ct = correction_timing.compute(teacher_described_confusion, learner_turns)

    row = {
        "scenario_id": scenario_id,
        "condition": condition,
        "seed": seed,
        "model_id": model_id,
        # Metric 1
        "rule_question_asking_rate": qa_rate,
        # Metric 2
        "rule_reasoning_trace_rate": rt_rate,
        # Metric 3
        "rule_target_error_preservation": tep["rule"],
        # Metric 4
        "rule_feedback_uptake_rate": fu["rule"],
        # Metric 5
        "rule_near_transfer_attempt": nta["rule"],
        # Metric 6
        "rule_premature_correctness": pc["rule"],
        # Metric 7
        "rule_role_drift_rate": rd["rule_drift_rate"],
        "rule_any_role_drift": rd["rule_any_drift"],
        # Metric 8
        "rule_over_technical_rate": otl["rule_over_technical_rate"],
        "rule_any_over_technical": otl["rule_any_over_technical"],
        # Metric 9
        "rule_bare_answer_rate": ur["rule_bare_answer_rate"],
        "rule_contradiction_rate": ur["rule_contradiction_rate"],
        "rule_any_unsupported": ur["rule_any_unsupported"],
        # Metric 10
        "rule_correction_timing_index": ct["correction_timing_index"],
        "rule_correction_turn_idx": ct["correction_turn_idx"],
        "rule_misconception_persists_to_end": ct["misconception_persists_to_end"],
        # Counts
        "n_learner_turns": len(learner_turns),
        "n_tutor_turns": len(tutor_turns),
    }

    # LLM-judge channels (optional)
    if ENABLE_LLM_JUDGE and learner_turns:
        # Sample first non-empty turn for per-turn judges
        sample_turn = next((t for t in learner_turns if t.strip()), "")
        row["llm_question_asking"] = call_judge(question_asking.llm_judge_prompt(sample_turn))
        row["llm_reasoning_trace"] = call_judge(reasoning_trace.llm_judge_prompt(sample_turn))
        row["llm_role_drift"] = call_judge(role_drift.llm_judge_prompt(sample_turn))
        row["llm_over_technical"] = call_judge(
            over_technical_language.llm_judge_prompt(sample_turn)
        )
        row["llm_unsupported_reasoning"] = call_judge(
            unsupported_reasoning.llm_judge_prompt(sample_turn)
        )
        if teacher_described_confusion and learner_turns:
            row["llm_target_error_preservation"] = call_judge(
                target_error_preservation.llm_judge_prompt(
                    teacher_described_confusion, learner_turns[0]
                )
            )
        if transfer_turn:
            transfer_probe = tutor_turns[-1] if tutor_turns else ""
            row["llm_near_transfer_attempt"] = call_judge(
                near_transfer_attempt.llm_judge_prompt(transfer_probe, transfer_turn)
            )

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pilot", "main"], default="main")
    args = parser.parse_args()

    output_dir = Path("outputs") / args.phase

    # Prefer model-specific subdirectories (e.g. outputs/main/gpt-4o-2024-11-20/C1/)
    # to avoid including legacy mock-data directories (C1/, C2/, C3/, C4/ at the top level).
    model_dirs = [d for d in output_dir.iterdir() if d.is_dir() and not d.name.startswith("C")]
    if model_dirs:
        json_files = []
        for md in sorted(model_dirs):
            json_files.extend(sorted(md.rglob("*.json")))
        print(f"Found {len(model_dirs)} model dirs, {len(json_files)} dialogue JSON files in {output_dir}")
    else:
        json_files = sorted(output_dir.rglob("*.json"))
        print(f"Found {len(json_files)} dialogue JSON files in {output_dir}")

    rows = []
    for jf in tqdm(json_files, desc="Computing trace metrics"):
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            print(f"Error reading {jf}: {e}")
            continue
        if data.get("exclusion_flag"):
            continue  # skip excluded dialogues
        row = compute_metrics_for_dialogue(data)
        rows.append(row)

    if not rows:
        print("No valid dialogues found. Run generation first.")
        return

    df = pd.DataFrame(rows)
    out_path = Path("outputs") / "automatic_trace_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")

    print("\n=== Metric means by condition ===")
    numeric_cols = [c for c in df.columns if c.startswith("rule_") and df[c].dtype in ["float64", "bool"]]
    summary = df.groupby("condition")[numeric_cols].mean().round(3)
    print(summary.to_string())

    if "model_id" in df.columns and df["model_id"].nunique() > 1:
        print("\n=== Metric means by model × condition ===")
        summary2 = df.groupby(["model_id", "condition"])[numeric_cols].mean().round(3)
        print(summary2.to_string())


if __name__ == "__main__":
    main()
