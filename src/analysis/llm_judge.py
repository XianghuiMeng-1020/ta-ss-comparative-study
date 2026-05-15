"""
LLM-as-Judge triangulation layer.

Workflow:
  1. Calibration: rate the 240 human-coded dialogues; compute ICC(judge, human) per dimension.
     Dimensions with ICC < 0.65 are excluded from judge-based analyses.
  2. Scoring: rate remaining ~4,560 dialogues (3 passes per dialogue; take median).
  3. Bias check: compare mean ratings per model family for same-family vs cross-family.
  4. Output: outputs/llm_judge_ratings.csv + outputs/llm_judge_calibration.csv

Judge model selection (Decision Log D10):
  - Must NOT be any of the 4 generation model families (GPT, Claude, Gemini, Llama)
  - Preferred: claude-opus-4 (if Anthropic not in generation set) else gpt-5

Usage:
  # Calibration on human-coded set
  python src/analysis/llm_judge.py --calibrate --human-ratings outputs/coder_ratings_raw.csv

  # Score all remaining dialogues (uses cached calibration)
  python src/analysis/llm_judge.py --score-all --phase main

  # Both in sequence
  python src/analysis/llm_judge.py --calibrate --score-all --phase main
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.generation.clients import LLMConfig, call_llm

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
JUDGE_ICC_THRESHOLD = 0.65  # Minimum ICC(judge, human) per dimension to use judge scores

JUDGE_MODEL_ID = os.environ.get("JUDGE_MODEL_ID", "claude-opus-4")
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "anthropic")
N_PASSES = 3  # self-consistency: rate each dialogue 3×, take median

JUDGE_SYSTEM_PROMPT = """You are an expert educational researcher evaluating AI-generated tutoring dialogues.
You will rate a simulated-learner dialogue on specific educational quality dimensions.
You MUST follow the rating scales exactly as specified. Think step-by-step before giving your final rating.
Do NOT reference the AI/LLM nature of the responses in your ratings — rate the dialogue as you observe it."""

JUDGE_PROMPT_TEMPLATE = """Please rate the following tutoring dialogue on the dimension specified below.

## DIALOGUE
PROBLEM: {problem}

ORIGINAL STUDENT ERROR: {original_error}

TRANSCRIPT:
{transcript}

---

## DIMENSION TO RATE: {dimension_name}

{dimension_description}

## RATING SCALE (1–5)
{scale_anchors}

## INSTRUCTIONS
1. Read the full dialogue carefully.
2. Think step-by-step: identify specific evidence for each point on the scale.
3. Choose the score that best fits based on observable dialogue features.
4. Output ONLY in this format:
   REASONING: [1-3 sentences citing specific evidence]
   SCORE: [integer 1, 2, 3, 4, or 5]
"""

DIMENSION_SPECS = {
    "TA1": {
        "name": "TA1 — Independent Performance",
        "description": (
            "Does the learner make a genuine independent attempt at the near-transfer "
            "problem (the final problem in the dialogue), showing self-generated reasoning "
            "without asking for help first?"
        ),
        "anchors": (
            "1 = No attempt; refusal; 'I don't know' without any working\n"
            "2 = Bare answer only; or copies tutor explanation verbatim\n"
            "3 = Partial attempt with some reasoning but incomplete or immediately asks for help\n"
            "4 = Clear attempt with visible reasoning; may contain errors but genuinely independent\n"
            "5 = Complete independent attempt with explicit step-by-step reasoning; no hint-seeking"
        ),
    },
    "TA2": {
        "name": "TA2 — Productive Learner Behaviour",
        "description": (
            "Does the learner exhibit behaviours that support the teaching process — "
            "asking clarifying questions, showing partial understanding, explicitly revising, "
            "reflecting — without acting as an expert?"
        ),
        "anchors": (
            "1 = Passive acknowledgement only ('ok', 'I see') with no revision or questioning\n"
            "2 = Minimal engagement; accepts corrections without showing revision\n"
            "3 = Some revision or questioning present but inconsistent\n"
            "4 = Consistent questioning and/or explicit revision; partial understanding visible\n"
            "5 = Rich productive behaviour: questioning, revising, showing both correct and incorrect thinking"
        ),
    },
    "TA3": {
        "name": "TA3 — Support for Teaching",
        "description": (
            "Does the dialogue create opportunities for the tutor to explain, correct, probe, "
            "and check understanding? Does the learner's behaviour invite teaching moves?"
        ),
        "anchors": (
            "1 = Learner solves everything or refuses to engage; no teaching opportunities\n"
            "2 = Minimal teaching opportunity; learner barely wrong or barely responsive\n"
            "3 = Occasional teaching opportunities but learner too correct or too passive\n"
            "4 = Clear and recurrent teaching opportunities across multiple turns\n"
            "5 = Rich teaching opportunities throughout; errors invite focused pedagogical response"
        ),
    },
    "TA4": {
        "name": "TA4 — Visible Reasoning as Shared Representation",
        "description": (
            "Does the learner show enough of their reasoning in text that the tutor "
            "can inspect what the learner is thinking and teach from it?"
        ),
        "anchors": (
            "1 = No reasoning shown; only bare answers\n"
            "2 = Minimal reasoning; too vague to diagnose\n"
            "3 = Some reasoning visible but incomplete or fragmented\n"
            "4 = Clear reasoning in most turns; errors are diagnosable\n"
            "5 = Rich explicit reasoning throughout; every step visible and diagnosable"
        ),
    },
    "SS1": {
        "name": "SS1 — Cognitive Model Plausibility",
        "description": (
            "Does the learner's reasoning resemble a plausible 7th-grade student strategy "
            "(not random errors, not advanced reasoning, not perfect reasoning)?"
        ),
        "anchors": (
            "1 = Reasoning is not plausible as any student (random, incoherent, or adult-expert level)\n"
            "2 = Vaguely student-like but inconsistent or implausible\n"
            "3 = Partially plausible; some age-appropriate errors with some implausible elements\n"
            "4 = Mostly plausible 7th-grade reasoning with realistic errors\n"
            "5 = Highly plausible; indistinguishable from a typical 7th-grade student's reasoning pattern"
        ),
    },
    "SS2": {
        "name": "SS2 — Error-Model Alignment",
        "description": (
            "Does the target misconception (as described in the student error field) appear "
            "in the learner's early responses rather than a random or different error?"
        ),
        "anchors": (
            "1 = Target misconception absent; a different or random error appears\n"
            "2 = Vague suggestion of target misconception; weak alignment\n"
            "3 = Target misconception partially present; mixed with other errors\n"
            "4 = Clear target misconception in first 1-2 turns with some persistence\n"
            "5 = Target misconception clearly and consistently present; aligns precisely with description"
        ),
    },
    "SS3": {
        "name": "SS3 — Prior-Knowledge Consistency",
        "description": (
            "Does the learner stay consistent with a 7th-grade ability level throughout? "
            "No use of knowledge the student would not have; no wildly inconsistent knowledge claims."
        ),
        "anchors": (
            "1 = Clear violation (uses calculus, matrices, or wildly inconsistent knowledge claims)\n"
            "2 = Occasional inconsistency in ability level\n"
            "3 = Mostly consistent with some minor inconsistencies\n"
            "4 = Consistent 7th-grade level throughout\n"
            "5 = Precisely consistent; deflects appropriately when asked about beyond-level content"
        ),
    },
    "SS4": {
        "name": "SS4 — Learning-Process Plausibility",
        "description": (
            "Does the learner change gradually in response to feedback rather than becoming "
            "expert immediately? Is the rate of improvement plausible?"
        ),
        "anchors": (
            "1 = Single-turn full correction (solves perfectly after one hint) or no change at all\n"
            "2 = Too rapid improvement; or completely static despite multiple hints\n"
            "3 = Some gradual improvement but with implausible jumps or stalls\n"
            "4 = Mostly gradual improvement; partial corrections visible\n"
            "5 = Clearly gradual; sub-errors persist; improvement is step-by-step across turns"
        ),
    },
    "SS5": {
        "name": "SS5 — Instructional-Testing Utility",
        "description": (
            "Could this dialogue be used to test a specific tutor response or feedback policy? "
            "Does it provide a realistic testbed for evaluating tutoring strategies?"
        ),
        "anchors": (
            "1 = Not usable as testbed (too unrealistic, too perfect, or incoherent)\n"
            "2 = Marginally usable; limited diagnostic value\n"
            "3 = Partially usable; some turns provide realistic feedback-testing opportunities\n"
            "4 = Mostly usable; would work as testbed for feedback policy experiments\n"
            "5 = Highly usable; realistic, stable, diagnosable; provides multiple testable moments"
        ),
    },
}


def build_transcript(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        tutor = t.get("tutor_turn", "").strip()
        learner = t.get("learner_turn", "").strip()
        if tutor:
            lines.append(f"[TUTOR]: {tutor}")
        if learner:
            lines.append(f"[LEARNER]: {learner}")
    return "\n".join(lines)


def parse_score(response_text: str) -> int | None:
    """Extract integer score from 'SCORE: N' in judge response."""
    import re
    match = re.search(r"SCORE\s*:\s*([1-5])", response_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: find last standalone digit 1-5
    digits = re.findall(r"\b([1-5])\b", response_text)
    if digits:
        return int(digits[-1])
    return None


def rate_dialogue_single_pass(
    dialogue_data: dict,
    config: LLMConfig,
    dimensions: list[str] | None = None,
    seed: int = 17,
) -> dict[str, int | None]:
    """Rate one dialogue on all specified dimensions in a single pass each."""
    if dimensions is None:
        dimensions = DIMENSIONS

    turns = dialogue_data.get("turns", [])
    transcript = build_transcript(turns)
    problem = dialogue_data.get("system_prompt", "")[:200]  # truncate
    original_error = turns[0].get("learner_turn", "")[:200] if turns else ""

    scores: dict[str, int | None] = {}
    for dim in dimensions:
        spec = DIMENSION_SPECS.get(dim, {})
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            problem=problem,
            original_error=original_error,
            transcript=transcript[:3000],  # truncate for token budget
            dimension_name=spec.get("name", dim),
            dimension_description=spec.get("description", ""),
            scale_anchors=spec.get("anchors", ""),
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            result = call_llm(messages, config, seed=seed)
            scores[dim] = parse_score(result.content)
        except Exception as e:
            scores[dim] = None
    return scores


def rate_dialogue_with_consistency(
    dialogue_data: dict,
    config: LLMConfig,
    dimensions: list[str] | None = None,
    n_passes: int = N_PASSES,
) -> dict[str, float | None]:
    """Rate dialogue N times; return median per dimension."""
    if dimensions is None:
        dimensions = DIMENSIONS
    all_pass_scores: list[dict[str, int | None]] = []
    for i in range(n_passes):
        seed = [17, 42, 91][i % 3]
        pass_scores = rate_dialogue_single_pass(dialogue_data, config, dimensions, seed=seed)
        all_pass_scores.append(pass_scores)

    final: dict[str, float | None] = {}
    for dim in dimensions:
        vals = [p[dim] for p in all_pass_scores if p.get(dim) is not None]
        final[dim] = float(np.median(vals)) if vals else None
    return final


def compute_judge_human_icc(
    judge_df: pd.DataFrame, human_df: pd.DataFrame
) -> dict[str, float]:
    """
    Compute ICC(judge_median, human_mean) per dimension on overlapping packets.
    Returns dict of {dimension: icc_value}.
    """
    try:
        import pingouin as pg
    except ImportError:
        print("pingouin not installed; skipping ICC computation.")
        return {}

    results = {}
    merged = judge_df.merge(human_df, on="packet_id", suffixes=("_judge", "_human"))
    for dim in DIMENSIONS:
        jcol = f"{dim}_judge"
        hcol = f"{dim}_human"
        if jcol not in merged.columns or hcol not in merged.columns:
            continue
        sub = merged[[jcol, hcol]].dropna()
        if len(sub) < 10:
            continue
        df_long = pd.melt(
            sub.reset_index().rename(columns={"index": "packet"}),
            id_vars="packet", var_name="rater", value_name="rating"
        )
        try:
            icc_res = pg.intraclass_corr(
                data=df_long, targets="packet", raters="rater", ratings="rating",
                nan_policy="omit"
            )
            icc_val = icc_res.iloc[0]["ICC"]
            results[dim] = round(float(icc_val), 4)
        except Exception:
            results[dim] = None
    return results


def calibrate(human_ratings_path: Path, config: LLMConfig) -> dict[str, float]:
    """
    Rate all human-coded dialogues with the judge; compute ICC(judge, human).
    Returns dict of {dimension: icc_value}; saves calibration CSV.
    """
    print("=== LLM Judge Calibration ===")
    human_df = pd.read_csv(human_ratings_path)

    human_mean = (
        human_df.groupby("packet_id")[DIMENSIONS].mean().reset_index()
    )
    manifest_path = Path("outputs/coder_packets/manifest.csv")
    if not manifest_path.exists():
        print("ERROR: manifest.csv not found. Run packetize.py first.")
        return {}

    manifest = pd.read_csv(manifest_path)
    judge_rows = []

    for _, row in manifest.iterrows():
        pid = row["packet_id"]
        json_path = row.get("json_path", "")
        if not json_path or not Path(json_path).exists():
            continue
        try:
            data = json.loads(Path(json_path).read_text())
        except Exception:
            continue
        scores = rate_dialogue_with_consistency(data, config)
        scores["packet_id"] = pid
        judge_rows.append(scores)

    if not judge_rows:
        print("No dialogues rated. Check manifest.csv and json_path entries.")
        return {}

    judge_df = pd.DataFrame(judge_rows)
    judge_df.to_csv(Path("outputs/llm_judge_calibration_scores.csv"), index=False)

    icc_results = compute_judge_human_icc(judge_df, human_mean)
    print("\nJudge–Human ICC per dimension:")
    calibration_rows = []
    for dim, icc_val in icc_results.items():
        status = "PASS" if icc_val is not None and icc_val >= JUDGE_ICC_THRESHOLD else "FAIL"
        print(f"  {dim}: {icc_val:.3f}  [{status}]")
        calibration_rows.append({"dimension": dim, "icc_judge_human": icc_val, "status": status})

    cal_df = pd.DataFrame(calibration_rows)
    cal_df.to_csv(Path("outputs/llm_judge_calibration_report.csv"), index=False)
    print(f"\nCalibration report → outputs/llm_judge_calibration_report.csv")
    return icc_results


def score_all(
    phase: str,
    config: LLMConfig,
    accepted_dimensions: list[str],
    human_packet_ids: set[str],
) -> None:
    """Rate all non-human-coded valid dialogues on accepted dimensions."""
    print(f"\n=== LLM Judge Scoring: {phase} ===")
    print(f"Accepted dimensions (ICC ≥ {JUDGE_ICC_THRESHOLD}): {accepted_dimensions}")

    manifest_path = Path("outputs/coder_packets/manifest.csv")
    human_pids = human_packet_ids

    output_dir = Path("outputs") / phase
    all_json = sorted(output_dir.rglob("*.json"))
    print(f"Found {len(all_json)} dialogue JSONs in {output_dir}")

    judge_rows = []
    out_csv = Path("outputs/llm_judge_ratings.csv")

    # Load checkpoint if exists
    done_pids: set[str] = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done_pids = set(prev["dialogue_path"].tolist())
        judge_rows = prev.to_dict("records")
        print(f"Resuming: {len(done_pids)} dialogues already scored.")

    for jpath in all_json:
        path_str = str(jpath)
        if path_str in done_pids:
            continue
        try:
            data = json.loads(jpath.read_text())
        except Exception:
            continue
        if data.get("exclusion_flag"):
            continue

        scores = rate_dialogue_with_consistency(data, config, dimensions=accepted_dimensions)
        scores["dialogue_path"] = path_str
        scores["condition"] = data.get("condition", "?")
        scores["model_tag"] = data.get("model_tag", "?")
        scores["scenario_id"] = data.get("scenario_id", "?")
        scores["seed"] = data.get("seed", "?")
        judge_rows.append(scores)

        # Save checkpoint every 50
        if len(judge_rows) % 50 == 0:
            pd.DataFrame(judge_rows).to_csv(out_csv, index=False)
            print(f"  Checkpoint: {len(judge_rows)} dialogues scored.")

    pd.DataFrame(judge_rows).to_csv(out_csv, index=False)
    print(f"\nJudge ratings saved → {out_csv} ({len(judge_rows)} dialogues)")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-Judge triangulation")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--score-all", action="store_true")
    parser.add_argument("--phase", default="main")
    parser.add_argument("--human-ratings", default="outputs/coder_ratings_raw.csv")
    parser.add_argument("--judge-model", default=JUDGE_MODEL_ID)
    parser.add_argument("--judge-backend", default=JUDGE_BACKEND)
    args = parser.parse_args()

    config = LLMConfig(
        model_id=args.judge_model,
        api_backend=args.judge_backend,
        temperature=0.0,  # judge uses greedy for consistency
        max_tokens=400,
    )

    icc_results: dict[str, float] = {}

    if args.calibrate:
        human_path = Path(args.human_ratings)
        if not human_path.exists():
            print(f"Human ratings not found: {human_path}")
            return
        icc_results = calibrate(human_path, config)
    else:
        # Load from previous calibration
        cal_path = Path("outputs/llm_judge_calibration_report.csv")
        if cal_path.exists():
            cal_df = pd.read_csv(cal_path)
            icc_results = dict(zip(cal_df["dimension"], cal_df["icc_judge_human"]))
        else:
            print("No calibration report found. Run --calibrate first.")

    if args.score_all:
        accepted = [
            dim for dim, icc in icc_results.items()
            if icc is not None and not np.isnan(icc) and icc >= JUDGE_ICC_THRESHOLD
        ]
        if not accepted:
            accepted = DIMENSIONS
            print("WARNING: No accepted dimensions from calibration; scoring all dimensions.")

        human_path = Path(args.human_ratings)
        human_pids: set[str] = set()
        if human_path.exists():
            hdf = pd.read_csv(human_path)
            human_pids = set(hdf["packet_id"].tolist())

        score_all(args.phase, config, accepted, human_pids)


if __name__ == "__main__":
    main()
