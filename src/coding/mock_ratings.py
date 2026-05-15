"""
Generate synthetic coder ratings for pipeline validation (P8 mock).

Simulates two coders rating 240 stratified dialogues.
Ratings are condition-informed (C1 higher on TA dims, C2 higher on SS dims)
with realistic coder noise, enabling end-to-end testing of P9-P11.

NOT for publication — replace with real coder ratings before analysis.
"""

import json
import random
from pathlib import Path

import pandas as pd

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
FAILURE_FLAGS = [
    "premature_expertise", "role_drift", "over_technical",
    "misconception_loss", "logical_inconsistency", "unsupported_reasoning",
]

USE_CASES = {
    "C1": "learning_by_teaching",
    "C2": "diagnostic_simulation",
    "C3": "reject",
    "C4": "reject",
}

# Mean ratings by condition × dimension (mock ground truth)
CONDITION_MEANS = {
    "C1": {"TA1": 4.2, "TA2": 4.0, "TA3": 3.8, "TA4": 3.9,
           "SS1": 2.8, "SS2": 2.2, "SS3": 2.5, "SS4": 2.3, "SS5": 2.6},
    "C2": {"TA1": 2.5, "TA2": 2.8, "TA3": 3.1, "TA4": 3.4,
           "SS1": 4.1, "SS2": 4.3, "SS3": 3.9, "SS4": 4.0, "SS5": 4.2},
    "C3": {"TA1": 2.8, "TA2": 2.5, "TA3": 2.9, "TA4": 2.6,
           "SS1": 2.6, "SS2": 2.1, "SS3": 2.4, "SS4": 2.5, "SS5": 2.3},
    "C4": {"TA1": 1.5, "TA2": 1.8, "TA3": 1.9, "TA4": 2.1,
           "SS1": 1.8, "SS2": 1.6, "SS3": 2.0, "SS4": 1.7, "SS5": 1.9},
}

FAILURE_RATES = {
    "C1": {"premature_expertise": 0.10, "role_drift": 0.08, "over_technical": 0.02,
           "misconception_loss": 0.05, "logical_inconsistency": 0.06, "unsupported_reasoning": 0.03},
    "C2": {"premature_expertise": 0.30, "role_drift": 0.07, "over_technical": 0.02,
           "misconception_loss": 0.25, "logical_inconsistency": 0.12, "unsupported_reasoning": 0.05},
    "C3": {"premature_expertise": 0.15, "role_drift": 0.20, "over_technical": 0.05,
           "misconception_loss": 0.35, "logical_inconsistency": 0.18, "unsupported_reasoning": 0.25},
    "C4": {"premature_expertise": 0.60, "role_drift": 0.05, "over_technical": 0.10,
           "misconception_loss": 0.70, "logical_inconsistency": 0.15, "unsupported_reasoning": 0.40},
}


def clamp(val: float, lo: int = 1, hi: int = 5) -> int:
    return max(lo, min(hi, round(val)))


def rate_dialogue(packet_id: str, condition: str, coder_id: str, rng: random.Random) -> dict:
    means = CONDITION_MEANS.get(condition, CONDITION_MEANS["C3"])
    noise_level = 0.4 if coder_id == "CODER_A" else 0.6

    row = {"packet_id": packet_id, "coder_id": coder_id}
    for dim in DIMENSIONS:
        mu = means.get(dim, 3.0)
        val = mu + rng.gauss(0, noise_level)
        row[dim] = clamp(val)

    # Use-case decision
    expected_uc = USE_CASES.get(condition, "reject")
    noise_flip = rng.random() < (0.15 if coder_id == "CODER_A" else 0.22)
    if noise_flip:
        alts = ["learning_by_teaching", "diagnostic_simulation", "feedback_policy_testing", "reject"]
        alts.remove(expected_uc)
        row["use_case_decision"] = rng.choice(alts)
    else:
        row["use_case_decision"] = expected_uc

    # Failure flags
    rates = FAILURE_RATES.get(condition, FAILURE_RATES["C3"])
    for flag in FAILURE_FLAGS:
        rate = rates.get(flag, 0.05)
        row[f"flag_{flag}"] = int(rng.random() < rate)

    row["notes"] = ""
    return row


def generate_mock_ratings(
    n_dialogues: int = 240,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Select n_dialogues from main generated_dialogues.csv (stratified by condition),
    generate mock ratings from two coders, and save.
    """
    gen_path = Path("outputs/generated_dialogues.csv")
    manifest_path = Path("outputs/coder_packets/manifest.csv")

    if not gen_path.exists():
        raise FileNotFoundError("Run generation first.")

    gen_df = pd.read_csv(gen_path)
    main_df = gen_df[(gen_df["phase"] == "main") & (~gen_df["exclusion_flag"].fillna(False))]

    rng = random.Random(seed)
    per_condition = n_dialogues // 4
    selected = []
    for cond in ["C1", "C2", "C3", "C4"]:
        pool = main_df[main_df["condition"] == cond]
        k = min(per_condition, len(pool))
        chosen = pool.sample(n=k, random_state=rng.randint(0, 2**31))
        selected.append(chosen)

    selected_df = pd.concat(selected, ignore_index=True)

    # Build synthetic manifest
    manifest_rows = []
    rating_rows = []

    manifest_dir = Path("outputs/coder_packets")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for _, row in selected_df.iterrows():
        import hashlib
        pid = hashlib.sha256(f"{row['scenario_id']}_{row['condition']}_{row['seed']}".encode()).hexdigest()[:8].upper()
        manifest_rows.append({
            "packet_id": pid,
            "true_condition": row["condition"],
            "scenario_id": row["scenario_id"],
            "seed": row["seed"],
            "json_path": row.get("json_path", ""),
            "calibration": False,
        })

        for coder in ["CODER_A", "CODER_B"]:
            r = rate_dialogue(pid, row["condition"], coder, rng)
            r["scenario_id"] = row["scenario_id"]
            r["true_condition"] = row["condition"]
            rating_rows.append(r)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved mock manifest ({len(manifest_df)} packets) → {manifest_path}")

    ratings_df = pd.DataFrame(rating_rows)
    ratings_path = Path("outputs/coder_ratings_raw.csv")
    ratings_df.to_csv(ratings_path, index=False)
    print(f"Saved mock ratings ({len(ratings_df)} rows) → {ratings_path}")
    print(f"Coders: {ratings_df['coder_id'].unique().tolist()}")
    print(f"Condition distribution:\n{manifest_df['true_condition'].value_counts().to_string()}")

    return ratings_df


if __name__ == "__main__":
    generate_mock_ratings()
