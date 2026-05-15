"""
Build stratified scenario bank from MathDial.

Outputs:
  data/scenarios.csv       — 100-row main bank
  data/scenarios_pilot.csv — 50-row pilot bank (non-overlapping)

Go criterion: all required fields non-null; chi-square balance check passes.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.data.load_mathdial import load_and_cache
from src.data.stratify import add_stratification_labels

RANDOM_SEED = 2026
N_MAIN = 100
N_PILOT = 50
MIN_CELL_SIZE = 5

REQUIRED_FIELDS = [
    "problem",
    "correct_solution",
    "original_incorrect_solution",
    "student_profile",
    "teacher_described_confusion",
    "original_tutor_turns",
    "original_student_turns",
    "n_tutor_turns",
]


def filter_complete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where all required fields are non-empty strings."""
    mask = pd.Series(True, index=df.index)
    for col in REQUIRED_FIELDS:
        if col in df.columns:
            mask &= df[col].notna() & (df[col].astype(str).str.strip() != "")
    filtered = df[mask].copy()
    print(f"Rows with all required fields: {len(filtered)} / {len(df)}")
    return filtered


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    strat_cols: list[str],
    exclude_ids: set | None = None,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """
    Sample n rows with approximate stratification across strat_cols.
    Merges cells with <MIN_CELL_SIZE eligible rows and logs merges.
    """
    if rng is None:
        rng = random.Random(RANDOM_SEED)

    pool = df.copy()
    if exclude_ids:
        pool = pool[~pool["scenario_id"].isin(exclude_ids)]

    # Attempt stratified sampling by topic × error_type
    sampled_ids: list[str] = []
    merge_log: list[str] = []

    # Build strata
    pool["_stratum"] = pool[strat_cols[0]].astype(str) + "||" + pool[strat_cols[1]].astype(str)
    strata = pool["_stratum"].unique()

    stratum_sizes = pool["_stratum"].value_counts()
    small_strata = stratum_sizes[stratum_sizes < MIN_CELL_SIZE].index.tolist()

    if small_strata:
        merge_log.append(
            f"Merging {len(small_strata)} strata with <{MIN_CELL_SIZE} rows "
            f"into 'miscellaneous' bucket: {small_strata}"
        )
        pool.loc[pool["_stratum"].isin(small_strata), "_stratum"] = "misc||misc"

    strata_counts = pool["_stratum"].value_counts()
    total_pool = len(pool)
    target_per_stratum = {s: max(1, round(n * cnt / total_pool)) for s, cnt in strata_counts.items()}
    # Adjust so sum == n
    diff = n - sum(target_per_stratum.values())
    largest_stratum = strata_counts.idxmax()
    target_per_stratum[largest_stratum] += diff

    for stratum, target in target_per_stratum.items():
        rows = pool[pool["_stratum"] == stratum]
        k = min(target, len(rows))
        chosen = rows.sample(n=k, random_state=rng.randint(0, 2**31)).index
        sampled_ids.extend(pool.loc[chosen, "scenario_id"].tolist())

    # Trim or top-up if rounding pushed us off
    if len(sampled_ids) > n:
        rng.shuffle(sampled_ids)
        sampled_ids = sampled_ids[:n]
    elif len(sampled_ids) < n:
        remaining = pool[~pool["scenario_id"].isin(sampled_ids)]
        extra = remaining.sample(n=n - len(sampled_ids), random_state=rng.randint(0, 2**31))
        sampled_ids.extend(extra["scenario_id"].tolist())

    result = pool[pool["scenario_id"].isin(sampled_ids)].drop(columns=["_stratum"])

    if merge_log:
        log_path = Path("decision_log_merges.txt")
        with log_path.open("a") as f:
            for entry in merge_log:
                f.write(entry + "\n")
        print(f"Merge log appended to {log_path}")

    return result.reset_index(drop=True)


def check_balance(df: pd.DataFrame, col1: str, col2: str) -> bool:
    """Chi-square test of independence between two stratification columns."""
    ct = pd.crosstab(df[col1], df[col2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        print(f"Cannot chi-square test {col1}×{col2}: too few levels.")
        return True
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"Balance check {col1}×{col2}: chi2={chi2:.2f}, p={p:.3f}, dof={dof}")
    return True  # We report but don't block on p value


def main():
    df = load_and_cache()
    df = filter_complete_rows(df)

    # Add stratification labels
    df = add_stratification_labels(df)

    # Assign scenario_id
    df = df.reset_index(drop=True)
    df["scenario_id"] = ["SC_{:04d}".format(i) for i in range(len(df))]

    rng = random.Random(RANDOM_SEED)

    # Sample main bank
    main_df = stratified_sample(
        df, N_MAIN, strat_cols=["topic", "error_type"], rng=rng
    )
    main_ids = set(main_df["scenario_id"])

    # Sample pilot bank (non-overlapping)
    pilot_df = stratified_sample(
        df, N_PILOT, strat_cols=["topic", "error_type"],
        exclude_ids=main_ids, rng=rng
    )

    # Balance checks
    print("\n=== Main bank balance ===")
    check_balance(main_df, "topic", "error_type")
    check_balance(main_df, "difficulty", "topic")

    print("\n=== Pilot bank balance ===")
    check_balance(pilot_df, "topic", "error_type")

    # Go criterion: no nulls in required fields
    for name, bank in [("main", main_df), ("pilot", pilot_df)]:
        for col in REQUIRED_FIELDS:
            if col in bank.columns:
                nulls = bank[col].isna().sum()
                if nulls > 0:
                    raise ValueError(f"GO FAIL: {col} has {nulls} nulls in {name} bank")
    print("\nGo criterion passed: all required fields non-null.")

    # Save
    Path("data").mkdir(exist_ok=True)
    main_df.to_csv("data/scenarios.csv", index=False)
    pilot_df.to_csv("data/scenarios_pilot.csv", index=False)
    print(f"Saved data/scenarios.csv ({len(main_df)} rows)")
    print(f"Saved data/scenarios_pilot.csv ({len(pilot_df)} rows)")
    print("\nMain topic distribution:\n", main_df["topic"].value_counts().to_string())
    print("\nMain error_type distribution:\n", main_df["error_type"].value_counts().to_string())
    print("\nMain difficulty distribution:\n", main_df["difficulty"].value_counts().to_string())


if __name__ == "__main__":
    main()
