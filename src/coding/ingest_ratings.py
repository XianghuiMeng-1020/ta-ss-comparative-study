"""
Ingest raw coder rating CSVs and produce coder_ratings_raw.csv.

Expects coder to return a filled rating_template.csv (or _calibration.csv)
with packet_id, coder_id, dimension scores, use_case_decision, and failure flags.

Merges ratings from multiple coders; joins with manifest to reveal true condition.
Output: outputs/coder_ratings_raw.csv
"""

import argparse
from pathlib import Path

import pandas as pd

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
FAILURE_FLAGS = [
    "premature_expertise", "role_drift", "over_technical",
    "misconception_loss", "logical_inconsistency", "unsupported_reasoning",
]

VALID_USE_CASES = {
    "learning_by_teaching", "diagnostic_simulation",
    "feedback_policy_testing", "reject",
}


def validate_ratings(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Basic validation and coercion of rating values."""
    errors = []

    # Dimension scores: expect 1–5 integer or blank
    for dim in DIMENSIONS:
        if dim in df.columns:
            non_null = df[dim].dropna()
            invalid = non_null[~non_null.astype(str).str.match(r"^[1-5]$")]
            if len(invalid) > 0:
                errors.append(f"{source_file}: {dim} has invalid values: {invalid.tolist()}")
            df[dim] = pd.to_numeric(df[dim], errors="coerce")

    # Use-case decision: expect one of valid strings
    if "use_case_decision" in df.columns:
        invalid_uc = df["use_case_decision"].dropna()
        invalid_uc = invalid_uc[~invalid_uc.isin(VALID_USE_CASES)]
        if len(invalid_uc) > 0:
            errors.append(f"{source_file}: invalid use_case_decision values: {invalid_uc.tolist()}")

    # Failure flags: expect 0 or 1 or blank
    for flag in FAILURE_FLAGS:
        col = f"flag_{flag}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if errors:
        for e in errors:
            print(f"WARNING: {e}")

    return df


def ingest(rating_files: list[Path], manifest_path: Path) -> pd.DataFrame:
    """Merge coder files and join with manifest."""
    manifest = pd.read_csv(manifest_path)

    all_ratings = []
    for rf in rating_files:
        df = pd.read_csv(rf)
        if "coder_id" not in df.columns or df["coder_id"].isna().all():
            print(f"WARNING: {rf} has no coder_id filled. Skipping.")
            continue
        df = validate_ratings(df, str(rf))
        all_ratings.append(df)

    if not all_ratings:
        raise ValueError("No valid rating files found.")

    combined = pd.concat(all_ratings, ignore_index=True)

    # Join with manifest to add true condition (AFTER ingestion)
    merged = combined.merge(
        manifest[["packet_id", "true_condition", "scenario_id", "seed"]],
        on="packet_id",
        how="left",
    )

    out_path = Path("outputs") / "coder_ratings_raw.csv"
    merged.to_csv(out_path, index=False)
    print(f"Saved {len(merged)} raw rating rows → {out_path}")
    print(f"Coders: {merged['coder_id'].unique().tolist()}")
    print(f"Packets rated: {merged['packet_id'].nunique()}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rating_files",
        nargs="+",
        help="Paths to filled rating_template CSV files from coders",
    )
    parser.add_argument(
        "--manifest",
        default="outputs/coder_packets/manifest.csv",
        help="Path to the locked manifest CSV",
    )
    args = parser.parse_args()

    rating_paths = [Path(f) for f in args.rating_files]
    ingest(rating_paths, Path(args.manifest))


if __name__ == "__main__":
    main()
