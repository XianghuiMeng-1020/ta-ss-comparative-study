"""
Ingest and validate v2 human annotation ratings (D1 HL1/HL2/HL3 + D2 EQ1).

Reads coder rating CSVs (filled rating_template.csv), validates ranges,
computes ICC, flags items needing arbitration (|rating_A - rating_B| >= 2).

Usage:
    python src/coding/ingest_ratings_v2.py \
        --ratings-dir outputs/coder_packets_qa \
        --output outputs/human_annotations_v2.csv \
        --icc-output outputs/icc_report_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DIMENSIONS = ["HL1", "HL2", "HL3", "EQ1"]
VALID_RANGES = {"HL1": (1, 5), "HL2": (1, 5), "HL3": (0, 1), "EQ1": (1, 5)}
ARBITRATION_THRESHOLD = 2


def load_and_validate(ratings_dir: Path) -> pd.DataFrame:
    rating_path = ratings_dir / "rating_template.csv"
    if not rating_path.exists():
        raise FileNotFoundError(f"Filled rating template not found: {rating_path}")

    df = pd.read_csv(rating_path)
    df = df.dropna(subset=["packet_id", "coder_id"])

    errors = []
    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        lo, hi = VALID_RANGES[dim]
        df[dim] = pd.to_numeric(df[dim], errors="coerce")
        invalid = df[(df[dim].notna()) & ((df[dim] < lo) | (df[dim] > hi))]
        if len(invalid) > 0:
            errors.append(f"{dim}: {len(invalid)} out-of-range values")

    if errors:
        print("[WARN] Validation errors:")
        for e in errors:
            print(f"  {e}")

    return df


def compute_agreement(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="packet_id", columns="coder_id", values=DIMENSIONS)

    agreement_rows = []
    coders = df["coder_id"].unique()
    if len(coders) < 2:
        print("[WARN] Only one coder found — skipping agreement computation.")
        return pd.DataFrame()

    coder_a, coder_b = coders[0], coders[1]

    for packet_id in df["packet_id"].unique():
        row_a = df[(df["packet_id"] == packet_id) & (df["coder_id"] == coder_a)]
        row_b = df[(df["packet_id"] == packet_id) & (df["coder_id"] == coder_b)]
        if row_a.empty or row_b.empty:
            continue

        row = {"packet_id": packet_id}
        needs_arbitration = False
        for dim in DIMENSIONS:
            a_val = row_a[dim].values[0] if dim in row_a.columns else None
            b_val = row_b[dim].values[0] if dim in row_b.columns else None
            row[f"{dim}_CoderA"] = a_val
            row[f"{dim}_CoderB"] = b_val
            if pd.notna(a_val) and pd.notna(b_val):
                diff = abs(float(a_val) - float(b_val))
                row[f"{dim}_diff"] = diff
                row[f"{dim}_mean"] = (float(a_val) + float(b_val)) / 2
                if diff >= ARBITRATION_THRESHOLD:
                    needs_arbitration = True
            else:
                row[f"{dim}_diff"] = None
                row[f"{dim}_mean"] = None
        row["needs_arbitration"] = needs_arbitration
        agreement_rows.append(row)

    return pd.DataFrame(agreement_rows)


def compute_icc(df: pd.DataFrame) -> pd.Series:
    """Compute ICC(2,k) for each dimension using pingouin."""
    try:
        import pingouin as pg
    except ImportError:
        print("[WARN] pingouin not installed — skip ICC. pip install pingouin")
        return pd.Series(dtype=float)

    results = {}
    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        sub = df[["packet_id", "coder_id", dim]].dropna()
        if sub.empty:
            continue
        try:
            icc_df = pg.intraclass_corr(
                data=sub, targets="packet_id", raters="coder_id", ratings=dim
            )
            icc2k = icc_df[icc_df["Type"] == "ICC2k"]["ICC"].values
            results[dim] = round(float(icc2k[0]), 4) if len(icc2k) else None
        except Exception as e:
            results[dim] = None
            print(f"  ICC error for {dim}: {e}")

    return pd.Series(results, name="ICC2k")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest v2 human annotation ratings")
    parser.add_argument("--ratings-dir", required=True,
                        help="Directory containing filled rating_template.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--icc-output", required=True)
    args = parser.parse_args()

    ratings_dir = Path(args.ratings_dir)
    df = load_and_validate(ratings_dir)
    print(f"Loaded {len(df)} rating rows | {df['coder_id'].nunique()} coders "
          f"| {df['packet_id'].nunique()} packets")

    icc = compute_icc(df)
    print("\nICC(2,k) per dimension:")
    print(icc.to_string())

    icc_fails = [dim for dim, val in icc.items() if val is not None and val < 0.75]
    if icc_fails:
        print(f"\n[WARN] ICC < 0.75 gate NOT met for: {icc_fails}")
        print("  Do NOT proceed to analysis for these dimensions.")
        print("  Schedule additional calibration session.")
    else:
        print("\nICC gate passed for all dimensions (≥ 0.75).")

    agreement = compute_agreement(df)
    if not agreement.empty:
        n_arbitrate = agreement["needs_arbitration"].sum()
        print(f"\nItems needing arbitration: {n_arbitrate}/{len(agreement)}")
        arb = agreement[agreement["needs_arbitration"]]
        if not arb.empty:
            print(arb[["packet_id", "needs_arbitration"] +
                       [c for c in arb.columns if "_diff" in c]].to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    icc_path = Path(args.icc_output)
    icc_path.parent.mkdir(parents=True, exist_ok=True)
    icc.to_csv(icc_path)

    agreement_path = output_path.with_name(output_path.stem + "_agreement.csv")
    if not agreement.empty:
        agreement.to_csv(agreement_path, index=False)
        print(f"Agreement table → {agreement_path}")

    print(f"\nRatings → {output_path}")
    print(f"ICC report → {icc_path}")


if __name__ == "__main__":
    main()
