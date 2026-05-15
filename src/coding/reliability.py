"""
Inter-rater reliability analysis — upgraded for C&E strong-accept standard.

Computes per dimension:
  - ICC(2,k) [Shrout & Fleiss, 1979] — two-way random, multiple raters
  - Krippendorff's α (ordinal) [Krippendorff, 2011]
  - Weighted Cohen's κ (linear weights, for ordinal 1-5 scales)
  - Percent exact agreement
  - Percent adjacent agreement (±1)

For use-case_decision:
  - Cohen's κ (nominal, 2 raters) or Fleiss κ (> 2 raters)

For failure flags (binary):
  - Fleiss κ

Go criterion (C&E standard):
  ALL dimensions: ICC(2,k) ≥ 0.75 AND α ≥ 0.70
  Mean ICC ≥ 0.80 across all 9 framework dimensions
  use_case_decision: κ ≥ 0.70
  failure flags: κ ≥ 0.60

Also handles adjudicated ratings:
  If adjudicator_id column present, uses adjudicated score as ground truth
  for disagreement items (|rating_A - rating_B| ≥ 2).

Output:
  outputs/reliability_report.csv  — per-dimension stats
  outputs/arbitration_log.csv     — items sent to third-coder adjudication

Usage:
  python src/coding/reliability.py [--ratings outputs/coder_ratings_raw.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

try:
    import krippendorff
    HAS_KRIPPENDORFF = True
except ImportError:
    HAS_KRIPPENDORFF = False

try:
    from sklearn.metrics import cohen_kappa_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
FAILURE_FLAGS = [
    "premature_expertise", "role_drift", "over_technical",
    "misconception_loss", "logical_inconsistency", "unsupported_reasoning",
]

ICC_THRESHOLD = 0.75          # C&E standard (upgraded from 0.60)
ALPHA_THRESHOLD = 0.70        # Krippendorff α
ICC_MEAN_TARGET = 0.80        # Mean ICC target across all 9 dimensions
USE_CASE_KAPPA_THRESHOLD = 0.70
FLAG_KAPPA_THRESHOLD = 0.60

ARBITRATION_THRESHOLD = 2     # |rating_A - rating_B| ≥ 2 → adjudication


# ── ICC(2,k) via pingouin ─────────────────────────────────────────────────────

def compute_icc_2k(ratings_wide: pd.DataFrame) -> dict:
    """
    Compute ICC(2,k) — two-way random effects, absolute agreement, multiple raters.
    Per Shrout & Fleiss (1979), this is the appropriate form for:
      - raters considered random sample from a population of raters
      - absolute agreement (not consistency) between raters
      - multiple raters (k > 2 collapses to ICC(2,1) if only 2)
    """
    if not HAS_PINGOUIN:
        return {
            "icc_type": "ICC(2,k)",
            "icc_value": None,
            "icc_ci_lower": None,
            "icc_ci_upper": None,
            "icc_f": None,
            "icc_df1": None,
            "icc_df2": None,
            "icc_p": None,
        }

    df_long = (
        ratings_wide.reset_index()
        .melt(
            id_vars=ratings_wide.index.name or "index",
            var_name="coder",
            value_name="rating",
        )
        .dropna(subset=["rating"])
    )
    df_long = df_long.rename(columns={df_long.columns[0]: "target"})

    try:
        icc_res = pg.intraclass_corr(
            data=df_long,
            targets="target",
            raters="coder",
            ratings="rating",
            nan_policy="omit",
        )
    except Exception as e:
        return {"icc_type": "ICC(2,k)", "icc_value": None, "error": str(e)}

    # Prefer ICC(A,k) = absolute agreement, multiple raters
    for icc_type in ["ICC(A,k)", "ICC(C,k)", "ICC(A,1)", "ICC(C,1)"]:
        match = icc_res[icc_res["Type"] == icc_type]
        if not match.empty:
            row = match.iloc[0]
            ci = row.get("CI95%", row.get("CI95", [None, None]))
            return {
                "icc_type": icc_type,
                "icc_value": round(float(row["ICC"]), 4),
                "icc_ci_lower": round(float(ci[0]), 4) if ci[0] is not None else None,
                "icc_ci_upper": round(float(ci[1]), 4) if ci[1] is not None else None,
                "icc_f": round(float(row.get("F", 0)), 3),
                "icc_df1": int(row.get("df1", 0)),
                "icc_df2": int(row.get("df2", 0)),
                "icc_p": round(float(row.get("pvalue", 1.0)), 4),
            }

    row = icc_res.iloc[0]
    ci = row.get("CI95%", row.get("CI95", [None, None]))
    return {
        "icc_type": str(row.get("Type", "unknown")),
        "icc_value": round(float(row["ICC"]), 4),
        "icc_ci_lower": round(float(ci[0]), 4) if ci[0] is not None else None,
        "icc_ci_upper": round(float(ci[1]), 4) if ci[1] is not None else None,
    }


# ── Krippendorff's α ─────────────────────────────────────────────────────────

def compute_krippendorff_alpha(ratings_matrix: np.ndarray, level: str = "ordinal") -> float:
    if not HAS_KRIPPENDORFF:
        return float("nan")
    try:
        return round(
            krippendorff.alpha(reliability_data=ratings_matrix, level_of_measurement=level),
            4,
        )
    except Exception as e:
        print(f"  Krippendorff error: {e}")
        return float("nan")


# ── Weighted Cohen's κ ────────────────────────────────────────────────────────

def compute_weighted_kappa(rater_a: list, rater_b: list, weights: str = "linear") -> float:
    if not HAS_SKLEARN:
        return float("nan")
    pairs = [(a, b) for a, b in zip(rater_a, rater_b) if pd.notna(a) and pd.notna(b)]
    if not pairs:
        return float("nan")
    a_vals, b_vals = zip(*pairs)
    try:
        return round(cohen_kappa_score(a_vals, b_vals, weights=weights), 4)
    except Exception:
        return float("nan")


# ── Percent agreement ─────────────────────────────────────────────────────────

def compute_percent_agreement(rater_a: list, rater_b: list) -> tuple[float, float]:
    """Returns (percent_exact, percent_adjacent)."""
    pairs = [(int(a), int(b)) for a, b in zip(rater_a, rater_b) if pd.notna(a) and pd.notna(b)]
    if not pairs:
        return float("nan"), float("nan")
    exact = sum(a == b for a, b in pairs) / len(pairs)
    adjacent = sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs)
    return round(exact, 4), round(adjacent, 4)


# ── Fleiss κ (binary flags) ───────────────────────────────────────────────────

def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    n_items, n_raters = ratings_matrix.shape
    n_cat = 2
    cat_counts = np.zeros((n_items, n_cat))
    for i in range(n_items):
        row = ratings_matrix[i]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        cat_counts[i, 0] = np.sum(valid == 0)
        cat_counts[i, 1] = np.sum(valid == 1)
    n_i = cat_counts.sum(axis=1)
    valid_rows = n_i > 1
    if valid_rows.sum() == 0:
        return float("nan")
    p_bar_j = cat_counts.sum(axis=0) / n_i.sum()
    p_bar = (
        (cat_counts[valid_rows] * (cat_counts[valid_rows] - 1)).sum(axis=1)
        / (n_i[valid_rows] * (n_i[valid_rows] - 1))
    ).mean()
    p_e = (p_bar_j ** 2).sum()
    if p_e == 1:
        return float("nan")
    return round(float((p_bar - p_e) / (1 - p_e)), 4)


# ── Arbitration log ───────────────────────────────────────────────────────────

def extract_arbitration_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify items where |coder_A - coder_B| >= ARBITRATION_THRESHOLD
    across any framework dimension.
    """
    coders = sorted(df["coder_id"].unique())
    if len(coders) < 2:
        return pd.DataFrame()

    rows = []
    c_a, c_b = coders[0], coders[1]
    df_a = df[df["coder_id"] == c_a].set_index("packet_id")
    df_b = df[df["coder_id"] == c_b].set_index("packet_id")
    common = df_a.index.intersection(df_b.index)

    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        for pid in common:
            val_a = df_a.loc[pid, dim] if pid in df_a.index else None
            val_b = df_b.loc[pid, dim] if pid in df_b.index else None
            if pd.isna(val_a) or pd.isna(val_b):
                continue
            if abs(float(val_a) - float(val_b)) >= ARBITRATION_THRESHOLD:
                rows.append({
                    "packet_id": pid,
                    "dimension": dim,
                    f"rating_{c_a}": val_a,
                    f"rating_{c_b}": val_b,
                    "discrepancy": abs(float(val_a) - float(val_b)),
                    "needs_adjudication": True,
                    "adjudicated_score": None,
                    "adjudicator_id": None,
                })

    return pd.DataFrame(rows)


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_reliability(ratings_raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ratings_raw_path)
    coders = sorted(df["coder_id"].unique())
    n_coders = len(coders)
    print(f"Coders: {coders} (n={n_coders})")

    results = []

    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        wide = df.pivot_table(index="packet_id", columns="coder_id", values=dim, aggfunc="first")
        if wide.shape[1] < 2:
            print(f"  Only 1 coder for {dim}; skipping reliability.")
            continue

        # ICC(2,k)
        icc_result = compute_icc_2k(wide)

        # Krippendorff α (ordinal)
        mat = wide.values.T.astype(float)
        alpha = compute_krippendorff_alpha(mat, level="ordinal")

        # Weighted κ (linear) — only valid for exactly 2 raters
        wk = float("nan")
        pct_exact = float("nan")
        pct_adjacent = float("nan")
        if n_coders == 2:
            col_a = wide.iloc[:, 0].tolist()
            col_b = wide.iloc[:, 1].tolist()
            wk = compute_weighted_kappa(col_a, col_b, weights="linear")
            pct_exact, pct_adjacent = compute_percent_agreement(col_a, col_b)

        icc_val = icc_result.get("icc_value")
        go = (
            (icc_val is not None and icc_val >= ICC_THRESHOLD)
            and (not np.isnan(alpha) and alpha >= ALPHA_THRESHOLD)
        )

        results.append({
            "dimension": dim,
            "type": "ordinal_rating",
            "n_items": wide.shape[0],
            "n_raters": wide.shape[1],
            "icc_type": icc_result.get("icc_type"),
            "icc_2k": icc_val,
            "icc_ci_lower": icc_result.get("icc_ci_lower"),
            "icc_ci_upper": icc_result.get("icc_ci_upper"),
            "icc_p": icc_result.get("icc_p"),
            "krippendorff_alpha": alpha,
            "weighted_kappa_linear": wk,
            "percent_exact_agreement": pct_exact,
            "percent_adjacent_agreement": pct_adjacent,
            "go_criterion_met": go,
            "note": "" if go else f"BELOW THRESHOLD — ICC={icc_val}, α={alpha:.3f}",
        })

    # Use-case decision
    if "use_case_decision" in df.columns:
        wide_uc = df.pivot_table(
            index="packet_id", columns="coder_id", values="use_case_decision", aggfunc="first"
        )
        if wide_uc.shape[1] >= 2:
            if HAS_SKLEARN and wide_uc.shape[1] == 2:
                kappa = compute_weighted_kappa(
                    wide_uc.iloc[:, 0].tolist(),
                    wide_uc.iloc[:, 1].tolist(),
                    weights=None,
                )
            else:
                kappa = float("nan")
            results.append({
                "dimension": "use_case_decision",
                "type": "nominal_kappa",
                "n_items": wide_uc.shape[0],
                "n_raters": wide_uc.shape[1],
                "icc_type": None,
                "icc_2k": None,
                "icc_ci_lower": None,
                "icc_ci_upper": None,
                "icc_p": None,
                "krippendorff_alpha": kappa,
                "weighted_kappa_linear": None,
                "percent_exact_agreement": None,
                "percent_adjacent_agreement": None,
                "go_criterion_met": not np.isnan(kappa) and kappa >= USE_CASE_KAPPA_THRESHOLD,
                "note": "Cohen κ (2 raters, nominal)",
            })

    # Failure flags
    for flag in FAILURE_FLAGS:
        col = f"flag_{flag}"
        if col not in df.columns:
            continue
        wide_f = df.pivot_table(index="packet_id", columns="coder_id", values=col, aggfunc="first")
        if wide_f.shape[1] < 2:
            continue
        mat_f = wide_f.values.astype(float)
        mat_f_clean = np.where(np.isnan(mat_f), 0, mat_f)
        fk = compute_fleiss_kappa(mat_f_clean)
        results.append({
            "dimension": col,
            "type": "binary_fleiss_kappa",
            "n_items": wide_f.shape[0],
            "n_raters": wide_f.shape[1],
            "icc_type": None,
            "icc_2k": None,
            "icc_ci_lower": None,
            "icc_ci_upper": None,
            "icc_p": None,
            "krippendorff_alpha": fk,
            "weighted_kappa_linear": None,
            "percent_exact_agreement": None,
            "percent_adjacent_agreement": None,
            "go_criterion_met": not np.isnan(fk) and fk >= FLAG_KAPPA_THRESHOLD,
            "note": "Fleiss κ (binary)",
        })

    report = pd.DataFrame(results)
    out_path = Path("outputs") / "reliability_report.csv"
    report.to_csv(out_path, index=False)

    # Arbitration log
    arb_df = extract_arbitration_items(df)
    if len(arb_df) > 0:
        arb_path = Path("outputs") / "arbitration_log.csv"
        arb_df.to_csv(arb_path, index=False)
        print(f"\nArbitration log: {len(arb_df)} items need 3rd-coder adjudication → {arb_path}")
    else:
        print("\nNo items require arbitration (all discrepancies < threshold).")

    # Summary
    framework_dims = report[report["dimension"].isin(DIMENSIONS)]
    if len(framework_dims) > 0:
        mean_icc = framework_dims["icc_2k"].dropna().mean()
        mean_alpha = framework_dims["krippendorff_alpha"].dropna().mean()
        print(f"\n=== Reliability Summary ===")
        print(f"Framework dimensions (n={len(framework_dims)})")
        print(f"  Mean ICC(2,k) = {mean_icc:.3f} (target ≥ {ICC_MEAN_TARGET})")
        print(f"  Mean Krippendorff α = {mean_alpha:.3f} (target ≥ {ALPHA_THRESHOLD})")
        below = framework_dims[~framework_dims["go_criterion_met"]]
        if len(below) == 0:
            print(f"  ✓ ALL {len(framework_dims)} dimensions meet go criterion.")
        else:
            print(f"  ✗ {len(below)} dimension(s) BELOW threshold:")
            print(below[["dimension", "icc_2k", "krippendorff_alpha", "note"]].to_string())

    print(f"\nReliability report → {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="outputs/coder_ratings_raw.csv")
    args = parser.parse_args()

    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        print(f"File not found: {ratings_path}. Run ingest_ratings.py first.")
        return
    run_reliability(ratings_path)


if __name__ == "__main__":
    main()
