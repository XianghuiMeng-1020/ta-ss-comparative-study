"""
Inter-rater reliability analysis.

Computes:
  - ICC(2,k) and Krippendorff α (ordinal) for TA1–TA4, SS1–SS5
  - Cohen κ for use_case_decision (when exactly 2 coders)
  - Fleiss κ for failure flags (binary)

Go criterion: ICC ≥ 0.6 or α ≥ 0.6 for core 9 dimensions.
Output: outputs/reliability_report.csv
"""

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

ICC_THRESHOLD = 0.60
ALPHA_THRESHOLD = 0.60


def compute_icc(ratings_wide: pd.DataFrame, dimension: str) -> dict:
    """
    Compute ICC(2,k) using pingouin.
    ratings_wide: one row per packet, one column per coder.
    """
    if not HAS_PINGOUIN:
        return {"icc_value": None, "icc_ci_lower": None, "icc_ci_upper": None}

    # ratings_wide index = packet_id; columns = coder_ids
    df_long = ratings_wide.reset_index().melt(
        id_vars=ratings_wide.index.name or "index",
        var_name="coder",
        value_name="rating",
    ).dropna()
    df_long = df_long.rename(columns={df_long.columns[0]: "target"})
    icc_res = pg.intraclass_corr(
        data=df_long, targets="target", raters="coder", ratings="rating",
        nan_policy="omit",
    )
    # pingouin >= 0.5: Type names are ICC(A,k), ICC(C,k), ICC(1,k) etc.
    # Prefer ICC(A,k) = two-way random, multiple raters (equivalent to ICC2k)
    for icc_type in ["ICC(A,k)", "ICC(C,k)", "ICC(A,1)", "ICC(1,1)"]:
        match = icc_res[icc_res["Type"] == icc_type]
        if not match.empty:
            row = match.iloc[0]
            ci = row["CI95"] if "CI95" in row.index else [None, None]
            return {
                "icc_value": round(float(row["ICC"]), 4),
                "icc_ci_lower": round(float(ci[0]), 4) if ci[0] is not None else None,
                "icc_ci_upper": round(float(ci[1]), 4) if ci[1] is not None else None,
            }
    row = icc_res.iloc[0]
    ci = row.get("CI95", [None, None])
    return {
        "icc_value": round(float(row["ICC"]), 4),
        "icc_ci_lower": round(float(ci[0]), 4) if ci[0] is not None else None,
        "icc_ci_upper": round(float(ci[1]), 4) if ci[1] is not None else None,
    }


def compute_krippendorff_alpha(ratings_matrix: np.ndarray, level: str = "ordinal") -> float:
    """
    Compute Krippendorff α.
    ratings_matrix: shape (n_raters, n_items), np.nan for missing.
    """
    if not HAS_KRIPPENDORFF:
        return float("nan")
    try:
        return round(krippendorff.alpha(reliability_data=ratings_matrix, level_of_measurement=level), 4)
    except Exception as e:
        print(f"Krippendorff error: {e}")
        return float("nan")


def compute_cohen_kappa(c1_ratings: list, c2_ratings: list) -> float:
    if not HAS_SKLEARN:
        return float("nan")
    valid = [(a, b) for a, b in zip(c1_ratings, c2_ratings) if pd.notna(a) and pd.notna(b)]
    if not valid:
        return float("nan")
    a, b = zip(*valid)
    return round(cohen_kappa_score(a, b), 4)


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Fleiss κ for binary ratings.
    ratings_matrix: shape (n_items, n_raters), values in {0, 1}.
    """
    n_items, n_raters = ratings_matrix.shape
    # P_j: proportion rating each category
    n_cat = 2
    # Build category counts matrix (n_items × n_cat)
    cat_counts = np.zeros((n_items, n_cat))
    for i in range(n_items):
        row = ratings_matrix[i]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        cat_counts[i, 0] = np.sum(valid == 0)
        cat_counts[i, 1] = np.sum(valid == 1)

    n_i = cat_counts.sum(axis=1)
    p_bar_j = cat_counts.sum(axis=0) / (n_i.sum())
    p_bar = ((cat_counts * (cat_counts - 1)).sum(axis=1) / (n_i * (n_i - 1))).mean()
    p_e = (p_bar_j ** 2).sum()
    if p_e == 1:
        return float("nan")
    kappa = (p_bar - p_e) / (1 - p_e)
    return round(float(kappa), 4)


def run_reliability(ratings_raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ratings_raw_path)
    coders = sorted(df["coder_id"].unique())
    print(f"Coders: {coders}")
    n_coders = len(coders)

    results = []

    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue
        # Pivot to wide: one row per packet, one col per coder
        wide = df.pivot_table(index="packet_id", columns="coder_id", values=dim, aggfunc="first")
        if wide.shape[1] < 2:
            print(f"Only 1 coder for {dim}; skipping ICC.")
            continue

        icc_res = compute_icc(wide, dim)
        mat = wide.values.T  # (n_raters, n_items)
        alpha = compute_krippendorff_alpha(mat.astype(float), level="ordinal")

        go = (
            (icc_res["icc_value"] is not None and icc_res["icc_value"] >= ICC_THRESHOLD)
            or alpha >= ALPHA_THRESHOLD
        )

        results.append({
            "dimension": dim,
            "type": "ordinal_rating",
            "n_items": wide.shape[0],
            "n_raters": wide.shape[1],
            "icc_2k": icc_res["icc_value"],
            "icc_ci_lower": icc_res["icc_ci_lower"],
            "icc_ci_upper": icc_res["icc_ci_upper"],
            "krippendorff_alpha": alpha,
            "go_criterion_met": go,
            "note": "" if go else "BELOW THRESHOLD — revise anchors",
        })

    # Use-case decision — Cohen κ (2 coders) or descriptive
    if "use_case_decision" in df.columns and n_coders == 2:
        wide_uc = df.pivot_table(
            index="packet_id", columns="coder_id", values="use_case_decision", aggfunc="first"
        )
        if wide_uc.shape[1] == 2:
            kappa = compute_cohen_kappa(
                wide_uc.iloc[:, 0].tolist(), wide_uc.iloc[:, 1].tolist()
            )
            results.append({
                "dimension": "use_case_decision",
                "type": "nominal_kappa",
                "n_items": wide_uc.shape[0],
                "n_raters": 2,
                "icc_2k": None,
                "icc_ci_lower": None,
                "icc_ci_upper": None,
                "krippendorff_alpha": kappa,
                "go_criterion_met": kappa >= 0.6 if not np.isnan(kappa) else False,
                "note": "Cohen κ (2 raters)",
            })

    # Failure flags — Fleiss κ
    for flag in FAILURE_FLAGS:
        col = f"flag_{flag}"
        if col not in df.columns:
            continue
        wide_f = df.pivot_table(
            index="packet_id", columns="coder_id", values=col, aggfunc="first"
        )
        if wide_f.shape[1] < 2:
            continue
        mat_f = wide_f.values.astype(float)
        # Impute NaN as 0 for Fleiss (conservative)
        mat_f_clean = np.where(np.isnan(mat_f), 0, mat_f)
        fk = compute_fleiss_kappa(mat_f_clean)
        results.append({
            "dimension": col,
            "type": "binary_fleiss_kappa",
            "n_items": wide_f.shape[0],
            "n_raters": wide_f.shape[1],
            "icc_2k": None,
            "icc_ci_lower": None,
            "icc_ci_upper": None,
            "krippendorff_alpha": fk,
            "go_criterion_met": fk >= 0.6 if not np.isnan(fk) else False,
            "note": "Fleiss κ (binary)",
        })

    report = pd.DataFrame(results)
    out_path = Path("outputs") / "reliability_report.csv"
    report.to_csv(out_path, index=False)
    print(f"\nSaved reliability report → {out_path}")

    failed = report[~report["go_criterion_met"]]
    if len(failed) > 0:
        print(f"\n⚠  {len(failed)} dimension(s) BELOW threshold:")
        print(failed[["dimension", "icc_2k", "krippendorff_alpha", "note"]].to_string())
    else:
        print("\n✓ All dimensions meet go criterion.")

    return report


def main():
    ratings_path = Path("outputs/coder_ratings_raw.csv")
    if not ratings_path.exists():
        print(f"File not found: {ratings_path}. Run ingest_ratings.py first.")
        return
    run_reliability(ratings_path)


if __name__ == "__main__":
    main()
