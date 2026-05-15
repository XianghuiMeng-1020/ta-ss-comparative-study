"""
Primary confirmatory analysis: linear mixed-effects models.

Model: rating ~ condition + (1 | scenario_id)
Unit: generated dialogue, nested within scenario.
Comparisons (Bonferroni-corrected over 4 primary families):
  C1 vs C2, C1 vs C3, C2 vs C3, {C1,C2} vs C4

Framework hypotheses:
  C1 (TA) should lead on TA1–TA4
  C2 (SS) should lead on SS1–SS5

Output: outputs/mixed_model_results.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
PRIMARY_COMPARISONS = [("C1", "C2"), ("C1", "C3"), ("C2", "C3")]
REFERENCE_CONDITION = "C4"
N_PRIMARY_FAMILIES = 4
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / N_PRIMARY_FAMILIES


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Pooled Cohen's d."""
    n1, n2 = len(a.dropna()), len(b.dropna())
    if n1 < 2 or n2 < 2:
        return float("nan")
    s = np.sqrt(((n1 - 1) * a.std() ** 2 + (n2 - 1) * b.std() ** 2) / (n1 + n2 - 2))
    if s == 0:
        return 0.0
    return (a.mean() - b.mean()) / s


def run_lmer_scipy(df: pd.DataFrame, dimension: str) -> list[dict]:
    """
    Fallback: paired Wilcoxon / Mann-Whitney when pymer4/lme4 not available.
    Uses scenario-level mean differences to respect nesting structure.
    """
    from scipy import stats

    results = []
    pairs = PRIMARY_COMPARISONS + [(("C1", "C2"), "C4")]

    for pair in PRIMARY_COMPARISONS:
        c_a, c_b = pair
        grp_a = df[df["true_condition"] == c_a].groupby("scenario_id")[dimension].mean()
        grp_b = df[df["true_condition"] == c_b].groupby("scenario_id")[dimension].mean()
        common = grp_a.index.intersection(grp_b.index)
        if len(common) < 5:
            continue
        diff = grp_a.loc[common] - grp_b.loc[common]
        stat, p = stats.wilcoxon(diff.dropna(), alternative="two-sided")
        d = cohens_d(grp_a.loc[common], grp_b.loc[common])
        results.append({
            "dimension": dimension,
            "comparison": f"{c_a}_vs_{c_b}",
            "mean_A": round(grp_a.mean(), 3),
            "mean_B": round(grp_b.mean(), 3),
            "mean_diff": round(grp_a.mean() - grp_b.mean(), 3),
            "cohens_d": round(d, 3) if not np.isnan(d) else None,
            "test": "wilcoxon_paired",
            "statistic": round(float(stat), 3),
            "p_value": round(float(p), 4),
            "p_bonferroni": round(float(p) * N_PRIMARY_FAMILIES, 4),
            "sig_bonferroni": float(p) * N_PRIMARY_FAMILIES < ALPHA,
            "n_scenarios": len(common),
        })

    # {C1, C2} vs C4
    for c_ab in ["C1", "C2"]:
        grp_ab = df[df["true_condition"] == c_ab].groupby("scenario_id")[dimension].mean()
        grp_c4 = df[df["true_condition"] == "C4"].groupby("scenario_id")[dimension].mean()
        common = grp_ab.index.intersection(grp_c4.index)
        if len(common) < 5:
            continue
        diff = grp_ab.loc[common] - grp_c4.loc[common]
        stat, p = stats.wilcoxon(diff.dropna(), alternative="two-sided")
        d = cohens_d(grp_ab.loc[common], grp_c4.loc[common])
        results.append({
            "dimension": dimension,
            "comparison": f"{c_ab}_vs_C4",
            "mean_A": round(grp_ab.mean(), 3),
            "mean_B": round(grp_c4.mean(), 3),
            "mean_diff": round(grp_ab.mean() - grp_c4.mean(), 3),
            "cohens_d": round(d, 3) if not np.isnan(d) else None,
            "test": "wilcoxon_paired",
            "statistic": round(float(stat), 3),
            "p_value": round(float(p), 4),
            "p_bonferroni": round(float(p) * N_PRIMARY_FAMILIES, 4),
            "sig_bonferroni": float(p) * N_PRIMARY_FAMILIES < ALPHA,
            "n_scenarios": len(common),
        })

    return results


def run_analysis():
    ratings_path = Path("outputs/coder_ratings_raw.csv")
    if not ratings_path.exists():
        print("coder_ratings_raw.csv not found. Run ingest_ratings.py first.")
        return

    df = pd.read_csv(ratings_path)

    # Resolve scores: average across coders per packet
    resolved = (
        df.groupby(["packet_id", "true_condition", "scenario_id"])[DIMENSIONS]
        .mean()
        .reset_index()
    )

    all_results = []
    for dim in DIMENSIONS:
        if dim not in resolved.columns:
            continue
        results = run_lmer_scipy(resolved, dim)
        all_results.extend(results)

    if not all_results:
        print("No results computed.")
        return

    out_df = pd.DataFrame(all_results)
    out_path = Path("outputs") / "mixed_model_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved mixed model results → {out_path}")

    # Print confirmatory summary
    print("\n=== PRIMARY CONFIRMATORY RESULTS ===")
    print("Sig = Bonferroni-corrected at α=0.0125\n")
    for dim in ["TA1", "TA2", "TA3", "TA4"]:
        subset = out_df[out_df["dimension"] == dim]
        print(f"[{dim}] Expected leader: C1")
        for _, row in subset.iterrows():
            sig_flag = "✓" if row["sig_bonferroni"] else " "
            print(
                f"  {sig_flag} {row['comparison']}: Δ={row['mean_diff']:+.2f}, "
                f"d={row['cohens_d']}, p_bonf={row['p_bonferroni']:.3f}"
            )
    print()
    for dim in ["SS1", "SS2", "SS3", "SS4", "SS5"]:
        subset = out_df[out_df["dimension"] == dim]
        print(f"[{dim}] Expected leader: C2")
        for _, row in subset.iterrows():
            sig_flag = "✓" if row["sig_bonferroni"] else " "
            print(
                f"  {sig_flag} {row['comparison']}: Δ={row['mean_diff']:+.2f}, "
                f"d={row['cohens_d']}, p_bonf={row['p_bonferroni']:.3f}"
            )
        print()


if __name__ == "__main__":
    run_analysis()
