"""
TOST equivalence testing: C3 vs C4 on TA and SS dimensions.

Two one-sided t-tests (TOST) per Lakens (2017).
Equivalence bound: d = 0.3 (pre-registered as a priori negligible effect, D11).

Null hypothesis: C3 and C4 differ by more than d = 0.3 (not equivalent).
Alternative (equivalence): C3 and C4 differ by less than d = 0.3 (equivalent).

Equivalence is concluded when BOTH one-sided tests are significant at α = 0.05.

Also reports power analysis: minimum detectable effect at current sample size.

Output: outputs/equivalence_test_results.csv

Usage:
  python src/analysis/equivalence_test.py [--ratings outputs/coder_ratings_raw.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
EQUIVALENCE_BOUND_D = 0.30  # Pre-registered (Decision Log D11)
ALPHA = 0.05


def tost(a: pd.Series, b: pd.Series, bound_d: float = EQUIVALENCE_BOUND_D) -> dict:
    """
    Two one-sided t-tests (TOST).
    Returns dict with p_lower, p_upper, equiv_p, is_equivalent, obs_d, bound_se.

    Per Lakens (2017), Supplementary Material:
      t_lower = (M_a - M_b - delta_upper) / SE_pooled
      t_upper = (M_a - M_b - delta_lower) / SE_pooled
    where delta_lower = -bound_d * pooled_sd, delta_upper = +bound_d * pooled_sd.
    """
    a, b = a.dropna(), b.dropna()
    n_a, n_b = len(a), len(b)
    if n_a < 5 or n_b < 5:
        return {
            "n_a": n_a, "n_b": n_b, "obs_d": float("nan"),
            "p_lower": float("nan"), "p_upper": float("nan"),
            "equiv_p": float("nan"), "is_equivalent": False,
            "interpretation": "insufficient_data",
        }

    m_a, m_b = a.mean(), b.mean()
    s_pool = np.sqrt(
        ((n_a - 1) * a.std(ddof=1) ** 2 + (n_b - 1) * b.std(ddof=1) ** 2)
        / (n_a + n_b - 2)
    )
    if s_pool == 0:
        return {
            "n_a": n_a, "n_b": n_b, "obs_d": 0.0,
            "p_lower": 0.0, "p_upper": 0.0,
            "equiv_p": 0.0, "is_equivalent": True,
            "interpretation": "zero_variance_trivially_equivalent",
        }

    obs_d = (m_a - m_b) / s_pool
    delta = bound_d * s_pool
    se = s_pool * np.sqrt(1 / n_a + 1 / n_b)
    df = n_a + n_b - 2

    # Lower test: H0: Δ ≤ -delta (test if mean_diff > -delta, i.e., not too negative)
    t_lower = (m_a - m_b - (-delta)) / se
    p_lower = stats.t.sf(t_lower, df=df)  # one-sided upper tail

    # Upper test: H0: Δ ≥ +delta (test if mean_diff < +delta, i.e., not too positive)
    t_upper = (m_a - m_b - delta) / se
    p_upper = stats.t.cdf(t_upper, df=df)  # one-sided lower tail

    equiv_p = max(p_lower, p_upper)
    is_equiv = equiv_p < ALPHA

    interpretation = "equivalent" if is_equiv else "not_equivalent"
    if abs(obs_d) > bound_d and not is_equiv:
        interpretation = "practically_different"

    return {
        "n_a": n_a,
        "n_b": n_b,
        "mean_a": round(float(m_a), 4),
        "mean_b": round(float(m_b), 4),
        "mean_diff": round(float(m_a - m_b), 4),
        "pooled_sd": round(float(s_pool), 4),
        "obs_d": round(float(obs_d), 4),
        "equivalence_bound_d": bound_d,
        "t_lower": round(float(t_lower), 4),
        "t_upper": round(float(t_upper), 4),
        "p_lower": round(float(p_lower), 4),
        "p_upper": round(float(p_upper), 4),
        "equiv_p": round(float(equiv_p), 4),
        "is_equivalent": bool(is_equiv),
        "df": df,
        "interpretation": interpretation,
    }


def min_detectable_effect(n_per_group: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Approximate minimum detectable d at given power (two-sample t-test)."""
    from scipy.stats import t as t_dist, norm
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return (z_alpha + z_beta) * np.sqrt(2 / n_per_group)


def run_equivalence(ratings_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ratings_path)

    # Average across coders
    resolved = (
        df.groupby(["packet_id", "true_condition"])[DIMENSIONS].mean().reset_index()
    )

    results = []
    for dim in DIMENSIONS:
        if dim not in resolved.columns:
            continue
        c3 = resolved[resolved["true_condition"] == "C3"][dim]
        c4 = resolved[resolved["true_condition"] == "C4"][dim]

        res = tost(c3, c4, EQUIVALENCE_BOUND_D)
        res["dimension"] = dim
        res["comparison"] = "C3_vs_C4"

        # Power analysis
        n_est = max(res.get("n_a", 0), 1)
        mde = min_detectable_effect(n_est)
        res["min_detectable_d_80pct"] = round(mde, 3)
        results.append(res)

    result_df = pd.DataFrame(results)
    out_path = Path("outputs/equivalence_test_results.csv")
    result_df.to_csv(out_path, index=False)

    print("\n=== TOST EQUIVALENCE TEST: C3 vs C4 ===")
    print(f"Equivalence bound: d = ±{EQUIVALENCE_BOUND_D} (pre-registered)")
    print(f"α = {ALPHA}\n")
    for _, row in result_df.iterrows():
        status = "EQUIVALENT" if row["is_equivalent"] else "NOT EQUIVALENT"
        print(
            f"  {row['dimension']}: obs_d = {row['obs_d']:.3f}, "
            f"equiv_p = {row['equiv_p']:.3f}  [{status}]  [{row['interpretation']}]"
        )

    n_equiv = result_df["is_equivalent"].sum()
    print(f"\n→ {n_equiv}/{len(result_df)} dimensions support C3 ≈ C4 equivalence claim.")
    print(f"\nEquivalence results → {out_path}")
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description="TOST equivalence testing: C3 vs C4")
    parser.add_argument("--ratings", default="outputs/coder_ratings_raw.csv")
    args = parser.parse_args()
    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        print(f"File not found: {ratings_path}")
        return
    run_equivalence(ratings_path)


if __name__ == "__main__":
    main()
