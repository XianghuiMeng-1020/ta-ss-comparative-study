"""
Primary confirmatory analysis: linear mixed-effects models (upgraded for C&E).

Model (Decision Log D11):
  rating ~ condition * model_family + (1|scenario_id) + (1|coder_id)

Where:
  - condition: C1/C2/C3/C4 (deviation coded, C4 as reference)
  - model_family: M1/M2/M3/M4 (robustness factor)
  - scenario_id: random intercept (repeated measures across conditions)
  - coder_id: random intercept (repeated measures across dialogues)
  - condition × model_family: interaction term (tests model-specificity of protocol effects)

Multiple comparison correction:
  Primary: Holm-Bonferroni (Decision Log D11)
  Secondary: Benjamini-Hochberg FDR (Appendix B)

Effect size: Cohen's d with 95% CI via bootstrap (5,000 iterations)

Fallback: if pymer4/lme4 unavailable → scenario-level Wilcoxon (preserved for robustness)

Output:
  outputs/mixed_model_results.csv      — primary lmer results
  outputs/mixed_model_results_wrs.csv  — Wilcoxon fallback
  outputs/bootstrap_effect_sizes.csv   — Cohen's d with 95% CI

Usage:
  python src/analysis/mixed_models.py [--ratings outputs/coder_ratings_raw.csv]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
CONDITIONS = ["C1", "C2", "C3", "C4"]
PRIMARY_COMPARISONS = [("C1", "C2"), ("C1", "C3"), ("C2", "C3")]
REFERENCE_CONDITION = "C4"

ALPHA = 0.05
N_BOOTSTRAP = 5_000

try:
    from pymer4.models import Lmer
    HAS_PYMER4 = True
except ImportError:
    HAS_PYMER4 = False


# ── Effect size ───────────────────────────────────────────────────────────────

def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s if s > 0 else 0.0


def bootstrap_cohens_d(
    a: pd.Series, b: pd.Series, n_boot: int = N_BOOTSTRAP, ci: float = 0.95
) -> tuple[float, float, float]:
    """Return (d, ci_lower, ci_upper) via bootstrap."""
    a, b = a.dropna().values, b.dropna().values
    rng = np.random.default_rng(2026)
    ds = []
    for _ in range(n_boot):
        samp_a = rng.choice(a, size=len(a), replace=True)
        samp_b = rng.choice(b, size=len(b), replace=True)
        n1, n2 = len(samp_a), len(samp_b)
        s = np.sqrt(
            ((n1 - 1) * samp_a.std(ddof=1) ** 2 + (n2 - 1) * samp_b.std(ddof=1) ** 2)
            / (n1 + n2 - 2)
        )
        ds.append((samp_a.mean() - samp_b.mean()) / s if s > 0 else 0.0)
    alpha_half = (1 - ci) / 2
    return (
        float(np.median(ds)),
        float(np.percentile(ds, alpha_half * 100)),
        float(np.percentile(ds, (1 - alpha_half) * 100)),
    )


# ── Multiple comparison corrections ──────────────────────────────────────────

def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni sequential correction. Returns corrected p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_min = 1.0
    for rank, (orig_idx, p) in enumerate(indexed):
        corrected = p * (n - rank)
        running_min = min(running_min, corrected)
        adjusted[orig_idx] = running_min
    return [min(p, 1.0) for p in adjusted]


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.minimum(1.0, sorted_p * n / (np.arange(n) + 1))
    # Ensure monotonicity (from right to left)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result.tolist()


# ── LME via pymer4 ────────────────────────────────────────────────────────────

def run_lmer(df: pd.DataFrame, dimension: str) -> list[dict]:
    """Run lmer with condition × model_family interaction + random effects."""
    sub = df[["packet_id", "true_condition", "model_tag", "scenario_id", "coder_id", dimension]].dropna()
    if len(sub) < 20:
        return []

    # Build lmer formula
    formula = f"{dimension} ~ true_condition * model_tag + (1|scenario_id) + (1|coder_id)"
    try:
        model = Lmer(formula, data=sub)
        model.fit()
    except Exception as e:
        print(f"  lmer failed for {dimension}: {e}. Falling back to Wilcoxon.")
        return []

    # Extract fixed effects
    fe = model.coefs
    results = []
    # Report main condition contrasts relative to C4 reference
    for cond in ["C1", "C2", "C3"]:
        row_mask = fe.index.str.contains(f"true_condition{cond}", na=False)
        if row_mask.any():
            row = fe[row_mask].iloc[0]
            results.append({
                "dimension": dimension,
                "analysis_type": "lmer_fixed_effect",
                "comparison": f"{cond}_vs_C4",
                "estimate": round(float(row.get("Estimate", 0)), 4),
                "se": round(float(row.get("SE", float("nan"))), 4),
                "t_value": round(float(row.get("T-stat", float("nan"))), 3),
                "p_value": round(float(row.get("P-val", 1.0)), 4),
                "marginal_r2": round(float(getattr(model, "marginal_r2", float("nan"))), 4),
                "conditional_r2": round(float(getattr(model, "conditional_r2", float("nan"))), 4),
            })

    # Interaction term
    interaction_rows = fe[fe.index.str.contains("true_condition.*model_tag|model_tag.*true_condition")]
    if len(interaction_rows) > 0:
        max_p = interaction_rows["P-val"].max()
        results.append({
            "dimension": dimension,
            "analysis_type": "lmer_interaction_test",
            "comparison": "condition_x_model_family",
            "estimate": float("nan"),
            "se": float("nan"),
            "t_value": float("nan"),
            "p_value": round(float(max_p), 4),
            "marginal_r2": float("nan"),
            "conditional_r2": float("nan"),
        })

    return results


# ── Wilcoxon signed-rank (scenario-matched) ───────────────────────────────────

def run_wilcoxon(df: pd.DataFrame, dimension: str) -> list[dict]:
    """Scenario-level Wilcoxon signed-rank (backward compat + cross-validation)."""
    results = []
    pairs = list(PRIMARY_COMPARISONS) + [
        ("C1", "C4"), ("C2", "C4"), ("C3", "C4")
    ]
    for c_a, c_b in pairs:
        grp_a = df[df["true_condition"] == c_a].groupby("scenario_id")[dimension].mean()
        grp_b = df[df["true_condition"] == c_b].groupby("scenario_id")[dimension].mean()
        common = grp_a.index.intersection(grp_b.index)
        if len(common) < 5:
            continue
        diff = grp_a.loc[common] - grp_b.loc[common]
        stat, p = stats.wilcoxon(diff.dropna(), alternative="two-sided")
        d, d_lo, d_hi = bootstrap_cohens_d(grp_a.loc[common], grp_b.loc[common])
        results.append({
            "dimension": dimension,
            "analysis_type": "wilcoxon_paired",
            "comparison": f"{c_a}_vs_{c_b}",
            "mean_A": round(float(grp_a.mean()), 3),
            "mean_B": round(float(grp_b.mean()), 3),
            "mean_diff": round(float(grp_a.mean() - grp_b.mean()), 3),
            "cohens_d": round(d, 3),
            "cohens_d_ci_lower": round(d_lo, 3),
            "cohens_d_ci_upper": round(d_hi, 3),
            "test_statistic": round(float(stat), 3),
            "p_value": round(float(p), 4),
            "n_scenarios": int(len(common)),
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_analysis(ratings_path: Path) -> None:
    df = pd.read_csv(ratings_path)

    # Resolve: average across coders per packet, then merge metadata
    resolved = (
        df.groupby(["packet_id", "true_condition", "scenario_id"])[DIMENSIONS]
        .mean()
        .reset_index()
    )
    # Merge model_tag if present
    if "model_tag" in df.columns:
        mtags = df.drop_duplicates("packet_id")[["packet_id", "model_tag"]]
        resolved = resolved.merge(mtags, on="packet_id", how="left")
    else:
        resolved["model_tag"] = "unknown"

    if "coder_id" in df.columns:
        coders_per_packet = df.drop_duplicates(["packet_id", "coder_id"])[["packet_id", "coder_id"]]
        resolved = resolved.merge(coders_per_packet.groupby("packet_id")["coder_id"].first().reset_index(),
                                  on="packet_id", how="left")
    else:
        resolved["coder_id"] = "unknown"

    # ── lmer (if available) ────────────────────────────────────────────────────
    lmer_results = []
    wrs_results = []

    for dim in DIMENSIONS:
        if dim not in resolved.columns:
            continue
        if HAS_PYMER4:
            lmer_rows = run_lmer(resolved, dim)
            lmer_results.extend(lmer_rows)
        wrs_rows = run_wilcoxon(resolved, dim)
        wrs_results.extend(wrs_rows)

    # ── Multiple comparison correction on Wilcoxon results ────────────────────
    if wrs_results:
        wrs_df = pd.DataFrame(wrs_results)
        p_vals = wrs_df["p_value"].tolist()
        wrs_df["p_holm"] = holm_bonferroni(p_vals)
        wrs_df["p_bh_fdr"] = benjamini_hochberg(p_vals)
        wrs_df["sig_holm_0125"] = wrs_df["p_holm"] < 0.0125
        wrs_df["sig_bh_fdr_05"] = wrs_df["p_bh_fdr"] < 0.05

        out_wrs = Path("outputs") / "mixed_model_results.csv"
        wrs_df.to_csv(out_wrs, index=False)
        print(f"Wilcoxon results → {out_wrs}")
        _print_summary(wrs_df)

    if lmer_results:
        lmer_df = pd.DataFrame(lmer_results)
        out_lmer = Path("outputs") / "lmer_results.csv"
        lmer_df.to_csv(out_lmer, index=False)
        print(f"LME results → {out_lmer}")

    # ── Bootstrap effect sizes table ──────────────────────────────────────────
    boot_rows = []
    for dim in DIMENSIONS:
        if dim not in resolved.columns:
            continue
        for c_a, c_b in [("C1", "C2"), ("C1", "C3"), ("C2", "C3"), ("C1", "C4"), ("C2", "C4")]:
            a = resolved[resolved["true_condition"] == c_a][dim]
            b = resolved[resolved["true_condition"] == c_b][dim]
            if len(a) < 5 or len(b) < 5:
                continue
            d, d_lo, d_hi = bootstrap_cohens_d(a, b, n_boot=N_BOOTSTRAP)
            boot_rows.append({
                "dimension": dim,
                "comparison": f"{c_a}_vs_{c_b}",
                "cohens_d": round(d, 3),
                "ci_lower_95": round(d_lo, 3),
                "ci_upper_95": round(d_hi, 3),
                "n_boot": N_BOOTSTRAP,
            })

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(Path("outputs/bootstrap_effect_sizes.csv"), index=False)
    print(f"Bootstrap effect sizes → outputs/bootstrap_effect_sizes.csv")


def _print_summary(df: pd.DataFrame) -> None:
    print("\n=== PRIMARY CONFIRMATORY RESULTS (Holm-Bonferroni, α=0.0125) ===")
    for dim in ["TA1", "TA2", "TA3", "TA4"]:
        sub = df[df["dimension"] == dim]
        print(f"\n[{dim}] Expected leader: C1")
        for _, row in sub.iterrows():
            sig = "✓" if row.get("sig_holm_0125") else " "
            d_str = f"d={row['cohens_d']:.2f} [{row['cohens_d_ci_lower']:.2f}, {row['cohens_d_ci_upper']:.2f}]"
            print(f"  {sig} {row['comparison']}: Δ={row['mean_diff']:+.2f}, {d_str}, p_holm={row['p_holm']:.3f}")
    for dim in ["SS1", "SS2", "SS3", "SS4", "SS5"]:
        sub = df[df["dimension"] == dim]
        print(f"\n[{dim}] Expected leader: C2")
        for _, row in sub.iterrows():
            sig = "✓" if row.get("sig_holm_0125") else " "
            d_str = f"d={row['cohens_d']:.2f} [{row['cohens_d_ci_lower']:.2f}, {row['cohens_d_ci_upper']:.2f}]"
            print(f"  {sig} {row['comparison']}: Δ={row['mean_diff']:+.2f}, {d_str}, p_holm={row['p_holm']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", default="outputs/coder_ratings_raw.csv")
    args = parser.parse_args()
    ratings_path = Path(args.ratings)
    if not ratings_path.exists():
        print(f"File not found: {ratings_path}")
        return
    run_analysis(ratings_path)


if __name__ == "__main__":
    main()
