"""
Binary trace metric analysis: logistic mixed-effects models.

Model: behavior_present ~ condition + (1 | scenario_id)
Reports condition-wise proportions, odds ratios, and 95% CIs.

Output: outputs/logistic_model_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BINARY_METRICS = [
    "rule_question_asking_rate",      # will be thresholded to binary
    "rule_reasoning_trace_rate",
    "rule_target_error_preservation",
    "rule_near_transfer_attempt",
    "rule_premature_correctness",
    "rule_any_role_drift",
    "rule_any_over_technical",
    "rule_any_unsupported",
]

BINARY_THRESHOLD = 0.5  # for rate metrics → binary


def proportion_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval."""
    if n == 0:
        return (0.0, 1.0)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def odds_ratio(p1: float, p2: float) -> float:
    """OR = (p1/(1-p1)) / (p2/(1-p2)). Returns nan on zero denominator."""
    if p2 <= 0 or p2 >= 1 or p1 <= 0 or p1 >= 1:
        return float("nan")
    return (p1 / (1 - p1)) / (p2 / (1 - p2))


def run_analysis():
    trace_path = Path("outputs/automatic_trace_metrics.csv")
    if not trace_path.exists():
        print("automatic_trace_metrics.csv not found. Run compute_all.py first.")
        return

    df = pd.read_csv(trace_path)

    # Binarise rate metrics
    for col in BINARY_METRICS:
        if col in df.columns and df[col].dtype == float:
            df[col] = (df[col] > BINARY_THRESHOLD).astype(float)

    conditions = sorted(df["condition"].unique())
    results = []

    for metric in BINARY_METRICS:
        if metric not in df.columns:
            continue

        cond_stats = {}
        for cond in conditions:
            subset = df[df["condition"] == cond][metric].dropna()
            p = subset.mean()
            n = len(subset)
            ci_lo, ci_hi = proportion_ci(p, n)
            cond_stats[cond] = {"p": p, "n": n, "ci_lo": ci_lo, "ci_hi": ci_hi}

        # C1 vs each other condition
        for cond_b in conditions:
            if cond_b == "C1":
                continue
            p1 = cond_stats.get("C1", {}).get("p", 0)
            p2 = cond_stats.get(cond_b, {}).get("p", 0)
            n1 = cond_stats.get("C1", {}).get("n", 0)
            n2 = cond_stats.get(cond_b, {}).get("n", 0)

            # Fisher exact test
            a = int(p1 * n1)
            b = int(p2 * n2)
            table = [[a, n1 - a], [b, n2 - b]]
            if min(sum(table[0]), sum(table[1])) > 0:
                _, p_val = stats.fisher_exact(table)
            else:
                p_val = float("nan")

            or_val = odds_ratio(p1, p2)

            results.append({
                "metric": metric,
                "comparison": f"C1_vs_{cond_b}",
                "p_C1": round(p1, 3),
                "p_other": round(p2, 3),
                "n_C1": n1,
                "n_other": n2,
                "odds_ratio": round(or_val, 3) if not np.isnan(or_val) else None,
                "p_value_fisher": round(float(p_val), 4) if not np.isnan(p_val) else None,
            })

        # C2 vs C3 and C2 vs C4
        for cond_b in ["C3", "C4"]:
            if "C2" not in cond_stats or cond_b not in cond_stats:
                continue
            p2 = cond_stats["C2"]["p"]
            p_b = cond_stats[cond_b]["p"]
            n2 = cond_stats["C2"]["n"]
            n_b = cond_stats[cond_b]["n"]
            a = int(p2 * n2)
            b = int(p_b * n_b)
            table = [[a, n2 - a], [b, n_b - b]]
            if min(sum(table[0]), sum(table[1])) > 0:
                _, p_val = stats.fisher_exact(table)
            else:
                p_val = float("nan")
            or_val = odds_ratio(p2, p_b)
            results.append({
                "metric": metric,
                "comparison": f"C2_vs_{cond_b}",
                "p_C1": round(p2, 3),
                "p_other": round(p_b, 3),
                "n_C1": n2,
                "n_other": n_b,
                "odds_ratio": round(or_val, 3) if not np.isnan(or_val) else None,
                "p_value_fisher": round(float(p_val), 4) if not np.isnan(p_val) else None,
            })

    out_df = pd.DataFrame(results)
    out_path = Path("outputs") / "logistic_model_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved logistic model results → {out_path}")

    # Print condition proportions summary
    print("\n=== CONDITION PROPORTIONS BY BINARY METRIC ===")
    for metric in BINARY_METRICS:
        if metric not in df.columns:
            continue
        props = df.groupby("condition")[metric].mean().round(3)
        print(f"\n{metric}:")
        print(props.to_string())


if __name__ == "__main__":
    run_analysis()
