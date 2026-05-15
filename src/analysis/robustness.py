"""
Robustness analysis.

Re-runs the main condition comparison separately within each split:
  by topic, error_type, difficulty, seed, self_correction_flag

For each split × dimension, reports the main effect direction and
whether it remains significant (non-parametric, no Bonferroni
within robustness — these are exploratory checks).

Output: outputs/robustness_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
PRIMARY_PAIRS = [("C1", "C2"), ("C1", "C3"), ("C2", "C3"), ("C1", "C4"), ("C2", "C4")]
SPLIT_COLS = ["topic", "error_type", "difficulty", "seed", "self_correction_flag"]


def scenario_level_comparison(
    df: pd.DataFrame, dim: str, cond_a: str, cond_b: str
) -> dict:
    """Scenario-matched Wilcoxon signed-rank on dimension means per scenario."""
    grp_a = df[df["true_condition"] == cond_a].groupby("scenario_id")[dim].mean()
    grp_b = df[df["true_condition"] == cond_b].groupby("scenario_id")[dim].mean()
    common = grp_a.index.intersection(grp_b.index)
    if len(common) < 5:
        return {"n": len(common), "mean_diff": None, "p_value": None, "direction": None}
    diff = (grp_a.loc[common] - grp_b.loc[common]).dropna()
    if diff.std() == 0:
        return {"n": len(common), "mean_diff": round(diff.mean(), 3), "p_value": 1.0,
                "direction": "="}
    try:
        stat, p = stats.wilcoxon(diff, alternative="two-sided")
    except Exception:
        return {"n": len(common), "mean_diff": round(diff.mean(), 3), "p_value": None,
                "direction": None}
    direction = "A>B" if diff.mean() > 0 else ("B>A" if diff.mean() < 0 else "=")
    return {
        "n": len(common),
        "mean_diff": round(diff.mean(), 3),
        "p_value": round(float(p), 4),
        "direction": direction,
    }


def run_analysis():
    ratings_path = Path("outputs/coder_ratings_raw.csv")
    scenarios_path = Path("data/scenarios.csv")
    trace_path = Path("outputs/automatic_trace_metrics.csv")

    if not ratings_path.exists():
        print("coder_ratings_raw.csv not found.")
        return

    df = pd.read_csv(ratings_path)
    resolved = (
        df.groupby(["packet_id", "true_condition", "scenario_id"])[DIMENSIONS]
        .mean()
        .reset_index()
    )

    # Merge scenario metadata for splits
    if scenarios_path.exists():
        scen_df = pd.read_csv(scenarios_path)[
            ["scenario_id", "topic", "error_type", "difficulty", "self_correction_flag"]
        ]
        resolved = resolved.merge(scen_df, on="scenario_id", how="left")

    # Merge seed from trace metrics
    if trace_path.exists():
        trace_df = pd.read_csv(trace_path)[["scenario_id", "condition", "seed"]].rename(
            columns={"condition": "true_condition"}
        )
        resolved = resolved.merge(trace_df.drop_duplicates(), on=["scenario_id", "true_condition"], how="left")

    results = []

    for split_col in SPLIT_COLS:
        if split_col not in resolved.columns:
            continue
        split_values = resolved[split_col].dropna().unique()

        for split_val in split_values:
            subset = resolved[resolved[split_col] == split_val]
            if len(subset) < 10:
                continue

            for dim in DIMENSIONS:
                if dim not in subset.columns:
                    continue
                for cond_a, cond_b in PRIMARY_PAIRS:
                    res = scenario_level_comparison(subset, dim, cond_a, cond_b)
                    results.append({
                        "split_col": split_col,
                        "split_val": split_val,
                        "dimension": dim,
                        "comparison": f"{cond_a}_vs_{cond_b}",
                        "n_scenarios": res["n"],
                        "mean_diff": res["mean_diff"],
                        "p_value": res["p_value"],
                        "direction": res["direction"],
                        "sig_p05": (
                            res["p_value"] < 0.05 if res["p_value"] is not None else False
                        ),
                    })

    if not results:
        print("No robustness results computed.")
        return

    out_df = pd.DataFrame(results)
    out_path = Path("outputs") / "robustness_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved robustness results ({len(out_df)} rows) → {out_path}")

    # Consistency summary
    print("\n=== ROBUSTNESS CONSISTENCY ===")
    print("For each dimension × comparison: fraction of splits where direction is consistent")
    consistency_rows = []
    for dim in DIMENSIONS:
        for cond_a, cond_b in [("C1", "C2"), ("C2", "C3")]:
            comp = f"{cond_a}_vs_{cond_b}"
            rows = out_df[(out_df["dimension"] == dim) & (out_df["comparison"] == comp)]
            if rows.empty:
                continue
            directions = rows["direction"].dropna()
            if len(directions) == 0:
                continue
            most_common = directions.value_counts().idxmax()
            pct = directions.value_counts(normalize=True).max()
            print(f"  {dim} {comp}: {most_common} in {pct*100:.0f}% of splits")
            consistency_rows.append({
                "dimension": dim,
                "comparison": comp,
                "dominant_direction": most_common,
                "consistency_pct": round(pct * 100, 1),
            })

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(Path("outputs/robustness_consistency.csv"), index=False)

    # Heatmap: direction consistency per dimension × comparison
    _plot_robustness_heatmap(consistency_df)


def _plot_robustness_heatmap(consistency_df: pd.DataFrame) -> None:
    """Plot heatmap of direction consistency rates across dimensions and comparisons."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("matplotlib not available; skipping heatmap.")
        return

    if consistency_df.empty:
        return

    pivot = consistency_df.pivot_table(
        index="dimension", columns="comparison", values="consistency_pct"
    )
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.8), 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=8,
                        color="black" if val > 70 else "white")

    plt.colorbar(im, ax=ax, label="Direction Consistency (%)")
    ax.set_title(
        "Robustness: Direction Consistency Across Splits\n"
        "(topic, error_type, difficulty, seed, model)",
        fontsize=10
    )
    plt.tight_layout()
    out_path = Path("outputs/robustness_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Robustness heatmap → {out_path}")


if __name__ == "__main__":
    run_analysis()
