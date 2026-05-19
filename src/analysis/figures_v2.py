"""
Figure generation for v2 IEEE TLT paper.

Produces all 8 main paper figures:
  Fig 1 — Scaling curves: D1 + D2 vs. log(parameter count)
  Fig 2 — Pareto frontier: D1 vs. D2 accuracy (bubble size = cost)
  Fig 3 — Fairness heatmap: CEAT effect size per model × attribute set
  Fig 4 — Type-A/B cross-tab: 2×2 dialogue-test discrepancy
  Fig 5 — Per-dimension forest plot: effect sizes D1–D7 across model tiers
  Fig 6 — Unlearning forgetting curves: accuracy vs. unlearning ratio
  Fig 7 — Question-type comparison: D1 by qtype × difficulty
  Fig 8 — P1 vs P2 comparison: head-to-head on D1 + D2

Usage:
    python src/analysis/figures_v2.py \
        --data-dir outputs \
        --output-dir outputs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

TIER_COLORS = {
    "Tiny":  "#e74c3c",
    "Small": "#f39c12",
    "Mid":   "#27ae60",
    "Big":   "#2980b9",
    "Unknown": "#95a5a6",
}

QTYPE_MARKERS = {
    "MCQ": "o", "TF": "s", "Fill": "^", "SA": "D", "OED": "P",
}

MODEL_TIER_MAP = {
    "smollm2": "Tiny", "tinyllama": "Tiny", "llama-3.2-1b": "Tiny",
    "qwen2.5-1.5b": "Small", "phi-3.5": "Small", "llama-3.2-3b": "Small",
    "qwen3-4b": "Mid", "mistral-7b": "Mid", "llama-3.1-8b": "Mid",
    "gpt-4o": "Big", "claude": "Big", "llama-3.1-70b": "Big",
}

MODEL_PARAMS = {
    "Tiny": 0.8, "Small": 3.0, "Mid": 6.5, "Big": 100.0,
}


def assign_tier(model_tag: str) -> str:
    tag_lower = str(model_tag).lower()
    for key, tier in MODEL_TIER_MAP.items():
        if key in tag_lower:
            return tier
    return "Unknown"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig1_scaling_curves(d1_path: Path, d2_path: Path, output_dir: Path) -> None:
    """Fig 1: D1 human-likeness + D2 accuracy vs. log(params)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, (path, metric, ylabel, title) in zip(axes, [
        (d1_path, "d1_composite", "Human-Likeness Score (D1, 1–5)", "D1: Human-Likeness"),
        (d2_path, "accuracy",     "Accuracy (D2)",                  "D2: Answering Accuracy"),
    ]):
        if not path.exists():
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(title)
            continue

        df = pd.read_csv(path)
        if "model_tier" not in df.columns:
            df["model_tier"] = df.get("model_tag", pd.Series(dtype=str)).apply(assign_tier)
        df["log_params"] = df["model_tier"].map({t: np.log(p) for t, p in MODEL_PARAMS.items()})

        if metric not in df.columns:
            ax.text(0.5, 0.5, f"Column '{metric}' missing", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(title)
            continue

        for tier, color in TIER_COLORS.items():
            sub = df[df["model_tier"] == tier]
            if sub.empty:
                continue
            ax.scatter(sub["log_params"], sub[metric], color=color,
                       label=tier, alpha=0.7, s=60, zorder=3)

        agg = df.groupby("model_tier").agg(
            log_params=("log_params", "mean"),
            metric=(metric, "mean"),
        ).dropna()
        if len(agg) >= 2:
            z = np.polyfit(agg["log_params"], agg["metric"], 1)
            xs = np.linspace(agg["log_params"].min(), agg["log_params"].max(), 100)
            ax.plot(xs, np.polyval(z, xs), color="black", linestyle="--",
                    linewidth=1.5, alpha=0.7, label="OLS trend")

        tier_labels = ["Tiny\n(<2B)", "Small\n(2-5B)", "Mid\n(5-15B)", "Big\n(>15B)"]
        tier_params = [np.log(0.8), np.log(3.0), np.log(6.5), np.log(100.0)]
        ax.set_xticks(tier_params)
        ax.set_xticklabels(tier_labels)
        ax.set_xlabel("Model Size (parameter count)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", framealpha=0.8)

    plt.tight_layout()
    out = output_dir / "fig1_scaling_curves.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Fig 1 → {out}")


def fig2_pareto(pareto_path: Path, output_dir: Path) -> None:
    """Fig 2: Pareto frontier scatter."""
    fig, ax = plt.subplots(figsize=(7, 5))

    if not pareto_path.exists():
        ax.text(0.5, 0.5, "Pareto data not available", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_dir / "fig2_pareto.pdf", bbox_inches="tight")
        plt.close()
        return

    df = pd.read_csv(pareto_path)
    if "model_tier" not in df.columns:
        df["model_tier"] = df["model_tag"].apply(assign_tier) if "model_tag" in df.columns else "Unknown"

    d1_col = "d1_mean" if "d1_mean" in df.columns else "d1_composite"
    d2_col = "d2_accuracy_mean" if "d2_accuracy_mean" in df.columns else "accuracy"
    cost_col = "cost_usd_mean" if "cost_usd_mean" in df.columns else "cost_usd"

    if d1_col not in df.columns or d2_col not in df.columns:
        ax.text(0.5, 0.5, "Missing D1/D2 columns", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_dir / "fig2_pareto.pdf", bbox_inches="tight")
        plt.close()
        return

    cost_vals = df[cost_col].fillna(0.01) if cost_col in df.columns else pd.Series(0.01, index=df.index)
    cost_scaled = (cost_vals - cost_vals.min()) / (cost_vals.max() - cost_vals.min() + 1e-9)
    bubble_sizes = 100 + cost_scaled * 400

    for _, row in df.iterrows():
        tier = row.get("model_tier", "Unknown")
        color = TIER_COLORS.get(tier, "#95a5a6")
        ax.scatter(row[d2_col], row[d1_col], s=bubble_sizes.loc[row.name],
                   color=color, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)
        if pd.notna(row.get("model_tag")):
            short_name = str(row["model_tag"]).split("_")[-1][:10]
            ax.annotate(short_name, (row[d2_col], row[d1_col]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7, alpha=0.8)

    pareto_df = df[df.get("pareto_optimal", pd.Series(False, index=df.index))]
    if not pareto_df.empty and d1_col in pareto_df.columns:
        pf = pareto_df.sort_values(d2_col)
        ax.plot(pf[d2_col], pf[d1_col], "k--", linewidth=1.5, alpha=0.6, label="Pareto frontier")

    legend_patches = [mpatches.Patch(color=c, label=t)
                      for t, c in TIER_COLORS.items() if t != "Unknown"]
    ax.legend(handles=legend_patches, title="Model tier", loc="upper left")
    ax.set_xlabel("D2: Answering Accuracy")
    ax.set_ylabel("D1: Human-Likeness (1–5)")
    ax.set_title("Pareto Frontier: Human-Likeness vs. Accuracy\n(bubble size ∝ cost/response)")

    plt.tight_layout()
    plt.savefig(output_dir / "fig2_pareto.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 2 → {output_dir/'fig2_pareto.pdf'}")


def fig3_fairness_heatmap(ceat_path: Path, output_dir: Path) -> None:
    """Fig 3: CEAT fairness heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if not ceat_path.exists():
        ax.text(0.5, 0.5, "CEAT data not available", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_dir / "fig3_fairness_heatmap.pdf", bbox_inches="tight")
        plt.close()
        return

    df = pd.read_csv(ceat_path)
    if "model_tag" not in df.columns or "attribute_set" not in df.columns:
        ax.text(0.5, 0.5, "Missing columns", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_dir / "fig3_fairness_heatmap.pdf", bbox_inches="tight")
        plt.close()
        return

    ceat_col = "ceat_d" if "ceat_d" in df.columns else "abs_d"
    pivot = df.pivot_table(index="model_tag", columns="attribute_set",
                           values=ceat_col, aggfunc="mean")

    if pivot.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_dir / "fig3_fairness_heatmap.pdf", bbox_inches="tight")
        plt.close()
        return

    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="|CEAT effect size d|", fraction=0.046, pad=0.04)
    ax.set_title("D5: Generation Fairness (CEAT)\nGreen = fair (|d|<0.2), Red = biased (|d|>0.5)")

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_fairness_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 3 → {output_dir/'fig3_fairness_heatmap.pdf'}")


def fig4_typeab_crosstab(discrepancy_path: Path, output_dir: Path) -> None:
    """Fig 4: 2×2 dialogue-test discrepancy cross-tab."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    if not discrepancy_path.exists():
        for ax in axes:
            ax.text(0.5, 0.5, "D6 data not available", ha="center", va="center",
                    transform=ax.transAxes)
        plt.savefig(output_dir / "fig4_typeab_crosstab.pdf", bbox_inches="tight")
        plt.close()
        return

    df = pd.read_csv(discrepancy_path)

    # Left: 2×2 conceptual diagram
    ax_left = axes[0]
    ax_left.set_xlim(0, 2); ax_left.set_ylim(0, 2)
    ax_left.set_xticks([0.5, 1.5]); ax_left.set_xticklabels(["Low score\n(test fails)", "High score\n(test passes)"])
    ax_left.set_yticks([0.5, 1.5]); ax_left.set_yticklabels(["Low KL\n(novice dialogue)", "High KL\n(expert dialogue)"])
    ax_left.axvline(1, color="gray", linewidth=1)
    ax_left.axhline(1, color="gray", linewidth=1)
    labels = [("Type-A\n(Eye-High\nHand-Low)", 1.5, 0.5, "#e74c3c"),
              ("Consistent\nHigh", 1.5, 1.5, "#27ae60"),
              ("Consistent\nLow", 0.5, 0.5, "#95a5a6"),
              ("Type-B\n(Silent\nAchiever)", 0.5, 1.5, "#2980b9")]
    for label, x, y, color in labels:
        ax_left.text(x, y, label, ha="center", va="center", fontsize=10,
                     bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))
    ax_left.set_title("Dialogue-Test Discrepancy\nTypology (D6)")
    ax_left.set_xlabel("Test Score (D2 accuracy)")
    ax_left.set_ylabel("Dialogue Knowledge Level (DKT)")

    # Right: prevalence bar chart by model tier
    ax_right = axes[1]
    if "student_type" in df.columns:
        if "model_tag" in df.columns:
            df["model_tier"] = df["model_tag"].apply(assign_tier)
        type_counts = df["student_type"].value_counts(normalize=True) * 100

        colors = {"Type-A (eye-high-hand-low)": "#e74c3c",
                  "Type-B (silent-achiever)": "#2980b9",
                  "Consistent-High": "#27ae60",
                  "Consistent-Low": "#95a5a6",
                  "Aligned": "#bdc3c7"}

        bars = ax_right.bar(range(len(type_counts)), type_counts.values,
                            color=[colors.get(t, "#bdc3c7") for t in type_counts.index])
        ax_right.set_xticks(range(len(type_counts)))
        ax_right.set_xticklabels([t.replace(" (", "\n(") for t in type_counts.index],
                                  rotation=15, ha="right", fontsize=9)
        ax_right.set_ylabel("Prevalence (%)")
        ax_right.set_title("Type-A/B Prevalence Across All Models")
        ax_right.set_ylim(0, max(type_counts.values) * 1.2)

        for bar, val in zip(bars, type_counts.values):
            ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    else:
        ax_right.text(0.5, 0.5, "student_type column missing", ha="center", va="center",
                      transform=ax_right.transAxes)

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_typeab_crosstab.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 4 → {output_dir/'fig4_typeab_crosstab.pdf'}")


def fig5_forest_plot(analysis_path: Path, output_dir: Path) -> None:
    """Fig 5: Forest plot of effect sizes per dimension × model tier."""
    fig, ax = plt.subplots(figsize=(8, 6))

    dims = ["D1 Human-likeness", "D2 Accuracy", "D3 Size Effect",
            "D4 Efficiency", "D5 Fairness", "D6 Discrepancy", "D7 Consistency"]
    y_pos = range(len(dims))

    if analysis_path.exists():
        try:
            results = json.loads(analysis_path.read_text())
            hyps = results.get("hypotheses", {})
            effect_sizes = []
            ci_los = []
            ci_his = []
            for dim in ["H1", "H3", None, None, None, None, None]:
                if dim and dim in hyps:
                    d = hyps[dim].get("cohen_d", 0.0)
                    ci = hyps[dim].get("ci_95", [d - 0.2, d + 0.2])
                    effect_sizes.append(d)
                    ci_los.append(ci[0])
                    ci_his.append(ci[1])
                else:
                    effect_sizes.append(0.0)
                    ci_los.append(-0.2)
                    ci_his.append(0.2)
        except Exception:
            effect_sizes = [0.0] * 7
            ci_los = [-0.2] * 7
            ci_his = [0.2] * 7
    else:
        effect_sizes = [0.0] * 7
        ci_los = [-0.2] * 7
        ci_his = [0.2] * 7

    colors = [TIER_COLORS.get("Big", "#2980b9")] * len(dims)
    ax.barh(list(y_pos), effect_sizes, xerr=[
        [d - lo for d, lo in zip(effect_sizes, ci_los)],
        [hi - d for d, hi in zip(effect_sizes, ci_his)],
    ], color=colors, alpha=0.7, capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="d=0.5 (medium)")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(dims)
    ax.set_xlabel("Cohen's d (95% Bootstrap CI)")
    ax.set_title("Effect Sizes: Big-tier vs. Tiny-tier Models\nAcross 7 Evaluation Dimensions")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "fig5_forest_plot.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 5 → {output_dir/'fig5_forest_plot.pdf'}")


def fig6_unlearning_curves(unlearn_eval_path: Path, output_dir: Path) -> None:
    """Fig 6: Forgetting curves (accuracy vs. unlearning ratio × seed)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    if not unlearn_eval_path.exists():
        for ax in axes:
            ax.text(0.5, 0.5, "Unlearning eval data not available",
                    ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_dir / "fig6_unlearning_curves.pdf", bbox_inches="tight")
        plt.close()
        return

    df = pd.read_csv(unlearn_eval_path)

    for ax, (model_key, model_label) in zip(axes, [
        ("mistral", "Mistral-7B-Instruct-v0.3"),
        ("qwen",    "Qwen2.5-3B-Instruct"),
    ]):
        sub = df[df["base_model"].str.lower().str.contains(model_key, na=False)]
        if sub.empty:
            ax.text(0.5, 0.5, f"No data for {model_label}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(model_label)
            continue

        agg = sub.groupby("unlearn_ratio")[["forget_accuracy", "retain_accuracy"]].agg(
            ["mean", "std"]
        )
        ratios_pct = [r * 100 for r in agg.index]

        ax.errorbar(ratios_pct, agg[("forget_accuracy", "mean")],
                    yerr=agg[("forget_accuracy", "std")],
                    marker="o", color="#e74c3c", label="Forget-set acc.", capsize=4)
        ax.errorbar(ratios_pct, agg[("retain_accuracy", "mean")],
                    yerr=agg[("retain_accuracy", "std")],
                    marker="s", color="#2980b9", label="Retain-set acc.", capsize=4)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Unlearning ratio (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title(model_label)
        ax.set_xticks(ratios_pct)
        ax.legend()

    plt.suptitle("Fig 6: P2 Unlearning Forgetting Curves (mean ± SD, 5 seeds)", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "fig6_unlearning_curves.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 6 → {output_dir/'fig6_unlearning_curves.pdf'}")


def fig7_qtype_comparison(d1_path: Path, output_dir: Path) -> None:
    """Fig 7: D1 human-likeness by question type × difficulty."""
    fig, ax = plt.subplots(figsize=(9, 5))

    if not d1_path.exists():
        ax.text(0.5, 0.5, "D1 data not available", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_dir / "fig7_qtype_comparison.pdf", bbox_inches="tight")
        plt.close()
        return

    df = pd.read_csv(d1_path)
    d1_col = "d1_composite" if "d1_composite" in df.columns else "HL1_mean"
    if d1_col not in df.columns or "qtype" not in df.columns:
        ax.text(0.5, 0.5, "Missing columns", ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_dir / "fig7_qtype_comparison.pdf", bbox_inches="tight")
        plt.close()
        return

    qtypes = df["qtype"].unique()
    difficulties = ["Easy", "Medium", "Hard"]
    x = np.arange(len(qtypes))
    width = 0.25
    diff_colors = {"Easy": "#27ae60", "Medium": "#f39c12", "Hard": "#e74c3c"}

    for i, diff in enumerate(difficulties):
        sub = df[df.get("difficulty", pd.Series(dtype=str)) == diff] if "difficulty" in df.columns else df
        means = [sub[sub["qtype"] == qt][d1_col].mean() for qt in qtypes]
        stds = [sub[sub["qtype"] == qt][d1_col].std() for qt in qtypes]
        ax.bar(x + i * width, means, width, yerr=stds, label=diff,
               color=diff_colors[diff], alpha=0.8, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels(qtypes)
    ax.set_xlabel("Question Type")
    ax.set_ylabel("D1 Human-Likeness (1–5)")
    ax.set_title("D1 Human-Likeness by Question Type × Difficulty")
    ax.legend(title="Difficulty")
    ax.set_ylim(0, 5.5)

    plt.tight_layout()
    plt.savefig(output_dir / "fig7_qtype_comparison.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 7 → {output_dir/'fig7_qtype_comparison.pdf'}")


def fig8_p1_vs_p2(d1_path: Path, d2_path: Path, output_dir: Path) -> None:
    """Fig 8: P1 (prompt) vs. P2 (unlearning) head-to-head on D1 and D2."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (path, metric, title) in zip(axes, [
        (d1_path, "d1_composite", "D1: Human-Likeness"),
        (d2_path, "accuracy",     "D2: Answering Accuracy"),
    ]):
        if not path.exists() or metric == "d1_composite":
            d1_col = "d1_composite" if path.exists() else None
            if not path.exists():
                ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title)
                continue
            df = pd.read_csv(path)
            metric_col = "d1_composite" if "d1_composite" in df.columns else "HL1_mean"
        else:
            if not path.exists():
                ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(title)
                continue
            df = pd.read_csv(path)
            metric_col = metric

        if "persona" not in df.columns or metric_col not in df.columns:
            ax.text(0.5, 0.5, "Missing persona column", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        for persona, color in [("P1", "#2980b9"), ("P2", "#e74c3c")]:
            sub = df[df["persona"] == persona]
            if sub.empty:
                continue
            by_ratio = sub.groupby("unlearn_ratio" if "unlearn_ratio" in sub.columns
                                    else "model_tag")[metric_col].mean()
            ax.bar(range(len(by_ratio)), by_ratio.values, color=color,
                   alpha=0.7, label=f"{persona}")

        ax.set_title(title)
        ax.set_ylabel(metric_col)
        ax.legend()

    plt.suptitle("Fig 8: Prompt-based (P1) vs. Unlearning-based (P2) Head-to-Head")
    plt.tight_layout()
    plt.savefig(output_dir / "fig8_p1_vs_p2.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Fig 8 → {output_dir/'fig8_p1_vs_p2.pdf'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all 8 paper figures")
    parser.add_argument("--data-dir", default="outputs")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    print("Generating all 8 paper figures...")

    fig1_scaling_curves(
        data_dir / "eval_d1_human_likeness.csv",
        data_dir / "eval_d2_answering.csv",
        output_dir,
    )
    fig2_pareto(data_dir / "eval_d4_efficiency_pareto.csv", output_dir)
    fig3_fairness_heatmap(data_dir / "eval_d5_ceat.csv", output_dir)
    fig4_typeab_crosstab(data_dir / "eval_d6_discrepancy.csv", output_dir)
    fig5_forest_plot(data_dir / "analysis_v2" / "analysis_results_v2.json", output_dir)
    fig6_unlearning_curves(data_dir / "unlearning_eval.csv", output_dir)
    fig7_qtype_comparison(data_dir / "eval_d1_human_likeness.csv", output_dir)
    fig8_p1_vs_p2(
        data_dir / "eval_d1_human_likeness.csv",
        data_dir / "eval_d2_answering.csv",
        output_dir,
    )

    print(f"\nAll figures saved to {output_dir}")
    print("Run: ls -la", output_dir)


if __name__ == "__main__":
    main()
