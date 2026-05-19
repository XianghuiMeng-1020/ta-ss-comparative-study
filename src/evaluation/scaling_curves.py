"""
D3 — Model-size effect (scaling curves).

Fits OLS regressions:
  D1 ~ log(params) + QType + difficulty
  D2 ~ log(params) + QType + difficulty

Identifies the "sweet spot" tier: highest D1 × (1 − D2) product.

Usage:
    python src/evaluation/scaling_curves.py \
        --d1 outputs/eval_d1_human_likeness.csv \
        --d2 outputs/eval_d2_answering.csv \
        --output outputs/eval_d3_scaling.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.human_likeness import MODEL_PARAMS_BILLIONS


def build_scaling_df(d1_path: Path, d2_path: Path) -> pd.DataFrame:
    d1 = pd.read_csv(d1_path)
    d2 = pd.read_csv(d2_path)

    merged = pd.merge(d1, d2, on=["model_tag", "qtype", "difficulty"], suffixes=("_d1", "_d2"))

    def get_params(tag: str) -> float:
        for key, val in MODEL_PARAMS_BILLIONS.items():
            if key.lower() in str(tag).lower():
                return val
        return None

    merged["params_B"] = merged["model_tag"].apply(get_params)
    merged = merged.dropna(subset=["params_B"])
    merged["log_params"] = np.log(merged["params_B"])

    d1_col = "d1_composite" if "d1_composite" in merged.columns else "HL1_mean"
    acc_col = "accuracy"

    merged["d1_score"] = merged[d1_col].astype(float)
    merged["d2_accuracy"] = merged[acc_col].astype(float)
    merged["naive_student_score"] = merged["d1_score"] * (1 - merged["d2_accuracy"])

    return merged


def fit_scaling_regressions(df: pd.DataFrame) -> dict:
    """OLS for D1 and D2 as a function of log(params)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("[WARN] statsmodels not installed. pip install statsmodels")
        return {}

    results = {}
    for outcome, col in [("D1_human_likeness", "d1_score"),
                         ("D2_accuracy", "d2_accuracy")]:
        if col not in df.columns:
            continue
        formula = f"{col} ~ log_params + C(qtype) + C(difficulty)"
        try:
            model = smf.ols(formula, data=df.dropna(subset=[col])).fit()
            results[outcome] = {
                "coef_log_params": round(model.params.get("log_params", 0), 4),
                "pval_log_params": round(model.pvalues.get("log_params", 1), 4),
                "r_squared": round(model.rsquared, 4),
                "aic": round(model.aic, 2),
                "n": int(model.nobs),
            }
            print(f"\n{outcome} ~ log(params):")
            print(f"  β={results[outcome]['coef_log_params']}, "
                  f"p={results[outcome]['pval_log_params']}, "
                  f"R²={results[outcome]['r_squared']}")
        except Exception as e:
            results[outcome] = {"error": str(e)}

    return results


def identify_sweet_spot(df: pd.DataFrame) -> pd.DataFrame:
    tier_boundaries = [(0, 2, "Tiny"), (2, 5, "Small"), (5, 15, "Mid"), (15, 999, "Big")]

    def assign_tier(p: float) -> str:
        for lo, hi, label in tier_boundaries:
            if lo <= p < hi:
                return label
        return "Unknown"

    df = df.copy()
    df["tier"] = df["params_B"].apply(assign_tier)

    sweet = df.groupby("tier")["naive_student_score"].mean().reset_index()
    sweet.columns = ["tier", "mean_naive_student_score"]
    sweet = sweet.sort_values("mean_naive_student_score", ascending=False)
    sweet["rank"] = range(1, len(sweet) + 1)
    return sweet


def main() -> None:
    parser = argparse.ArgumentParser(description="D3 Scaling curves analysis")
    parser.add_argument("--d1", required=True)
    parser.add_argument("--d2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = build_scaling_df(Path(args.d1), Path(args.d2))
    print(f"Scaling dataset: {len(df)} rows | {df['model_tag'].nunique()} models")

    reg_results = fit_scaling_regressions(df)

    sweet_spot = identify_sweet_spot(df)
    print("\nNaive-student sweet spot by tier:")
    print(sweet_spot.to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    reg_path = output_path.with_name(output_path.stem + "_regression.json")
    import json
    reg_path.write_text(json.dumps(reg_results, indent=2))
    sweet_spot.to_csv(output_path.with_name(output_path.stem + "_sweetspot.csv"), index=False)

    print(f"\nD3 scaling data → {output_path}")
    print(f"D3 regression   → {reg_path}")


if __name__ == "__main__":
    main()
