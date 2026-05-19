"""
D4 — Efficiency evaluation and Pareto frontier analysis.

Metrics:
  - Wall-clock latency (ms/response) — logged by qa_runner.py
  - Tokens per second (output_tokens / latency_ms * 1000)
  - USD per response — from cost_usd column
  - Pareto frontier over {-D1, -D2_accuracy, D4_cost}

Usage:
    python src/evaluation/efficiency.py \
        --responses outputs/qa_responses_*.csv \
        --d1 outputs/eval_d1_human_likeness.csv \
        --d2 outputs/eval_d2_answering.csv \
        --output outputs/eval_d4_efficiency.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def load_efficiency_data(response_paths: list[str]) -> pd.DataFrame:
    dfs = []
    for p in response_paths:
        if not Path(p).exists():
            continue
        dfs.append(pd.read_csv(p, usecols=lambda c: c in {
            "item_id", "model_tag", "qtype", "difficulty", "seed",
            "latency_ms", "output_tokens", "cost_usd",
        }))
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return df


def compute_efficiency_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tokens_per_sec"] = df["output_tokens"] / (df["latency_ms"] / 1000 + 1e-9)

    agg = df.groupby(["model_tag", "qtype"]).agg(
        latency_mean_ms=("latency_ms", "mean"),
        latency_median_ms=("latency_ms", "median"),
        latency_p95_ms=("latency_ms", lambda x: x.quantile(0.95)),
        tokens_per_sec_mean=("tokens_per_sec", "mean"),
        cost_usd_mean=("cost_usd", "mean"),
        cost_usd_sum=("cost_usd", "sum"),
        n=("latency_ms", "count"),
    ).reset_index()

    for col in ["latency_mean_ms", "latency_median_ms", "latency_p95_ms",
                "tokens_per_sec_mean", "cost_usd_mean"]:
        agg[col] = agg[col].round(3)

    return agg


def compute_pareto_frontier(
    efficiency_df: pd.DataFrame,
    d1_df: pd.DataFrame,
    d2_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Non-dominated sorting over objectives:
      maximize D1 (human-likeness)
      maximize D2 × (1 - accuracy) [more novice-appropriate errors]
      minimize cost_usd
    """
    d1_col = "d1_composite" if "d1_composite" in d1_df.columns else "HL1_mean"

    d1_agg = d1_df.groupby("model_tag")[d1_col].mean().reset_index()
    d1_agg.columns = ["model_tag", "d1_mean"]

    d2_agg = d2_df.groupby("model_tag")["accuracy"].mean().reset_index()
    d2_agg.columns = ["model_tag", "d2_accuracy_mean"]

    eff_agg = efficiency_df.groupby("model_tag")[["cost_usd_mean", "latency_mean_ms"]].mean()

    merged = d1_agg.merge(d2_agg, on="model_tag", how="outer") \
                   .merge(eff_agg, on="model_tag", how="outer")

    merged["novice_score"] = merged["d1_mean"] * (1 - merged["d2_accuracy_mean"])
    merged = merged.dropna(subset=["d1_mean", "cost_usd_mean"])

    def is_dominated(row, candidates: pd.DataFrame) -> bool:
        for _, other in candidates.iterrows():
            if (other["d1_mean"] >= row["d1_mean"] and
                    other["novice_score"] >= row["novice_score"] and
                    other["cost_usd_mean"] <= row["cost_usd_mean"] and
                    not (other["d1_mean"] == row["d1_mean"] and
                         other["novice_score"] == row["novice_score"] and
                         other["cost_usd_mean"] == row["cost_usd_mean"])):
                return True
        return False

    merged["pareto_dominated"] = merged.apply(
        lambda r: is_dominated(r, merged), axis=1
    )
    merged["pareto_optimal"] = ~merged["pareto_dominated"]
    merged = merged.sort_values("novice_score", ascending=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="D4 Efficiency evaluation")
    parser.add_argument("--responses", nargs="+", required=True)
    parser.add_argument("--d1", required=True)
    parser.add_argument("--d2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = []
    for pattern in args.responses:
        paths.extend(glob.glob(pattern))

    df = load_efficiency_data(paths)
    if df.empty:
        print("No response data loaded.")
        return

    eff_df = compute_efficiency_stats(df)
    print("Efficiency stats:")
    print(eff_df[["model_tag", "qtype", "latency_mean_ms",
                  "tokens_per_sec_mean", "cost_usd_mean"]].to_string(index=False))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eff_df.to_csv(output_path, index=False)

    pareto_path = output_path.with_name(output_path.stem + "_pareto.csv")
    d1_df = pd.read_csv(args.d1)
    d2_df = pd.read_csv(args.d2)
    pareto_df = compute_pareto_frontier(eff_df, d1_df, d2_df)
    pareto_df.to_csv(pareto_path, index=False)

    print(f"\nPareto-optimal models:")
    print(pareto_df[pareto_df["pareto_optimal"]][
        ["model_tag", "d1_mean", "d2_accuracy_mean", "cost_usd_mean", "pareto_optimal"]
    ].to_string(index=False))

    print(f"\nD4 efficiency → {output_path}")
    print(f"D4 Pareto     → {pareto_path}")


if __name__ == "__main__":
    main()
