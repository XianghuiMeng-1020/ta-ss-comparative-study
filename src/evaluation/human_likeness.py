"""
D1 — Human-likeness evaluation.

Two sub-scores:
  HL1: 1–5 Likert "resembles a real student" (from human annotations)
  HL2: Error plausibility 1–5 (from human annotations)
  HL3: Turing binary (0=AI, 1=human student; from human annotations)
  HL_auto: Automated proxy — perplexity under GPT-2-based student-language model

Usage:
    # Aggregate from Qualtrics export (human annotations):
    python src/evaluation/human_likeness.py \
        --annotations outputs/human_annotations_v2.csv \
        --output outputs/eval_d1_human_likeness.csv

    # Compute automated proxy (no human labels needed):
    python src/evaluation/human_likeness.py \
        --responses outputs/qa_responses_gpt-4o.csv \
        --auto-proxy \
        --output outputs/eval_d1_proxy_gpt4o.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PARAMS_BILLIONS = {
    "SmolLM2-360M-Instruct":           0.36,
    "TinyLlama-1.1B-Chat-v1.0":        1.1,
    "Llama-3.2-1B-Instruct":           1.0,
    "Qwen2.5-1.5B-Instruct":           1.5,
    "Phi-3.5-mini-instruct":           3.8,
    "Llama-3.2-3B-Instruct":           3.0,
    "Qwen3-4B":                        4.0,
    "Mistral-7B-Instruct-v0.3":        7.0,
    "Llama-3.1-8B-Instruct":           8.0,
    "gpt-4o-2024-11-20":             200.0,
    "claude-sonnet-4-5-20250929":     70.0,
    "Llama-3.1-70B-Instruct":         70.0,
}


def aggregate_human_ratings(annotations_path: Path) -> pd.DataFrame:
    """
    Aggregate Qualtrics coder export into per-response D1 scores.
    Expected columns: item_id, coder_id, HL1, HL2, HL3
    """
    df = pd.read_csv(annotations_path)
    required = {"item_id", "coder_id", "HL1", "HL2", "HL3"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation file missing columns: {missing}")

    grouped = df.groupby("item_id").agg(
        HL1_mean=("HL1", "mean"),
        HL1_sd=("HL1", "std"),
        HL2_mean=("HL2", "mean"),
        HL2_sd=("HL2", "std"),
        HL3_mean=("HL3", "mean"),
        n_coders=("coder_id", "nunique"),
    ).reset_index()

    grouped["d1_composite"] = (grouped["HL1_mean"] + grouped["HL2_mean"]) / 2
    return grouped


def compute_icc_d1(annotations_path: Path) -> pd.Series:
    """Compute ICC(2,k) for HL1, HL2, HL3 using pingouin."""
    try:
        import pingouin as pg
    except ImportError:
        print("[WARN] pingouin not installed. pip install pingouin")
        return pd.Series(dtype=float)

    df = pd.read_csv(annotations_path)
    results = {}
    for dim in ["HL1", "HL2", "HL3"]:
        if dim not in df.columns:
            continue
        icc_df = pg.intraclass_corr(
            data=df, targets="item_id", raters="coder_id", ratings=dim
        )
        icc2k = icc_df[icc_df["Type"] == "ICC2k"]["ICC"].values
        results[dim] = round(float(icc2k[0]), 4) if len(icc2k) else None
    return pd.Series(results, name="ICC2k")


def compute_auto_proxy(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Automated human-likeness proxy: perplexity under GPT-2.
    Lower perplexity among student-language corpora → more student-like.
    Scaled to 1–5: ppl_score = 5 − 4 × (ppl − ppl_min) / (ppl_max − ppl_min)
    """
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    except ImportError:
        print("[WARN] transformers not installed — auto proxy unavailable.")
        responses_df["hl_auto_ppl"] = None
        responses_df["hl_auto_score"] = None
        return responses_df

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    ppls = []
    for response in responses_df["response"].fillna(""):
        if not response.strip():
            ppls.append(None)
            continue
        try:
            import torch
            enc = tokenizer(response, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                loss = model(**enc, labels=enc["input_ids"]).loss
            ppls.append(float(torch.exp(loss).item()))
        except Exception:
            ppls.append(None)

    responses_df = responses_df.copy()
    responses_df["hl_auto_ppl"] = ppls
    valid = [p for p in ppls if p is not None]
    if valid:
        ppl_min, ppl_max = min(valid), max(valid)
        responses_df["hl_auto_score"] = responses_df["hl_auto_ppl"].apply(
            lambda p: round(5 - 4 * (p - ppl_min) / max(ppl_max - ppl_min, 1e-9), 3)
            if p is not None else None
        )
    else:
        responses_df["hl_auto_score"] = None
    return responses_df


def get_model_params(model_tag: str) -> float:
    """Return parameter count in billions; None if unknown."""
    for key, val in MODEL_PARAMS_BILLIONS.items():
        if key.lower() in model_tag.lower() or model_tag.lower() in key.lower():
            return val
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 Human-likeness evaluation")
    parser.add_argument("--annotations", help="Qualtrics annotation CSV")
    parser.add_argument("--responses", help="qa_responses CSV for auto-proxy")
    parser.add_argument("--auto-proxy", action="store_true")
    parser.add_argument("--icc", action="store_true", help="Compute ICC only")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.icc and args.annotations:
        icc = compute_icc_d1(Path(args.annotations))
        print("ICC(2,k) for D1 dimensions:")
        print(icc.to_string())
        icc.to_csv(output_path.with_suffix(".icc.csv"))
        return

    if args.annotations:
        df = aggregate_human_ratings(Path(args.annotations))
        df["params_B"] = df.get("model_tag", pd.Series(dtype=str)).apply(
            lambda t: get_model_params(str(t))
        )
        df.to_csv(output_path, index=False)
        print(f"D1 human ratings aggregated → {output_path}")
        print(df[["item_id", "HL1_mean", "HL2_mean", "HL3_mean",
                   "d1_composite", "n_coders"]].describe().to_string())

    elif args.auto_proxy and args.responses:
        df = pd.read_csv(args.responses)
        df = compute_auto_proxy(df)
        df.to_csv(output_path, index=False)
        print(f"D1 auto-proxy → {output_path}")
        print(df[["model_tag", "qtype", "difficulty", "hl_auto_score"]].groupby(
            ["model_tag", "qtype"]
        )["hl_auto_score"].mean().round(3).to_string())
    else:
        parser.error("Provide --annotations or (--responses + --auto-proxy)")


if __name__ == "__main__":
    main()
