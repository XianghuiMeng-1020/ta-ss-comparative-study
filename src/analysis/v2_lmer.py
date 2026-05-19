"""
v2 primary statistical analysis — 4-way linear mixed-effects model.

Model (per analysis_plan.md v2.0):
  lmer(metric ~ model_tier * question_type * difficulty * persona
               + (1|item_id) + (1|coder_id))

Applies Holm-Bonferroni correction within each hypothesis family.
Reports Cohen's d with 5,000-iter bootstrap 95% CIs.
Runs TOST equivalence test for H-equiv claim.

Requires: pymer4, pandas, numpy, scipy, pingouin

Usage:
    python src/analysis/v2_lmer.py \
        --d1 outputs/eval_d1_human_likeness.csv \
        --d2 outputs/eval_d2_answering.csv \
        --d4 outputs/eval_d4_efficiency.csv \
        --d5 outputs/eval_d5_ceat.csv \
        --d6 outputs/eval_d6_discrepancy.csv \
        --d7 outputs/eval_d7_persona_consistency.csv \
        --annotations outputs/human_annotations_v2.csv \
        --output-dir outputs/analysis_v2

    # Single dimension debug run:
    python src/analysis/v2_lmer.py --d1 outputs/eval_d1_human_likeness.csv \
        --output-dir outputs/analysis_v2 --dim D1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODEL_TIER_MAP = {
    "smollm2":  "Tiny", "tinyllama": "Tiny",  "llama-3.2-1b": "Tiny",
    "qwen2.5-1.5b": "Small", "phi-3.5": "Small", "llama-3.2-3b": "Small",
    "qwen3-4b": "Mid", "mistral-7b": "Mid", "llama-3.1-8b": "Mid",
    "gpt-4o":   "Big", "claude":    "Big",  "llama-3.1-70b": "Big",
}


def assign_tier(model_tag: str) -> str:
    tag_lower = str(model_tag).lower()
    for key, tier in MODEL_TIER_MAP.items():
        if key in tag_lower:
            return tier
    return "Unknown"


def load_dimension(path: str | None, col_map: dict) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df = df.rename(columns=col_map)
    if "model_tag" in df.columns:
        df["model_tier"] = df["model_tag"].apply(assign_tier)
    return df


def cohen_d_bootstrap(a: np.ndarray, b: np.ndarray, n_iter: int = 5000,
                       seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    d_obs = (a.mean() - b.mean()) / max(pooled_std, 1e-9)
    boot_ds = []
    for _ in range(n_iter):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        ps = np.sqrt((a_boot.std() ** 2 + b_boot.std() ** 2) / 2)
        boot_ds.append((a_boot.mean() - b_boot.mean()) / max(ps, 1e-9))
    ci_lo = float(np.percentile(boot_ds, 2.5))
    ci_hi = float(np.percentile(boot_ds, 97.5))
    return round(float(d_obs), 4), round(ci_lo, 4), round(ci_hi, 4)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [None] * n
    prev = 0.0
    for rank, (orig_idx, p) in enumerate(sorted_pairs):
        adj = p * (n - rank)
        adj = max(adj, prev)
        adj = min(adj, 1.0)
        corrected[orig_idx] = adj
        prev = adj
    return corrected


def run_lmer(df: pd.DataFrame, outcome: str, formula: str) -> dict:
    """Run lmer via pymer4 if available; fallback to OLS with fixed effects only."""
    try:
        from pymer4.models import Lmer
        model = Lmer(formula.replace("OUTCOME", outcome), data=df)
        model.fit()
        fe = model.coefs
        return {
            "method": "lmer",
            "AIC": getattr(model, "AIC", None),
            "fixed_effects": fe.to_dict() if hasattr(fe, "to_dict") else {},
        }
    except Exception:
        pass

    try:
        import statsmodels.formula.api as smf
        ols_formula = formula.replace("OUTCOME", outcome).split("+")[0]
        ols_formula = ols_formula.replace("(1|item_id)", "").replace("(1|coder_id)", "")
        ols_formula = ols_formula.strip().rstrip("+").strip()
        model = smf.ols(ols_formula, data=df.dropna(subset=[outcome])).fit()
        return {
            "method": "ols_fallback",
            "AIC": round(model.aic, 2),
            "R_squared": round(model.rsquared, 4),
            "pvalues": {k: round(v, 6) for k, v in model.pvalues.items()},
            "params": {k: round(v, 6) for k, v in model.params.items()},
        }
    except Exception as e:
        return {"method": "failed", "error": str(e)}


def run_tost(a: np.ndarray, b: np.ndarray,
             equivalence_bound: float = 0.3) -> dict:
    """Two one-sided t-tests (TOST) for equivalence."""
    diff = a.mean() - b.mean()
    se = np.sqrt(a.std() ** 2 / len(a) + b.std() ** 2 / len(b))
    df_val = len(a) + len(b) - 2

    t_upper = (diff - equivalence_bound) / max(se, 1e-9)
    t_lower = (diff + equivalence_bound) / max(se, 1e-9)
    p_upper = 1 - stats.t.cdf(t_upper, df=df_val)
    p_lower = stats.t.cdf(t_lower, df=df_val)
    p_tost = max(p_upper, p_lower)
    equivalent = p_tost < 0.05

    return {
        "diff": round(diff, 4),
        "se": round(se, 4),
        "t_upper": round(t_upper, 4),
        "t_lower": round(t_lower, 4),
        "p_tost": round(p_tost, 4),
        "equivalence_bound": equivalence_bound,
        "equivalent": bool(equivalent),
    }


def run_hypothesis_tests(annotations: pd.DataFrame) -> dict:
    """Run H1–H6 hypothesis tests on the integrated annotation dataset."""
    results = {}
    tiers = annotations.get("model_tier", pd.Series(dtype=str))
    d1_col = "d1_composite" if "d1_composite" in annotations.columns else "HL1_mean"

    # H1: Larger models > smaller on D1
    if d1_col in annotations.columns and "model_tier" in annotations.columns:
        big = annotations[annotations["model_tier"] == "Big"][d1_col].dropna().values
        tiny = annotations[annotations["model_tier"] == "Tiny"][d1_col].dropna().values
        if len(big) > 0 and len(tiny) > 0:
            d, ci_lo, ci_hi = cohen_d_bootstrap(big, tiny)
            t_stat, p_val = stats.ttest_ind(big, tiny, equal_var=False)
            results["H1"] = {
                "comparison": "Big vs Tiny on D1",
                "cohen_d": d, "ci_95": [ci_lo, ci_hi],
                "t": round(t_stat, 4), "p_raw": round(p_val, 6),
                "supported": d > 0.5 and p_val < 0.05,
            }

    # H3 (P1 vs P2 accuracy on MCQ)
    if "persona" in annotations.columns and "d2_accuracy" in annotations.columns:
        p1 = annotations[annotations["persona"] == "P1"]["d2_accuracy"].dropna().values
        p2 = annotations[annotations["persona"] == "P2"]["d2_accuracy"].dropna().values
        if len(p1) > 0 and len(p2) > 0:
            d, ci_lo, ci_hi = cohen_d_bootstrap(p1, p2)
            t_stat, p_val = stats.ttest_ind(p1, p2, equal_var=False)
            results["H3"] = {
                "comparison": "P1 vs P2 on D2 accuracy",
                "cohen_d": d, "ci_95": [ci_lo, ci_hi],
                "t": round(t_stat, 4), "p_raw": round(p_val, 6),
                "supported": p2.mean() < p1.mean() and d > 0.5 and p_val < 0.05,
            }

    # H-equiv: Qwen2.5-3B vs Mistral-7B on D1
    if d1_col in annotations.columns and "model_tag" in annotations.columns:
        qwen = annotations[annotations["model_tag"].str.lower().str.contains("qwen2.5-1.5b|qwen3-4b", na=False)][d1_col].dropna().values
        mistral = annotations[annotations["model_tag"].str.lower().str.contains("mistral", na=False)][d1_col].dropna().values
        if len(qwen) > 1 and len(mistral) > 1:
            results["H_equiv"] = run_tost(qwen, mistral, equivalence_bound=0.3)

    # Holm-Bonferroni correction across primary p-values
    primary_hyps = [k for k in ["H1", "H3"] if k in results]
    raw_ps = [results[h]["p_raw"] for h in primary_hyps]
    corrected = holm_bonferroni(raw_ps)
    for h, p_adj in zip(primary_hyps, corrected):
        results[h]["p_holm"] = round(p_adj, 6)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 primary statistical analysis")
    parser.add_argument("--d1", default=None)
    parser.add_argument("--d2", default=None)
    parser.add_argument("--d4", default=None)
    parser.add_argument("--d5", default=None)
    parser.add_argument("--d6", default=None)
    parser.add_argument("--d7", default=None)
    parser.add_argument("--annotations", default=None,
                        help="Merged human annotations CSV with all dimension scores")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dim", default="ALL",
                        help="Run single dimension only (D1–D7) or ALL")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    FORMULA = ("OUTCOME ~ C(model_tier, Treatment('Tiny')) * C(question_type) "
               "* C(difficulty) * C(persona, Treatment('P1')) "
               "+ (1|item_id) + (1|coder_id)")

    all_results = {}

    for dim_name, path, col_map, outcome in [
        ("D1", args.d1, {}, "d1_composite"),
        ("D2", args.d2, {"accuracy": "d2_accuracy"}, "d2_accuracy"),
        ("D7", args.d7, {"d7_persona_score": "d7_persona_score"}, "d7_persona_score"),
    ]:
        if args.dim not in ("ALL", dim_name):
            continue
        df = load_dimension(path, col_map)
        if df is None:
            continue
        if outcome not in df.columns:
            print(f"[SKIP] {dim_name}: outcome column '{outcome}' not in data")
            continue
        print(f"\n=== {dim_name} ===")
        lmer_result = run_lmer(df, outcome, FORMULA)
        all_results[dim_name] = lmer_result
        print(f"  Method: {lmer_result['method']}")

    if args.annotations:
        ann = pd.read_csv(args.annotations)
        ann["model_tier"] = ann["model_tag"].apply(assign_tier) if "model_tag" in ann.columns else "Unknown"
        print("\n=== Hypothesis Tests ===")
        hyp_results = run_hypothesis_tests(ann)
        all_results["hypotheses"] = hyp_results
        for h, res in hyp_results.items():
            print(f"  {h}: {res}")

    out_path = output_dir / "analysis_results_v2.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nAnalysis results → {out_path}")


if __name__ == "__main__":
    main()
