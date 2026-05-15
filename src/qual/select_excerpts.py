"""
Qualitative excerpt selection for failure memo.

Automatically selects 4 × 6 = 24 representative dialogues:
  1. high-alignment   — C1/C2 in top 10% of their primary framework dimensions
  2. low-alignment    — bottom 10% of their primary framework dimensions
  3. max-disagreement — coders differ by ≥ 2 points on any core dimension
  4. failure          — any LLM validity failure flag = 1

Each excerpt is annotated with the framework dimension or LLM validity threat
that motivated its selection, per the failure memo template.

Output: outputs/qualitative_excerpts.csv (with dialogue text included)
        outputs/qualitative_failure_memo_scaffold.md (text scaffold for memo)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

N_PER_CATEGORY = 6

TA_DIMS = ["TA1", "TA2", "TA3", "TA4"]
SS_DIMS = ["SS1", "SS2", "SS3", "SS4", "SS5"]
FAILURE_FLAGS = [
    "premature_expertise", "role_drift", "over_technical",
    "misconception_loss", "logical_inconsistency", "unsupported_reasoning",
]

# Map failure flags to Yuan/Mannekote threat labels
FLAG_TO_THREAT = {
    "premature_expertise":    "Competence paradox (Yuan et al., 2026)",
    "role_drift":             "Role instability (Yuan et al., 2026; Mannekote et al., 2025)",
    "over_technical":         "Competence paradox / over-capable learner (Yuan et al., 2026)",
    "misconception_loss":     "Epistemic fidelity failure (Yuan et al., 2026)",
    "logical_inconsistency":  "Logical inconsistency (Yuan et al., 2026)",
    "unsupported_reasoning":  "Unsupported reasoning / authenticity (BEA / Mannekote et al., 2025)",
}

CONDITION_PRIMARY_DIMS = {
    "C1": TA_DIMS,
    "C2": SS_DIMS,
    "C3": TA_DIMS + SS_DIMS,  # expected low on all
    "C4": TA_DIMS + SS_DIMS,
}


def load_dialogue_text(json_path: str) -> str:
    try:
        data = json.loads(Path(json_path).read_text())
    except Exception:
        return "(dialogue not available)"
    lines = []
    for t in data.get("turns", []):
        tutor = t.get("tutor_turn", "").strip()
        learner = t.get("learner_turn", "").strip()
        if tutor:
            lines.append(f"[TUTOR]   {tutor}")
        if learner:
            lines.append(f"[LEARNER] {learner}")
    return "\n".join(lines)


def select_high_alignment(resolved_df: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for cond in ["C1", "C2"]:
        dims = CONDITION_PRIMARY_DIMS.get(cond, [])
        available = [d for d in dims if d in resolved_df.columns]
        if not available:
            continue
        subset = resolved_df[resolved_df["true_condition"] == cond].copy()
        subset["_primary_mean"] = subset[available].mean(axis=1)
        threshold = subset["_primary_mean"].quantile(0.90)
        top = subset[subset["_primary_mean"] >= threshold].head(n // 2)
        top = top.assign(excerpt_category="high_alignment", framework_anchor=f"Top 10% on {available}")
        rows.append(top)
    return pd.concat(rows, ignore_index=True).head(n) if rows else pd.DataFrame()


def select_low_alignment(resolved_df: pd.DataFrame, n: int) -> pd.DataFrame:
    rows = []
    for cond in ["C1", "C2"]:
        dims = CONDITION_PRIMARY_DIMS.get(cond, [])
        available = [d for d in dims if d in resolved_df.columns]
        if not available:
            continue
        subset = resolved_df[resolved_df["true_condition"] == cond].copy()
        subset["_primary_mean"] = subset[available].mean(axis=1)
        threshold = subset["_primary_mean"].quantile(0.10)
        bottom = subset[subset["_primary_mean"] <= threshold].head(n // 2)
        bottom = bottom.assign(
            excerpt_category="low_alignment",
            framework_anchor=f"Bottom 10% on {available}",
        )
        rows.append(bottom)
    return pd.concat(rows, ignore_index=True).head(n) if rows else pd.DataFrame()


def select_max_disagreement(ratings_raw: pd.DataFrame, resolved_df: pd.DataFrame, n: int) -> pd.DataFrame:
    all_dims = TA_DIMS + SS_DIMS
    available = [d for d in all_dims if d in ratings_raw.columns]
    if not available:
        return pd.DataFrame()

    coders = ratings_raw["coder_id"].unique()
    if len(coders) < 2:
        return pd.DataFrame()

    # Max pairwise difference per packet
    wide = ratings_raw.pivot_table(
        index="packet_id", columns="coder_id", values=available, aggfunc="first"
    )
    max_diffs = []
    for pid in wide.index:
        row = wide.loc[pid]
        diffs = []
        for dim in available:
            col_vals = [row.get((dim, c), np.nan) for c in coders]
            col_vals = [v for v in col_vals if not np.isnan(v)]
            if len(col_vals) >= 2:
                diffs.append(max(col_vals) - min(col_vals))
        max_diffs.append({"packet_id": pid, "max_coder_diff": max(diffs) if diffs else 0})

    diff_df = pd.DataFrame(max_diffs)
    disagreement = diff_df[diff_df["max_coder_diff"] >= 2].head(n)
    result = disagreement.merge(
        resolved_df[["packet_id", "true_condition", "scenario_id"]].drop_duplicates(),
        on="packet_id", how="left",
    )
    result["excerpt_category"] = "max_disagreement"
    result["framework_anchor"] = "Coder disagreement ≥ 2 points"
    return result.head(n)


def select_failure_cases(ratings_raw: pd.DataFrame, n: int) -> pd.DataFrame:
    flag_cols = [f"flag_{f}" for f in FAILURE_FLAGS if f"flag_{f}" in ratings_raw.columns]
    if not flag_cols:
        return pd.DataFrame()

    has_failure = ratings_raw[flag_cols].fillna(0).sum(axis=1) > 0
    failed = ratings_raw[has_failure].copy()
    if failed.empty:
        return pd.DataFrame()

    # Identify which flag was triggered
    def first_flag(row):
        for fc in flag_cols:
            if row.get(fc, 0) == 1:
                flag_name = fc.replace("flag_", "")
                return FLAG_TO_THREAT.get(flag_name, flag_name)
        return ""

    failed["framework_anchor"] = failed.apply(first_flag, axis=1)
    failed["excerpt_category"] = "failure"
    return failed.groupby("packet_id").first().reset_index().head(n)


def generate_memo_scaffold(excerpts_df: pd.DataFrame) -> str:
    lines = [
        "# Qualitative Failure Memo — Scaffold",
        "",
        "Framework references: Blair et al. (2007); Koedinger et al. (2015); ",
        "Yuan et al. (2026); Mannekote et al. (2025).",
        "",
        "Each excerpt is bound to ONE framework dimension or ONE LLM validity threat.",
        "Fill in [EXCERPT TEXT], [ANALYSIS], and [IMPLICATION] for each case.",
        "",
    ]
    for cat in ["high_alignment", "low_alignment", "max_disagreement", "failure"]:
        subset = excerpts_df[excerpts_df["excerpt_category"] == cat]
        cat_label = cat.replace("_", " ").title()
        lines.append(f"## {cat_label} Cases (n={len(subset)})")
        lines.append("")
        for _, row in subset.iterrows():
            lines.append(f"### Case {row.get('packet_id', '?')} — {row.get('true_condition', '?')}")
            lines.append(f"**Framework anchor**: {row.get('framework_anchor', '')}")
            lines.append(f"**Scenario**: {row.get('scenario_id', '')}")
            lines.append("")
            lines.append("[EXCERPT TEXT]")
            lines.append("")
            lines.append("> **Analysis**: [Connect to framework dimension or LLM threat]")
            lines.append("> **Implication**: [What this suggests for protocol design or evaluation]")
            lines.append("")
    return "\n".join(lines)


def main():
    ratings_path = Path("outputs/coder_ratings_raw.csv")
    manifest_path = Path("outputs/coder_packets/manifest.csv")

    if not ratings_path.exists():
        print("coder_ratings_raw.csv not found.")
        return

    ratings_raw = pd.read_csv(ratings_path)
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

    all_dims = TA_DIMS + SS_DIMS
    available_dims = [d for d in all_dims if d in ratings_raw.columns]
    resolved = (
        ratings_raw.groupby(["packet_id", "true_condition", "scenario_id"])[available_dims]
        .mean()
        .reset_index()
    )

    # Merge dialogue JSON path from manifest
    if not manifest.empty and "json_path" in manifest.columns:
        resolved = resolved.merge(
            manifest[["packet_id", "json_path"]].drop_duplicates(),
            on="packet_id", how="left",
        )

    all_excerpts = []

    high = select_high_alignment(resolved, N_PER_CATEGORY)
    low = select_low_alignment(resolved, N_PER_CATEGORY)
    disagree = select_max_disagreement(ratings_raw, resolved, N_PER_CATEGORY)
    failure = select_failure_cases(ratings_raw, N_PER_CATEGORY)

    for df_part in [high, low, disagree, failure]:
        if df_part is not None and not df_part.empty:
            all_excerpts.append(df_part)

    if not all_excerpts:
        print("No excerpts selected. Run coder ingestion first.")
        return

    excerpts_df = pd.concat(all_excerpts, ignore_index=True)

    # Add dialogue text
    if "json_path" in excerpts_df.columns:
        excerpts_df["dialogue_text"] = excerpts_df["json_path"].apply(
            lambda p: load_dialogue_text(p) if pd.notna(p) else ""
        )

    out_path = Path("outputs") / "qualitative_excerpts.csv"
    excerpts_df.to_csv(out_path, index=False)
    print(f"Saved {len(excerpts_df)} excerpts → {out_path}")

    # Memo scaffold
    memo_text = generate_memo_scaffold(excerpts_df)
    memo_path = Path("outputs") / "qualitative_failure_memo_scaffold.md"
    memo_path.write_text(memo_text)
    print(f"Saved memo scaffold → {memo_path}")


if __name__ == "__main__":
    main()
