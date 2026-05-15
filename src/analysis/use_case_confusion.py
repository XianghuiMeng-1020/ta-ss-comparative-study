"""
Use-case decision analysis.

Produces 4×4 confusion matrix: actual condition × coder-assigned use-case.
Reports diagonal proportions (correct-use-case assignment rate) and reject rates.
Key layer: upgrades study beyond ordinary rating to use-case-decision-grade evaluation.

Output: outputs/use_case_confusion_matrix.csv, outputs/use_case_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

USE_CASES = ["learning_by_teaching", "diagnostic_simulation", "feedback_policy_testing", "reject"]
CONDITIONS = ["C1", "C2", "C3", "C4"]

EXPECTED_MAPPING = {
    "C1": "learning_by_teaching",
    "C2": "diagnostic_simulation",
    "C3": "reject",   # expected to fail or be mixed
    "C4": "reject",
}


def majority_vote(group: pd.DataFrame, col: str) -> str:
    """Resolve multi-coder use_case_decision by majority vote."""
    counts = group[col].dropna().value_counts()
    if counts.empty:
        return "reject"
    return counts.idxmax()


def run_analysis():
    ratings_path = Path("outputs/coder_ratings_raw.csv")
    if not ratings_path.exists():
        print("coder_ratings_raw.csv not found.")
        return

    df = pd.read_csv(ratings_path)
    if "use_case_decision" not in df.columns:
        print("No use_case_decision column found.")
        return

    # Resolve: majority vote per packet
    resolved = (
        df.groupby(["packet_id", "true_condition", "scenario_id"])
        .apply(lambda g: majority_vote(g, "use_case_decision"))
        .reset_index(name="resolved_use_case")
    )

    # Save resolved decisions
    resolved_path = Path("outputs") / "use_case_decisions.csv"
    resolved.to_csv(resolved_path, index=False)
    print(f"Saved use case decisions → {resolved_path}")

    # Confusion matrix
    cm = pd.crosstab(
        resolved["true_condition"],
        resolved["resolved_use_case"],
        rownames=["Actual Condition"],
        colnames=["Assigned Use Case"],
    )

    # Reindex to standard order
    cm = cm.reindex(index=CONDITIONS, columns=USE_CASES, fill_value=0)
    cm_path = Path("outputs") / "use_case_confusion_matrix.csv"
    cm.to_csv(cm_path)
    print(f"\nConfusion matrix → {cm_path}")
    print(cm.to_string())

    # Summary metrics
    summary_rows = []
    for cond in CONDITIONS:
        if cond not in cm.index:
            continue
        row = cm.loc[cond]
        total = row.sum()
        expected_uc = EXPECTED_MAPPING.get(cond, "")
        correct_count = row.get(expected_uc, 0)
        reject_count = row.get("reject", 0)
        summary_rows.append({
            "condition": cond,
            "expected_use_case": expected_uc,
            "n_packets": total,
            "n_correct_assignment": correct_count,
            "pct_correct_assignment": round(correct_count / total * 100, 1) if total > 0 else 0,
            "n_reject": reject_count,
            "pct_reject": round(reject_count / total * 100, 1) if total > 0 else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path("outputs") / "use_case_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nUse-case summary → {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_analysis()
