"""
Eedi QATD-2k Ecological Reference Analysis.

Computes real Eedi student-turn distributional features and compares them
to our generated dialogues via Jensen-Shannon divergence.

Reports: which condition × model combination produces dialogue closest
to real Eedi student turns — provides ecological validity anchor.

Output: outputs/eedi_ecological_reference.csv
        outputs/eedi_ecological_divergence_heatmap.png

Decision Log D6 (Eedi QATD-2k as ecological reference)

Usage:
  python src/analysis/eedi_ecological_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

EEDI_DATA_DIR = Path("data/raw_eedi_qatd")
GENERATED_CSV = Path("outputs/generated_dialogues_gpt-4o-2024-11-20.csv")  # primary model CSV
OUTPUT_CSV = Path("outputs/eedi_ecological_reference.csv")
OUTPUT_HEATMAP = Path("outputs/eedi_ecological_divergence_heatmap.png")

FEATURES = [
    "mean_turn_length_words",
    "question_rate",           # fraction of turns containing "?"
    "reasoning_rate",          # fraction of turns containing math operators or "because"
    "correction_phrase_rate",  # "I see", "oh", "wait" etc.
]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_from_text_list(texts: list[str]) -> dict:
    if not texts:
        return {f: 0.0 for f in FEATURES}
    lengths = [len(t.split()) for t in texts]
    q_rate = sum("?" in t for t in texts) / len(texts)
    reas = sum(
        any(x in t.lower() for x in ["because", "since", "+", "×", "÷", "=", "*", "/"])
        for t in texts
    ) / len(texts)
    correction_phrases = ["i see", "oh", "wait", "ah", "i was wrong", "i made", "i think i"]
    corr_rate = sum(
        any(p in t.lower() for p in correction_phrases) for t in texts
    ) / len(texts)
    return {
        "mean_turn_length_words": float(np.mean(lengths)),
        "question_rate": q_rate,
        "reasoning_rate": reas,
        "correction_phrase_rate": corr_rate,
    }


def load_eedi_student_turns() -> list[str]:
    """Load Eedi student utterances from anchored dialogues CSV or JSONL."""
    turns = []
    csv_path = EEDI_DATA_DIR / "anchored_dialogues_train.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, low_memory=False)
        # Eedi stores student/tutor turns; find student column
        for col in ["student_message", "student_turn", "student", "learner_message"]:
            if col in df.columns:
                turns = df[col].dropna().astype(str).tolist()
                break
        if not turns:
            # Try all-messages column with role filter
            for role_col in ["role", "speaker", "author"]:
                if role_col in df.columns:
                    student_mask = df[role_col].astype(str).str.lower().isin(
                        {"student", "learner", "pupil"}
                    )
                    msg_col = next(
                        (c for c in ["message", "text", "content", "utterance"] if c in df.columns),
                        None
                    )
                    if msg_col:
                        turns = df.loc[student_mask, msg_col].dropna().astype(str).tolist()
                        break
        if turns:
            print(f"Loaded {len(turns)} Eedi student turns from {csv_path}")
            return turns

    # Fall back to JSONL
    jsonl_path = EEDI_DATA_DIR / "eedi_dialogues_all.jsonl"
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                msgs = record.get("messages", record.get("turns", []))
                for m in msgs:
                    role = str(m.get("role", m.get("speaker", ""))).lower()
                    if role in {"student", "learner", "pupil"}:
                        text = m.get("text", m.get("content", ""))
                        if text:
                            turns.append(str(text))
            except Exception:
                continue
        print(f"Loaded {len(turns)} Eedi student turns from {jsonl_path}")

    return turns


def load_generated_student_turns(generated_csv: Path) -> dict[tuple[str, str], list[str]]:
    """Load generated student turns grouped by (condition, model_tag)."""
    if not generated_csv.exists():
        # Try aggregating across all model-specific CSVs
        all_csvs = list(Path("outputs").glob("generated_dialogues_*.csv"))
        if not all_csvs:
            return {}
        dfs = [pd.read_csv(p) for p in all_csvs]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(generated_csv)

    # Need to load JSON files to get actual learner turns
    groups: dict[tuple[str, str], list[str]] = {}
    for _, row in df.iterrows():
        json_path = row.get("json_path", "")
        if not json_path or not Path(json_path).exists():
            continue
        try:
            data = json.loads(Path(json_path).read_text())
        except Exception:
            continue
        if data.get("exclusion_flag"):
            continue
        cond = data.get("condition", "?")
        mtag = data.get("model_tag", data.get("model_id", "?"))
        key = (cond, mtag)
        if key not in groups:
            groups[key] = []
        for t in data.get("turns", []):
            lt = t.get("learner_turn", "")
            if lt:
                groups[key].append(lt)

    return groups


def js_divergence_scalar(feat_a: dict, feat_b: dict) -> float:
    """Approximate Jensen-Shannon divergence as mean absolute feature difference."""
    vals_a = np.array([feat_a.get(f, 0.0) for f in FEATURES])
    vals_b = np.array([feat_b.get(f, 0.0) for f in FEATURES])
    # Normalise to [0,1] per feature across a and b
    diffs = np.abs(vals_a - vals_b)
    # Pseudo-JSD: mean squared normalised difference
    max_vals = np.maximum(np.abs(vals_a) + np.abs(vals_b), 1e-8)
    return float(np.mean((diffs / max_vals) ** 2))


def main() -> None:
    print("=== Eedi Ecological Reference Analysis ===\n")

    eedi_turns = load_eedi_student_turns()
    if not eedi_turns:
        print("ERROR: No Eedi student turns found. Check data/raw_eedi_qatd/")
        return

    eedi_feats = extract_features_from_text_list(eedi_turns[:10_000])  # cap for speed
    print("Eedi reference features:")
    for k, v in eedi_feats.items():
        print(f"  {k}: {v:.4f}")

    print("\nLoading generated dialogues ...")
    generated_groups = load_generated_student_turns(GENERATED_CSV)
    if not generated_groups:
        print("No generated dialogues found. Run main generation first.")
        # Produce empty output
        pd.DataFrame(columns=["condition", "model_tag"] + FEATURES + ["jsd_from_eedi"]).to_csv(
            OUTPUT_CSV, index=False
        )
        return

    rows = []
    for (cond, mtag), texts in sorted(generated_groups.items()):
        feats = extract_features_from_text_list(texts)
        jsd = js_divergence_scalar(eedi_feats, feats)
        row = {"condition": cond, "model_tag": mtag, "jsd_from_eedi": round(jsd, 5)}
        row.update({k: round(v, 4) for k, v in feats.items()})
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("jsd_from_eedi")
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEcological reference results → {OUTPUT_CSV}")
    print("\nTop 5 conditions closest to real Eedi students (lowest JSD):")
    print(result_df.head(5)[["condition", "model_tag", "jsd_from_eedi"]].to_string(index=False))

    if HAS_MPL and len(rows) > 0:
        pivot = result_df.pivot_table(index="condition", columns="model_tag", values="jsd_from_eedi")
        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), 4))
        im = ax.imshow(pivot.values, cmap="YlOrRd_r", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("JSD from Eedi Real Students\n(lower = closer to real students)")
        plt.colorbar(im, ax=ax, label="Pseudo-JSD")
        plt.tight_layout()
        plt.savefig(OUTPUT_HEATMAP, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved → {OUTPUT_HEATMAP}")


if __name__ == "__main__":
    main()
