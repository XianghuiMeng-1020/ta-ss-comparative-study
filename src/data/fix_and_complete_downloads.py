"""
Fix and complete the 3 supplementary dataset downloads.

Fixes:
  1. Bridge  – re-save bridge_all.jsonl with proper ndarray serialization
  2. Eedi    – download with correct config names ('anchored-dialogues' +
               'dq-question-metadata')
  3. TalkMoves – compile all xlsx transcript files + train_data_504.xlsx
               into a single talkmoves_all.csv
"""

import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# JSON encoder that handles numpy types
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, cls=NumpyEncoder, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path.name}")


# ---------------------------------------------------------------------------
# Fix 1 – Bridge: re-write JSONL with numpy-safe encoder
# ---------------------------------------------------------------------------

def fix_bridge_jsonl() -> None:
    print("\n=== Fix [1/3] Bridge: re-serialise JSONL ===")
    out_dir = DATA / "raw_bridge"
    all_records: list[dict] = []
    for split in ("train", "validation", "test"):
        df = pd.read_csv(out_dir / f"{split}.csv")
        all_records.extend(df.to_dict(orient="records"))
    save_jsonl(all_records, out_dir / "bridge_all.jsonl")

    # Also write a quick data-quality summary
    train_df = pd.read_csv(out_dir / "train.csv")
    summary = (
        f"Bridge dataset summary\n"
        f"  Splits  : train={419}, validation={71}, test={210}\n"
        f"  Columns : {train_df.columns.tolist()}\n"
        f"  Error types: {sorted(train_df['e'].dropna().unique().tolist())}\n"
        f"  Lesson topics: {sorted(train_df['lesson_topic'].dropna().unique().tolist())}\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("  Bridge fix complete.")


# ---------------------------------------------------------------------------
# Fix 2 – Eedi: download with correct config names
# ---------------------------------------------------------------------------

def download_eedi_configs() -> None:
    print("\n=== Fix [2/3] Eedi QATD-2k: downloading configs ===")
    out_dir = DATA / "raw_eedi_qatd"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = ["anchored-dialogues", "dq-question-metadata"]
    all_dialogues: list[dict] = []

    for cfg in configs:
        print(f"  Loading config: {cfg} …")
        ds = load_dataset("Eedi/Question-Anchored-Tutoring-Dialogues-2k", cfg)
        print(f"    Splits: {list(ds.keys())}")
        for split, dataset in ds.items():
            df = dataset.to_pandas()
            df["_split"] = split
            df["_config"] = cfg
            fname = f"{cfg.replace('-', '_')}_{split}.csv"
            df.to_csv(out_dir / fname, index=False)
            print(f"    {cfg}/{split}: {len(df):,} rows, cols: {list(df.columns)}")
            if cfg == "anchored-dialogues":
                all_dialogues.extend(df.to_dict(orient="records"))

    # Save combined dialogue file
    if all_dialogues:
        save_jsonl(all_dialogues, out_dir / "eedi_dialogues_all.jsonl")

    # Write provenance
    (out_dir / "PROVENANCE.md").write_text(
        "# Eedi Question-Anchored Tutoring Dialogues (QATD-2k)\n\n"
        "- Source: https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k\n"
        "- Paper: Zent, Smith, Woodhead (2025). PIIvot: A Lightweight NLP "
        "Anonymization Framework. EMNLP 2025.\n"
        "- License: CC BY-NC-SA 4.0\n"
        "- Configs downloaded: anchored-dialogues, dq-question-metadata\n"
        "- Description: 2,000 real 1:1 chat-based math tutoring sessions from "
        "the Eedi platform (UK, students age ~12, Nov 2021 – Feb 2025), with "
        "talk-move predictions and PII anonymization.\n"
        "- Relevance: Large-scale real platform data enabling cross-platform "
        "robustness analysis against MathDial and Bridge.\n",
        encoding="utf-8",
    )
    print("  Eedi QATD-2k download complete.")


# ---------------------------------------------------------------------------
# Fix 3 – TalkMoves: compile xlsx transcripts → combined CSV
# ---------------------------------------------------------------------------

def compile_talkmoves() -> None:
    print("\n=== Fix [3/3] TalkMoves: compiling transcript files ===")
    repo = DATA / "raw_talkmoves" / "TalkMoves_repo"
    out_dir = DATA / "raw_talkmoves"

    # 1. Use the pre-compiled training split (train_data_504.xlsx)
    main_file = repo / "data" / "train_data_504.xlsx"
    if main_file.exists():
        df_main = pd.read_excel(main_file)
        df_main.to_csv(out_dir / "talkmoves_train_data.csv", index=False)
        print(f"  train_data_504.xlsx: {len(df_main):,} sentences → talkmoves_train_data.csv")
    else:
        print("  WARNING: train_data_504.xlsx not found")
        df_main = pd.DataFrame()

    # 2. Also compile individual transcript xlsx files for reference
    subset_dfs: list[pd.DataFrame] = []
    xlsx_files = sorted(repo.rglob("Subset */*.xlsx"))
    print(f"  Found {len(xlsx_files)} individual transcript files")

    for f in xlsx_files:
        try:
            df = pd.read_excel(f, engine="openpyxl")
            # Normalise columns
            df.columns = [str(c).strip() for c in df.columns]
            df["source_file"] = f.name
            df["subset"] = f.parent.name
            subset_dfs.append(df)
        except Exception as exc:
            print(f"    SKIP {f.name}: {exc}")

    if subset_dfs:
        df_combined = pd.concat(subset_dfs, ignore_index=True)
        df_combined.to_csv(out_dir / "talkmoves_transcripts_all.csv", index=False)
        print(
            f"  Combined {len(subset_dfs)} transcripts → "
            f"{len(df_combined):,} rows → talkmoves_transcripts_all.csv"
        )

    # 3. Also save the student-task TSV (model training data format)
    tsv_file = repo / "data" / "train_student.tsv"
    if tsv_file.exists():
        df_tsv = pd.read_csv(tsv_file, sep="\t")
        df_tsv.to_csv(out_dir / "talkmoves_student_labels.csv", index=False)
        print(f"  train_student.tsv: {len(df_tsv):,} rows → talkmoves_student_labels.csv")

    # Write provenance
    (out_dir / "PROVENANCE.md").write_text(
        "# TalkMoves Dataset\n\n"
        "- Source: https://github.com/SumnerLab/TalkMoves\n"
        "- Paper: Suresh et al. (2022). The TalkMoves Dataset: K-12 "
        "Mathematics Lesson Transcripts Annotated for Teacher and Student "
        "Discursive Moves. LREC 2022.\n"
        "- License: CC BY-NC-SA 4.0\n"
        "- Files:\n"
        "    talkmoves_train_data.csv   – 504 training sentences with talk-move labels\n"
        "    talkmoves_transcripts_all.csv – all raw transcript xlsx files compiled\n"
        "    talkmoves_student_labels.csv  – sentence-pair + student move labels\n"
        "    TalkMoves_repo/              – full repository clone\n"
        "- Description: 567 human-annotated K-12 mathematics lesson "
        "transcripts with sentence-level labels for 10 discursive moves "
        "(accountable talk theory) and dialog-act labels.\n"
        "- Relevance: Classroom-level ecological context and cross-validated "
        "talk-move taxonomy bridging tutor-move categories in MathDial to "
        "broader classroom discourse research.\n",
        encoding="utf-8",
    )
    print("  TalkMoves compile complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    errors: list[str] = []

    for label, fn in [
        ("Bridge JSONL fix", fix_bridge_jsonl),
        ("Eedi QATD-2k download", download_eedi_configs),
        ("TalkMoves compile", compile_talkmoves),
    ]:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {label}: {exc}")
            traceback.print_exc()
            errors.append(f"{label}: {exc}")

    print("\n" + "=" * 60)
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All fixes and downloads completed successfully.")

    # Print final directory snapshot
    print("\nFinal data directory:")
    for d in sorted(DATA.iterdir()):
        if d.is_dir():
            files = list(d.iterdir())
            print(f"  {d.name}/  ({len(files)} items)")
