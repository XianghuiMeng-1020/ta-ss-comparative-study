"""
Download the 3 supplementary real-world tutoring datasets for the
Computers & Education cross-dataset analysis.

Datasets:
  1. Bridge (rose-e-wang/bridge) – NAACL 2024
     700 real math tutoring conversations with student error type,
     teacher strategy and intention annotations.
     License: CC-BY-NC-4.0

  2. Eedi QATD-2k (Eedi/Question-Anchored-Tutoring-Dialogues-2k) – EMNLP 2025
     2,000 real one-on-one chat-based math tutoring dialogues from the
     Eedi platform (UK, age ~12), with talk-move annotations and
     anonymised PII.
     License: CC BY-NC-SA 4.0

  3. TalkMoves (SumnerLab/TalkMoves via GitHub) – LREC 2022
     567 annotated K-12 mathematics lesson transcripts with teacher and
     student discursive move labels (accountable talk theory).
     License: CC BY-NC-SA 4.0

All three are real human-produced corpora; NO synthetic data.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):,} rows → {path}")


# ---------------------------------------------------------------------------
# Dataset 1 – Bridge
# ---------------------------------------------------------------------------

def download_bridge() -> None:
    print("\n=== [1/3] Bridge (rose-e-wang/bridge) ===")
    out_dir = DATA / "raw_bridge"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("rose-e-wang/bridge", trust_remote_code=True)
    print(f"  Splits: {list(ds.keys())}")

    all_records: list[dict] = []
    for split, dataset in ds.items():
        df = dataset.to_pandas()
        df["_split"] = split
        save_csv(df, out_dir / f"{split}.csv")
        all_records.extend(df.to_dict(orient="records"))
        print(f"  {split}: {len(df)} rows, columns: {list(df.columns)}")

    save_jsonl(all_records, out_dir / "bridge_all.jsonl")

    # Write a brief provenance note
    readme = out_dir / "PROVENANCE.md"
    readme.write_text(
        "# Bridge Dataset\n\n"
        "- Source: https://huggingface.co/datasets/rose-e-wang/bridge\n"
        "- Paper: Wang et al. (2024). Bridging the Novice-Expert Gap via "
        "Models of Decision-Making. NAACL 2024.\n"
        "- License: CC-BY-NC-4.0\n"
        "- Description: 700 real math tutoring conversations annotated with "
        "student error type (`e`), teacher remediation strategy (`z_what`), "
        "and teacher intention (`z_why`).\n"
        "- Relevance: Provides expert-labelled error-type taxonomy and "
        "novice-vs-expert tutor responses – directly usable as a third "
        "ecological validity check alongside MathDial scenarios.\n",
        encoding="utf-8",
    )
    print("  Bridge download complete.")


# ---------------------------------------------------------------------------
# Dataset 2 – Eedi QATD-2k
# ---------------------------------------------------------------------------

def download_eedi() -> None:
    print("\n=== [2/3] Eedi QATD-2k (Eedi/Question-Anchored-Tutoring-Dialogues-2k) ===")
    out_dir = DATA / "raw_eedi_qatd"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "Eedi/Question-Anchored-Tutoring-Dialogues-2k",
        trust_remote_code=True,
    )
    print(f"  Splits / configs: {list(ds.keys())}")

    all_records: list[dict] = []
    for split, dataset in ds.items():
        df = dataset.to_pandas()
        df["_split"] = split
        save_csv(df, out_dir / f"{split}.csv")
        all_records.extend(df.to_dict(orient="records"))
        print(f"  {split}: {len(df)} rows, columns: {list(df.columns)}")

    save_jsonl(all_records, out_dir / "eedi_all.jsonl")

    readme = out_dir / "PROVENANCE.md"
    readme.write_text(
        "# Eedi Question-Anchored Tutoring Dialogues (QATD-2k)\n\n"
        "- Source: https://huggingface.co/datasets/Eedi/Question-Anchored-Tutoring-Dialogues-2k\n"
        "- Paper: Zent, Smith, Woodhead (2025). PIIvot: A Lightweight NLP "
        "Anonymization Framework. EMNLP 2025.\n"
        "- License: CC BY-NC-SA 4.0\n"
        "- Description: 2,000 real 1:1 chat-based math tutoring sessions from "
        "the Eedi platform (UK, students age ~12), with talk-move predictions "
        "and PII anonymization.\n"
        "- Relevance: Large-scale real platform data from a distinct student "
        "population (UK, younger) and delivery mode (async chat) – enables "
        "cross-platform robustness analysis.\n",
        encoding="utf-8",
    )
    print("  Eedi QATD-2k download complete.")


# ---------------------------------------------------------------------------
# Dataset 3 – TalkMoves
# ---------------------------------------------------------------------------

def download_talkmoves() -> None:
    print("\n=== [3/3] TalkMoves (SumnerLab/TalkMoves) ===")
    out_dir = DATA / "raw_talkmoves"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/SumnerLab/TalkMoves.git"
    clone_target = out_dir / "TalkMoves_repo"

    if clone_target.exists():
        print(f"  Repo already cloned at {clone_target}; pulling latest …")
        subprocess.run(
            ["git", "-C", str(clone_target), "pull", "--ff-only"],
            check=True,
        )
    else:
        print(f"  Cloning {repo_url} …")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_target)],
            check=True,
        )

    # Locate annotation CSV files in the repo and copy to flat directory
    csv_files = list(clone_target.rglob("*.csv"))
    json_files = list(clone_target.rglob("*.json"))
    print(f"  Found {len(csv_files)} CSV + {len(json_files)} JSON files in repo")

    for f in csv_files + json_files:
        dest = out_dir / f.name
        import shutil
        shutil.copy2(f, dest)
        print(f"    Copied {f.name} ({f.stat().st_size // 1024} KB)")

    readme = out_dir / "PROVENANCE.md"
    readme.write_text(
        "# TalkMoves Dataset\n\n"
        "- Source: https://github.com/SumnerLab/TalkMoves\n"
        "- Paper: Suresh et al. (2022). The TalkMoves Dataset: K-12 "
        "Mathematics Lesson Transcripts Annotated for Teacher and Student "
        "Discursive Moves. LREC 2022.\n"
        "- License: CC BY-NC-SA 4.0\n"
        "- Description: 567 human-annotated K-12 mathematics lesson "
        "transcripts with sentence-level labels for ten discursive moves "
        "(accountable talk theory) and utterance-level dialog-act labels.\n"
        "- Relevance: Provides classroom-level ecological context and a "
        "cross-validated talk-move taxonomy that bridges our tutor-move "
        "categories in MathDial to broader classroom discourse research.\n",
        encoding="utf-8",
    )
    print("  TalkMoves download complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Downloading 3 supplementary real-world tutoring datasets …")
    errors: list[str] = []

    for name, fn in [
        ("Bridge", download_bridge),
        ("Eedi QATD-2k", download_eedi),
        ("TalkMoves", download_talkmoves),
    ]:
        try:
            fn()
        except Exception as exc:
            print(f"\n[ERROR] {name}: {exc}", file=sys.stderr)
            errors.append(f"{name}: {exc}")

    print("\n" + "=" * 60)
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All 3 datasets downloaded successfully.")
        print(f"Data directory: {DATA}")
