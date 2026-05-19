"""
QA item packetizer for v2 augmented human annotation.

Generates blinded coder packets for 480 single-response items (MCQ/TF/Fill/SA).
Each packet contains:
  - The question (blinded: concept only, not full domain)
  - The model's response (with model identity stripped)
  - Rating scales for D1 (human-likeness) and D2 (explanation quality)

Packet format matches the existing OED packet style for consistency.

Output:
  outputs/coder_packets_qa/{sha8}_qa_packet.txt   — blinded item response
  outputs/coder_packets_qa/manifest.csv           — sha8 → model/item mapping (LOCKED)
  outputs/coder_packets_qa/rating_template.csv    — blank rating sheet

Dimensions coded:
  HL1: Overall student-likeness (1–5 Likert)
  HL2: Error plausibility (1–5 Likert)
  HL3: Turing binary (0=AI, 1=human student)
  EQ1: Explanation quality (1–5 Likert; SA/OED only)

Usage:
    python src/coding/qa_packetize.py \
        --responses outputs/qa_responses_*.csv \
        --n 480 \
        --stratify-by model_tag qtype \
        --output-dir outputs/coder_packets_qa
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import random
from pathlib import Path

import pandas as pd

QA_DIMENSIONS = ["HL1", "HL2", "HL3", "EQ1"]

PACKET_HEADER = """
╔══════════════════════════════════════════════════════════════════════╗
║  QA CODER PACKET — ID: {packet_id}                                  ║
║  Study: Naive Student Simulation — v2 Human Annotation              ║
║  MODEL IDENTITY AND ITEM SOURCE HAVE BEEN REMOVED                   ║
╚══════════════════════════════════════════════════════════════════════╝

QUESTION TYPE: {qtype}
CONCEPT AREA:  {concept}
DIFFICULTY:    {difficulty}

QUESTION:
{question}

──────────────────────────────────────────────────────────────────────
STUDENT RESPONSE
──────────────────────────────────────────────────────────────────────
{response}

──────────────────────────────────────────────────────────────────────
RATING INSTRUCTIONS
──────────────────────────────────────────────────────────────────────

HL1 — Overall student-likeness (1–5):
  5 = Clearly sounds like a real novice student
  4 = Mostly student-like with minor inconsistencies
  3 = Partially student-like; some expert or AI tells
  2 = Mostly sounds like an AI or expert
  1 = Clearly an AI / expert-level response

HL2 — Error plausibility (1–5):
  5 = Errors are exactly what a novice student would make
  4 = Errors are realistic; minor implausibilities
  3 = Some plausible errors; some odd choices
  2 = Errors are unlikely or unrealistic for a student
  1 = No errors, or errors that no student would make

HL3 — Turing judgment (forced-choice):
  1 = This was written by a HUMAN student
  0 = This was written by an AI

EQ1 — Explanation quality [SA/OED ONLY; skip for MCQ/TF] (1–5):
  5 = Rich, clear reasoning showing some understanding
  4 = Adequate reasoning with minor gaps
  3 = Partial reasoning; key steps missing or confused
  2 = Minimal reasoning; mostly guesses
  1 = No real reasoning; incoherent

──────────────────────────────────────────────────────────────────────
END OF QA PACKET {packet_id}
──────────────────────────────────────────────────────────────────────
"""

LEAKAGE_TERMS = [
    "as an ai", "as an llm", "as a language model", "i was trained",
    "my training data", "gpt", "claude", "mistral", "llama", "qwen",
    "phi-3", "smollm", "tinyllama",
]


def strip_model_identity(text: str) -> str:
    text_lower = text.lower()
    for term in LEAKAGE_TERMS:
        if term in text_lower:
            text = text.replace(term, "[REDACTED]")
            text = text.replace(term.upper(), "[REDACTED]")
    return text


def make_packet_id(row: pd.Series, idx: int) -> str:
    key = f"qa_{idx}_{row.get('item_id', '')}_{row.get('model_tag', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:8].upper()


def sample_stratified(df: pd.DataFrame, n: int, stratify_cols: list[str],
                      seed: int = 42) -> pd.DataFrame:
    groups = df.groupby(stratify_cols)
    n_groups = len(groups)
    per_group = max(1, n // n_groups)

    sampled = []
    rng = random.Random(seed)
    for _, group_df in groups:
        n_sample = min(per_group, len(group_df))
        sampled.append(group_df.sample(n_sample, random_state=seed))

    combined = pd.concat(sampled, ignore_index=True)
    if len(combined) > n:
        combined = combined.sample(n, random_state=seed)
    return combined.reset_index(drop=True)


def write_packet(row: pd.Series, packet_id: str, output_dir: Path) -> None:
    response_clean = strip_model_identity(str(row.get("response", "")))
    packet_text = PACKET_HEADER.format(
        packet_id=packet_id,
        qtype=row.get("qtype", ""),
        concept=row.get("concept", ""),
        difficulty=row.get("difficulty", ""),
        question=row.get("question", ""),
        response=response_clean,
    )
    dest = output_dir / f"{packet_id}_qa_packet.txt"
    dest.write_text(packet_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="QA item packetizer for D1/D2 coding")
    parser.add_argument("--responses", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=480)
    parser.add_argument("--stratify-by", nargs="+", default=["model_tag", "qtype"])
    parser.add_argument("--output-dir", default="outputs/coder_packets_qa")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = []
    for pattern in args.responses:
        paths.extend(glob.glob(pattern))

    dfs = [pd.read_csv(p) for p in paths if Path(p).exists()]
    if not dfs:
        print("No response CSVs found.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["response"])
    df = df[df["response"].str.strip().ne("")]
    print(f"Total responses: {len(df)}")

    stratify_cols = [c for c in args.stratify_by if c in df.columns]
    sampled = sample_stratified(df, args.n, stratify_cols, seed=args.seed)
    print(f"Sampled: {len(sampled)} items for coding")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rating_rows = []

    for idx, row in sampled.iterrows():
        packet_id = make_packet_id(row, idx)
        write_packet(row, packet_id, output_dir)
        manifest_rows.append({
            "packet_id":  packet_id,
            "item_id":    row.get("item_id"),
            "model_tag":  row.get("model_tag"),
            "model_id":   row.get("model_id"),
            "qtype":      row.get("qtype"),
            "difficulty": row.get("difficulty"),
            "concept":    row.get("concept"),
            "seed":       row.get("seed"),
        })
        for coder_id in ["CoderA", "CoderB"]:
            rating_rows.append({
                "packet_id": packet_id,
                "coder_id":  coder_id,
                "qtype":     row.get("qtype"),
                "HL1": "", "HL2": "", "HL3": "",
                "EQ1": "",
                "notes": "",
            })

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    pd.DataFrame(rating_rows).to_csv(output_dir / "rating_template.csv", index=False)

    print(f"\n{len(sampled)} packets written → {output_dir}")
    print(f"Manifest (KEEP SECRET from coders): {output_dir}/manifest.csv")
    print(f"Rating template: {output_dir}/rating_template.csv")

    qtype_counts = sampled["qtype"].value_counts()
    model_counts = sampled["model_tag"].value_counts()
    print(f"\nQType distribution:\n{qtype_counts.to_string()}")
    print(f"\nModel distribution:\n{model_counts.to_string()}")


if __name__ == "__main__":
    main()
