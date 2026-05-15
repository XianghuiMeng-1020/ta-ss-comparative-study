"""
Generate blinded coder packets for human coding.

Strips condition labels, hashes filenames, and produces:
  outputs/coder_packets/{sha8}_packet.txt   — blinded dialogue text
  outputs/coder_packets/manifest.csv        — sha8 → true condition mapping (LOCKED, not shared)
  outputs/coder_packets/rating_template.csv — blank rating sheet for coders

Dimensions coded (per codebook.docx):
  Framework-based (9 dimensions): TA1–TA4, SS1–SS5
  Use-case decision: learning_by_teaching / diagnostic_simulation /
                     feedback_policy_testing / reject
  Failure flags (6): premature_expertise, role_drift, over_technical,
                     misconception_loss, logical_inconsistency, unsupported_reasoning
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

DIMENSIONS = ["TA1", "TA2", "TA3", "TA4", "SS1", "SS2", "SS3", "SS4", "SS5"]
USE_CASES = ["learning_by_teaching", "diagnostic_simulation", "feedback_policy_testing", "reject"]
FAILURE_FLAGS = [
    "premature_expertise", "role_drift", "over_technical",
    "misconception_loss", "logical_inconsistency", "unsupported_reasoning",
]

PACKET_HEADER = """
╔══════════════════════════════════════════════════════════════════════╗
║  CODER PACKET — ID: {packet_id}                                     ║
║  Study: Framework-Grounded Evaluation of Simulated Learner Protocols ║
║  CONDITION LABEL HAS BEEN REMOVED — do not infer from filename       ║
╚══════════════════════════════════════════════════════════════════════╝

PROBLEM:
{problem}

ORIGINAL STUDENT ERROR / CONFUSION:
{original_incorrect_solution}

──────────────────────────────────────────────────────────────────────
DIALOGUE TRANSCRIPT
──────────────────────────────────────────────────────────────────────
{dialogue_text}

──────────────────────────────────────────────────────────────────────
END OF PACKET {packet_id}
──────────────────────────────────────────────────────────────────────
"""


def dialogue_to_text(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        tutor = t.get("tutor_turn", "").strip()
        learner = t.get("learner_turn", "").strip()
        if tutor:
            lines.append(f"[TUTOR]:   {tutor}")
        if learner:
            lines.append(f"[LEARNER]: {learner}")
        lines.append("")
    return "\n".join(lines)


def packet_id(json_path: str) -> str:
    return hashlib.sha256(json_path.encode()).hexdigest()[:8].upper()


def sample_for_coding(
    phase: str,
    n: int,
    calibration: bool,
    seed: int = 2026,
) -> list[Path]:
    """Stratified sample of n dialogue JSON files, balanced across conditions."""
    base = Path("outputs") / phase
    all_files = sorted(base.rglob("*.json"))
    valid = []
    for jf in all_files:
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        if not data.get("exclusion_flag", False):
            valid.append((jf, data.get("condition", "?")))

    rng = random.Random(seed + (1 if calibration else 0))
    rng.shuffle(valid)

    per_condition = n // 4
    buckets: dict[str, list[Path]] = {"C1": [], "C2": [], "C3": [], "C4": []}
    for jf, cond in valid:
        if cond in buckets and len(buckets[cond]) < per_condition:
            buckets[cond].append(jf)

    selected = [jf for files in buckets.values() for jf in files]
    return selected


def generate_packets(
    phase: str,
    n: int,
    calibration: bool,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = sample_for_coding(phase, n, calibration)
    print(f"Selected {len(selected)} dialogues for coder packets.")

    manifest_rows = []
    rating_rows = []

    for jf in selected:
        data = json.loads(jf.read_text())
        pid = packet_id(str(jf))

        # Fetch scenario fields (embedded in system_prompt fallback)
        problem = ""
        incorrect = ""
        sys_p = data.get("system_prompt", "")
        in_problem = False
        for line in sys_p.splitlines():
            if line.strip() == "PROBLEM:":
                in_problem = True
                continue
            if in_problem and line.strip():
                problem = line.strip()
                in_problem = False
        # Try turns
        turns = data.get("turns", [])
        # Generate dialogue text
        dialogue_text = dialogue_to_text(turns)

        packet_text = PACKET_HEADER.format(
            packet_id=pid,
            problem=problem or "(see dialogue)",
            original_incorrect_solution=incorrect or "(see first learner turn)",
            dialogue_text=dialogue_text,
        )

        packet_path = output_dir / f"{pid}_packet.txt"
        packet_path.write_text(packet_text)

        manifest_rows.append({
            "packet_id": pid,
            "true_condition": data.get("condition"),
            "scenario_id": data.get("scenario_id"),
            "seed": data.get("seed"),
            "json_path": str(jf),
            "calibration": calibration,
        })

        row = {"packet_id": pid, "coder_id": ""}
        for dim in DIMENSIONS:
            row[dim] = ""
        row["use_case_decision"] = ""
        for flag in FAILURE_FLAGS:
            row[f"flag_{flag}"] = ""
        row["notes"] = ""
        rating_rows.append(row)

    # Manifest — locked, not shared with coders
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Manifest saved (DO NOT SHARE WITH CODERS): {manifest_path}")

    # Rating template for coders
    rating_df = pd.DataFrame(rating_rows)
    suffix = "_calibration" if calibration else ""
    rating_path = output_dir / f"rating_template{suffix}.csv"
    rating_df.to_csv(rating_path, index=False)
    print(f"Rating template: {rating_path}")
    print(f"Packets written to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="main")
    parser.add_argument("--n", type=int, default=240, help="Number of dialogues to sample")
    parser.add_argument("--calibration", action="store_true",
                        help="Generate calibration set (binary trace labels only)")
    parser.add_argument("--output-dir", default="outputs/coder_packets")
    args = parser.parse_args()

    generate_packets(
        phase=args.phase,
        n=args.n,
        calibration=args.calibration,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
