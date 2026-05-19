"""
Item bank builder for v2 study design.

Produces data/item_bank/{mcq,tf,fill,sa}/items.jsonl with standardised schema.

Sources:
  MCQ (600):
    - 300 Python MCQ: Jiajia's dataset (cloned from GEMLab-HKU/Unlearn_and_Relearn)
      Expected path: data/raw_jiajia_mcq/python_mcqs.jsonl
      Fallback: generate 300 template stubs for manual completion.
    - 300 Math MCQ: derived from data/scenarios.csv (MathDial) + LLM-generated extensions.
  TF  (300): auto-derived from MCQ correct/distractors.
  Fill(300): cloze-deletion from MathDial reasoning steps.
  SA  (300): open-prompt from MathDial reasoning steps.

Run:
    python src/data/build_item_bank.py --mcq-python 300 --mcq-math 300 --tf 300 --fill 300 --sa 300
    python src/data/build_item_bank.py --verify       # counts and schema check only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

BANK_ROOT = Path("data/item_bank")
SCHEMA_REQUIRED = {"item_id", "qtype", "domain", "concept", "question", "correct_answer",
                   "difficulty_raw", "source"}

OPTION_LABELS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def make_item_id(text: str, qtype: str, idx: int) -> str:
    digest = hashlib.md5(f"{qtype}{idx}{text}".encode()).hexdigest()[:8].upper()
    return f"{qtype}_{digest}"


def write_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(items)} items → {path}")


def validate_items(items: list[dict]) -> list[str]:
    errors = []
    ids_seen: set[str] = set()
    for i, item in enumerate(items):
        missing = SCHEMA_REQUIRED - item.keys()
        if missing:
            errors.append(f"Item {i}: missing fields {missing}")
        if item.get("item_id") in ids_seen:
            errors.append(f"Item {i}: duplicate item_id {item['item_id']}")
        ids_seen.add(item.get("item_id"))
    return errors


# ---------------------------------------------------------------------------
# MCQ — Python (from Jiajia dataset)
# ---------------------------------------------------------------------------

def load_python_mcqs(n: int) -> list[dict]:
    """Load Python MCQ from Jiajia dataset; fall back to stubs if not present."""
    raw_path = Path("data/raw_jiajia_mcq/python_mcqs.jsonl")
    items = []
    if raw_path.exists():
        with raw_path.open() as f:
            for line in f:
                row = json.loads(line)
                items.append(row)
        items = items[:n]
        return [_normalise_python_mcq(row, i) for i, row in enumerate(items)]

    print(f"  [WARN] {raw_path} not found — generating {n} stub Python MCQ items.")
    print(f"  Clone https://github.com/GEMLab-HKU/Unlearn_and_Relearn and copy the MCQ jsonl there.")
    return _generate_python_stubs(n)


def _normalise_python_mcq(row: dict, idx: int) -> dict:
    question = row.get("question", row.get("question_description", ""))
    options_raw = row.get("options", row.get("choices", {}))
    if isinstance(options_raw, list):
        options = {OPTION_LABELS[i]: opt for i, opt in enumerate(options_raw[:4])}
    else:
        options = options_raw
    correct = row.get("correct_answer", row.get("answer", "A"))
    if isinstance(correct, int):
        correct = OPTION_LABELS[correct]

    return {
        "item_id": make_item_id(question, "MCQ", idx),
        "qtype": "MCQ",
        "domain": "python_programming",
        "concept": row.get("concept", row.get("category", "python_general")),
        "question": question,
        "options": options,
        "correct_answer": str(correct).upper(),
        "explanation": row.get("explanation", ""),
        "difficulty_raw": row.get("difficulty", None),
        "source": "jiajia_python_mcq",
        "unlearning_eligible": True,
    }


def _generate_python_stubs(n: int) -> list[dict]:
    concepts = [
        "functions", "loops", "data_types", "lists", "dictionaries",
        "operators", "classes", "exceptions", "file_io", "modules",
    ]
    stubs = []
    for i in range(n):
        concept = concepts[i % len(concepts)]
        stubs.append({
            "item_id": make_item_id(f"stub_python_{i}", "MCQ", i),
            "qtype": "MCQ",
            "domain": "python_programming",
            "concept": concept,
            "question": f"[STUB — REPLACE] Python {concept} question {i+1}",
            "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
            "correct_answer": "A",
            "explanation": "",
            "difficulty_raw": None,
            "source": "stub",
            "unlearning_eligible": True,
        })
    return stubs


# ---------------------------------------------------------------------------
# MCQ — Math (from MathDial scenarios)
# ---------------------------------------------------------------------------

def build_math_mcqs(n: int) -> list[dict]:
    scenarios_path = Path("data/scenarios.csv")
    if not scenarios_path.exists():
        print(f"  [WARN] {scenarios_path} not found — generating {n} stub math MCQ items.")
        return _generate_math_stubs(n)

    df = pd.read_csv(scenarios_path)
    items = []
    for i, row in df.iterrows():
        if len(items) >= n:
            break
        problem = str(row.get("problem", ""))
        if not problem or len(problem) < 20:
            continue
        correct = str(row.get("correct_answer", row.get("answer", "UNKNOWN")))
        distractors = _make_math_distractors(correct, row)
        options = {"A": correct}
        options.update({OPTION_LABELS[j + 1]: d for j, d in enumerate(distractors[:3])})
        keys = list(options.keys())
        random.shuffle(keys)
        shuffled = {k: options[k] for k in keys}
        correct_key = [k for k, v in shuffled.items() if v == correct][0]

        items.append({
            "item_id": make_item_id(problem, "MCQ_math", i),
            "qtype": "MCQ",
            "domain": "mathematics",
            "concept": str(row.get("topic", "math_general")),
            "question": f"Solve the following problem and choose the correct answer:\n\n{problem}",
            "options": shuffled,
            "correct_answer": correct_key,
            "explanation": str(row.get("original_incorrect_solution", "")),
            "difficulty_raw": str(row.get("difficulty", None)),
            "error_type": str(row.get("error_type", "")),
            "source": "mathdial_scenario",
            "scenario_id": str(row.get("scenario_id", "")),
            "unlearning_eligible": False,
        })

    while len(items) < n:
        items.extend(_generate_math_stubs(n - len(items)))

    return items[:n]


def _make_math_distractors(correct: str, row) -> list[str]:
    try:
        val = float(re.sub(r"[^\d.\-]", "", correct))
        return [str(round(val * 2, 2)), str(round(val + 5, 2)), str(round(val / 2, 2))]
    except Exception:
        return ["Option B", "Option C", "Option D"]


def _generate_math_stubs(n: int) -> list[dict]:
    stubs = []
    topics = ["fractions", "algebra", "geometry", "percentages", "multi_step"]
    for i in range(n):
        topic = topics[i % len(topics)]
        stubs.append({
            "item_id": make_item_id(f"stub_math_{i}", "MCQ_math", 9000 + i),
            "qtype": "MCQ",
            "domain": "mathematics",
            "concept": topic,
            "question": f"[STUB — REPLACE] Math {topic} MCQ question {i+1}",
            "options": {"A": "Correct answer", "B": "Distractor 1", "C": "Distractor 2", "D": "Distractor 3"},
            "correct_answer": "A",
            "explanation": "",
            "difficulty_raw": None,
            "source": "stub",
            "unlearning_eligible": False,
        })
    return stubs


# ---------------------------------------------------------------------------
# TF — from MCQ
# ---------------------------------------------------------------------------

def build_tf_from_mcq(mcq_items: list[dict], n: int) -> list[dict]:
    tf_items = []
    random.seed(42)
    for i, mcq in enumerate(mcq_items):
        if len(tf_items) >= n:
            break
        correct_key = mcq["correct_answer"]
        correct_text = mcq["options"].get(correct_key, "")
        if not correct_text or correct_text.startswith("[STUB"):
            continue

        distractor_keys = [k for k in mcq["options"] if k != correct_key]
        if not distractor_keys:
            continue
        distractor_text = mcq["options"][random.choice(distractor_keys)]

        # True statement (correct answer is true)
        tf_items.append({
            "item_id": make_item_id(correct_text, "TF_T", i),
            "qtype": "TF",
            "domain": mcq["domain"],
            "concept": mcq["concept"],
            "question": (
                f"Context: {mcq['question']}\n\n"
                f"Statement: {correct_text}\n\nIs this statement True or False?"
            ),
            "correct_answer": "True",
            "source_item_id": mcq["item_id"],
            "difficulty_raw": mcq.get("difficulty_raw"),
            "source": f"derived_from_mcq:{mcq['source']}",
            "unlearning_eligible": mcq.get("unlearning_eligible", False),
        })

        if len(tf_items) >= n:
            break

        # False statement (distractor is false)
        tf_items.append({
            "item_id": make_item_id(distractor_text, "TF_F", i),
            "qtype": "TF",
            "domain": mcq["domain"],
            "concept": mcq["concept"],
            "question": (
                f"Context: {mcq['question']}\n\n"
                f"Statement: {distractor_text}\n\nIs this statement True or False?"
            ),
            "correct_answer": "False",
            "source_item_id": mcq["item_id"],
            "difficulty_raw": mcq.get("difficulty_raw"),
            "source": f"derived_from_mcq:{mcq['source']}",
            "unlearning_eligible": mcq.get("unlearning_eligible", False),
        })

    return tf_items[:n]


# ---------------------------------------------------------------------------
# Fill and SA — from MathDial reasoning steps
# ---------------------------------------------------------------------------

def build_fill_and_sa(n_fill: int, n_sa: int) -> tuple[list[dict], list[dict]]:
    scenarios_path = Path("data/scenarios.csv")
    fill_items: list[dict] = []
    sa_items: list[dict] = []

    if not scenarios_path.exists():
        print(f"  [WARN] {scenarios_path} not found — generating stubs.")
        return _fill_stubs(n_fill), _sa_stubs(n_sa)

    df = pd.read_csv(scenarios_path)

    for i, row in df.iterrows():
        problem = str(row.get("problem", ""))
        if not problem or len(problem) < 20:
            continue

        concept = str(row.get("topic", "math_general"))
        difficulty = str(row.get("difficulty", None))
        domain = "mathematics"
        source = "mathdial_scenario"
        scenario_id = str(row.get("scenario_id", ""))
        incorrect = str(row.get("original_incorrect_solution", ""))

        if len(fill_items) < n_fill:
            blanked, answer = _cloze_deletion(problem, incorrect)
            if blanked and answer:
                fill_items.append({
                    "item_id": make_item_id(blanked, "Fill", i),
                    "qtype": "Fill",
                    "domain": domain,
                    "concept": concept,
                    "question": blanked,
                    "correct_answer": answer,
                    "difficulty_raw": difficulty,
                    "source": source,
                    "scenario_id": scenario_id,
                    "unlearning_eligible": False,
                })

        if len(sa_items) < n_sa:
            sa_q = _make_sa_question(problem, concept)
            sa_items.append({
                "item_id": make_item_id(sa_q, "SA", i),
                "qtype": "SA",
                "domain": domain,
                "concept": concept,
                "question": sa_q,
                "correct_answer": str(row.get("correct_answer", row.get("answer", ""))),
                "difficulty_raw": difficulty,
                "source": source,
                "scenario_id": scenario_id,
                "unlearning_eligible": False,
            })

        if len(fill_items) >= n_fill and len(sa_items) >= n_sa:
            break

    while len(fill_items) < n_fill:
        fill_items.extend(_fill_stubs(n_fill - len(fill_items)))
    while len(sa_items) < n_sa:
        sa_items.extend(_sa_stubs(n_sa - len(sa_items)))

    return fill_items[:n_fill], sa_items[:n_sa]


def _cloze_deletion(problem: str, incorrect: str) -> tuple[str, str]:
    nums = re.findall(r"\b\d+(?:\.\d+)?\b", problem)
    if not nums:
        return "", ""
    target = nums[-1]
    blanked = problem.replace(target, "_______", 1)
    if blanked == problem:
        return "", ""
    return (
        f"Fill in the blank with the missing number:\n\n{blanked}",
        target,
    )


def _make_sa_question(problem: str, concept: str) -> str:
    return (
        f"Solve the following {concept} problem. Show your reasoning step by step.\n\n{problem}"
    )


def _fill_stubs(n: int) -> list[dict]:
    return [{
        "item_id": make_item_id(f"stub_fill_{i}", "Fill", 8000 + i),
        "qtype": "Fill",
        "domain": "mathematics",
        "concept": "arithmetic",
        "question": f"[STUB — REPLACE] Fill-in-blank question {i+1}: The answer to 3 × _______ = 12 is?",
        "correct_answer": "4",
        "difficulty_raw": None,
        "source": "stub",
        "unlearning_eligible": False,
    } for i in range(n)]


def _sa_stubs(n: int) -> list[dict]:
    return [{
        "item_id": make_item_id(f"stub_sa_{i}", "SA", 7000 + i),
        "qtype": "SA",
        "domain": "mathematics",
        "concept": "arithmetic",
        "question": f"[STUB — REPLACE] Short-answer question {i+1}: What is 15% of 200?",
        "correct_answer": "30",
        "difficulty_raw": None,
        "source": "stub",
        "unlearning_eligible": False,
    } for i in range(n)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build item bank for v2 study")
    parser.add_argument("--mcq-python", type=int, default=300)
    parser.add_argument("--mcq-math", type=int, default=300)
    parser.add_argument("--tf", type=int, default=300)
    parser.add_argument("--fill", type=int, default=300)
    parser.add_argument("--sa", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true",
                        help="Only verify existing item bank files, do not rebuild")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.verify:
        for qtype, fname in [("mcq","mcq"), ("tf","tf"), ("fill","fill"), ("sa","sa")]:
            path = BANK_ROOT / fname / "items.jsonl"
            if not path.exists():
                print(f"MISSING: {path}")
                continue
            items = [json.loads(l) for l in path.open()]
            errs = validate_items(items)
            qtype_counts = {}
            for it in items:
                qtype_counts[it.get("qtype","?")] = qtype_counts.get(it.get("qtype","?"), 0) + 1
            print(f"{path}: {len(items)} items | types: {qtype_counts} | errors: {len(errs)}")
            for e in errs[:5]:
                print(f"  {e}")
        return

    print("=== Building MCQ Python ===")
    py_mcq = load_python_mcqs(args.mcq_python)
    print(f"  Loaded {len(py_mcq)} Python MCQ items")

    print("=== Building MCQ Math ===")
    math_mcq = build_math_mcqs(args.mcq_math)
    print(f"  Built {len(math_mcq)} Math MCQ items")

    all_mcq = py_mcq + math_mcq
    write_jsonl(all_mcq, BANK_ROOT / "mcq" / "items.jsonl")

    print("=== Building TF ===")
    tf_items = build_tf_from_mcq(all_mcq, args.tf)
    write_jsonl(tf_items, BANK_ROOT / "tf" / "items.jsonl")

    print("=== Building Fill + SA ===")
    fill_items, sa_items = build_fill_and_sa(args.fill, args.sa)
    write_jsonl(fill_items, BANK_ROOT / "fill" / "items.jsonl")
    write_jsonl(sa_items, BANK_ROOT / "sa" / "items.jsonl")

    print("\n=== Validation ===")
    for items, label in [(all_mcq, "MCQ"), (tf_items, "TF"),
                         (fill_items, "Fill"), (sa_items, "SA")]:
        errs = validate_items(items)
        stub_count = sum(1 for it in items if it.get("source") == "stub")
        print(f"  {label}: {len(items)} items | stubs: {stub_count} | schema errors: {len(errs)}")

    print("\nDone. Review stub items before P1 generation.")


if __name__ == "__main__":
    main()
