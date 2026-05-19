"""
Generate 300 Python MCQ items via GPT-4o-mini and save to
data/raw_jiajia_mcq/python_mcqs.jsonl

Topics aligned with intro-CS / Jiajia-unlearning dataset scope.
Run:
    python src/data/generate_python_mcqs.py
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

OUTDIR = Path("data/raw_jiajia_mcq")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUTDIR / "python_mcqs.jsonl"

TOPICS = [
    ("variables_and_types",       25, "variable assignment, data types (int, float, str, bool), type casting"),
    ("operators",                 20, "arithmetic, comparison, logical, bitwise, assignment operators"),
    ("strings",                   25, "string methods (upper, lower, split, join, strip, replace), slicing, f-strings"),
    ("lists",                     30, "list creation, indexing, slicing, append, pop, sort, list comprehensions"),
    ("tuples_and_sets",           15, "tuple immutability, set operations (union, intersection, difference)"),
    ("dictionaries",              25, "dict creation, get, update, keys/values/items, dict comprehensions"),
    ("control_flow",              25, "if/elif/else, nested conditions, short-circuit evaluation"),
    ("loops",                     30, "for loop, while loop, break, continue, range, enumerate, zip"),
    ("functions",                 30, "def, return, default args, *args, **kwargs, scope (LEGB)"),
    ("exceptions",                20, "try/except/finally, raise, common exceptions (ValueError, TypeError, etc.)"),
    ("file_io",                   15, "open, read, write, with statement, modes"),
    ("modules_and_imports",       15, "import, from...import, as alias, common stdlib modules"),
    ("classes_and_oop",           25, "class, __init__, self, inheritance, super, @staticmethod, @classmethod"),
]

SYSTEM_PROMPT = """You are an expert Python instructor creating multiple-choice quiz questions for introductory-level students.
Each question must be:
- Concise (3–5 lines max for question + code if any)
- Have exactly 4 options labeled A, B, C, D
- Have ONE unambiguously correct answer
- Test a concept a naive beginner student commonly gets wrong
- Written in English

Output ONLY a valid JSON object with these exact keys:
{
  "question": "<question text, may include short code snippet>",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "<A|B|C|D>",
  "explanation": "<1-2 sentence explanation of why the correct answer is right>",
  "concept": "<one of the topic keywords>",
  "difficulty": "<easy|medium|hard>"
}
"""


def generate_batch(topic: str, description: str, n: int, client: OpenAI) -> list[dict]:
    items = []
    difficulties = ["easy"] * (n // 3) + ["medium"] * (n // 3) + ["hard"] * (n - 2 * (n // 3))
    random.shuffle(difficulties)

    for i, diff in enumerate(difficulties):
        user_msg = (
            f"Generate a {diff} Python MCQ about: {topic} ({description}). "
            f"This is question {i+1} of {n} on this topic. Make it distinct from typical examples."
        )
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.9,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                )
                raw = json.loads(resp.choices[0].message.content)
                raw["concept"] = raw.get("concept", topic)
                raw["difficulty"] = raw.get("difficulty", diff)
                raw["source_topic"] = topic
                items.append(raw)
                break
            except Exception as e:
                print(f"    attempt {attempt+1} failed: {e}")
                time.sleep(2)

    return items


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        dotenv = Path(".env")
        if dotenv.exists():
            for line in dotenv.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in env or .env")

    client = OpenAI(api_key=api_key)
    all_items: list[dict] = []

    # Resume if partial file exists
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            all_items = [json.loads(l) for l in f if l.strip()]
        print(f"Resuming — {len(all_items)} items already generated.")

    target = 300
    generated_per_topic = {topic: 0 for topic, _, _ in TOPICS}
    for item in all_items:
        t = item.get("source_topic", "")
        if t in generated_per_topic:
            generated_per_topic[t] += 1

    for topic, quota, description in TOPICS:
        already = generated_per_topic.get(topic, 0)
        remaining = quota - already
        if remaining <= 0:
            print(f"  {topic}: already at quota ({already}/{quota}), skipping.")
            continue
        if len(all_items) >= target:
            break

        print(f"\n=== {topic}: generating {remaining} items ({description[:50]}…) ===")
        new_items = generate_batch(topic, description, remaining, client)
        all_items.extend(new_items)

        # Save incrementally
        with OUT_PATH.open("w") as f:
            for it in all_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  Saved {len(all_items)} total items so far.")

    all_items = all_items[:target]
    with OUT_PATH.open("w") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\nDone — {len(all_items)} Python MCQ items saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
