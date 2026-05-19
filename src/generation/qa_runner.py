"""
QA generation runner for v2 study design.

Generates naive-student responses to item bank items (MCQ, TF, Fill, SA) using
any LLM client in src/generation/clients/. Handles:
  - Checkpoint-resume (skips existing JSONs)
  - Cost tracking
  - Latency measurement (for D4 efficiency)
  - Demographic variant injection (for D5 CEAT)
  - Stratified sampling by difficulty tier

Output structure:
  outputs/qa/{phase}/{model_tag}/{qtype}/seed{seed}/{item_id}.json

Aggregated CSV:
  outputs/qa_responses_{model_tag}.csv

Usage:
    # Full P1 generation (single model):
    python src/generation/qa_runner.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --backend vllm \
        --qtypes MCQ TF Fill SA \
        --seeds 17 42 91 \
        --phase main

    # With demographic variants (D5 CEAT):
    python src/generation/qa_runner.py \
        --model gpt-4o-2024-11-20 \
        --backend openai \
        --qtypes MCQ \
        --seeds 42 \
        --phase ceat \
        --demo-variants gender race national ses

    # IRT baseline (Mistral-7B only, single seed):
    python src/generation/qa_runner.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --backend vllm \
        --qtypes MCQ TF Fill SA \
        --seeds 42 \
        --phase irt_baseline
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.generation.clients import LLMConfig, call_llm, infer_backend
from src.generation.personas.prompt_based import build_naive_student_prompt

BANK_ROOT = Path("data/item_bank")
OUTPUT_ROOT = Path("outputs/qa")
PERSONA_VERSION = "v2.0"

DEMO_VARIANT_CONTEXTS: dict[str, dict[str, str]] = {
    "gender": {
        "male":    "The student is a male student.",
        "female":  "The student is a female student.",
        "neutral": "The student is a student.",
        "omitted": "",
    },
    "race": {
        "White":    "The student is a White student.",
        "Asian":    "The student is an Asian student.",
        "Black":    "The student is a Black student.",
        "Hispanic": "The student is a Hispanic student.",
    },
    "national": {
        "US":      "The student is from the United States.",
        "Chinese": "The student is from China.",
        "Indian":  "The student is from India.",
        "British": "The student is from the United Kingdom.",
    },
    "ses": {
        "high-income":    "The student comes from a high-income family.",
        "low-income":     "The student comes from a low-income family.",
        "unspecified":    "The student comes from an average family.",
        "working-class":  "The student comes from a working-class family.",
    },
}

AUTOGRADE_QTYPES = {"MCQ", "TF"}
JUDGE_QTYPES = {"Fill", "SA"}


def model_tag(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_").replace(".", "_")


def load_items(qtype: str) -> list[dict]:
    paths = [
        BANK_ROOT / qtype.lower() / "items_with_difficulty.jsonl",
        BANK_ROOT / qtype.lower() / "items.jsonl",
    ]
    for p in paths:
        if p.exists():
            return [json.loads(l) for l in p.open(encoding="utf-8")]
    raise FileNotFoundError(
        f"No item bank found for {qtype}. Run src/data/build_item_bank.py first."
    )


def generate_response(
    item: dict,
    qtype: str,
    config: LLMConfig,
    seed: int,
    demographic_context: str = "",
) -> dict:
    system_prompt = build_naive_student_prompt(
        item, qtype=qtype,
        demographic_context=demographic_context or None,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please attempt the question now."},
    ]

    t0 = time.perf_counter()
    try:
        result = call_llm(messages, config, seed=seed)
        latency_ms = (time.perf_counter() - t0) * 1000
        response_text = result.content
        input_tokens = result.input_tokens
        output_tokens = result.output_tokens
        cost_usd = result.cost_usd
        error = None
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        error = str(e)

    correct, grade_method = _autograde(item, response_text, qtype)

    return {
        "item_id":            item["item_id"],
        "qtype":              qtype,
        "domain":             item.get("domain", ""),
        "concept":            item.get("concept", ""),
        "difficulty":         item.get("difficulty", item.get("difficulty_raw", "")),
        "question":           item.get("question", ""),
        "correct_answer":     item.get("correct_answer", ""),
        "persona":            "P1",
        "model_id":           config.model_id,
        "model_tag":          model_tag(config.model_id),
        "seed":               seed,
        "persona_version":    PERSONA_VERSION,
        "demographic_context": demographic_context,
        "generated_at":       datetime.utcnow().isoformat(),
        "response":           response_text,
        "correct":            correct,
        "grade_method":       grade_method,
        "input_tokens":       input_tokens,
        "output_tokens":      output_tokens,
        "latency_ms":         round(latency_ms, 1),
        "cost_usd":           round(cost_usd, 8),
        "error":              error,
        "system_prompt":      system_prompt,
    }


def _autograde(item: dict, response: str, qtype: str) -> tuple[int | None, str]:
    """Auto-grade MCQ and TF; returns (correct_int_or_None, grade_method_str)."""
    gold = str(item.get("correct_answer", "")).strip().upper()
    resp = response.strip()

    if qtype == "MCQ":
        for label in ["A", "B", "C", "D"]:
            if f"my answer is: {label.lower()}" in resp.lower():
                return (1 if label == gold else 0, "pattern_match")
        for label in ["A", "B", "C", "D"]:
            if resp.upper().startswith(label) or f"\n{label}." in resp or f"({label})" in resp.upper():
                return (1 if label == gold else 0, "prefix_match")
        return (None, "ungraded")

    if qtype == "TF":
        resp_lower = resp.lower()
        if resp_lower.startswith("true") or "is true" in resp_lower:
            return (1 if gold == "TRUE" else 0, "pattern_match")
        if resp_lower.startswith("false") or "is false" in resp_lower:
            return (1 if gold == "FALSE" else 0, "pattern_match")
        return (None, "ungraded")

    return (None, "judge_required")


def save_response(data: dict, output_dir: Path) -> Path:
    mtag = data["model_tag"]
    qtype = data["qtype"]
    seed = data["seed"]
    item_id = data["item_id"]
    demo_suffix = ""
    if data.get("demographic_context"):
        demo_suffix = "_" + data["demographic_context"].split()[2].lower()[:8]
    dest = output_dir / mtag / qtype / f"seed{seed}" / f"{item_id}{demo_suffix}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return dest


def aggregate_to_csv(output_dir: Path, mtag: str) -> None:
    rows = []
    for json_path in sorted((output_dir / mtag).rglob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        rows.append({
            k: data.get(k) for k in [
                "item_id", "qtype", "domain", "concept", "difficulty",
                "persona", "model_id", "model_tag", "seed",
                "demographic_context", "generated_at",
                "correct", "grade_method",
                "input_tokens", "output_tokens", "latency_ms", "cost_usd", "error",
            ]
        })
    if not rows:
        return
    df = pd.DataFrame(rows)
    out_csv = Path("outputs") / f"qa_responses_{mtag}.csv"
    df.to_csv(out_csv, index=False)
    total = len(df)
    graded = df["correct"].notna().sum()
    acc = df[df["correct"].notna()]["correct"].mean()
    total_cost = df["cost_usd"].sum()
    print(f"\nAggregated {total} responses for {mtag} → {out_csv}")
    print(f"Graded: {graded}/{total} | Accuracy (auto-graded): {acc:.3f}")
    print(f"Total cost: ${total_cost:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="QA runner for v2 naive-student study")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "openai", "openrouter", "anthropic",
                                 "google", "vllm"])
    parser.add_argument("--qtypes", nargs="+", default=["MCQ"],
                        choices=["MCQ", "TF", "Fill", "SA"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 91])
    parser.add_argument("--phase", default="main",
                        choices=["main", "irt_baseline", "ceat", "p2_eval"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--demo-variants", nargs="*", default=[],
                        choices=["gender", "race", "national", "ses"],
                        help="Run demographic variants for CEAT (D5)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    backend = args.backend if args.backend != "auto" else infer_backend(args.model)
    config = LLMConfig(
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_backend=backend,
    )

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    mtag = model_tag(args.model)
    print(f"Model: {args.model} | Backend: {backend} | Phase: {args.phase}")

    demo_contexts: list[str] = [""]
    if args.demo_variants:
        demo_contexts = []
        for attr_set in args.demo_variants:
            for cond_label, ctx in DEMO_VARIANT_CONTEXTS[attr_set].items():
                demo_contexts.append(ctx)

    total_cost = 0.0
    for qtype in args.qtypes:
        items = load_items(qtype)
        print(f"\n[{qtype}] {len(items)} items loaded")

        jobs = [
            (item, seed, demo_ctx)
            for item in items
            for seed in args.seeds
            for demo_ctx in demo_contexts
        ]
        print(f"  Jobs: {len(jobs)} (items × seeds × demo_variants)")

        skipped = 0
        for item, seed, demo_ctx in tqdm(jobs, desc=f"[{mtag}][{qtype}]"):
            demo_suffix = ""
            if demo_ctx:
                demo_suffix = "_" + demo_ctx.split()[2].lower()[:8]
            dest = (output_dir / mtag / qtype / f"seed{seed}"
                    / f"{item['item_id']}{demo_suffix}.json")
            if dest.exists():
                skipped += 1
                continue

            data = generate_response(item, qtype, config, seed, demo_ctx)
            save_response(data, output_dir)
            total_cost += data.get("cost_usd", 0.0)

        print(f"  Skipped (checkpoint): {skipped}")

    print(f"\nSession cost: ${total_cost:.4f}")
    aggregate_to_csv(output_dir, mtag)


if __name__ == "__main__":
    main()
