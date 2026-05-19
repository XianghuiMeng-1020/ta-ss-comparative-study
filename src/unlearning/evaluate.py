"""
P2 unlearning evaluation — accuracy and F1 on forget/retain sets.

Replicates Table-1 of Jiajia et al. (2026) and extends to Qwen2.5-3B.

Usage:
    python src/unlearning/evaluate.py \
        --adapter-dir outputs/unlearning/mistral_7b_instruct_v0_3/ratio30_seed42 \
        --items data/item_bank/mcq/items.jsonl \
        --output outputs/unlearning_eval.csv

    # All adapters:
    python src/unlearning/evaluate.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

OUTPUT_ROOT = Path("outputs/unlearning")
ITEMS_PATH = Path("data/item_bank/mcq/items.jsonl")


def evaluate_adapter(
    adapter_dir: Path,
    items_path: Path = ITEMS_PATH,
    max_items: int = 200,
    batch_size: int = 4,
) -> dict:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError:
        return {"error": "transformers or peft not installed"}

    meta_path = adapter_dir / "training_metrics.json"
    if not meta_path.exists():
        return {"error": f"training_metrics.json not found in {adapter_dir}"}

    meta = json.loads(meta_path.read_text())
    base_model_id = meta["base_model"]
    unlearn_ratio = meta["unlearn_ratio"]
    seed = meta["seed"]
    n_forget = meta["n_forget"]

    all_items = [json.loads(l) for l in items_path.open()]

    import random
    rng = random.Random(seed)
    eligible = [it for it in all_items if it.get("unlearning_eligible", True)]
    n_forget_target = max(1, int(len(eligible) * unlearn_ratio))
    forget_ids = set(it["item_id"] for it in rng.sample(eligible, n_forget_target))
    forget_items = [it for it in eligible if it["item_id"] in forget_ids][:max_items]
    retain_items = [it for it in eligible if it["item_id"] not in forget_ids][:max_items]

    print(f"Loading base: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()

    def batch_accuracy(items: list[dict]) -> float:
        correct = 0
        for item in items:
            from src.unlearning.unlearn import format_mcq_prompt
            prompt = format_mcq_prompt(item, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)
            with torch.no_grad():
                logits = peft_model(**inputs).logits[:, -1, :]
            option_logits = {}
            for label in ["A", "B", "C", "D"]:
                tok = tokenizer.encode(" " + label, add_special_tokens=False)
                if tok:
                    option_logits[label] = logits[0, tok[0]].item()
            if not option_logits:
                continue
            predicted = max(option_logits, key=option_logits.get)
            if predicted == item.get("correct_answer", "A").upper():
                correct += 1
        return correct / len(items) if items else 0.0

    forget_acc = batch_accuracy(forget_items)
    retain_acc = batch_accuracy(retain_items)

    return {
        "adapter_dir":    str(adapter_dir),
        "base_model":     base_model_id,
        "unlearn_ratio":  unlearn_ratio,
        "seed":           seed,
        "n_forget_eval":  len(forget_items),
        "n_retain_eval":  len(retain_items),
        "forget_accuracy": round(forget_acc, 4),
        "retain_accuracy": round(retain_acc, 4),
        "forgetting_gap":  round(retain_acc - forget_acc, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate P2 unlearned adapters")
    parser.add_argument("--adapter-dir", help="Single adapter directory")
    parser.add_argument("--items", default=str(ITEMS_PATH))
    parser.add_argument("--output", default="outputs/unlearning_eval.csv")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all adapters in outputs/unlearning/")
    args = parser.parse_args()

    results = []
    if args.all:
        for metrics_path in sorted(OUTPUT_ROOT.rglob("training_metrics.json")):
            adapter_dir = metrics_path.parent
            print(f"\nEvaluating {adapter_dir}")
            result = evaluate_adapter(adapter_dir, Path(args.items))
            results.append(result)
            print(f"  forget_acc={result.get('forget_accuracy', 'N/A')} "
                  f"retain_acc={result.get('retain_accuracy', 'N/A')}")
    elif args.adapter_dir:
        result = evaluate_adapter(Path(args.adapter_dir), Path(args.items))
        results.append(result)
        print(json.dumps(result, indent=2))
    else:
        parser.error("Provide --adapter-dir or --all")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults → {args.output}")
        if "forget_accuracy" in df.columns:
            summary = df.groupby(["base_model", "unlearn_ratio"])[
                ["forget_accuracy", "retain_accuracy", "forgetting_gap"]
            ].mean().round(4)
            print(summary.to_string())


if __name__ == "__main__":
    main()
