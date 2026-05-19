"""
Unlearning core — LoRA + KL teacher-distribution forgetting.

Faithfully reproduces Jiajia et al. (2026) hyperparameters for Mistral-7B and
applies the same procedure to Qwen2.5-3B for cross-family replication.

Hyperparameters (locked per decision_log.md D19):
  - β  = 0.1        (KL divergence weight)
  - lr = 1e-4
  - epochs = 20
  - batch_size = 8
  - LoRA r = 8, α = 32
  - N_distractors = 3 (per MCQ item: replace correct with 3 plausible distractors)
  - Retain strength = 1.0

Usage:
    python src/unlearning/unlearn.py \
        --base-model mistralai/Mistral-7B-Instruct-v0.3 \
        --forget-items data/item_bank/mcq/items.jsonl \
        --unlearn-ratio 0.30 \
        --seed 42 \
        --output-dir outputs/unlearning/mistral_7b/ratio30_seed42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch

SUPPORTED_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "lora_target_modules": ["q_proj", "v_proj"],
        "max_length": 512,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "lora_target_modules": ["q_proj", "v_proj"],
        "max_length": 512,
    },
}


def load_forget_items(items_path: Path, unlearn_ratio: float, seed: int) -> list[dict]:
    """Sample `unlearn_ratio` fraction of MCQ items as the forget set."""
    all_items = [json.loads(l) for l in items_path.open()]
    eligible = [it for it in all_items if it.get("unlearning_eligible", True)]
    rng = random.Random(seed)
    n_forget = max(1, int(len(eligible) * unlearn_ratio))
    forget_set = rng.sample(eligible, n_forget)
    retain_set = [it for it in eligible if it not in forget_set]
    return forget_set, retain_set


def build_teacher_distribution(item: dict, n_distractors: int = 3) -> list[dict]:
    """
    Replace correct answer with N plausible distractors.
    Returns modified items for teacher-distribution construction.
    """
    options = item.get("options", {})
    correct_key = item.get("correct_answer", "A")
    distractors = [v for k, v in options.items() if k != correct_key]
    if len(distractors) < n_distractors:
        distractors = distractors + ["[distractor]"] * (n_distractors - len(distractors))
    return distractors[:n_distractors]


def format_mcq_prompt(item: dict, tokenizer) -> str:
    """Format MCQ item as an instruction-tuning prompt."""
    options = item.get("options", {})
    opts_str = "\n".join(f"  {k}. {v}" for k, v in options.items())
    question = item.get("question", "")
    return (
        f"Answer the following Python programming multiple-choice question:\n\n"
        f"{question}\n\nOptions:\n{opts_str}\n\nAnswer:"
    )


def run_unlearning(
    base_model_id: str,
    forget_items: list[dict],
    retain_items: list[dict],
    output_dir: Path,
    beta: float = 0.1,
    lr: float = 1e-4,
    epochs: int = 20,
    batch_size: int = 8,
    lora_r: int = 8,
    lora_alpha: int = 32,
    retain_strength: float = 1.0,
    n_distractors: int = 3,
    max_length: int = 512,
) -> dict:
    """
    Main unlearning training loop.
    Returns dict of training metrics (loss curves, final accuracy).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        raise ImportError(
            "Required packages not installed. Run:\n"
            "  pip install transformers peft accelerate bitsandbytes"
        ) from e

    model_config = SUPPORTED_MODELS.get(base_model_id, {
        "lora_target_modules": ["q_proj", "v_proj"],
        "max_length": max_length,
    })

    print(f"Loading base model: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=model_config["lora_target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    metrics = {"forget_loss": [], "retain_loss": [], "total_loss": []}

    model.train()
    for epoch in range(epochs):
        epoch_forget_loss = 0.0
        epoch_retain_loss = 0.0
        n_batches = 0

        random.shuffle(forget_items)
        for b_start in range(0, len(forget_items), batch_size):
            batch_forget = forget_items[b_start: b_start + batch_size]

            forget_loss = _compute_forget_loss(
                model, tokenizer, batch_forget,
                beta=beta, n_distractors=n_distractors,
                max_length=max_length,
            )

            retain_batch = random.sample(
                retain_items, min(batch_size, len(retain_items))
            )
            retain_loss = _compute_retain_loss(
                model, tokenizer, retain_batch, max_length=max_length
            )

            total_loss = forget_loss + retain_strength * retain_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_forget_loss += forget_loss.item()
            epoch_retain_loss += retain_loss.item()
            n_batches += 1

        avg_f = epoch_forget_loss / max(n_batches, 1)
        avg_r = epoch_retain_loss / max(n_batches, 1)
        avg_t = avg_f + retain_strength * avg_r
        metrics["forget_loss"].append(round(avg_f, 4))
        metrics["retain_loss"].append(round(avg_r, 4))
        metrics["total_loss"].append(round(avg_t, 4))

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | forget={avg_f:.4f} retain={avg_r:.4f} total={avg_t:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")

    return metrics


def _compute_forget_loss(
    model, tokenizer, batch: list[dict],
    beta: float, n_distractors: int, max_length: int,
) -> torch.Tensor:
    """
    KL-divergence loss against teacher distribution (distractors).
    Encourages model to predict distractors instead of correct answer.
    """
    total_loss = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
    for item in batch:
        prompt = format_mcq_prompt(item, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length,
                           truncation=True).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)

        distractors = build_teacher_distribution(item, n_distractors)
        teacher_probs = torch.zeros_like(probs)
        for d_text in distractors:
            d_tokens = tokenizer.encode(" " + d_text, add_special_tokens=False)
            if d_tokens:
                teacher_probs[0, d_tokens[0]] += 1.0 / len(distractors)

        teacher_probs = teacher_probs / (teacher_probs.sum() + 1e-9)
        kl = (teacher_probs * (teacher_probs.log() - probs.log() + 1e-9)).sum()
        total_loss = total_loss + beta * kl

    return total_loss / max(len(batch), 1)


def _compute_retain_loss(
    model, tokenizer, batch: list[dict], max_length: int,
) -> torch.Tensor:
    """Standard cross-entropy loss to preserve non-forgotten knowledge."""
    total_loss = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
    for item in batch:
        prompt = format_mcq_prompt(item, tokenizer)
        correct_text = item["options"].get(item["correct_answer"], "")
        full_text = prompt + " " + correct_text
        inputs = tokenizer(full_text, return_tensors="pt", max_length=max_length,
                           truncation=True).to(model.device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss = total_loss + outputs.loss

    return total_loss / max(len(batch), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unlearning training run")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forget-items", required=True, help="Path to items.jsonl")
    parser.add_argument("--unlearn-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    items_path = Path(args.forget_items)
    forget_items, retain_items = load_forget_items(items_path, args.unlearn_ratio, args.seed)
    print(f"Forget set: {len(forget_items)} items | Retain set: {len(retain_items)} items")

    output_dir = Path(args.output_dir)
    metrics = run_unlearning(
        base_model_id=args.base_model,
        forget_items=forget_items,
        retain_items=retain_items,
        output_dir=output_dir,
        beta=args.beta,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps({
        "base_model": args.base_model,
        "unlearn_ratio": args.unlearn_ratio,
        "seed": args.seed,
        "n_forget": len(forget_items),
        "n_retain": len(retain_items),
        "hyperparameters": {
            "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        },
        "metrics": metrics,
    }, indent=2))
    print(f"Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
