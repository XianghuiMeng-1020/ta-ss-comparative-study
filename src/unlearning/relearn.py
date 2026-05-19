"""
Relearning loop — coach-agent re-exposure for P2 teachable-agent interaction.

Mirrors Jiajia et al. (2026) Stage 3: iterative {response → judge → coach feedback → LoRA update}.

Usage:
    python src/unlearning/relearn.py \
        --adapter-dir outputs/unlearning/mistral_7b_instruct_v0_3/ratio30_seed42 \
        --items data/item_bank/mcq/items.jsonl \
        --coach-model gpt-4o-2024-11-20 \
        --rounds 3 \
        --output-dir outputs/relearning/mistral_7b/ratio30_seed42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COACH_SYSTEM_PROMPT = """\
You are an expert tutor coaching a student who has just answered a question incorrectly.
Your goal is to help the student understand their mistake and learn the correct concept.

Guidelines:
- Diagnose the specific misconception in the student's answer.
- Provide a clear, targeted correction (2–3 sentences).
- Ask a follow-up question to check understanding.
- Do NOT simply give the answer — guide the student to reason through it.
"""


def get_coach_feedback(
    item: dict,
    student_response: str,
    coach_model: str,
    coach_backend: str = "openai",
) -> str:
    """Call the coach LLM to generate instructional feedback."""
    from src.generation.clients import LLMConfig, call_llm

    config = LLMConfig(
        model_id=coach_model,
        temperature=0.3,
        max_tokens=300,
        api_backend=coach_backend,
    )
    messages = [
        {"role": "system", "content": COACH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {item.get('question', '')}\n"
                f"Correct answer: {item.get('correct_answer', '')}\n"
                f"Student's response: {student_response}\n\n"
                "Please provide targeted coaching feedback."
            ),
        },
    ]
    result = call_llm(messages, config, seed=42)
    return result.content


def run_relearning_round(
    peft_model,
    tokenizer,
    item: dict,
    coach_feedback: str,
    lr: float = 5e-6,
    max_length: int = 512,
) -> float:
    """One LoRA fine-tuning step on coach-student dialogue pair."""
    import torch
    from src.unlearning.unlearn import format_mcq_prompt

    question_prompt = format_mcq_prompt(item, tokenizer)
    full_text = (
        question_prompt + "\n\n"
        "[Coach feedback]: " + coach_feedback + "\n\n"
        "[Correct answer]: " + item.get("correct_answer", "") + "\n"
        + item.get("options", {}).get(item.get("correct_answer", "A"), "")
    )

    inputs = tokenizer(full_text, return_tensors="pt",
                       max_length=max_length, truncation=True).to(peft_model.device)
    peft_model.train()
    optimizer = torch.optim.AdamW(peft_model.parameters(), lr=lr)
    outputs = peft_model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    peft_model.eval()
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 relearning coach loop")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--items", default="data/item_bank/mcq/items.jsonl")
    parser.add_argument("--coach-model", default="gpt-4o-2024-11-20")
    parser.add_argument("--coach-backend", default="openai")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError:
        print("transformers and peft required. pip install transformers peft")
        return

    adapter_dir = Path(args.adapter_dir)
    meta = json.loads((adapter_dir / "training_metrics.json").read_text())
    base_model_id = meta["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    peft_model.eval()

    all_items = [json.loads(l) for l in Path(args.items).open()]
    forget_ids_path = adapter_dir / "forget_item_ids.json"
    if forget_ids_path.exists():
        forget_ids = set(json.loads(forget_ids_path.read_text()))
        target_items = [it for it in all_items if it["item_id"] in forget_ids]
    else:
        target_items = all_items[:50]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    for round_idx in range(args.rounds):
        print(f"\n=== Relearning Round {round_idx + 1}/{args.rounds} ===")
        round_losses = []
        for item in target_items[:20]:
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
            predicted = max(option_logits, key=option_logits.get) if option_logits else "?"
            student_response = f"I think the answer is {predicted}."

            feedback = get_coach_feedback(
                item, student_response, args.coach_model, args.coach_backend
            )
            loss = run_relearning_round(peft_model, tokenizer, item, feedback, lr=args.lr)
            round_losses.append(loss)

        avg_loss = sum(round_losses) / max(len(round_losses), 1)
        print(f"  Round {round_idx+1} avg loss: {avg_loss:.4f}")
        logs.append({"round": round_idx + 1, "avg_loss": round(avg_loss, 4)})

    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "relearning_log.json").write_text(json.dumps(logs, indent=2))
    print(f"\nRelearned model → {output_dir}")


if __name__ == "__main__":
    main()
