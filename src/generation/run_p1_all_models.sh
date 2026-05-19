#!/usr/bin/env bash
# P1 Generation Script — all 12 models, 4 QTypes, 3 seeds
# Run from repo root:  bash src/generation/run_p1_all_models.sh
#
# Prerequisites:
#   1. Build item bank: python src/data/build_item_bank.py
#   2. vLLM running locally for Tiny/Small/Mid models:
#      vllm serve MODEL_ID --port 8000 --enable-lora
#   3. API keys in .env for Big-tier models.
#
# Each model is run sequentially within its tier.
# Resume-safe: qa_runner.py skips existing output JSONs.

set -euo pipefail

SEEDS="17 42 91"
QTYPES="MCQ TF Fill SA"
PHASE="main"

# ── Tiny tier ──────────────────────────────────────────────────────────────
echo "=== Tiny tier ==="
for MODEL in \
    "HuggingFaceTB/SmolLM2-360M-Instruct" \
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    "meta-llama/Llama-3.2-1B-Instruct"; do
  python src/generation/qa_runner.py \
    --model "$MODEL" --backend vllm \
    --qtypes $QTYPES --seeds $SEEDS --phase $PHASE
done

# ── Small tier ─────────────────────────────────────────────────────────────
echo "=== Small tier ==="
for MODEL in \
    "Qwen/Qwen2.5-1.5B-Instruct" \
    "microsoft/Phi-3.5-mini-instruct" \
    "meta-llama/Llama-3.2-3B-Instruct"; do
  python src/generation/qa_runner.py \
    --model "$MODEL" --backend vllm \
    --qtypes $QTYPES --seeds $SEEDS --phase $PHASE
done

# ── Mid tier ───────────────────────────────────────────────────────────────
echo "=== Mid tier ==="
for MODEL in \
    "Qwen/Qwen3-4B" \
    "mistralai/Mistral-7B-Instruct-v0.3" \
    "meta-llama/Llama-3.1-8B-Instruct"; do
  python src/generation/qa_runner.py \
    --model "$MODEL" --backend vllm \
    --qtypes $QTYPES --seeds $SEEDS --phase $PHASE
done

# ── Big tier ───────────────────────────────────────────────────────────────
echo "=== Big tier — MCQ/TF/Fill/SA only (OED from existing 4800 dialogues) ==="
python src/generation/qa_runner.py \
    --model gpt-4o-2024-11-20 --backend openai \
    --qtypes $QTYPES --seeds $SEEDS --phase $PHASE

python src/generation/qa_runner.py \
    --model claude-sonnet-4-5-20250929 --backend anthropic \
    --qtypes $QTYPES --seeds $SEEDS --phase $PHASE

python src/generation/qa_runner.py \
    --model meta-llama/Llama-3.1-70B-Instruct --backend openrouter \
    --qtypes $QTYPES --seeds $SEEDS --phase $PHASE

# ── CEAT demographic variants (D5) — Big tier only for cost efficiency ─────
echo "=== CEAT demographic variants (D5) ==="
python src/generation/qa_runner.py \
    --model gpt-4o-2024-11-20 --backend openai \
    --qtypes MCQ --seeds 42 --phase ceat \
    --demo-variants gender race national ses

echo "=== P1 generation complete ==="
echo "Run item bank verification: python src/data/build_item_bank.py --verify"
echo "Run IRT calibration: python src/data/irt_calibration.py --responses outputs/irt_baseline --qtype MCQ"
