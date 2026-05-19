"""
Orchestrates all P2 unlearning training runs.

Runs: 2 models × 3 unlearning ratios × 5 seeds = 30 training jobs
Hardware: 1× A100 40GB recommended; RTX 4090 (24GB) minimum.
Estimated GPU-hours: ~80h total (≈2.5h per job on A100).

Usage:
    # Full run (all models, ratios, seeds):
    python src/unlearning/train_runner.py --all

    # Single model:
    python src/unlearning/train_runner.py \
        --models mistral --ratios 0.10 0.30 0.50 --seeds 42 2026 123 56 1

    # Resume (skips completed jobs):
    python src/unlearning/train_runner.py --all --resume

Output:
    outputs/unlearning/{model_tag}/ratio{pct}_seed{seed}/
        adapter_model.bin
        adapter_config.json
        training_metrics.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen3b":  "Qwen/Qwen2.5-3B-Instruct",
}

RATIOS = [0.10, 0.30, 0.50]
SEEDS  = [42, 2026, 123, 56, 1]

ITEMS_PATH = "data/item_bank/mcq/items.jsonl"
OUTPUT_ROOT = Path("outputs/unlearning")


def model_tag(model_id: str) -> str:
    return model_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def job_output_dir(model_id: str, ratio: float, seed: int) -> Path:
    mtag = model_tag(model_id)
    pct = int(ratio * 100)
    return OUTPUT_ROOT / mtag / f"ratio{pct:02d}_seed{seed}"


def run_job(model_id: str, ratio: float, seed: int, resume: bool = True) -> None:
    out_dir = job_output_dir(model_id, ratio, seed)

    if resume and (out_dir / "adapter_config.json").exists():
        print(f"  SKIP (already done): {out_dir}")
        return

    cmd = [
        sys.executable, "src/unlearning/unlearn.py",
        "--base-model",     model_id,
        "--forget-items",   ITEMS_PATH,
        "--unlearn-ratio",  str(ratio),
        "--seed",           str(seed),
        "--output-dir",     str(out_dir),
        "--beta",           "0.1",
        "--lr",             "1e-4",
        "--epochs",         "20",
        "--batch-size",     "8",
        "--lora-r",         "8",
        "--lora-alpha",     "32",
    ]

    pct = int(ratio * 100)
    print(f"\n>>> Training: {model_id} | ratio={pct}% | seed={seed}")
    print(f"    Output: {out_dir}")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [ERROR] Job failed: {model_id} ratio={pct}% seed={seed}")
    else:
        print(f"  [DONE] {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 unlearning training orchestrator")
    parser.add_argument("--all", action="store_true",
                        help="Run all models × ratios × seeds")
    parser.add_argument("--models", nargs="+", choices=list(MODELS.keys()),
                        default=list(MODELS.keys()))
    parser.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip already-completed jobs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print jobs without running them")
    args = parser.parse_args()

    if args.all:
        selected_models = list(MODELS.values())
    else:
        selected_models = [MODELS[m] for m in args.models]

    total = len(selected_models) * len(args.ratios) * len(args.seeds)
    print(f"Total jobs: {total}")
    print(f"Models: {[m.split('/')[-1] for m in selected_models]}")
    print(f"Ratios: {args.ratios}")
    print(f"Seeds:  {args.seeds}")

    for model_id in selected_models:
        for ratio in args.ratios:
            for seed in args.seeds:
                if args.dry_run:
                    out = job_output_dir(model_id, ratio, seed)
                    done = (out / "adapter_config.json").exists()
                    print(f"  {'[DONE]' if done else '[TODO]'} {out}")
                else:
                    run_job(model_id, ratio, seed, resume=args.resume)

    print("\nAll jobs complete. Run evaluation:")
    print("  python src/unlearning/evaluate.py --all")


if __name__ == "__main__":
    main()
