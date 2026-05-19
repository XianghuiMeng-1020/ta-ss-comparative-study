"""
Full P1 pipeline orchestrator — runs everything from item bank to coder packets.

Phases (each with checkpoint-resume):
  1. Wait for Python MCQs (data/raw_jiajia_mcq/python_mcqs.jsonl, 300 items)
  2. Rebuild item bank
  3. IRT proxy calibration (GPT-4o-mini baseline, assigns E/M/H)
  4. P1 generation — 6 API-accessible models (Tier 1 Big + Tier 2 Mid)
  5. Aggregate CSVs + generate coder packets

Usage:
    cd /path/to/project
    source .venv/bin/activate
    export OPENAI_API_KEY=... OPENROUTER_API_KEY=...
    python scripts/orchestrate_p1.py [--smoke-test] [--skip-irt] [--models M1 M2 ...]

Note: Small-model tiers (T3/T4) require local vLLM and are NOT run here.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


# ------- Model roster (API-accessible only) -----------------------------------
# Format: (label, model_id, backend)
API_MODELS = [
    # Tier 1 — Big
    ("M1_GPT4o",    "gpt-4o-2024-11-20",                 "openai"),
    ("M2_Claude",   "claude-sonnet-4-5-20250929",         "openrouter"),
    ("M3_Gemini",   "gemini-2.5-pro",                     "openrouter"),
    # Tier 2 — Mid
    ("M4_Llama70B", "meta-llama/Llama-3.1-70B-Instruct", "openrouter"),
    ("M5_Qwen72B",  "Qwen/Qwen2.5-72B-Instruct",         "openrouter"),
    ("M6_DeepSeek", "deepseek-ai/DeepSeek-V3",            "openrouter"),
]

SEEDS = [17, 42, 91]
QTYPES = ["MCQ", "TF", "Fill", "SA"]
MAX_TOKENS = 500


def run_cmd(cmd: list[str], desc: str) -> int:
    """Run a subprocess, stream output, return exit code."""
    import os
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD:  {' '.join(cmd)}")
    print('='*60)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] Step failed with exit code {result.returncode}: {desc}")
    return result.returncode


def wait_for_python_mcqs(target: int = 300, timeout_min: int = 30) -> bool:
    """Block until python_mcqs.jsonl reaches target count."""
    mcq_path = PROJECT_ROOT / "data/raw_jiajia_mcq/python_mcqs.jsonl"
    deadline = time.time() + timeout_min * 60

    while time.time() < deadline:
        if mcq_path.exists():
            count = sum(1 for _ in mcq_path.open())
            print(f"\r  Python MCQs: {count}/{target}", end="", flush=True)
            if count >= target:
                print(f"\n  Done — {count} Python MCQ items ready.")
                return True
        time.sleep(15)

    print(f"\n  [TIMEOUT] Only {count if mcq_path.exists() else 0}/{target} items after {timeout_min} min.")
    return False


def phase_rebuild_item_bank() -> bool:
    rc = run_cmd(
        [sys.executable, "src/data/build_item_bank.py",
         "--mcq-python", "300", "--mcq-math", "300",
         "--tf", "300", "--fill", "300", "--sa", "300"],
        "Rebuild item bank (all qtypes)"
    )
    if rc == 0:
        run_cmd([sys.executable, "src/data/build_item_bank.py", "--verify"],
                "Verify item bank")
    return rc == 0


def phase_irt_proxy(sample: int | None = None) -> bool:
    cmd = [sys.executable, "src/data/irt_proxy_api.py",
           "--qtypes", "MCQ", "TF", "Fill", "SA",
           "--n-reps", "5"]
    if sample:
        cmd += ["--sample", str(sample)]
    rc = run_cmd(cmd, "IRT proxy calibration (GPT-4o-mini baseline, all qtypes)")
    return rc == 0


def phase_p1_generation(
    models: list[tuple[str, str, str]],
    qtypes: list[str],
    seeds: list[int],
    smoke_test: bool = False,
) -> None:
    for label, model_id, backend in models:
        for qtype in qtypes:
            seed_args = [str(seeds[0])] if smoke_test else [str(s) for s in seeds]
            cmd = [
                sys.executable, "src/generation/qa_runner.py",
                "--model", model_id,
                "--backend", backend,
                "--qtypes", qtype,
                "--seeds", *seed_args,
                "--phase", "main",
                "--max-tokens", str(MAX_TOKENS),
                "--temperature", "0.7",
            ]
            desc = f"P1 generation: {label} | {qtype} | seeds={seed_args}"
            rc = run_cmd(cmd, desc)
            if rc != 0:
                print(f"  [WARN] {label}/{qtype} failed, continuing with next job...")


def phase_ceat_variants(
    models: list[tuple[str, str, str]],
    smoke_test: bool = False,
) -> None:
    """Run CEAT demographic variants for D5 fairness evaluation (MCQ only)."""
    seed_args = ["42"] if smoke_test else [str(s) for s in SEEDS]
    for label, model_id, backend in models[:3]:  # Big tier only for CEAT
        cmd = [
            sys.executable, "src/generation/qa_runner.py",
            "--model", model_id,
            "--backend", backend,
            "--qtypes", "MCQ",
            "--seeds", *seed_args,
            "--phase", "ceat",
            "--demo-variants", "gender", "race", "national", "ses",
            "--max-tokens", str(MAX_TOKENS),
        ]
        run_cmd(cmd, f"CEAT variants: {label}")


def phase_packetize() -> bool:
    rc = run_cmd(
        [sys.executable, "src/coding/qa_packetize.py"],
        "Generate coder packets for human annotation"
    )
    return rc == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 pipeline orchestrator")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run minimal smoke test (1 seed, 5 items per type)")
    parser.add_argument("--skip-wait", action="store_true",
                        help="Skip waiting for Python MCQs (assume already done)")
    parser.add_argument("--skip-irt", action="store_true",
                        help="Skip IRT calibration (assume already done)")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip item bank rebuild")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model labels to run (default: all API models). "
                             f"Options: {[m[0] for m in API_MODELS]}")
    parser.add_argument("--qtypes", nargs="*", default=QTYPES,
                        choices=QTYPES)
    parser.add_argument("--no-ceat", action="store_true",
                        help="Skip CEAT demographic variant generation")
    args = parser.parse_args()

    models = API_MODELS
    if args.models:
        models = [m for m in API_MODELS if m[0] in args.models]
        if not models:
            print(f"No models matched. Options: {[m[0] for m in API_MODELS]}")
            sys.exit(1)

    print(f"\n{'#'*60}")
    print("P1 Pipeline Orchestrator — IEEE TLT v2 Study")
    print(f"  Models: {[m[0] for m in models]}")
    print(f"  Qtypes: {args.qtypes}")
    print(f"  Seeds:  {SEEDS if not args.smoke_test else [SEEDS[0]]}")
    print(f"  Smoke:  {args.smoke_test}")
    print(f"{'#'*60}\n")

    # Phase 1: Wait for Python MCQs
    if not args.skip_wait:
        print("\n[Phase 1] Waiting for Python MCQ generation...")
        ok = wait_for_python_mcqs(target=300, timeout_min=40)
        if not ok:
            print("[WARN] Proceeding with partial Python MCQ set.")

    # Phase 2: Rebuild item bank
    if not args.skip_build:
        print("\n[Phase 2] Rebuilding item bank...")
        phase_rebuild_item_bank()

    # Phase 3: IRT proxy calibration
    if not args.skip_irt:
        print("\n[Phase 3] IRT proxy calibration...")
        sample = 30 if args.smoke_test else None
        phase_irt_proxy(sample=sample)

    # Phase 4: P1 generation
    print("\n[Phase 4] P1 generation for API models...")
    phase_p1_generation(
        models=models,
        qtypes=args.qtypes,
        seeds=SEEDS,
        smoke_test=args.smoke_test,
    )

    # Phase 5: CEAT demographic variants
    if not args.no_ceat:
        print("\n[Phase 5] CEAT demographic variants...")
        phase_ceat_variants(models=models, smoke_test=args.smoke_test)

    # Phase 6: Coder packets
    print("\n[Phase 6] Generating coder packets...")
    phase_packetize()

    print("\n" + "="*60)
    print("Orchestration complete.")
    print("Next step: distribute coder_packets/ to human annotators.")
    print("="*60)


if __name__ == "__main__":
    main()
