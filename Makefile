.PHONY: install scenarios pilot main main-all bridge trace trace-pilot packets reliability \
        analyze ablations equivalence judge excerpts all clean smoke

PYTHON := python

# ── Models for multi-model generation ─────────────────────────────────────────
M1 := gpt-4o-2024-11-20
M2 := claude-sonnet-4-5-20250929
M3 := gemini-2.5-pro
M4 := meta-llama/Llama-3.1-70B-Instruct

# ── Install ────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Data ──────────────────────────────────────────────────────────────────────
scenarios:
	$(PYTHON) src/data/load_mathdial.py
	$(PYTHON) src/data/build_scenario_bank.py
	$(PYTHON) src/data/stratify.py

bridge-scenarios:
	$(PYTHON) src/data/build_bridge_scenarios.py

# ── Smoke test (50 scenarios × 4 models, costs ~$20) ─────────────────────────
smoke:
	$(PYTHON) src/generation/runner.py --phase pilot --scenarios data/scenarios_pilot.csv \
	    --seeds 17 42 --model $(M1) --backend auto
	$(PYTHON) src/generation/runner.py --phase pilot --scenarios data/scenarios_pilot.csv \
	    --seeds 17 42 --model $(M2) --backend auto
	$(PYTHON) src/generation/runner.py --phase pilot --scenarios data/scenarios_pilot.csv \
	    --seeds 17 42 --model $(M3) --backend auto
	$(PYTHON) src/generation/runner.py --phase pilot --scenarios data/scenarios_pilot.csv \
	    --seeds 17 42 --model $(M4) --backend auto

# ── Pilot (50 scenarios × 1 model; for codebook calibration) ─────────────────
pilot:
	$(PYTHON) src/generation/runner.py --phase pilot --scenarios data/scenarios_pilot.csv \
	    --seeds 17 42 --model $(M1) --backend auto

# ── Main generation — run once per model (supports checkpoint-resume) ─────────
main-m1:
	$(PYTHON) src/generation/runner.py --phase main --scenarios data/scenarios.csv \
	    --seeds 17 42 91 --model $(M1) --backend auto

main-m2:
	$(PYTHON) src/generation/runner.py --phase main --scenarios data/scenarios.csv \
	    --seeds 17 42 91 --model $(M2) --backend auto

main-m3:
	$(PYTHON) src/generation/runner.py --phase main --scenarios data/scenarios.csv \
	    --seeds 17 42 91 --model $(M3) --backend auto

main-m4:
	$(PYTHON) src/generation/runner.py --phase main --scenarios data/scenarios.csv \
	    --seeds 17 42 91 --model $(M4) --backend auto

# Run all 4 models sequentially (recommended; parallel risks API rate limits)
main-all: main-m1 main-m2 main-m3 main-m4

# Legacy alias
main: main-m1

# ── Bridge cross-dataset replication ─────────────────────────────────────────
bridge-m1:
	$(PYTHON) src/generation/runner.py --phase bridge --scenarios data/scenarios_bridge.csv \
	    --seeds 17 42 --model $(M1) --backend auto

bridge-m2:
	$(PYTHON) src/generation/runner.py --phase bridge --scenarios data/scenarios_bridge.csv \
	    --seeds 17 42 --model $(M2) --backend auto

bridge: bridge-scenarios bridge-m1 bridge-m2

# ── Trace Metrics ─────────────────────────────────────────────────────────────
trace:
	$(PYTHON) src/trace_metrics/compute_all.py --phase main

trace-pilot:
	$(PYTHON) src/trace_metrics/compute_all.py --phase pilot

trace-bridge:
	$(PYTHON) src/trace_metrics/compute_all.py --phase bridge

# ── Human coding support ──────────────────────────────────────────────────────
packets:
	$(PYTHON) src/coding/packetize.py --phase main --n 240
	$(PYTHON) src/coding/packetize.py --phase main --n 60 --calibration

reliability:
	$(PYTHON) src/coding/reliability.py

# ── LLM-as-Judge ─────────────────────────────────────────────────────────────
judge:
	$(PYTHON) src/analysis/llm_judge.py --calibrate --human-ratings outputs/coder_ratings_raw.csv
	$(PYTHON) src/analysis/llm_judge.py --score-all --phase main

judge-calibrate:
	$(PYTHON) src/analysis/llm_judge.py --calibrate --human-ratings outputs/coder_ratings_raw.csv

# ── Primary analysis ──────────────────────────────────────────────────────────
analyze:
	$(PYTHON) src/analysis/mixed_models.py
	$(PYTHON) src/analysis/logistic_models.py
	$(PYTHON) src/analysis/use_case_confusion.py
	$(PYTHON) src/analysis/reference_similarity.py
	$(PYTHON) src/analysis/robustness.py
	$(PYTHON) src/analysis/equivalence_test.py

# ── Ablations (8 elements) ────────────────────────────────────────────────────
ablations:
	$(PYTHON) src/analysis/ablations.py --n-scenarios 25 --seeds 17 42 --model $(M1) --backend auto

# ── Ecological reference (Eedi) ───────────────────────────────────────────────
eedi-reference:
	$(PYTHON) src/analysis/eedi_ecological_reference.py

# ── Qualitative ──────────────────────────────────────────────────────────────
excerpts:
	$(PYTHON) src/qual/select_excerpts.py

vignette:
	$(PYTHON) src/qual/teacher_vignette_analysis.py

# ── Full pipeline (sequential) ────────────────────────────────────────────────
all: scenarios bridge-scenarios smoke main-all bridge trace trace-bridge \
     packets reliability judge analyze ablations eedi-reference excerpts

# ── Clean generated outputs ───────────────────────────────────────────────────
clean:
	find outputs/ -name "*.json" -delete
	rm -f outputs/generated_dialogues_*.csv outputs/automatic_trace_metrics*.csv
	@echo "Cleaned outputs (kept reports and code)"
