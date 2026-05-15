.PHONY: install pilot main trace code analyze all clean

PYTHON := python

install:
	pip install -e ".[dev]"

# ── Data ──────────────────────────────────────────────────────────────
scenarios:
	$(PYTHON) src/data/load_mathdial.py
	$(PYTHON) src/data/build_scenario_bank.py
	$(PYTHON) src/data/stratify.py

# ── Generation ────────────────────────────────────────────────────────
pilot:
	$(PYTHON) src/generation/runner.py --phase pilot --scenarios data/scenarios_pilot.csv --seeds 17 42

main:
	$(PYTHON) src/generation/runner.py --phase main --scenarios data/scenarios.csv --seeds 17 42 91

# ── Trace Metrics ─────────────────────────────────────────────────────
trace:
	$(PYTHON) src/trace_metrics/compute_all.py --phase main

trace-pilot:
	$(PYTHON) src/trace_metrics/compute_all.py --phase pilot

# ── Coding ────────────────────────────────────────────────────────────
packets:
	$(PYTHON) src/coding/packetize.py --phase main --n 240
	$(PYTHON) src/coding/packetize.py --phase main --n 120 --calibration

reliability:
	$(PYTHON) src/coding/reliability.py

# ── Analysis ──────────────────────────────────────────────────────────
analyze:
	$(PYTHON) src/analysis/mixed_models.py
	$(PYTHON) src/analysis/logistic_models.py
	$(PYTHON) src/analysis/use_case_confusion.py
	$(PYTHON) src/analysis/reference_similarity.py
	$(PYTHON) src/analysis/robustness.py

ablations:
	$(PYTHON) src/analysis/ablations.py --n-scenarios 25 --seeds 17 42 91

# ── Qualitative ───────────────────────────────────────────────────────
excerpts:
	$(PYTHON) src/qual/select_excerpts.py

# ── Full pipeline ─────────────────────────────────────────────────────
all: scenarios pilot main trace packets analyze ablations excerpts

clean:
	find outputs/ -name "*.json" -delete
	rm -f outputs/generated_dialogues.csv outputs/automatic_trace_metrics.csv
