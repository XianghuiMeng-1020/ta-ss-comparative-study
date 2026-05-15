# Framework-Grounded Comparative Evaluation of Teachable Agent vs Student Simulation Protocols

## Study Overview

This is a **public-dataset-based methodological comparative analysis** that evaluates how different simulated learner role protocols produce different forms of evidence for different educational uses. No human participants are recruited.

**Core claim**: Teachable Agent (TA) and Student Simulation (SS) protocols are not interchangeable LLM learner prompts. Under matched public tutoring cases (MathDial), they produce structurally different simulated learner evidence when evaluated by purpose-specific frameworks.

## Theoretical Frameworks

| Framework | Source | Role in Study |
|-----------|--------|---------------|
| Simulated learner evaluation | Koedinger et al. (2015) | Purpose-aligned evaluation dimensions (SS1–SS5) |
| Teachable Agent principles | Blair et al. (2007) | TA evaluation dimensions (TA1–TA4) |
| LLM validity checks | Yuan et al. (2026) | Failure flag taxonomy |

## Dataset

**MathDial** (Macina et al., 2023, EMNLP Findings) — `eth-nlped/mathdial` on HuggingFace, CC BY-4.0 license.  
~2,861 one-to-one teacher–student tutoring dialogues grounded in math reasoning problems.

## Experimental Design

- **100 scenarios** × **4 conditions** × **3 seeds** = **1,200 generated dialogues** (main)
- **50 scenarios** × **4 conditions** × **2 seeds** = **400 generated dialogues** (pilot)

### Conditions

| ID | Name | Description |
|----|------|-------------|
| C1 | Teachable Agent | Blair et al. protocol: clarification, revision, independent performance, near-transfer |
| C2 | Student Simulation | Koedinger et al. protocol: target misconception, gradual learning, prior-knowledge stability |
| C3 | Generic Learner | Baseline: "Act as a math student." No structural constraints |
| C4 | No-role Assistant | Control: standard helpful-assistant behaviour |

## Ethical Considerations

- No human participants; coders = research staff (not study subjects)
- MathDial is public domain (CC BY-4.0); no new data collection
- IRB exempt determination to be filed for journal submission

## Reproducibility

All prompts, model IDs, temperatures, seeds, and exclusion rules are frozen in `prompt_protocols.md` before main generation. See `decision_log.md` for all methodological decisions.

```bash
# Install dependencies
pip install -e ".[dev]"

# Build scenario bank
make scenarios

# Run pilot
make pilot

# Run main experiment
make main

# Compute trace metrics
make trace

# Full analysis
make analyze
```

## Directory Structure

```
ta_comparative_analysis/
├── src/                    # Source code
│   ├── data/               # Data loading and scenario bank
│   ├── generation/         # LLM dialogue generation
│   ├── trace_metrics/      # Automatic behavioural metrics
│   ├── coding/             # Human coding support
│   ├── analysis/           # Statistical analysis
│   └── qual/               # Qualitative excerpt selection
├── prompts/                # Frozen prompt files (read-only after P2)
├── data/                   # Scenario CSVs
├── outputs/                # Generated dialogues and results
├── notebooks/              # Analysis notebook
└── reports/                # Paper draft and memos
```

## Key References

- Macina et al. (2023). MathDial. EMNLP Findings.
- Koedinger et al. (2015). Methods for evaluating simulated learners. AIED Workshop.
- Blair et al. (2007). Pedagogical agents for learning by teaching. Educational Technology.
- Yuan et al. (2026). Towards valid student simulation with LLMs. arXiv:2601.05473.
- Mannekote et al. (2025). Can LLMs reliably simulate human learner actions? AAAI Workshop.
