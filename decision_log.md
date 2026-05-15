# Decision Log — TA vs Student Simulation Comparative Study

All methodological decisions are timestamped here. This file is append-only after P2 prompt freeze.

---

## D1. Generation LLM

**Decision**: Single frontier LLM
**Model ID**: `gpt-4o-2024-11-20`  ← FILL IN before P2
**Temperature**: 0.7  
**max_tokens per turn**: 600  
**Seeds**: [17, 42, 91] (main); [17, 42] (pilot)  
**Rationale**: Single-model approach maximises condition parity and minimises cost (~$100–150 for 1,200 dialogues). Model-identity robustness check on 25-scenario subset with a second model (e.g., `claude-sonnet-4-5`) will be run post-main to guard against model-specific confounds.  
**Status**: OPEN — confirm model ID before P2  
**Logged**: 2026-05-15

---

## D2. Tutor Role Implementation

**Decision**: MathDial human tutor turn-by-turn replay  
**Rule**: Each MathDial original tutor turn is replayed verbatim as the `user` message to the learner LLM. Turn budget = length of original conversation + 1 transfer-probe turn. All four conditions (C1–C4) receive identical tutor stimuli. This eliminates tutor variation as a confound and provides maximum scenario parity.  
**Near-transfer**: A single transfer probe is auto-generated per scenario (same structure, different numbers) and appended after the final tutor turn. All conditions receive it; only C1 protocol requires an independent attempt.  
**Rationale**: Strongest defensible claim of scenario parity; reviewers cannot attribute condition differences to differential tutor behaviour. Alternative (LLM tutor) would require a tutor-sensitivity check sub-experiment (+1–2 weeks).  
**Status**: LOCKED  
**Logged**: 2026-05-15

---

## D3. Stratification Merge Rules

**Date**: 2026-05-15  
**Status**: LOCKED (recorded after P1 execution)

Strata with <5 eligible rows were merged into a `misc||misc` bucket before stratified sampling. This affected the following (topic × error_type) cells:

**Main bank (100 scenarios)**:
- fractions||unit_error, fractions||setup_error, arithmetic||setup_error, geometry||unit_error, geometry||conceptual_misconception, percentages||setup_error, multi_step||setup_error, algebra_words||miscellaneous, geometry||setup_error (9 strata merged)

**Pilot bank (50 scenarios)**:
- geometry||arithmetic_error, fractions||unit_error, geometry||unit_error, geometry||conceptual_misconception, fractions||setup_error, percentages||setup_error, arithmetic||setup_error, multi_step||setup_error, algebra_words||miscellaneous, geometry||setup_error (10 strata merged)

**Rationale**: algebra_words contained only 1 row total in MathDial train split. geometry is represented by only 3 rows in the main bank; these rare strata are reported as a study limitation.

**Balance check results** (chi-square independence tests, both passed):
- Main: topic×error_type χ²=8.27, p=0.990 (df=20); difficulty×topic χ²=11.54, p=0.173 (df=8)
- Pilot: topic×error_type χ²=7.61, p=0.994 (df=20)

---

## D4. Prompt Version History

| Version | Date | Change | Commit Hash |
|---------|------|--------|-------------|
| v0.1 | 2026-05-15 | Initial draft | — |
| v1.0 | 2026-05-15 | Freeze after P2 — all 4 condition prompts locked | c2c41e6145c0471c40d2a9c1e43d393cf77ff789 |

---

## D5. Model Robustness Subcheck

After main generation, run 25-scenario subset under a second model. Log model ID and date here when activated.

---

*Instructions: Append new decisions below with date, rationale, and status. Never delete or backdate entries.*
