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

*To be filled when scenario bank is built (P1).*  
Pattern: if any (topic × error_type) cell has <5 rows after sampling, merge with nearest conceptual neighbour and record here.

---

## D4. Prompt Version History

| Version | Date | Change | Commit Hash |
|---------|------|--------|-------------|
| v0.1 | 2026-05-15 | Initial draft | — |
| v1.0 | PENDING | Freeze after P3 pilot review | — |

---

## D5. Model Robustness Subcheck

After main generation, run 25-scenario subset under a second model. Log model ID and date here when activated.

---

*Instructions: Append new decisions below with date, rationale, and status. Never delete or backdate entries.*
