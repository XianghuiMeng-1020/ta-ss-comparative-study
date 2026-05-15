# Qualitative Failure Memo
## Framework-Grounded Comparative Evaluation of TA vs Student Simulation Protocols

**Version**: DRAFT  
**Authors**: [PI name, RA name]  
**Date**: [date]

This memo reports qualitative evidence from the 24 selected excerpts (6 per category).
Each excerpt is bound to exactly ONE framework dimension or ONE LLM validity threat.
This memo supports the Failure Analysis section of the paper.

---

## Overview

Excerpt categories and counts:

| Category | n | Selection criterion |
|----------|---|---------------------|
| High alignment | 6 | Top 10% on condition-primary framework dimensions |
| Low alignment | 6 | Bottom 10% on condition-primary framework dimensions |
| Max disagreement | 6 | Coder score difference ≥ 2 points on any core dimension |
| Failure | 6 | Any failure flag = 1 |

---

## Part 1 — High Alignment Cases

*These excerpts demonstrate what C1/C2 look like when the protocol succeeds.*

### Case H1 — [Packet ID] — C1 Teachable Agent

**Framework anchor**: TA1 (Independent Performance — Blair et al., 2007)

> [Paste relevant dialogue excerpt here. Approximately 4–8 turns.]

**Analysis**: [Describe how this dialogue exemplifies independent performance. Reference TA1 definition. Be specific about which turns show the criterion.]

**Implication**: [What does this tell us about C1 protocol design? What makes it succeed here?]

---

### Case H2 — [Packet ID] — C1 Teachable Agent

**Framework anchor**: TA2 (Productive Learner Behaviour — Blair et al., 2007)

> [Excerpt]

**Analysis**: [...]

**Implication**: [...]

---

### Case H3 — [Packet ID] — C2 Student Simulation

**Framework anchor**: SS2 (Error-Model Alignment — Koedinger et al., 2015)

> [Excerpt]

**Analysis**: [...]

**Implication**: [...]

---

### Case H4 — [Packet ID] — C2 Student Simulation

**Framework anchor**: SS4 (Learning-Process Plausibility — Koedinger et al., 2015)

> [Excerpt]

**Analysis**: [...]

**Implication**: [...]

---

### Cases H5–H6 — [Additional high-alignment cases]

[...]

---

## Part 2 — Low Alignment Cases

*These excerpts show where protocols fail to achieve their intended purpose.*

### Case L1 — [Packet ID] — C1 Teachable Agent

**Framework anchor**: TA1 failure (Independent Performance absent)

> [Excerpt]

**Analysis**: [Describe how and why independent performance fails here. Is it premature correctness? Role drift? Protocol failure?]

**Implication**: [What protocol adjustment might address this failure?]

---

### Cases L2–L6

[Repeat structure for each case, always binding to one framework dimension.]

---

## Part 3 — Maximum Disagreement Cases

*These excerpts show where the two frameworks produce contested evidence.*

### Case D1 — [Packet ID] — [Condition]

**Framework anchor**: [dimension where coders disagreed most]  
**Coder score range**: [e.g., Coder A = 4, Coder B = 2]

> [Excerpt]

**Analysis of disagreement**: [Why did coders disagree? Is the indicator ambiguous in this case? Which anchor definition applies?]

**Resolution**: [Which score is more defensible, and why? What does this reveal about the codebook?]

---

### Cases D2–D6

[...]

---

## Part 4 — Failure Cases

*These excerpts show LLM validity threats (Yuan et al., 2026; Mannekote et al., 2025).*

### Case F1 — [Packet ID] — [Condition]

**LLM validity threat**: Competence paradox (Yuan et al., 2026)  
**Failure flag**: `premature_expertise`

> [Excerpt]

**Analysis**: [Describe how premature expertise manifests. Quote the specific turn where the model reveals full competence prematurely.]

**Implication for validity**: [What does this mean for using this protocol in educational systems? When is this a fatal flaw vs. a recoverable issue?]

---

### Case F2 — [Packet ID] — [Condition]

**LLM validity threat**: Role instability (Yuan et al., 2026; Mannekote et al., 2025)  
**Failure flag**: `role_drift`

> [Excerpt showing tutor-language in learner turn]

**Analysis**: [...]

**Implication for validity**: [...]

---

### Case F3 — [Packet ID] — [Condition]

**LLM validity threat**: Epistemic fidelity failure (Yuan et al., 2026)  
**Failure flag**: `misconception_loss`

> [Excerpt]

**Analysis**: [...]

**Implication for validity**: [...]

---

### Cases F4–F6

[Bind to: over_technical / logical_inconsistency / unsupported_reasoning]

---

## Part 5 — Cross-Cutting Observations

[2–3 paragraphs synthesising patterns across all 24 cases. Address:]

1. Which conditions show the highest failure rates and why?
2. Are there failure modes that appear across conditions (suggesting LLM-level problems)?
3. What do disagreement cases reveal about the codebook's edge cases?
4. What are the boundary conditions for each protocol's effectiveness?

---

## Appendix: Excerpt Selection Methodology

Excerpts were selected programmatically by `src/qual/select_excerpts.py`:
- High/low alignment: top/bottom 10% on mean of condition-primary framework dimensions
- Max disagreement: dialogue with largest coder score differential (≥2 points on any dimension)
- Failure: any failure flag = 1 (as rated by majority vote of 2 coders)

Auto-selection was reviewed by PI before finalising. Cases were replaced if they were
duplicates or if the primary dimension could not be clearly identified.
