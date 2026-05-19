# Analysis Plan v2 — Pre-committed to GitHub Before P2 Unlearning Training
## How Naive is "Naive"? A Multi-Model, Multi-Question-Type, Multi-Dimension Evaluation of LLM-based Student Simulation

**Plan version**: v2.0 (replaces v1.0, commit 27a554e2ed884eda245d2f9f9111751b235101ba)
**Supersedes**: Original TA vs SS comparative analysis (C1–C4 protocol comparison)
**Corresponding author**: [BLINDED FOR REVIEW]
**Target venue**: IEEE Transactions on Learning Technologies (Regular Paper)

> This document must be committed to the public GitHub repository BEFORE any P2 (machine
> unlearning) training runs begin. The commit timestamp is the pre-registration anchor for
> all confirmatory hypotheses. Deviations are reported in decision_log.md.

---

## 1. Have any data been collected for this study already?

Partially. The following data exist from the prior TA vs SS study (v1 design) and are
carried forward as the **open-ended dialogue** slice of the n-axis:

- 4,800 dialogues from 4 frontier models (GPT-4o, Claude Sonnet-4.5, Gemini-2.0-Flash,
  Llama-3.1-70B) × 4 conditions × 100 MathDial scenarios × 3 seeds
- 10 automatic trace metrics computed for all 4,800 dialogues
- 240-dialogue human coding packets (partially coded; re-annotated with augmented codebook)

All NEW data collection — item bank construction, smaller-model P1 generation, machine
unlearning (P2) training and evaluation, augmented human annotations — begins AFTER this
plan is committed.

---

## 2. Research Questions and Hypotheses

### Core Research Question

When LLMs are deployed as naive student simulators, how do they differ across (a) model
size and family, (b) question type and difficulty, (c) simulation strategy (prompt-based
vs. machine-unlearning-based), and (d) seven evaluation dimensions spanning human-likeness,
task performance, efficiency, fairness, and dialogue-test alignment?

### Hypotheses

**H1 (Scaling direction — human-likeness)**: Larger models produce significantly higher
human-likeness ratings (D1) than models ≤ 3B parameters on MCQ and Short-Answer items,
with Cohen's d > 0.5.

**H2 (Smaller-model efficiency dominance)**: Models ≤ 3B parameters achieve Pareto-superior
positions on the (D1, D2, D4) frontier relative to 7B–70B models for at least one question
type × difficulty cell.

**H3 (Unlearning advantage over prompting for novice-level stability)**: Unlearning-based
simulation (P2) on Mistral-7B achieves significantly lower answering accuracy (D2) on
forgotten knowledge components than prompt-based simulation (P1) on the same model, with
Cohen's d > 0.5 — confirming that prompting alone cannot suppress underlying competence.

**H4 (Cross-question-type robustness)**: The rank ordering of models on D1 (human-likeness)
is consistent (Spearman ρ ≥ 0.7) across at least 3 of 5 question types.

**H5 (Fairness — no size-fairness trade-off)**: CEAT effect sizes (D5) do not decrease
monotonically with model size; at least one sub-7B model shows CEAT comparable to or lower
than GPT-4o on ≥ 2 of 4 demographic attribute sets.

**H6 (Dialogue-test discrepancy prevalence)**: Type-A and Type-B misalignment profiles
(D6) occur at non-trivial rates (> 10% of simulation instances) in at least 3 models,
and their prevalence differs significantly across model sizes (Kruskal-Wallis p < .05).

**Equivalence claim (secondary)**: Prompt-based P1 on Qwen2.5-3B is statistically
equivalent to Mistral-7B-P1 on D1 human-likeness (TOST; bound d = 0.3), supporting
the use of small models as drop-in replacements.

---

## 3. Design Matrix

### m-axis: 12 LLMs as Naive Students (4 tiers)

| Tier | Models | Params | Inference |
|------|--------|--------|-----------|
| Tiny | SmolLM2-360M, TinyLlama-1.1B, Llama-3.2-1B | < 2B | vLLM local |
| Small | Qwen2.5-1.5B, Phi-3.5-mini-3.8B, Llama-3.2-3B | 1.5–4B | vLLM local |
| Mid | Qwen3-4B, Mistral-7B-Instruct-v0.3 (anchor), Llama-3.1-8B | 4–8B | vLLM local |
| Big | GPT-4o-2024-11-20, Claude-Sonnet-4-5, Gemini-2.0-Flash, Llama-3.1-70B | 70B+ | API |

Note: Big-tier models reuse existing 4,800 dialogues for open-ended slice; net-new
generation required only for MCQ/TF/Fill/SA question types.

### n-axis: 5 Question Types × 3 Difficulty Levels

| QType | Code | Item source | N items |
|-------|------|-------------|---------|
| Multiple-Choice | MCQ | Jiajia Python subset (300) + MathDial-derived (300) | 600 |
| True-False | TF | Auto-derived from MCQ correct/distractor pairs | 300 |
| Fill-in-blank | Fill | Auto-derived from MathDial reasoning steps | 300 |
| Short-Answer | SA | MathDial reasoning steps (new) | 300 |
| Open-Ended Dialogue | OED | 100 MathDial scenarios × tutor replay | 100 scenarios |

Difficulty: Easy / Medium / Hard — labels assigned by 2PL IRT calibrated on Mistral-7B
baseline (see t05-irt-calibration in decision_log).

### Simulation strategy axis

| Code | Strategy | Applied to |
|------|----------|------------|
| P1 | Prompt-based naive student (naive_student_prompt.txt) | All 12 models |
| P2 | Machine-unlearning-based (LoRA + KL forgetting) | Mistral-7B + Qwen2.5-3B |

### Total new generation

12 models × 4 QTypes × 3 difficulty × P1 × 3 seeds × 10 items/cell = **4,320 condition-cells**
→ 43,200 individual item responses (net-new)

P2: 2 models × 3 unlearning ratios × 5 seeds × 600 MCQ items = **18,000 responses** (GPU)

Existing 4,800 OED dialogues carried forward under P1 for Big-tier models.

---

## 4. Seven Evaluation Dimensions

### D1 — Human-likeness

**Definition**: The degree to which a simulated naive-student response is indistinguishable
from a genuine novice student response.

**Operationalization**:
- Human 1–5 Likert rating ("How much does this response resemble a real student?")
- Binary forced-choice Turing task ("Human student or AI simulation?")
- Pooled across 2 coders; κ ≥ 0.65 required before analysis
- Automated proxy: perplexity under a student-language model (GPT-2 fine-tuned on
  MOOC forum posts) — correlation with human ratings reported

**Human coding subset**: 480 single-response items (stratified across models and QTypes)

### D2 — Answering Performance

**Definition**: How accurately and coherently the model responds to items as a novice
student — in the desirable case, with predictable errors at lower difficulty and plausible
partial-credit patterns.

**Operationalization**:
- MCQ/TF/Fill: exact-match accuracy (auto-graded vs. gold key)
- SA/OED: explanation-quality 1–5 (LLM-judge, calibrated ICC ≥ 0.65 vs. human coders)
- Partial-credit rubric: correct answer (2 pts) / plausible error (1 pt) / nonsense (0 pt)
- Target profile for a good naive student: medium accuracy at Easy, low at Hard; errors
  semantically plausible (not random)

### D3 — Model-Size Effect

**Definition**: How D1 and D2 vary as a function of log(parameter count).

**Operationalization**:
- OLS regression: D1 ~ log(params) + QType + difficulty; slopes reported per QType
- Non-linear test: add log(params)^2 term; report AIC comparison
- "Sweet spot" identification: the param count tier with highest D1 × (1 − D2) product
  (most human-like while also most novice-appropriate in accuracy)

### D4 — Efficiency

**Definition**: Latency and cost per response as a function of model size and quality.

**Operationalization**:
- Wall-clock latency (ms/response) measured during generation with per-turn timestamps
- Tokens/sec computed from latency + output token count
- USD/response from cost_estimate_usd() (LLMConfig, extended with small-model rates)
- Pareto frontier: non-dominated set over (D1, D2, −cost_per_response)
- Plotted as scatter in D1 vs. D2 space, bubble size = latency

### D5 — Generation Fairness (CEAT)

**Definition**: Degree to which simulated-student responses contain implicit demographic
stereotypes, measured via Contextualized Embedding Association Test (Peng et al., 2025).

**Operationalization**:
- 4 demographic attribute sets: gender (male/female target words), race (White/Asian/
  Black/Hispanic), national (US/Chinese/Indian/British), SES (high-income/low-income)
- For each attribute set: construct 200 sentence pairs from model responses;
  compute CEAT effect size d per Peng et al. (2025) using sentence-transformers
- "Fair" threshold: |d| < 0.2 (small effect, per Cohen 1988)
- Fairness score per model = fraction of attribute sets with |d| < 0.2

**Demographic injection**: 4 × 4 = 16 prompt variants per item; models generate responses
under each variant; CEAT computed on the distribution shift
(see prompts/demographic_variants/)

### D6 — Dialogue-Test Discrepancy

**Definition**: Mismatch between a model's conversational knowledge level (estimated
from dialogue) and its formal assessment score (from D2).

**Operationalization**:
- Knowledge level from dialogue: estimated via DKT-derived knowledge state at each turn
  (reusing lab dialogue-based knowledge tracing model); final-turn estimate = KL-score
- Discrepancy Δ = z(test-score) − z(KL-score); computed per model × item-set
- Type-A profile (eye-high-hand-low): Δ < −1 (KL-score high, test-score low)
- Type-B profile (silent high-achiever): Δ > +1 (KL-score low, test-score high)
- Prevalence rate per model reported; compared across tiers with Kruskal-Wallis

### D7 — Persona Consistency

**Definition**: Stability of the naive-student persona across a multi-turn dialogue.

**Operationalization**:
- Role-drift rate: fraction of turns where model shifts from student to teacher/expert
  (extended from src/trace_metrics/role_drift.py)
- Expert-leakage rate: fraction of turns with jargon above 7th-grade Flesch-Kincaid
  reading level
- Turn-to-turn knowledge stability: cosine similarity of consecutive-turn knowledge
  state estimates (from DKT model); lower variance = more stable persona
- Minimum: single-item responses lack multi-turn data; D7 computed only over OED slice

---

## 5. Statistical Analysis Plan

### Primary model (H1–H3, H6)

```
lmer(metric ~ model_tier * question_type * difficulty * persona
             + (1 | item_id) + (1 | coder_id),
     data = ratings, REML = TRUE)
```

Estimated via `pymer4`. Deviation coding with Mistral-7B-P1 as reference.
Contrasts: Holm-Bonferroni corrected within each hypothesis family.

### Equivalence (H-equiv)

TOST with d = 0.3 bound; 90% CI on Qwen2.5-3B vs. Mistral-7B D1 means.

### Pareto analysis (H2)

Non-dominated sorting (NSGA-II style) over {−D1, −D2, D4_cost} per model.
All models on the frontier reported as "Pareto-optimal naive-student configurations".

### Effect sizes

Cohen's d with 5,000-iteration bootstrap 95% CIs (continuing pipeline in
outputs/bootstrap_effect_sizes.csv).

### Multiple comparison families

| Family | Hypotheses | Correction | α threshold |
|--------|-----------|-----------|-------------|
| F1 | H1 (D1 vs. model tier) | Holm-Bonferroni | 0.0125 |
| F2 | H3 (P1 vs. P2 accuracy) | Holm-Bonferroni | 0.05 |
| F3 | H5 (CEAT comparisons) | Benjamini-Hochberg | 0.05 FDR |
| F4 | H6 (Kruskal-Wallis post-hoc) | Dunn with BH | 0.05 FDR |

---

## 6. Exclusion Rules (extended from v1)

| Code | Trigger | Action |
|------|---------|--------|
| E1 | Empty learner turn (< 5 chars) | Drop turn; ≥ 2 such turns → exclude dialogue |
| E2 | Non-domain content | Exclude item response |
| E3 | Unsafe / refusal response | Exclude item response, log |
| E4 | Token overflow (> 800 tokens single turn) | Truncate; if answer lost → exclude |
| E5 | Missing transfer response in OED | Regenerate once; fails → exclude |
| E6 | Condition leakage (role label in output) | Exclude response |
| E7 (new) | Tiny-model incoherence (< 3 real words / sentence) | Exclude response |
| E8 (new) | Unlearning collapse (accuracy < 5% on retain set) | Exclude unlearning ratio; document |

Exclusion rates reported per model tier and question type in Table A1 (Appendix).

---

## 7. Sample Size and Power

**P1 item responses**: 12 models × 4 QTypes × 3 difficulty × 10 items × 3 seeds = 4,320 cells
→ 43,200 responses; human coded subset = 480 items (10 per model per QType)

**P2 unlearning**: 2 models × 3 ratios × 5 seeds × 600 MCQ = 18,000 responses

**Human annotation**: 480 item-responses + 240 OED dialogues = 720 items total;
2 coders, 100% overlap (720 each); ICC ≥ 0.75 gate per dimension

**Power (D1 primary)**: For n = 480 human-annotated items across 12 models (40/model),
minimum detectable effect size d ≈ 0.41 at 80% power, α = 0.0125 (Holm). Preliminary
trace-metric data suggest D1 differences of d > 0.8 between Tiny and Big tiers.

---

## 8. What Will Be Reported Regardless of Outcome

- All 7 dimensions, all model × QType × difficulty cells (supplementary tables)
- All exclusion rates per model and question type
- All CEAT effect sizes per demographic attribute set per model
- Full Type-A/Type-B prevalence table
- IRT difficulty calibration results and item bank summary
- P2 unlearning forgetting curves (accuracy vs. unlearning ratio × seed)

---

## 9. Open Materials Commitment

| Artifact | Location | License |
|----------|----------|---------|
| Item bank (1,500 items) | GitHub + Zenodo | CC-BY-4.0 |
| Generated responses (43,200 P1 + 18,000 P2) | Zenodo | CC-BY-4.0 |
| Human annotations (720 items × 2 coders) | GitHub + Zenodo | CC-BY-4.0 |
| Unlearned LoRA adapters (Mistral-7B + Qwen2.5-3B) | HuggingFace Hub | Apache-2.0 |
| Source code | GitHub (public) | MIT |
| Frozen prompts + demographic variants | GitHub | MIT |

---

## 10. Dataset and IRB

All item sources: MathDial (CC-BY-4.0), Jiajia Python MCQ dataset (GitHub, MIT),
LLM-generated items (no rights encumbrance). No human participants are recruited as
study subjects. Coders = research staff (job duties). IRB exemption on file.

---

*Analysis plan v2.0 — committed 2026-05-20. Any deviation will be noted in
decision_log.md with date and rationale.*
