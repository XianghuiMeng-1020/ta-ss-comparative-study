# Analysis Plan — Pre-committed to GitHub Before Data Collection
## Framework-Grounded Comparative Evaluation of Teachable Agent and Student Simulation Protocols

**Commit date**: [INSERT BEFORE GENERATION BEGINS]
**Commit hash**: [INSERT]
**GitHub URL**: github.com/[REPO]/blob/[COMMIT]/reports/analysis_plan.md
**Corresponding author**: [BLINDED FOR REVIEW]

> This document was committed to the public GitHub repository before any real LLM API
> calls were made for the main experiment. The UTC timestamp of the commit is publicly
> verifiable on GitHub and serves as a pre-data-collection record of all hypotheses,
> analysis specifications, exclusion rules, and stopping criteria.

---

## 1. Have any data been collected for this study already?

No. This analysis plan is committed before any real LLM API calls are made for the main experiment.
Pilot data generated with the `mock_runner.py` (using MathDial original student turns as synthetic
proxies) was used for pipeline validation only and will not be analysed or reported as
experimental results.

---

## 2. What is the main question being asked or hypothesis being tested in this study?

**Core research question**: Do Teachable Agent (TA) and Student Simulation (SS) protocols, when
implemented as LLM system prompts, produce structurally different simulated-learner evidence
under matched public tutoring scenarios, and is this difference stable across multiple LLM
families and datasets?

**Hypotheses** (directional, Holm-Bonferroni corrected at α = 0.0125 within each family):

- **H1 (TA superiority)**: C1 (Teachable Agent protocol) produces significantly higher ratings
  than C2, C3, and C4 on all four TA dimensions (TA1–TA4; Blair et al., 2007), with Cohen's d > 0.5.
- **H2 (SS superiority)**: C2 (Student Simulation protocol) produces significantly higher ratings
  than C3 and C4 on all five SS dimensions (SS1–SS5; Koedinger et al., 2015), with Cohen's d > 0.5.
- **H3 (Cross-model stability)**: The direction of H1 and H2 effects replicates in ≥ 3 of 4
  LLM families (GPT-4o, Claude Sonnet, Gemini Pro, Llama-3.1-70B).
- **H4 (Cross-dataset stability)**: The direction of H1 and H2 effects replicates in ≥ 70% of
  primary comparisons when the same experiment is run on Bridge scenarios (Macina et al., 2024).
- **H5 (Ablation separability)**: Each structural protocol element, when removed individually,
  produces a statistically significant reduction (p < .05, uncorrected) in its target trace metric
  relative to the full protocol baseline (8 ablation conditions).

**Equivalence claim** (secondary, exploratory):
- C3 (generic learner) and C4 (no-role assistant) are statistically equivalent on both TA and SS
  dimensions (TOST; equivalence bound d = 0.3).

---

## 3. Describe the key dependent variable(s) specifying how they will be measured

**Primary DV: Framework-based ratings** (human coders)
- 9 dimensions: TA1–TA4 (Blair et al., 2007) + SS1–SS5 (Koedinger et al., 2015)
- Scale: 1–5 ordinal, anchored as specified in Codebook v1.1
- Coders: 2 independent coders, 100% overlap, 240 dialogues
- Reliability requirement: ICC(2,k) ≥ 0.75 on all 9 dimensions before proceeding

**Secondary DV: Automatic trace metrics** (all valid dialogues)
- 10 metrics: question_asking_rate, reasoning_trace_rate, target_error_preservation,
  feedback_uptake_rate, near_transfer_attempt_rate, premature_correctness_rate,
  role_drift_rate, over_technical_language_rate, unsupported_reasoning_rate,
  correction_timing_index
- Computed over ALL valid dialogues (n ≈ 4,800 main + 1,280 Bridge)

**Tertiary DV: Use-case decision** (human coders)
- 4-category nominal: learning_by_teaching / diagnostic_simulation / feedback_policy_testing / reject
- Coders assign one label per dialogue; Cohen's κ ≥ 0.70 required

---

## 4. How many observations will be collected or what will determine sample size?

**Main experiment**: 100 scenarios × 4 conditions × 3 seeds × 4 models = 4,800 dialogues

**Human coding subset**: 4 models × 4 conditions × 15 dialogues = 240 dialogues coded
(both coders rate all 240; 100% overlap)

**Bridge cross-dataset replication**: 80 scenarios × 4 conditions × 2 seeds × 2 models = 1,280 dialogues

**Ablation subset**: 25 scenarios × 8 ablation variants × 2 seeds × 1 model = 400 dialogues

**Power analysis**: For the primary Wilcoxon signed-rank test on scenario-matched means (n=100
scenarios), minimum detectable effect size is d ≈ 0.32 at 80% power and α = 0.0125 (Bonferroni-
corrected). Our preliminary trace metric differences (from mock pipeline) suggest d > 1.0 for
primary TA/SS comparisons, providing substantial power margin.

**Stopping rule**: All 100 main scenarios will be run. No sequential stopping. Dialogue-level
exclusions follow pre-registered rules E1–E6 only.

---

## 5. Anything else you would like to pre-specify?

### Exclusion rules (also frozen in prompt_protocols.md v1.0, commit c2c41e6)
| Code | Trigger | Action |
|------|---------|--------|
| E1 | Empty learner turn (< 5 chars) | Drop turn; ≥ 2 such turns → exclude dialogue |
| E2 | Non-math content | Exclude dialogue |
| E3 | Unsafe / refusal response | Exclude dialogue, log |
| E4 | Token overflow (> 800 tokens in single turn) | Truncate; if answer lost → exclude |
| E5 | Missing C1 transfer response | Regenerate once; if fails → exclude |
| E6 | Condition leakage | Exclude dialogue |

No additional exclusion rules may be added after pilot generation begins.

### Analysis pipeline
1. Human coding (ICC ≥ 0.75 gate) → primary lmer analysis → H1/H2
2. Automatic trace metrics (all dialogues) → secondary correlation with human ratings
3. LLM-judge calibration (ICC ≥ 0.65 gate) → generalization analysis on n=4,800
4. Bridge replication → H4
5. Ablation analysis → H5
6. TOST equivalence → C3/C4 equivalence claim
7. Robustness grids → topic/error_type/difficulty/model/seed splits

### Primary analysis model
```
lmer(rating ~ condition * model_family + (1|scenario_id) + (1|coder_id))
```
Estimated with REML. Contrasts: deviation coding with C4 as reference.
Significance: Holm-Bonferroni corrected over 4 primary contrast families.

### What will count as a positive result
H1 supported if: C1 > C2, C1 > C3, C1 > C4 on ALL of TA1–TA4, all p_holm < .0125, all d > 0.5.
H2 supported if: C2 > C3, C2 > C4 on ALL of SS1–SS5, all p_holm < .0125, all d > 0.5.
Partial support reported if some (not all) dimensions meet threshold.

### What will be reported regardless of outcome
- All 9 dimensions, all comparisons, in main or supplementary tables
- All exclusion rates by condition and model
- Full ablation results (all 8)
- All robustness splits

---

## 6. Additional information

**Datasets**: MathDial (Macina et al., 2023; CC-BY-4.0), Bridge (CC-BY-NC-4.0),
Eedi QATD-2k (CC-BY-NC-SA-4.0), TalkMoves (CC-BY-NC-SA-4.0).

**No human participants**: All data are pre-existing public corpora. Human coders are
research staff performing job duties. IRB exemption determination filed separately.

**LLM-as-judge**: Judge ratings are secondary only. Main confirmatory results use human
ratings exclusively. Judge-human calibration ICC ≥ 0.65 is required before judge ratings
are used for any inference.

**Open materials**: All code, prompts, and (de-identified) dialogue data will be shared on
GitHub and Zenodo upon acceptance. The GitHub commit URL for this analysis plan will be
cited in the paper's Methods section.

**Target venue**: *Computers & Education* (Special Issue: GenAI Enhanced Learning, 2026).

---

*This analysis plan was prepared on 2026-05-15. Any deviations from this plan will be
explicitly noted in the paper's Methods section and decision_log.md.*
