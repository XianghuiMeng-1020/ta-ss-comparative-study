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
**Status**: LOCKED — generation started 2026-05-16
**Logged**: 2026-05-15
**Updated**: 2026-05-16 — OpenRouter used for M2-M4 (single API key, OpenAI-compatible); M3 re-routed from gemini-2.5-pro-preview (thinking model, incompatible with 600 token limit) to google/gemini-2.0-flash-001 (stable, fast); M4 estimated ~16h runtime at ~50s/dialogue; checkpoint-resume enabled

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

---

## D7. Multi-Model Generation Strategy (C&E External Validity Requirement)

**Decision**: Generate all 1,200 main dialogues under 4 LLM families; include 1 open-source model for reproducibility.
**Logged**: 2026-05-15
**Status**: LOCKED

| ID | Model ID | Family | Type | Role |
|----|----------|--------|------|------|
| M1 | `gpt-4o-2024-11-20` | OpenAI | Closed | Primary (original plan) |
| M2 | `claude-sonnet-4-5-20250929` | Anthropic | Closed | Cross-family replication |
| M3 | `gemini-2.5-pro` → routed to `google/gemini-2.0-flash-001` via OpenRouter | Google | Closed | Cross-RLHF replication |
| M4 | `meta-llama/Llama-3.1-70B-Instruct` | Meta | Open | Reproducibility anchor |

**Total main**: 4 models × 4 conditions × 3 seeds × 100 scenarios = **4,800 dialogues**
**Fallback**: If M4 exclusion rate > 30% on pilot, replace with `Qwen2.5-72B-Instruct` or `DeepSeek-V3`.
**Cost estimate**: $1,000–1,500 USD total across all 4 models.
**Rationale**: Single-model designs are systematically challenged by C&E reviewers as model-specific artefacts. Four-family design enables condition × model interaction test; open-source model ensures long-term reproducibility.

---

## D8. Bridge Scenario Construction Rules

**Decision**: Use Bridge (rose-e-wang/bridge) expert-tutor conversations to seed 80-scenario cross-dataset replication.
**Logged**: 2026-05-15
**Status**: LOCKED

**Tutor selection**: Expert tutor transcripts only (Bridge `expertise_level == "expert"` or equivalent field). Novice-tutor transcripts excluded to match MathDial's controlled-tutor design.

**Field mapping (Bridge → MathDial schema)**:
| Bridge field | MathDial equivalent | Notes |
|---|---|---|
| `e` (error type) | `error_type` | See taxonomy below |
| `z_what` (teacher strategy) | tutor_move | Not used in learner prompt; structural reference only |
| Dialogue turns | Tutor turns for replay | Expert tutor turns replayed verbatim (D2 rule) |
| Student utterances | `incorrect_solution` | First student utterance used as incorrect-solution seed |

**Error-type taxonomy mapping**:
| Bridge `e` value | MathDial `error_type` equivalent |
|---|---|
| conceptual | conceptual_misconception |
| procedural | procedural_slip |
| arithmetic | arithmetic_error |
| setup | setup_error |
| other | miscellaneous |

**Sampling**: 80 scenarios stratified by mapped error type; strata < 5 rows merged to misc bucket.
**Execution**: 80 × 4 conditions × 2 seeds × 2 models (M1+M2) = **1,280 dialogues**, ~$300.
**Analysis**: Separate `mixed_models.py` run on Bridge data; direction-consistency rate ≥ 70% required for confirmatory claim.

---

## D9. Coder Recruitment and Training Protocol

**Decision**: Recruit 2 research-assistant coders; 100% overlap on 240-dialogue main coding set.
**Logged**: 2026-05-15
**Status**: LOCKED

**Coder profiles**:
- Coder A: Educational psychology / learning sciences background; familiar with misconception coding
- Coder B: NLP / educational technology background; familiar with dialogue structure analysis

**Compensation**: $25–30/hr; estimated 80–120 hrs per coder; total ~$5,600.
**Training schedule** (Week 6, 20 hrs/person):
1. Session 1 (4 hrs): Blair (2007) + Koedinger (2015) framework study
2. Session 2 (4 hrs): Codebook v1.0 walkthrough with positive/negative examples
3. Session 3 (3 hrs): TalkMoves 30-utterance warm-up (κ ≥ 0.65 required)
4. Session 4 (5 hrs): 30-dialogue calibration round → ICC → discussion → codebook v1.1
5. Session 5 (4 hrs): 30-dialogue second calibration → must achieve ICC ≥ 0.75 to proceed

**Main coding** (Weeks 7–9): 240 dialogues × 100% overlap (both coders rate all 240).
**Arbitration**: Items with |rating_A − rating_B| ≥ 2 → third coder (faculty member) adjudicates; adjudicated score = ground truth.
**Platform**: Qualtrics survey (one survey per dialogue packet; conditions masked by packetize.py).
**Blind protocol**: Coders do not see condition labels, model IDs, or other coder ratings until all 240 are complete.

---

## D10. LLM-as-Judge Configuration

**Decision**: Use Claude Opus 4 (or GPT-5) as judge model on remaining ~4,500 dialogues after human coding.
**Logged**: 2026-05-15
**Status**: LOCKED

**Judge model**: Must be from a different family than all 4 generation models to avoid self-preference bias. Preferred: `claude-opus-4` (if not in generation set) or `gpt-5` (if Opus 4 is in generation set).
**Calibration**: Judge evaluated on all 240 human-coded dialogues. ICC(judge, human) ≥ 0.65 per dimension required to include that dimension's judge scores in Generalization analysis; otherwise dimension is excluded from judge-based analysis.
**Self-consistency**: Each dialogue rated 3× by judge; median score used.
**Bias check**: Compare mean ratings per condition for dialogues generated by same vs different model family as judge. Flag if systematic bias ≥ 0.5 points.
**Reporting**: Main Results table = human ratings (n=240). "Generalization to full sample" subsection = judge ratings (n=4,800), explicitly labelled.
**Cost estimate**: ~$200–400 for 4,500 × 9 dimensions × 3 passes.

---

## D11. Statistical Analysis Upgrades

**Decision**: Upgrade primary analysis from Wilcoxon to linear mixed-effects model; add Holm-Bonferroni and TOST.
**Logged**: 2026-05-15
**Status**: LOCKED

**Primary model**: `lmer(rating ~ condition * model_family + (1|scenario_id) + (1|coder_id), data=ratings)`
  - condition: C1/C2/C3/C4 (main factor)
  - model_family: M1/M2/M3/M4 (robustness factor)
  - scenario_id: random intercept (repeated measures across conditions)
  - coder_id: random intercept (repeated measures across dialogues)
  - Implementation: `pymer4` (Python wrapper for R's `lme4`)

**Multiple comparison correction**:
  - Primary: Holm-Bonferroni (family = 4 primary contrasts)
  - Secondary: Benjamini-Hochberg FDR (reported in supplement)
  - Family of comparisons disclosed in Methods

**Effect size**: Cohen's d with 95% CI via bootstrap (5,000 iterations)

**Equivalence test**: Two one-sided t-tests (TOST) for C3 vs C4 on TA1–TA4 and SS1–SS5. Equivalence bound = d = 0.3 (small effect; chosen a priori). Supports "generic baseline is statistically equivalent to no-role baseline" claim.

**Power analysis**: Post-hoc G*Power calculation reported for n=240 (human) and n=4,800 (judge). Minimum detectable effect size documented.

---

## D12. Ablation Expansion (8 Elements)

**Decision**: Expand ablation set from 4 to 8, removing one structural element at a time.
**Logged**: 2026-05-15
**Status**: LOCKED

| Ablation ID | Protocol | Element Removed | Expected Drop |
|---|---|---|---|
| C1_ablate_transfer | C1 | Near-transfer requirement | TA1 |
| C1_ablate_revise | C1 | Explicit revision requirement | TA2 |
| C1_ablate_questions | C1 | Clarifying-question requirement | TA2/TA3 |
| C1_ablate_student_stance | C1 | Student-stance maintenance | TA3 |
| C2_ablate_misconception | C2 | Target misconception injection | SS2 |
| C2_ablate_gradual | C2 | Gradual learning constraint | SS4 |
| C2_ablate_verbalize | C2 | Verbalize-reasoning requirement | SS1/TA4 |
| C2_ablate_prior_knowledge | C2 | Prior-knowledge boundary | SS3 |

**Execution**: 25 scenarios × M1 × 2 seeds × 8 ablations = 400 dialogues (~$150).
**Analysis**: Compare ablation vs full-protocol on corresponding metric using paired Wilcoxon; report % drop.

---

## D13. Analysis Plan — GitHub Timestamp (replacing OSF pre-registration)

**Decision**: Commit analysis plan to public GitHub repo before data generation begins.
**Logged**: 2026-05-15
**Status**: OPEN — commit pending before main generation run

**Rationale**: Provides a verifiable UTC timestamp that hypotheses H1–H5 and analysis
specifications were fixed before data collection. Functionally equivalent to OSF
AsPredicted for audit purposes; requires ~5 minutes vs ~30 minutes for OSF.

**Hypotheses specified**:
- H1: C1 > all others on TA1–TA4 (Holm-Bonferroni corrected, α=0.0125)
- H2: C2 > C3 and C2 > C4 on SS1–SS5 (Holm-Bonferroni corrected, α=0.0125)
- H3: Protocol effects replicate in direction across ≥ 3 of 4 model families
- H4: Protocol effects replicate in direction across MathDial and Bridge datasets (≥ 70% consistency)
- H5: Each ablation element removal reduces its target metric below full-protocol baseline

**Document**: `reports/analysis_plan.md` (committed to GitHub public repo)
**Commit hash**: 27a554e2ed884eda245d2f9f9111751b235101ba
**Commit date**: 2026-05-15
**GitHub URL**: github.com/XianghuiMeng-1020/ta-ss-comparative-study/blob/27a554e2ed884eda245d2f9f9111751b235101ba/reports/analysis_plan.md

**In paper**: Methods §3.0 "Analysis Plan and Transparency"

---

## D14. IRB Classification

**Decision**: Study is IRB exempt — no human participants; public datasets; coders are research staff.
**Logged**: 2026-05-15
**Status**: OPEN — exempt determination pending

**Rationale for exemption**:
- Primary data: MathDial, Bridge, Eedi QATD-2k, TalkMoves — all publicly available, consented, de-identified
- No new data collection from students or tutors
- Coders = research staff performing job duties (not human subjects)
- Teacher vignette study (Phase 6) — minimal risk; separate IRB determination filed

**IRB number**: TBD — insert after determination letter received.

---

## D15. Open Materials and Reproducibility Package

**Decision**: Full open-materials package on GitHub + Zenodo before submission.
**Logged**: 2026-05-15
**Status**: OPEN — complete before Week 16

**Components**:
| Artifact | Location | License |
|---|---|---|
| Source code | GitHub (public) | MIT |
| Generated dialogues (4,800) | Zenodo (de-identified) | CC-BY-4.0 |
| Human coder ratings (240) | GitHub / Zenodo | CC-BY-4.0 |
| LLM judge ratings (4,800) | Zenodo | CC-BY-4.0 |
| Frozen prompts | GitHub | MIT |
| Datasheet for Datasets | `data/DATASHEET.md` | CC-BY-4.0 |
| Model cards | `MODELCARDS.md` | CC-BY-4.0 |
| Code snapshot DOI | Zenodo | MIT |

**`CITATION.cff`**: Included in repo root.
**arXiv preprint**: Submitted simultaneously with journal submission (C&E allows preprints).

---

## D6. Supplementary Datasets for Cross-Dataset Generalization (Computers & Education)

**Decision**: Add 3 real public tutoring datasets alongside MathDial (primary)  
**Logged**: 2026-05-15  
**Status**: LOCKED — all 3 datasets downloaded and verified  

### Rationale
Computers & Education reviewers expect cross-dataset robustness. Three supplementary corpora from distinct platforms, populations, and modalities strengthen the claim that TA/SS protocol differences are not MathDial-specific artefacts. All three are real human-produced data; NO synthetic or simulated data included.

### Datasets Selected

| # | Name | Source | License | Records | Key Annotations | Rationale |
|---|------|---------|---------|---------|-----------------|-----------|
| D1 | **MathDial** (primary) | eth-nlped/mathdial (HF) | CC-BY-4.0 | 2,861 dialogues | Student confusion, teacher move taxonomy, grade-7 | Locked primary dataset |
| D2 | **Bridge** | rose-e-wang/bridge (HF) | CC-BY-NC-4.0 | 700 conversations | Student error type (`e`), teacher strategy (`z_what`), teacher intention (`z_why`) | Expert-annotated error taxonomy; novice vs. expert tutor pairs; NAACL 2024 |
| D3 | **Eedi QATD-2k** | Eedi/Question-Anchored-Tutoring-Dialogues-2k (HF) | CC-BY-NC-SA 4.0 | 1,971 interventions / 68,717 messages | Talk-move predictions, question metadata, PII-anonymised | Large-scale real 1:1 platform data, UK students age ~12, async chat; EMNLP 2025 |
| D4 | **TalkMoves** | SumnerLab/TalkMoves (GitHub) | CC-BY-NC-SA 4.0 | 567 transcripts / 237,537 utterances | 10 discursive move labels, student-move labels | K-12 classroom discourse with cross-validated talk-move taxonomy; LREC 2022 |

### Exclusion Rationale (datasets considered but rejected)
- **CIMA** (kstats/CIMA): English-to-Italian language tutoring — not math, not comparable  
- **NCTE Transcripts** (Demszky & Hill 2022): Requires manual Google Form sign-up per user — not directly downloadable  
- **ASSISTments**: Student problem logs without dialogue turns — structural mismatch  

### Usage Plan
- **Bridge** (D2): Provide scenario seeds for cross-dataset replication of main 100-scenario experiment (up to 100 matched error-type scenarios); compare protocol differences on `e`-labelled subsets  
- **Eedi QATD-2k** (D3): Use as ecological validity reference for talk-move distribution comparison; compare tutor-move repertoire to protocol-generated tutor behaviours  
- **TalkMoves** (D4): Validate talk-move taxonomy alignment; use `Teacher Tag` labels as reference distribution for coding calibration  

### File Locations
```
data/raw_bridge/          train.csv, validation.csv, test.csv, bridge_all.jsonl
data/raw_eedi_qatd/       anchored_dialogues_train.csv, anchored_dialogues_test.csv,
                          dq_question_metadata_train.csv, eedi_dialogues_all.jsonl
data/raw_talkmoves/       talkmoves_train_data.csv, talkmoves_transcripts_all.csv,
                          talkmoves_student_labels.csv, TalkMoves_repo/
```

---

## D16. Full Study Reframe — IEEE TLT Target

**Decision**: Reframe study from "TA vs SS protocol comparison" (C&E target) to "multi-model, multi-question-type, multi-dimension naive-student simulation comparative analysis" (IEEE TLT target).
**Logged**: 2026-05-20
**Status**: LOCKED — analysis_plan.md v2.0 committed; all subsequent work follows v2 design.

**Rationale**: Jiajia's AIED 2026 paper (unlearn-and-relearn, Mistral-7B, Python MCQ) directly addresses the single-protocol student simulation question from a model-weight perspective. Our study is most novel as a comparative benchmark that:
(a) answers Jiajia's three stated limitations (single model → 12 models; MCQ-only → 5 QTypes; Python-only → Python + Math);
(b) adds 6 new evaluation dimensions beyond accuracy (D1, D3, D4, D5, D6, D7);
(c) provides the first Pareto-frontier characterization of naive-student model configurations.

**Target venue change**: Computers & Education → IEEE Transactions on Learning Technologies (Regular Paper, ~14 double-column pages). IEEE TLT has broader scope for multi-model benchmarking studies.

**Preserved assets from v1 design**:
- 4,800 OED dialogues → n-axis OED slice under P1 (Big-tier models)
- 10 trace metrics → contribute to D7 and D2 explanation quality
- 240-dialogue human coding infrastructure → extended for D1 human-likeness

**New components logged below (D17–D25)**

---

## D17. Model Roster (m-axis)

**Decision**: 12 models across 4 tiers; Big-tier reuses existing data.
**Logged**: 2026-05-20
**Status**: LOCKED

| ID | Model | Params | Tier | Inference |
|----|-------|--------|------|-----------|
| M01 | SmolLM2-360M | 0.36B | Tiny | vLLM |
| M02 | TinyLlama-1.1B | 1.1B | Tiny | vLLM |
| M03 | Llama-3.2-1B | 1B | Tiny | vLLM |
| M04 | Qwen2.5-1.5B-Instruct | 1.5B | Small | vLLM |
| M05 | Phi-3.5-mini-instruct | 3.8B | Small | vLLM |
| M06 | Llama-3.2-3B | 3B | Small | vLLM |
| M07 | Qwen3-4B | 4B | Mid | vLLM |
| M08 | Mistral-7B-Instruct-v0.3 | 7B | Mid (anchor) | vLLM |
| M09 | Llama-3.1-8B-Instruct | 8B | Mid | vLLM |
| M10 | gpt-4o-2024-11-20 | ~200B | Big | OpenAI API |
| M11 | claude-sonnet-4-5-20250929 | ~70B | Big | Anthropic API |
| M12 | Llama-3.1-70B-Instruct | 70B | Big | OpenRouter |

Note: Gemini-2.0-Flash removed from Big roster (replaced by Claude-Sonnet in v2 to
match Jiajia's set for consistency). Gemini dialogues in outputs/ available as supplement.

---

## D18. Item Bank Construction

**Decision**: 1,500 items across 5 QTypes; IRT-calibrated difficulty labels.
**Logged**: 2026-05-20
**Status**: OPEN — to be completed before P1 generation (W3–4)

| QType | N | Source | File |
|-------|---|--------|------|
| MCQ | 600 | 300 Python (Jiajia dataset) + 300 Math (MathDial-derived + LLM-generated) | data/item_bank/mcq/items.jsonl |
| TF | 300 | Auto-derived from MCQ correct/distractor pairs | data/item_bank/tf/items.jsonl |
| Fill | 300 | MathDial reasoning steps (cloze deletion) | data/item_bank/fill/items.jsonl |
| SA | 300 | MathDial reasoning steps (open prompts) | data/item_bank/sa/items.jsonl |
| OED | 100 scenarios | Existing MathDial 100-scenario bank (data/scenarios.csv) | data/scenarios.csv (reused) |

Difficulty calibration: 2PL IRT via `py-irt` on Mistral-7B baseline P1 responses.

---

## D19. Persona Implementation

**Decision**: Two simulation strategies (P1 prompt-based, P2 unlearning-based).
**Logged**: 2026-05-20
**Status**: LOCKED

P1 prompt: prompts/naive_student_prompt.txt (unified across all QTypes; QType-specific
format suffixes in prompts/format_suffixes/).
P2 unlearning: src/unlearning/ (LoRA + KL; β=0.1, lr=1e-4, 20 epochs, r=8, α=32).
P2 models: Mistral-7B (primary), Qwen2.5-3B (cross-family replication).
P2 unlearning ratios: 10%, 30%, 50% × 5 seeds each.

---

## D20. Demographic Injection for CEAT (D5)

**Decision**: 4 attribute sets × 4 conditions each = 16 prompt variants.
**Logged**: 2026-05-20
**Status**: LOCKED

| Attribute set | Conditions |
|---------------|-----------|
| Gender | male / female / gender-neutral / omitted |
| Race | White / Asian / Black / Hispanic |
| National | US / Chinese / Indian / British |
| SES | high-income / low-income / unspecified / working-class |

CEAT procedure: verbatim per Peng et al. (2025) — prompt-engineered target/attribute
word extraction via RAG; cosine similarity of sentence-transformer embeddings.
Model: sentence-transformers/all-mpnet-base-v2.

---

## D21. Statistical Model v2

**Decision**: Replace 2-way lmer (condition × model_family) with 4-way lmer.
**Logged**: 2026-05-20
**Status**: LOCKED

Primary: lmer(metric ~ model_tier * question_type * difficulty * persona
              + (1|item_id) + (1|coder_id))

Pareto: non-dominated sorting on {D1, D2, −D4_cost}; visualized as frontier plot.
TOST: d = 0.3 bound; applied to H-equiv claim (Qwen2.5-3B vs. Mistral-7B on D1).

---

## D22. Human Annotation v2 (augmented codebook)

**Decision**: Extend existing 240-dialogue codebook with D1 human-likeness dimension;
add 480 new single-response coding packets.
**Logged**: 2026-05-20
**Status**: OPEN — training sessions scheduled for W8–9

New codebook dimensions added to existing TA1–TA4 / SS1–SS5:
- HL1: Overall student-likeness (1–5 Likert)
- HL2: Error plausibility (1–5 Likert)
- HL3: Turing binary (0 = AI, 1 = human student)
- EQ1: Explanation quality for SA/OED (1–5 Likert)

ICC ≥ 0.75 gate per dimension before analysis.

---

## D23. Reproducibility Package

**Decision**: Full open materials on GitHub + Zenodo + HuggingFace Hub.
**Logged**: 2026-05-20
**Status**: OPEN — complete before W16

| Artifact | Location | License |
|----------|----------|---------|
| Item bank (1,500 items) | GitHub + Zenodo | CC-BY-4.0 |
| P1 responses (43,200) | Zenodo | CC-BY-4.0 |
| P2 responses (18,000) | Zenodo | CC-BY-4.0 |
| Human annotations (720 items) | GitHub + Zenodo | CC-BY-4.0 |
| Unlearned LoRA adapters | HuggingFace Hub | Apache-2.0 |
| Source code | GitHub (public) | MIT |
| Frozen prompts | GitHub | MIT |

---

## D24. P1 API Execution — Model Roster Adjustment

**Decision**: Adjusted model roster to match API availability. Tier 1/2 models now include
6 API-accessible models; Tier 3/4 (vLLM) deferred to GPU server setup.
**Logged**: 2026-05-20
**Status**: LOCKED — executing as of 2026-05-20 00:57 UTC+8

**Running via API (P1 executed 2026-05-20):**

| Label | Model ID | Backend | Tier |
|-------|----------|---------|------|
| M1 | gpt-4o-2024-11-20 | OpenAI | Big |
| M2 | claude-sonnet-4-5-20250929 | OpenRouter | Big |
| M3 | gemini-2.5-pro (via gemini-2.5-pro-preview-05-06) | OpenRouter | Big |
| M4 | meta-llama/Llama-3.1-70B-Instruct | OpenRouter | Mid |
| M5 | Qwen/Qwen2.5-72B-Instruct | OpenRouter | Mid |
| M6 | deepseek-ai/DeepSeek-V3 | OpenRouter | Mid |

**Deferred to GPU server (require vLLM):**
- Tiny: SmolLM2-360M, TinyLlama-1.1B, Llama-3.2-1B
- Small: Qwen2.5-1.5B, Phi-3.5-mini, Llama-3.2-3B
- Mid: Qwen3-4B, Mistral-7B-Instruct (anchor), Llama-3.1-8B

**Item bank**: 1500 items with fast-path difficulty injection (Python MCQ: GPT-4o-mini
self-labelled; Math MCQ/TF/Fill/SA: MathDial difficulty_raw metadata). Full IRT proxy
(irt_proxy_api.py) available when resources allow.

**Generation**: 6 models × 4 QTypes × 3 seeds × 600 items (MCQ), 300 items (TF/Fill/SA)
running in parallel with checkpoint-resume. Monitor: scripts/wait_and_packetize.py.

---

*Append new decisions below with date, rationale, status. Never delete or backdate entries.*
