# IEEE TLT Submission Checklist
## "How Naive is 'Naive'?" — v2 Reframe

Use this list as the final gate before hitting "Submit" in the IEEE Author Portal.
All boxes must be ✅ before submission.

---

## Pre-Submission Review (W17–W18)

### Paper Completeness
- [ ] All 10 sections written and internally consistent
- [ ] All `[RESULTS PENDING]` placeholders replaced with actual numbers
- [ ] All `[INSERT]` placeholders replaced (GitHub commit hash, Zenodo DOI, HF Hub URLs)
- [ ] 14 pages ± 1 page (check with LaTeX word count)
- [ ] Abstract ≤ 250 words (IEEE TLT requirement)
- [ ] All 8 figures present, labeled, referenced in text
- [ ] All tables numbered and referenced
- [ ] References complete and formatted per IEEEtran style
- [ ] No spelling/grammar errors (run Grammarly pass)

### Statistical Completeness
- [ ] All 6 hypotheses (H1–H6 + H-equiv) reported with test statistic, p, d, CI
- [ ] Holm-Bonferroni p_adj values reported alongside raw p values
- [ ] Bootstrap CI column in Table of effect sizes (5,000 iterations confirmed)
- [ ] TOST result for H-equiv (Qwen2.5-3B vs. Mistral-7B on D1) reported
- [ ] Pre-registration commit hash in §Statistical Analysis
- [ ] Exclusion rates table (Table A1) in Appendix

### Data and Code
- [ ] Zenodo DOI confirmed and live
- [ ] All item-bank stubs replaced with real items (run: `python src/data/build_item_bank.py --verify`)
- [ ] Human ICC ≥ 0.75 for all annotated dimensions
- [ ] Unlearning adapters uploaded to HuggingFace Hub
- [ ] README.md updated with v2 instructions and links
- [ ] Git tag `v2.0-tlt-submission` created

### Figures
- [ ] Fig 1: Scaling curves (D1 + D2 vs. log params)
- [ ] Fig 2: Pareto frontier (D1 vs. D2 vs. cost)
- [ ] Fig 3: CEAT fairness heatmap
- [ ] Fig 4: Type-A/B cross-tab
- [ ] Fig 5: Forest plot (effect sizes D1–D7)
- [ ] Fig 6: Unlearning forgetting curves (Mistral + Qwen)
- [ ] Fig 7: D1 by QType × Difficulty
- [ ] Fig 8: P1 vs. P2 head-to-head
- [ ] All figures in PDF vector format; fonts embedded

---

## Response to Anticipated Reviewers

### Anticipated Concern 1: Scope too broad
**Pre-planned response**: The 7 dimensions directly address the protégé effect's
practical requirements (fidelity, performance, efficiency, fairness). Each dimension
has a distinct operational definition and statistical model. The Pareto analysis
synthesises them into a single actionable recommendation—this is the paper's core
contribution rather than a weakness.

### Anticipated Concern 2: Comparison to real student data
**Pre-planned response**: We acknowledge that LLM simulators may not match real
student error distributions. The paper frames this as a simulator evaluation, not
a claim of ecological validity. Type-A/B profiles are presented as hypotheses for
real-student research, not empirical claims about actual students.

### Anticipated Concern 3: P2 limited to 2 models
**Pre-planned response**: GPU constraints (80 GPU-hours) limited P2 to Mistral-7B
(the original Jiajia anchor) and Qwen2.5-3B (cross-family replication). All training
code and hyperparameters are released, enabling replication on additional models.

### Anticipated Concern 4: CEAT validity for student-side generation
**Pre-planned response**: We directly extend Peng et al. (2025), who validated CEAT
for AI-generated educational content. We follow their procedure verbatim (word lists,
sentence-transformer model, effect size threshold) and release word lists for replication.

### Anticipated Concern 5: Human-likeness circularity (judge is AI, not real students)
**Pre-planned response**: D1 uses *human* raters (2 coders), not AI judges. The GPT-2
perplexity proxy is explicitly labelled as an automated *supplementary* metric and not
used in any confirmatory hypothesis test.

---

## IEEE Author Portal Submission Steps

1. Go to: https://mc.manuscriptcentral.com/tlt-ieee
2. Create submission: "Regular Paper"
3. Upload:
   - Main PDF (reports/tlt_paper_draft.pdf)
   - Figures (individual PDF files from outputs/figures/)
   - Cover letter (see template below)
   - Supplementary materials (Appendix: item bank sample, full exclusion table,
     per-model per-dimension tables)
4. Enter keywords: teachable agents, student simulation, large language models,
   machine unlearning, human-likeness, fairness, knowledge tracing
5. Suggest editors: [Look up IEEE TLT area editors handling AI in education]
6. Suggest reviewers: [3–5 researchers in ITS/AI-Ed with LLM/simulation expertise]
7. Conflict of interest declarations: [Complete per institution]

---

## Cover Letter Template

Dear IEEE TLT Editors,

We submit "How Naive Is 'Naive'? A Multi-Model, Multi-Question-Type, Multi-Dimension
Evaluation of LLM-based Student Simulation for Teachable-Agent Research" for
consideration as a Regular Paper in IEEE Transactions on Learning Technologies.

This paper addresses a critical gap in teachable-agent research: how do we choose
the right LLM configuration for naive-student simulation? We present the first
systematic comparison across 12 models, 5 question types, and 7 evaluation dimensions,
including the first CEAT-based fairness audit and dialogue-test discrepancy analysis
for TA simulation.

The study is fully pre-registered (GitHub commit [INSERT]), and all code, data, and
unlearned model adapters are publicly released (Zenodo DOI [INSERT]).

This manuscript is not under review elsewhere. No human participants were recruited;
IRB exemption is on file.

We believe this paper is well-suited for IEEE TLT's readership of learning technology
researchers and educational AI practitioners.

Sincerely,
[Authors — blinded for review]
