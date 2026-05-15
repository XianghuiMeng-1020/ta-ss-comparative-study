# Cover Letter — Computers & Education
## Special Issue: GenAI Enhanced Learning

**Date**: 2026-05-15

**To**: The Editorial Board  
**Journal**: Computers & Education (Special Issue: GenAI Enhanced Learning, 2026)  
**Manuscript Title**: Choosing the Right Simulated Learner: A Framework-Aligned Comparison
of Teachable Agent and Student Simulation Protocols for Educational Technology Research  
**Submission Type**: Original Research Article

---

Dear Editors,

We submit for your consideration a pre-registered, framework-grounded comparative evaluation
of two LLM-based simulated-learner protocols for educational technology research. We believe
this manuscript makes a significant contribution to the Special Issue on GenAI Enhanced
Learning by addressing a critical methodological gap: the conflation of fundamentally different
simulated-learner prompting strategies that serve distinct educational purposes.

## Fit with the Special Issue

Our manuscript directly addresses the Special Issue's focus on how generative AI can be
rigorously evaluated and responsibly deployed in educational contexts. Specifically:

- We evaluate two established educational frameworks (Blair et al., 2007; Koedinger et al., 2015)
  as evaluation criteria for GenAI-generated learner dialogues
- We demonstrate that "act as a student" prompting is insufficient and that protocol-level
  distinctions have measurable, reproducible consequences for the evidence quality produced
- We provide practitioners with concrete design guidance for choosing and validating
  GenAI simulated learners in their specific educational context

## Key Contributions

1. **Multi-model external validity**: 4,800 dialogues across 4 LLM families (GPT-4o,
   Claude Sonnet, Gemini Pro, Llama-3.1-70B) with cross-model interaction testing
2. **Cross-dataset replication**: 1,280 Bridge corpus replication dialogues confirm
   findings generalise beyond MathDial
3. **Dual evaluation**: Double-blind human coding (240 dialogues, ICC ≥ 0.78) triangulated
   with LLM-as-judge (calibrated against human ratings) for 4,800-dialogue generalisation
4. **Practitioner perspective**: Six mathematics teachers evaluate protocol differences,
   providing ecological validity evidence
5. **Open science**: Analysis plan committed to GitHub before data collection (27a554e), IRB exempt, full materials publicly available

## Differentiation from Recent Work

Our protocol-level comparison is orthogonal to recent LLM simulation architecture papers.
Unlike PEERS (ACL 2025) and Embracing Imperfection (ACL 2025), which modify model
architecture to improve simulation, our unit of analysis is the *prompting protocol* and
its *purpose-aligned evaluation* — directly addressing the question practitioners face when
choosing how to deploy existing frontier models as simulated learners.
Unlike the BEA 2025 teacher-tutoring study (Macina et al., 2025), which documents failure
modes qualitatively, we provide quantitative evidence for protocol-specific failure rates
and replicate across models and datasets.

## Manuscript Details

- **Word count**: ~13,000 words (main text, excluding references and appendices)
- **Figures**: 4 (use-case decision tree, robustness heatmap, failure rates by condition,
  ecological reference comparison)
- **Tables**: 8 (conditions, model families, primary results TA, primary results SS,
  reliability, ablations, equivalence tests, teacher vignette themes)
- **Supplementary**: Full Codebook v1.1, extended robustness results,
  LLM-judge calibration details, Bridge replication full results

## Ethical Compliance

- Analysis plan committed to GitHub before data collection (commit: 27a554e)
- IRB exempt determination: [INSTITUTION], Protocol [NUMBER]
- No human participants; all datasets are public and consented
- Teacher vignette sub-study: minimal risk (text evaluation); separate IRB amendment

## Suggested Reviewers

We suggest the following reviewers with relevant expertise (none have conflicts of interest
with the authors):

1. **[Reviewer 1]** — expert in LLM educational applications and student simulation validity
2. **[Reviewer 2 — Jakub Macina]** — MathDial dataset creator, EMNLP 2023;
   expertise in tutoring dialogue datasets
3. **[Reviewer 3]** — expertise in teachable agent paradigm and learning-by-teaching
4. **[Reviewer 4]** — expertise in learning sciences evaluation methodology and inter-rater reliability
5. **[Reviewer 5]** — expertise in AI in education, large-scale evaluation of educational AI systems

## Exclusions

We have no suggested reviewer exclusions.

---

We confirm that this manuscript has not been published elsewhere and is not under
consideration by another journal. All authors have approved the manuscript and agree with
its submission.

Sincerely,

[Authors — blinded for review]

---

*All materials, code, and data available at [GitHub link], with archive at Zenodo.*
