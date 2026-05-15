# IRB Exempt Determination Request
## Template — Complete Institutional Details Before Submission

**Date**: [DATE]
**To**: Institutional Review Board, [INSTITUTION NAME]
**From**: [PI NAME], [DEPARTMENT], [INSTITUTION]
**Re**: Request for Exempt Determination — "Framework-Grounded Comparative Evaluation of Teachable Agent and Student Simulation Protocols"

---

## Study Summary

We request an exempt determination for a methodological study comparing two LLM-based
simulated-learner prompting protocols (Teachable Agent and Student Simulation) using
pre-existing public datasets. This study does NOT involve recruitment of human participants
and does NOT collect new primary data from students or tutors.

---

## Exemption Category Sought

**45 CFR 46.104(d)(2)** — Research involving the use of educational tests, survey procedures,
interview procedures, or observation of public behavior, when:
- (i) information obtained is recorded in such a manner that human subjects cannot be identified,
  directly or through identifiers linked to the subjects; or
- (ii) any disclosure of the human subjects' responses outside the research could not reasonably
  place the subjects at risk.

**Additionally applicable**: 45 CFR 46.104(d)(4) — Secondary research for which consent is not
required (publicly available data).

---

## Why This Study Qualifies for Exemption

### 1. No Human Participants Recruited

This study uses exclusively pre-existing, publicly available datasets:

| Dataset | Source | License | Human Data Type |
|---------|--------|---------|----------------|
| MathDial | Macina et al. (2023), HuggingFace `eth-nlped/mathdial` | CC-BY-4.0 | Anonymised tutor-student dialogue transcripts |
| Bridge | Wang et al. (2024), HuggingFace `rose-e-wang/bridge` | CC-BY-NC-4.0 | Anonymised tutoring conversations |
| Eedi QATD-2k | Eedi Ltd, HuggingFace `Eedi/Question-Anchored-Tutoring-Dialogues-2k` | CC-BY-NC-SA-4.0 | PII-anonymised student-tutor chat logs |
| TalkMoves | SumnerLab/TalkMoves (GitHub) | CC-BY-NC-SA-4.0 | Anonymised classroom discourse transcripts |

All datasets were publicly released by their original creators, who obtained appropriate consent
from participants and performed de-identification before public release. We use these datasets
as scenario seeds for LLM dialogue generation; we do not analyse the original human participants'
data beyond selecting scenario templates.

### 2. No New Data Collection from Students or Tutors

The study generates synthetic dialogue data by prompting LLMs (GPT-4o, Claude Sonnet,
Gemini Pro, Llama-3.1-70B) to act as simulated learners. The "learner" in each dialogue is
an AI model, not a human. No students participate in the study, no tutors are recruited, and
no educational interventions are administered.

### 3. Human Coders Are Research Staff (Not Human Subjects)

Two graduate research assistants will rate subsets of the AI-generated dialogues for quality
and protocol adherence. These raters are:
- Research staff employed by [INSTITUTION] performing standard research duties
- Not subjects of the research (their data is not the focus of any research question)
- Consenting adults performing job duties under employment agreements

This activity is analogous to standard data annotation work and does not constitute human
subjects research.

### 4. Teacher Vignette Study (Phase 6 — Separate Amendment If Required)

Phase 6 involves brief (30-minute) remote interviews with 4–6 practising mathematics teachers
who will evaluate short anonymous dialogue excerpts. This component:
- Is minimal risk (viewing text, providing opinions — no sensitive information)
- Involves adult volunteers
- Requires separate IRB determination (exempt under Category 2)

**We will file a separate amendment or new protocol for the teacher vignette study if your
office determines it requires separate review.**

---

## Research Objectives

The study compares how different LLM prompting protocols (Teachable Agent protocol derived
from Blair et al., 2007; Student Simulation protocol derived from Koedinger et al., 2015)
produce different forms of AI-generated learner dialogue when evaluated against purpose-specific
educational frameworks. Findings will inform best practices for researchers using LLMs to
simulate students in educational technology development.

---

## Data Security and Privacy

- All datasets used are publicly available and already de-identified
- Generated AI dialogues contain no personal information (AI-generated text)
- Human coder annotations will be stored on encrypted institutional servers
- No re-identification of dataset participants is attempted or possible
- Results will be published in aggregate form only

---

## Principal Investigator Attestation

I attest that this research does not involve human subjects as defined under 45 CFR 46.102(e),
and I believe it qualifies for exempt determination. I commit to notifying the IRB if the study
scope changes in ways that might alter this determination.

**PI Signature**: ___________________________  **Date**: __________
**Department Chair/Dean's Representative**: ___________________________  **Date**: __________

---

## Attachments

- [ ] Study protocol summary (this document)
- [ ] Dataset provenance documentation (see `data/DATASHEET.md`)
- [ ] Codebook v1.1 (human coder instrument)
- [ ] Coder consent/employment agreement template
- [ ] Analysis plan with GitHub commit timestamp (see `reports/analysis_plan.md`)

---

*For questions, contact: [PI NAME], [EMAIL], [PHONE]*
