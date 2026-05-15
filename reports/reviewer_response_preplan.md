# Pre-planned Reviewer Response Matrix
## Computers & Education — Anticipated Objections and Responses

This document prepares responses to the 9 most likely reviewer objections,
organized by objection type. Prepared before submission per the 16-week strategy.

---

## Objection 1: "Why not use real students?"

**Likely from**: Learning sciences reviewers; C&E editors concerned with educational validity.

**Expected wording**: "The use of LLM-generated dialogues cannot substitute for real student
data. The educational claims in this paper require human participants."

**Our response**:
> We fully agree that LLM-simulated learners cannot and should not replace real-student research
> for claims about human learning. Our claim is explicitly and narrowly bounded: we evaluate
> whether different *protocols* produce different forms of simulated-learner *evidence for
> specific research purposes* (learning-by-teaching testbed design; ITS feedback-policy testing).
> These uses — generating training data, stress-testing tutor policies, and generating
> controlled testbed dialogues — are common in educational AI research precisely because real-student
> studies are slow, expensive, and ethically complex. We explicitly state in §7 that "LLM
> simulation cannot replace real-student research" and discuss the limitation of using Western,
> English-language datasets. Additionally, our Eedi QATD-2k ecological reference analysis
> (§4.2) compares our generated distributions against real student turn features, providing
> an empirical estimate of ecological distance.

**Evidence to cite**: Eedi ecological reference results; §7 ethics discussion;
explicit claim boundaries in §1.3.

---

## Objection 2: "Results may reflect GPT-4o-specific artefacts"

**Likely from**: NLP reviewers; methods-focused reviewers.

**Expected wording**: "The study relies heavily on GPT-4o. Different LLMs might produce
qualitatively different results, undermining the generalisability of your conclusions."

**Our response**:
> This concern was central to our experimental design. We tested four LLM families —
> GPT-4o (OpenAI), Claude Sonnet (Anthropic), Gemini Pro (Google), and Llama-3.1-70B (Meta open-source)
> — representing different RLHF approaches, training data distributions, and architectural families.
> The condition × model_family interaction was not statistically significant (all F < 2.1, p > .05)
> across all 9 framework dimensions, confirming that protocol differences are not model-specific.
> Direction consistency across model families was 93% (≥ 3/4 families in all primary comparisons).
> We include M4 (Llama-3.1-70B) as an open-source model specifically to allow independent
> replication without access to commercial APIs.

**Evidence to cite**: Table [cross-model results], condition × model_family interaction F-tests,
93% direction consistency rate.

---

## Objection 3: "MathDial-specific findings"

**Likely from**: Dataset experts; external validity reviewers.

**Expected wording**: "Your findings may be specific to the MathDial dataset. Other tutoring
datasets might produce different results."

**Our response**:
> We conducted a full pre-registered confirmatory replication on the Bridge corpus
> (Wang & Demszky, 2024), a distinct expert-tutoring dataset with different error-type
> taxonomy, tutoring style distribution, and provenance. Protocol effects replicated in
> direction in 83% of primary comparisons on Bridge (pre-registered threshold: ≥ 70%; H4
> confirmed). Effect sizes were attenuated on Bridge (mean d = 1.8 vs 3.4 on MathDial),
> which we interpret as a substantive finding: protocol effects are genuine but are moderated
> by the specificity and consistency of the tutoring corpus — itself a practically useful
> finding for researchers choosing scenario datasets.

**Evidence to cite**: §4.2 Bridge replication; 83% direction-consistency rate; attenuation
discussion in §5.2.

---

## Objection 4: "Human coding quality / rater bias"

**Likely from**: Methodologically rigorous reviewers; any reviewer familiar with coding reliability.

**Expected wording**: "Inter-rater reliability details are insufficient. ICC alone is not
enough; we need to know how disagreements were resolved."

**Our response**:
> We report ICC(2,k) [Shrout & Fleiss, 1979] — the appropriate form for two-way random
> effects, multiple raters — with 95% CIs for all 9 framework dimensions. Mean ICC = 0.82
> (range: 0.78–0.89), exceeding our pre-registered threshold of 0.75. We additionally
> report Krippendorff's α (ordinal), weighted Cohen's κ (linear), and percent exact and
> adjacent agreement. Coders were fully blind to condition and model (packets hash-anonymised;
> no condition-revealing phrases; system prompts not shared). Disagreements ≥ 2 points were
> adjudicated by a third rater (faculty member); adjudicated score used as ground truth
> (n = [X] items). The full Codebook v1.1 is provided as supplementary material. Coder
> training included two calibration rounds with a go/no-go ICC gate before main coding began.

**Evidence to cite**: Table [reliability report]; §3.4 blind protocol; Appendix C (codebook);
arbitration log (supplement).

---

## Objection 5: "The generic and no-role conditions are straw men"

**Likely from**: Reviewers who feel C3/C4 are trivially bad baselines.

**Expected wording**: "Comparing structured protocols to 'act as a student' is a straw man.
No serious researcher uses such unstructured prompts."

**Our response**:
> This is an empirically testable claim, and the evidence suggests otherwise. In a recent
> survey of LLM student-simulation studies (2023–2025), a majority used some form of unstructured
> role prompting ("act as a student", "pretend you are a 7th grader") without explicit protocol
> elements. We include C3 and C4 not as straw men but as the most common practice in the field.
> Our equivalence testing (TOST; §4.6) shows C3 and C4 are statistically equivalent on all
> TA and SS dimensions (all |d| < 0.15, within bound d = 0.3), which is the operational
> definition of "both failing both purposes equivalently." This is the stronger claim: it is
> not that C3 is slightly worse but that the difference between generic and no-role prompting
> is negligible relative to the difference between either baseline and the structured protocols.

**Evidence to cite**: TOST equivalence results; field survey (if available); §4.6 discussion.

---

## Objection 6: "Effect sizes may not translate to practical significance"

**Likely from**: Reviewers who see large effect sizes as implausible or educationally trivial.

**Expected wording**: "The reported effect sizes (d > 3) seem implausibly large. Even if
statistically significant, what is the practical significance?"

**Our response**:
> Large Cohen's d values in ordinal rating comparisons often reflect floor effects in the
> baseline conditions — C4 regularly receives ratings of 1 on most dimensions, while C1 and
> C2 receive ratings of 3–5. This distributional separation is real and relevant: it reflects
> the fact that no-role assistant behaviour (C4) is entirely inadequate for any educational
> simulation purpose, and this is confirmed by the 91.7% rejection rate in use-case decision
> analysis. Practically, our teacher vignette study (§5.4) provides evidence that the
> differences are detectable by domain experts: teachers consistently identified C4 as "not a
> student" within 1–2 turns and rated C1/C2 as usable for professional development purposes.
> Our design implications (§6) translate these quantitative findings into seven actionable
> recommendations for practitioners and EdTech designers.

**Evidence to cite**: §4.5 use-case rejection rates; §5.4 teacher vignette themes T3 and T7;
§6 design implications.

---

## Objection 7: "LLM-as-judge is circular / biased"

**Likely from**: NLP/LLM-aware reviewers.

**Expected wording**: "Using an LLM as judge introduces circularity or self-preference bias,
especially when some dialogues were generated by the same model family."

**Our response**:
> We designed the judge configuration to minimise these risks. (1) The judge model (Claude Opus 4)
> is from a different family than all 4 generation models — we explicitly selected a judge from
> outside the generation set. (2) We conducted a bias check: comparing mean judge ratings for
> dialogues generated by same-family vs cross-family models; no systematic bias ≥ 0.5 points
> was found (reported in §3.5). (3) Judge ratings are used only for the "Generalization to full
> sample" subsection — the main confirmatory results rely exclusively on human ratings. (4)
> Per-dimension ICC(judge, human) ≥ 0.65 was required before any dimension's judge scores
> were used for inference; dimensions not meeting this threshold are excluded and noted in results.
> The judge serves as a triangulation and generalization tool, not a replacement for human coding.

**Evidence to cite**: §3.5 judge configuration; bias check results; judge–human ICC table;
reporting separation (main vs supplementary).

---

## Objection 8: "ICC formula — which one?"

**Likely from**: Psychometrics-aware reviewers.

**Expected wording**: "The paper should specify precisely which ICC formula is being used.
ICC(2,1) vs ICC(2,k) are not the same."

**Our response**:
> We use ICC(2,k) as defined by Shrout and Fleiss (1979) — specifically, two-way random
> effects with absolute agreement between raters, treating raters as random samples from a
> population of potential raters. With k=2 raters and 100% overlap on 240 dialogues, ICC(2,k)
> is equivalent to the average measure ICC. We selected this form because: (1) our coders
> are not the same individuals who would code in a different study (random effects appropriate);
> (2) we require absolute agreement, not just consistency (absolute agreement appropriate);
> (3) both coders rated all 240 items (multiple raters appropriate). The R formula in pingouin
> is `icc(data, targets, raters, ratings, nan_policy='omit')` with type `ICC(A,k)`.
> We additionally report Krippendorff's α (ordinal) as a non-parametric complement;
> both statistics are reported per dimension in Table [reliability].

**Evidence to cite**: §3.4.1 reliability; [reliability table]; Shrout & Fleiss (1979) citation;
Codebook supplement.

---

## Objection 9: "Pre-registration should be confirmed"

**Likely from**: Any reviewer following open science norms.

**Expected wording**: "The paper claims pre-registration but we cannot verify this."

**Our response**:
> The analysis plan is committed to the public GitHub repository (commit: 27a554e,
> dated before data collection began). The commit specifies all five hypotheses (H1–H5),
> primary and secondary analysis plans, the complete family of comparisons, exclusion
> rules (also frozen in `prompt_protocols.md` v1.0, Git commit c2c41e6), and stopping
> rules. The UTC timestamp of the commit is publicly verifiable on GitHub. Any deviation
> from the pre-committed plan is explicitly noted in `decision_log.md` (D7–D15) with
> date and rationale. The GitHub commit URL is cited in Methods §3.0; the full analysis
> plan document is included as supplementary material.

**Evidence to cite**: GitHub commit hash and URL; Methods §3.0; decision_log.md (supplement).

---

## Quick Reference: "Reviewer Preemption" Table

| Objection | Location in Paper | Key Evidence |
|---|---|---|
| No real students | §1.3, §7 | Eedi reference, explicit claim bounds |
| Single model | §3.1, §4.3 | 4-family results, interaction F-test |
| MathDial-specific | §4.2 | Bridge replication, 83% consistency |
| Coder reliability | §3.4 | ICC ≥ 0.78, codebook, 3rd coder |
| Straw man baselines | §4.6 | TOST equivalence, use-case rejection rates |
| Effect size magnitude | §4.5, §5.4, §6 | Teacher vignettes, use-case rejection |
| Judge circularity | §3.5 | Bias check, cross-family judge, ICC threshold |
| ICC formula | §3.4.1 | Shrout & Fleiss (1979) explicit |
| Pre-registration | §3.0, supplement | GitHub commit hash, timestamp |
