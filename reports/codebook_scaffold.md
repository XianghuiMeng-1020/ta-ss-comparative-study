# Codebook — Framework-Grounded Evaluation of Simulated Learner Protocols

**Version**: v1.0-DRAFT  
**Status**: DRAFT — freeze after P4 pilot reliability round  
**Framework sources**: Blair et al. (2007); Koedinger et al. (2015)

This codebook is the operational guide for all human coders. Each dimension
maps directly to a published framework principle. Coders use observable
dialogue indicators, not intuition.

---

## General Instructions

1. Read the full dialogue transcript in the coder packet before rating any dimension.
2. Rate each dimension independently (do not let one rating influence another).
3. Use the 1–5 scale anchors. Half-points are not allowed.
4. Fill the yes/no decision rules BEFORE entering a numeric score.
5. Write a note for any rating of 1 or 5, and for all ratings where you are uncertain.
6. Blind condition: you will not see condition labels. Do not infer conditions.

---

## PART A — Teachable Agent Dimensions (Blair et al., 2007)

### TA1 — Independent Performance

**Framework source**: Blair et al. (2007) — Principle: independent agent performance.  
**Definition**: After the teaching dialogue, does the learner make a genuine independent attempt at the near-transfer problem, showing self-generated reasoning without requesting help?

**Observable indicators**:
- Learner response to final (transfer) prompt contains reasoning steps
- Learner does NOT ask for a hint before attempting
- Learner does NOT reproduce a tutor-given formula verbatim without adaptation

**Decision tree**:
- Does the last learner turn respond to a new problem? → if NO, rate 1
- Does it contain reasoning steps (not just a bare answer)? → if NO, rate 2
- Is the reasoning self-generated (not just echoing the tutor)? → if YES, rate 4–5

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | No attempt; refusal; "I don't know" without any working |
| 2 | Bare answer only; or copies tutor explanation verbatim |
| 3 | Partial attempt with some reasoning but incomplete or immediately asks for help |
| 4 | Clear attempt with visible reasoning; may contain errors but is genuinely independent |
| 5 | Complete independent attempt with explicit step-by-step reasoning; no hint-seeking |

**Positive example (score 5)**: "Ok so this time there are 48 items total. If 3/4 went to group A, that's 48×3÷4 = 36. So group B gets 48−36 = 12."  
**Negative example (score 1)**: "I'm not sure how to do this one. Can you help?"  
**Edge case**: Learner attempts but makes the same error as the original problem → rate 3 (attempt is genuine even if unsuccessful).

---

### TA2 — Productive Learner Behaviour

**Framework source**: Blair et al. (2007) — Principle: productive learner behaviour.  
**Definition**: Does the learner exhibit behaviours that support the teaching process — asking clarifying questions, showing partial understanding, explicitly revising, reflecting — without acting as an expert?

**Observable indicators**:
- Learner asks at least one genuine clarifying question
- Learner explicitly states a revision ("I see, I was wrong about X")
- Learner shows partial understanding (gets some steps right, others wrong)
- Learner does NOT immediately solve everything correctly

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | Passive acknowledgement only ("ok", "I see") with no revision or questioning |
| 2 | Minimal engagement; accepts corrections without showing revision |
| 3 | Some revision or questioning present but inconsistent |
| 4 | Consistent questioning and/or explicit revision; partial understanding visible |
| 5 | Rich productive behaviour: asking questions, revising, showing both correct and incorrect thinking |

**Positive example (score 5)**: "Wait, I thought you multiplied both sides by 2 first? Oh, I see — you need to divide, not multiply. Let me redo: 24÷2 = 12, so…"  
**Negative example (score 1)**: "OK. Got it."

---

### TA3 — Support for Teaching

**Framework source**: Blair et al. (2007) — Principle: teaching-supportive environments.  
**Definition**: Does the dialogue create opportunities for the tutor to explain, correct, probe, and check understanding? Does the learner's behaviour invite teaching moves?

**Observable indicators**:
- Learner errors or confusions are visible and correctable
- Learner responses leave room for tutor to probe further
- Dialogue does NOT short-circuit to a correct solution that requires no further teaching
- Tutor has something substantive to respond to in each turn

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | Learner solves everything or refuses to engage; no teaching opportunities |
| 2 | Minimal teaching opportunity; learner barely wrong or barely responsive |
| 3 | Occasional teaching opportunities but learner too correct or too passive |
| 4 | Clear and recurrent teaching opportunities across multiple turns |
| 5 | Rich teaching opportunities throughout; errors invite focused pedagogical response |

---

### TA4 — Visible Reasoning as Shared Representation

**Framework source**: Blair et al. (2007) — Principle: shared representation (adapted to text-based dialogue).  
**Definition**: Does the learner show enough of their reasoning in text form that the tutor (or a researcher) can inspect what the learner is thinking and teach from it?

**Observable indicators**:
- Learner writes out steps, not just answers
- Learner explains WHY they did an operation
- Reasoning errors are diagnosable from the learner's output

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | No reasoning shown; only bare answers |
| 2 | Minimal reasoning; reasoning is too vague to diagnose |
| 3 | Some reasoning visible but incomplete or fragmented |
| 4 | Clear reasoning in most turns; errors are diagnosable |
| 5 | Rich explicit reasoning throughout; every step visible and diagnosable |

---

## PART B — Student Simulation Dimensions (Koedinger et al., 2015)

### SS1 — Cognitive Model Plausibility

**Framework source**: Koedinger et al. (2015) — precise theory of learning goal.  
**Definition**: Does the learner's reasoning resemble a plausible 7th-grade student strategy (not random errors, not advanced reasoning, not perfect reasoning)?

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | Reasoning is not plausible as any student (random, incoherent, or adult-expert level) |
| 2 | Vaguely student-like but inconsistent or implausible |
| 3 | Partially plausible; some age-appropriate errors with some implausible elements |
| 4 | Mostly plausible 7th-grade reasoning with realistic errors |
| 5 | Highly plausible; indistinguishable from a typical 7th-grade student's reasoning pattern |

---

### SS2 — Error-Model Alignment

**Framework source**: Koedinger et al. (2015) — error-model goal.  
**Definition**: Does the target misconception (from MathDial metadata) appear in the learner's responses rather than a random or different error?

**Coder note**: The coder packet includes the target misconception label. Check whether the learner's first 1–2 turns reflect this specific misconception.

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | Target misconception absent; a different or random error appears |
| 2 | Vague suggestion of target misconception; weak alignment |
| 3 | Target misconception partially present; mixed with other errors |
| 4 | Clear target misconception in first 1–2 turns with some persistence |
| 5 | Target misconception clearly and consistently present; aligns precisely with label |

---

### SS3 — Prior-Knowledge Consistency

**Framework source**: Koedinger et al. (2015) — prior-knowledge goal.  
**Definition**: Does the learner stay consistent with the assigned ability level (7th grade) throughout? No use of knowledge the student would not have; no wildly inconsistent knowledge claims.

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | Clear violation (e.g., uses calculus) or wildly inconsistent knowledge claims |
| 2 | Occasional inconsistency in ability level |
| 3 | Mostly consistent with some minor inconsistencies |
| 4 | Consistent 7th-grade level throughout |
| 5 | Precisely consistent; deflects appropriately when asked about beyond-level content |

---

### SS4 — Learning-Process Plausibility

**Framework source**: Koedinger et al. (2015) — learning-process goal.  
**Definition**: Does the learner change gradually in response to feedback rather than becoming expert immediately? Is the rate of improvement plausible?

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | Single-turn full correction (solves perfectly after one hint) or no change at all |
| 2 | Too rapid improvement; or completely static despite multiple hints |
| 3 | Some gradual improvement but with implausible jumps or stalls |
| 4 | Mostly gradual improvement; partial corrections visible |
| 5 | Clearly gradual; sub-errors persist; improvement is step-by-step across turns |

---

### SS5 — Instructional-Testing Utility

**Framework source**: Koedinger et al. (2015) — instructional-testing goal.  
**Definition**: Could this dialogue be used to test a specific tutor response or feedback policy? Does it provide a realistic testbed for evaluating tutoring strategies?

**5-point scale**:
| Score | Description |
|-------|-------------|
| 1 | Not usable as testbed (too unrealistic, too perfect, or incoherent) |
| 2 | Marginally usable; limited diagnostic value |
| 3 | Partially usable; some turns provide realistic feedback-testing opportunities |
| 4 | Mostly usable; would work as testbed for feedback policy experiments |
| 5 | Highly usable; realistic, stable, diagnosable; provides multiple testable moments |

---

## PART C — Use-Case Decision

After rating all 9 dimensions, make ONE use-case decision per dialogue.

**Options**:
- `learning_by_teaching` — This dialogue would work as a Teachable Agent learning-by-teaching environment (C1 target use)
- `diagnostic_simulation` — This dialogue would work for diagnosing student misconceptions (C2 target use)
- `feedback_policy_testing` — This dialogue would work for testing feedback policies (C2 secondary use)
- `reject` — This dialogue should be rejected for any of the above uses

**Reject if ANY of the following is present**:
- Severe premature expertise (solved correctly in turn 1–2)
- Clear role drift (learner became tutor)
- Logical inconsistency (learner contradicted themselves without acknowledgement)
- Unsupported reasoning (bare answers throughout)
- Misconception completely lost

---

## PART D — Failure Flags (Binary: 1 = present, 0 = absent)

| Flag | Definition | Trigger |
|------|-----------|---------|
| `premature_expertise` | Learner achieves correct answer before ≤2 substantive tutor turns | Score 1 if yes |
| `role_drift` | Learner adopts tutor/teacher language ("Let me explain…", "Remember that…") | Score 1 if ≥1 turn affected |
| `over_technical` | Learner uses beyond-7th-grade vocabulary (calculus, matrices, etc.) | Score 1 if ≥1 turn affected |
| `misconception_loss` | Target misconception disappears without tutor addressing it | Score 1 if yes |
| `logical_inconsistency` | Learner gives contradictory answers across turns without acknowledging | Score 1 if ≥1 pair of contradictory turns |
| `unsupported_reasoning` | Learner gives answers with no reasoning steps in ≥50% of turns | Score 1 if yes |

**LLM validity threat references**:
- premature_expertise, over_technical: Competence paradox (Yuan et al., 2026)
- role_drift: Role instability (Yuan et al., 2026; Mannekote et al., 2025)
- misconception_loss: Epistemic fidelity (Yuan et al., 2026)
- logical_inconsistency: Logical inconsistency (Yuan et al., 2026)
- unsupported_reasoning: Authenticity (Mannekote et al., 2025; BEA workshop)

---

## References

Blair, K., Schwartz, D.L., Biswas, G., & Leelawong, K. (2007). Pedagogical agents for learning by teaching: Teachable Agents. *Educational Technology*, 47(1), 56–61.

Koedinger, K.R., Matsuda, N., MacLellan, C.J., & McLaughlin, E.A. (2015). Methods for evaluating simulated learners: Examples from SimStudent. *AIED Workshop Proceedings*.

Yuan, Z., Xiao, Y., Li, M., Xuan, W., Tong, R., Diab, M., & Mitchell, T. (2026). Towards valid student simulation with large language models. *arXiv:2601.05473*.

Mannekote, A., et al. (2025). Can LLMs reliably simulate human learner actions? *LearnDialogue / AAAI Workshop*.
