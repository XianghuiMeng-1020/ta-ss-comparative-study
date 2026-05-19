# Datasheet for Dataset — v2
## IEEE TLT: Multi-Model, Multi-Question-Type, Multi-Dimension Naive-Student Simulation

Following Gebru et al. (2021) "Datasheets for Datasets" (CACM).

---

## Motivation

**For what purpose was the dataset created?**
To benchmark 12 LLMs as naive-student simulators across 5 question types, 3 difficulty
levels, and 7 evaluation dimensions — enabling the first multi-model, multi-dimension
comparison for teachable-agent research.

**Who created the dataset and on whose behalf?**
[Blinded for review]. Created for academic research.

**Who funded the creation?**
[Blinded for review].

---

## Composition

The dataset has six components:

### Component 1: Item Bank (1,500 items)

| Split | QType | Domain | N | Source |
|-------|-------|--------|---|--------|
| MCQ-Python | MCQ | Python programming | 300 | Jiajia et al. (2026) dataset subset |
| MCQ-Math | MCQ | Grade-7 mathematics | 300 | MathDial + LLM-assisted generation |
| TF | True-False | Mixed | 300 | Auto-derived from MCQ |
| Fill | Fill-in-blank | Mathematics | 300 | MathDial cloze deletion |
| SA | Short-Answer | Mathematics | 300 | MathDial open-response |
| OED | Open-Ended Dialogue | Mathematics | 100 scenarios | MathDial (reused) |

**IRT calibration**: 2PL IRT difficulty labels (Easy/Medium/Hard) calibrated on
Mistral-7B-Instruct-v0.3 P1 baseline responses. See `src/data/irt_calibration.py`.

**Location**: `data/item_bank/{mcq,tf,fill,sa,open}/items_with_difficulty.jsonl`

**Schema** (per item):
```json
{
  "item_id":       "MCQ_A1B2C3D4",
  "qtype":         "MCQ",
  "domain":        "python_programming",
  "concept":       "functions",
  "question":      "...",
  "options":       {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "B",
  "explanation":   "...",
  "difficulty_raw": null,
  "difficulty":    "Medium",
  "irt_a":         1.23,
  "irt_b":         0.12,
  "source":        "jiajia_python_mcq",
  "unlearning_eligible": true
}
```

### Component 2: P1 Responses (43,200 expected)

12 models × 4 QTypes × 3 difficulty × 10 items × 3 seeds.
Big-tier OED responses (4,800 dialogues) from prior study reused.

**Location**: `outputs/qa/main/{model_tag}/{qtype}/seed{N}/{item_id}.json`
**Aggregated CSV**: `outputs/qa_responses_{model_tag}.csv`

**Schema** (per response):
```json
{
  "item_id": "...", "qtype": "MCQ", "domain": "...", "concept": "...",
  "difficulty": "Medium", "question": "...", "correct_answer": "B",
  "persona": "P1", "model_id": "...", "model_tag": "...", "seed": 42,
  "persona_version": "v2.0", "demographic_context": "",
  "generated_at": "2026-06-01T00:00:00",
  "response": "...", "correct": 0, "grade_method": "pattern_match",
  "input_tokens": 120, "output_tokens": 85,
  "latency_ms": 234.5, "cost_usd": 0.00024, "error": null,
  "system_prompt": "..."
}
```

### Component 3: P2 Responses (18,000 expected)

2 models × 3 unlearning ratios × 5 seeds × 600 MCQ items.

**Location**: `outputs/qa/p2_eval/`
**Adapter logs**: `outputs/unlearning/{model_tag}/ratio{N}_seed{S}/training_metrics.json`

### Component 4: CEAT Demographic Variant Responses

GPT-4o × 16 demographic conditions × 600 MCQ items × 1 seed = 9,600 responses.

**Location**: `outputs/qa/ceat/`

### Component 5: Human Annotations (720 items)

- 240 OED dialogues (re-coded with augmented codebook HL1/HL2/HL3)
- 480 single-response items (HL1, HL2, HL3, EQ1)
- 2 coders, 100% overlap; ICC(2,k) computed per dimension.

**Location**: `outputs/human_annotations_v2.csv`
**Codebook**: `src/coding/codebook_v2.md` (to be drafted)
**ICC report**: `outputs/icc_report_v2.csv`

### Component 6: OED Dialogues (4,800; from prior study)

Reused as open-ended dialogue slice. See v1 DATASHEET.md for full description.

---

## Collection Process

**How was data collected?**
- Item bank: adapted from public datasets (CC-BY) and auto-derived via Python scripts.
- LLM responses: API calls (OpenAI, Anthropic) and local vLLM inference.
- Human annotations: trained coders recruited as research staff; paid $25–30/hr.

**Over what time period?**
Item bank: 2026-W3–W4. P1 generation: 2026-W5–W8. P2 training: 2026-W5–W9.
Human coding: 2026-W9–W11.

---

## Preprocessing / Cleaning

**Exclusion rules** (see analysis_plan.md v2.0 §6):
- E1: Empty response (< 5 chars) → exclude
- E2: Non-domain content → exclude
- E3: Unsafe/refusal → exclude, log
- E4: Token overflow → truncate; if answer lost → exclude
- E7: Tiny-model incoherence (< 3 real words/sentence) → exclude
- E8: Unlearning collapse (accuracy < 5% on retain set) → exclude ratio

**Stub items** (marked `source: "stub"` in items.jsonl): require manual review
before data collection begins. Run: `python src/data/build_item_bank.py --verify`

---

## Uses

**What tasks is the dataset intended to support?**
- Comparative evaluation of LLM naive-student simulation quality
- Benchmarking LLM human-likeness in educational settings
- Fairness auditing of student-side LLM generation
- Dialogue-test discrepancy analysis for knowledge tracing research

**Who are the intended dataset users?**
Educational AI researchers, ITS developers, dialogue-based knowledge tracing researchers.

**Is there anything that could affect data users?**
CEAT demographic variant responses involve racially/nationally coded prompts.
These should not be used to perpetuate stereotypes. They are included solely
for bias auditing under controlled conditions.

---

## Distribution

| Component | License | Repository |
|-----------|---------|-----------|
| Item bank | CC-BY-4.0 | GitHub + Zenodo |
| P1 responses | CC-BY-4.0 | Zenodo |
| P2 responses | CC-BY-4.0 | Zenodo |
| Human annotations | CC-BY-4.0 | GitHub + Zenodo |
| Unlearned LoRA adapters | Apache-2.0 | HuggingFace Hub |
| Source code | MIT | GitHub |

Zenodo DOI: [INSERT before submission]
HuggingFace Hub: [INSERT after training]

---

## Maintenance

Maintained by the corresponding author. Issues filed at GitHub repo.
Planned update: after peer-review (add reviewer-requested additional models/items).

---

*Datasheet v2.0 — 2026-05-20*
