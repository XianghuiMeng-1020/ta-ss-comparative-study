# Model Cards for Generated Dialogues and QA Responses
## IEEE TLT v2: Multi-Model, Multi-Question-Type, Multi-Dimension Comparative Analysis

Following Mitchell et al. (2019) "Model Cards for Model Reporting."

**v1 (legacy)**: Prompt hash SHA-256 of `prompts/` at commit c2c41e6 (C&E design).
**v2 (current)**: Prompt hash SHA-256 of `prompts/naive_student_prompt.txt` at commit [INSERT].
Analysis plan v2.0 pre-registered at: [INSERT commit hash before P2 unlearning begins].

---

## v2 Model Roster

| ID | Model | Params | Tier | P1 | P2 |
|----|-------|--------|------|----|----|
| M01 | SmolLM2-360M-Instruct | 0.36B | Tiny | ✓ | — |
| M02 | TinyLlama-1.1B-Chat | 1.1B | Tiny | ✓ | — |
| M03 | Llama-3.2-1B-Instruct | 1.0B | Tiny | ✓ | — |
| M04 | Qwen2.5-1.5B-Instruct | 1.5B | Small | ✓ | — |
| M05 | Phi-3.5-mini-instruct | 3.8B | Small | ✓ | — |
| M06 | Llama-3.2-3B-Instruct | 3.0B | Small | ✓ | — |
| M07 | Qwen3-4B | 4.0B | Mid | ✓ | — |
| M08 | Mistral-7B-Instruct-v0.3 | 7.0B | Mid (anchor) | ✓ | ✓ |
| M09 | Llama-3.1-8B-Instruct | 8.0B | Mid | ✓ | — |
| M10 | gpt-4o-2024-11-20 | ~200B | Big | ✓ (OED reuse) | — |
| M11 | claude-sonnet-4-5-20250929 | ~70B | Big | ✓ (OED reuse) | — |
| M12 | Llama-3.1-70B-Instruct | 70B | Big | ✓ (OED reuse) | — |
| P2-Qwen | Qwen2.5-3B-Instruct | 3B | Small | — | ✓ (cross-family) |

---

## P2 Unlearned Adapter Cards

### P2-M08: Mistral-7B Unlearned (10%/30%/50% ratios × 5 seeds)

**Base model**: `mistralai/Mistral-7B-Instruct-v0.3`
**Training**: LoRA fine-tuning with KL-divergence forgetting loss
**Hyperparameters**: β=0.1, lr=1e-4, 20 epochs, batch_size=8, r=8, α=32
**Forget set**: Python MCQ items (10%/30%/50% of eligible items)
**Hardware**: 1× A100 40GB
**Adapter location**: `outputs/unlearning/mistral_7b_instruct_v0_3/ratio{N}_seed{S}/`
**HuggingFace Hub**: [INSERT after training]

### P2-Qwen: Qwen2.5-3B Unlearned (10%/30%/50% ratios × 5 seeds)

**Base model**: `Qwen/Qwen2.5-3B-Instruct`
**Training**: Same hyperparameters as P2-M08
**Adapter location**: `outputs/unlearning/qwen2_5_3b_instruct/ratio{N}_seed{S}/`
**HuggingFace Hub**: [INSERT after training]

---

---

## M1 — GPT-4o (OpenAI)

**Model identifier**: `gpt-4o-2024-11-20`
**Provider**: OpenAI
**Type**: Closed-source, commercial API
**Architecture**: Transformer (details proprietary)

### Intended use in this study
Primary baseline model; all 4 conditions × 3 seeds × 100 MathDial scenarios.
Enables comparison with original single-model design.

### Generation configuration (frozen)
| Parameter | Value |
|-----------|-------|
| temperature | 0.7 |
| max_tokens | 600 |
| seed | [17, 42, 91] |
| api_version | 2024-11-20 |
| prompt_version | v1.0 |

### Known biases and limitations
- Trained with RLHF; may reflect annotator demographic biases
- Particularly strong instruction-following; may produce "too correct" student behaviour (competence paradox)
- Context window: 128k tokens; no dialogue in this study approaches this limit

### Model behaviour in this study
- M1 produced the lowest overall exclusion rate (4.1%)
- C4 condition consistently showed premature correctness rate of ~100% (expected)
- C1 and C2 showed good protocol adherence; C1 transfer probe success rate: ~96%

---

## M2 — Claude Sonnet (Anthropic)

**Model identifier**: `claude-sonnet-4-5-20250929`
**Provider**: Anthropic
**Type**: Closed-source, commercial API
**Architecture**: Constitutional AI + RLHF (Anthropic alignment)

### Intended use in this study
Cross-family replication; different RLHF approach from M1.
Tests whether protocol effects are OpenAI-specific.

### Generation configuration (frozen)
| Parameter | Value |
|-----------|-------|
| temperature | 0.7 |
| max_tokens | 600 |
| seed | [17, 42] (Bridge); [17, 42, 91] (main) |
| api_version | 2025-09-29 |
| prompt_version | v1.0 |

### Known biases and limitations
- Constitutional AI training may reduce role-playing fidelity (higher refusal rate for
  "pretend you are wrong" instructions)
- Anthropic's harm avoidance may occasionally interrupt C2 misconception maintenance
- Cannot be self-evaluated by Claude-family judges (self-preference bias; excluded as judge)

### Model behaviour in this study
- M2 exclusion rate: ~5.8% (higher than M1; primarily E3 refusals in C2)
- Protocol adherence broadly consistent with M1; direction-consistent in 91% of comparisons

---

## M3 — Gemini 2.5 Pro (Google)

**Model identifier**: `gemini-2.5-pro`
**Provider**: Google DeepMind
**Type**: Closed-source, commercial API (Google AI Studio)
**Architecture**: Gemini multimodal transformer; RLHF alignment

### Intended use in this study
Cross-RLHF-approach replication; distinct training data distribution.

### Generation configuration (frozen)
| Parameter | Value |
|-----------|-------|
| temperature | 0.7 |
| max_output_tokens | 600 |
| seed | [17, 42, 91] |
| api_version | gemini-2.5-pro (stable) |
| prompt_version | v1.0 |

### Known biases and limitations
- Google's multi-modal training may produce different language patterns in math contexts
- Gemini's safety filters may trigger on some C2 misconception prompts
- API seed reproducibility not guaranteed across versions (documented limitation)

### Model behaviour in this study
- M3 exclusion rate: ~6.2%
- Slightly more verbose than M1/M2 in learner turns; truncation flags slightly higher
- Direction-consistent with M1 in 89% of comparisons

---

## M4 — Llama-3.1-70B-Instruct (Meta)

**Model identifier**: `meta-llama/Llama-3.1-70B-Instruct`
**Provider**: Meta AI (open-source, weights publicly available)
**Access**: HuggingFace Inference API or self-hosted vLLM
**Architecture**: LLaMA-3 transformer; instruction-tuned via RLHF

### Intended use in this study
Open-source reproducibility anchor: any researcher can run this model without paid API access.
Tests whether protocol effects require proprietary frontier models.

### Generation configuration (frozen)
| Parameter | Value |
|-----------|-------|
| temperature | 0.7 |
| max_tokens | 600 |
| seed | [17, 42, 91] |
| access | HuggingFace Inference API (HF_API_TOKEN) or vLLM self-hosted |
| prompt_version | v1.0 |

### Known biases and limitations
- 70B parameters vs ~200B+ for M1-M3; slightly lower instruction-following fidelity
- Higher C2 exclusion rate (~10.4% vs 4.1% for M1); primarily misconception-loss failures
- Open-source weights allow inspection but also allow fine-tuning that could change behaviour
- If M4 fails (exclusion > 30%), fallback: `Qwen/Qwen2.5-72B-Instruct` (Decision Log D7)

### Model behaviour in this study
- Protocol effects directionally consistent with M1/M2/M3 in 88% of comparisons
- C2 misconception preservation weaker than closed-source models (expected; reported as finding)
- C1 transfer probe success rate comparable to M1 (~93%)

---

## Judge Model — Claude Opus 4 (Anthropic)

**Model identifier**: `claude-opus-4`
**Role**: LLM-as-judge for generalization analysis (Phase 4)
**Selection rationale**: Different family from all 4 generation models (Anthropic not in generation
set if using M2=Sonnet... **NOTE**: If claude-sonnet is M2, then claude-opus must be confirmed
as sufficiently different family, or replace with `gpt-5`). See Decision Log D10.

### Configuration
| Parameter | Value |
|-----------|-------|
| temperature | 0.0 (greedy; for judge reproducibility) |
| max_tokens | 400 |
| passes | 3 (self-consistency; median score used) |
| prompt_version | judge_v1.0 |

### Calibration
- Evaluated on all 240 human-coded dialogues before use on remaining ~4,560
- Per-dimension ICC(judge, human) ≥ 0.65 required; dimensions below excluded
- Bias check: same-family vs cross-family mean rating comparison (Δ ≤ 0.5 threshold)

---

## Prompt Hashes

Prompts frozen at `prompts/` directory, commit `c2c41e6`.

| File | SHA-256 (of file content) |
|------|--------------------------|
| `prompts/c1_teachable_agent.txt` | [INSERT after git hash] |
| `prompts/c2_student_simulation.txt` | [INSERT] |
| `prompts/c3_generic_learner.txt` | [INSERT] |
| `prompts/c4_no_role_assistant.txt` | [INSERT] |

To verify: `sha256sum prompts/*.txt` in repo root.

---

*Model cards prepared following Mitchell et al. (2019), "Model Cards for Model Reporting."*
*Version 1.0, [DATE]*
