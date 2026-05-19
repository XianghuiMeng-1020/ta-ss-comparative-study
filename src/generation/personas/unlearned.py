"""
P2 — Machine-unlearning-based naive student persona.

This module wraps inference with an unlearned LoRA adapter, producing naive-student
responses without any explicit persona prompt. The model's weights have been modified
via the LoRA + KL forgetting procedure (src/unlearning/unlearn.py) to suppress specific
knowledge components, so the prompt need only instruct the model to attempt the question.

Usage (after unlearning is complete):
    from src.generation.personas.unlearned import UnlearnedStudentClient
    client = UnlearnedStudentClient.from_adapter(
        base_model="mistralai/Mistral-7B-Instruct-v0.3",
        adapter_path="outputs/unlearning/mistral_7b/ratio30_seed42",
    )
    response = client.respond(item)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PERSONA_VERSION = "v2.0"

MINIMAL_ATTEMPT_PROMPT = """\
Please attempt the following question. Show your reasoning, even if you are unsure.

{question}"""


@dataclass
class UnlearnedStudentClient:
    """
    Thin wrapper around a vLLM-served unlearned adapter.
    Delegates actual inference to src.generation.clients.vllm_client.
    """
    base_model: str
    adapter_path: str
    unlearning_ratio: float
    seed: int
    temperature: float = 0.7
    max_tokens: int = 400

    @classmethod
    def from_adapter(
        cls,
        base_model: str,
        adapter_path: str,
        unlearning_ratio: float = 0.3,
        seed: int = 42,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> "UnlearnedStudentClient":
        if not Path(adapter_path).exists():
            raise FileNotFoundError(
                f"Adapter path not found: {adapter_path}. "
                f"Run src/unlearning/train_runner.py first."
            )
        return cls(
            base_model=base_model,
            adapter_path=adapter_path,
            unlearning_ratio=unlearning_ratio,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def respond(self, item: dict, qtype: str = "MCQ") -> dict:
        """
        Generate a naive-student response for a single item.
        Returns a dict compatible with qa_runner.py result format.
        """
        from src.generation.personas.prompt_based import _format_question

        question_text = _format_question(item, qtype)
        prompt = MINIMAL_ATTEMPT_PROMPT.format(question=question_text)

        try:
            from src.generation.clients.vllm_client import call_vllm_with_adapter
            t0 = time.time()
            result = call_vllm_with_adapter(
                prompt=prompt,
                adapter_path=self.adapter_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            latency_ms = (time.time() - t0) * 1000
            content = result.content
        except ImportError:
            content = "[P2 inference requires vLLM with LoRA adapter support]"
            latency_ms = 0.0

        return {
            "persona": "P2",
            "model_id": self.base_model,
            "adapter_path": self.adapter_path,
            "unlearning_ratio": self.unlearning_ratio,
            "seed": self.seed,
            "item_id": item.get("item_id", ""),
            "qtype": qtype,
            "response": content,
            "latency_ms": latency_ms,
            "prompt": prompt,
        }

    @staticmethod
    def get_persona_id() -> str:
        return "P2"

    @staticmethod
    def get_persona_label() -> str:
        return "Unlearning-based Naive Student"
