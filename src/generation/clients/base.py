"""
Base types shared across all LLM client implementations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMConfig:
    model_id: str = "gpt-4o-2024-11-20"
    temperature: float = 0.7
    max_tokens: int = 600
    api_backend: str = "openai"  # "openai" | "anthropic" | "google" | "vllm"

    def cost_estimate_usd(self, input_tokens: int, output_tokens: int) -> float:
        """Rough cost estimate; override per-model if needed."""
        rates = {
            "gpt-4o-2024-11-20":          (2.50, 10.00),  # $/M tokens in/out
            "claude-sonnet-4-5-20250929":  (3.00, 15.00),
            "gemini-2.5-pro":              (1.25,  5.00),
            "meta-llama/Llama-3.1-70B-Instruct": (0.59, 0.79),
            # Qwen fallback
            "Qwen/Qwen2.5-72B-Instruct":   (0.40, 0.40),
        }
        in_rate, out_rate = rates.get(self.model_id, (1.00, 3.00))
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


@dataclass
class GenerationResult:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    finish_reason: str
    cost_usd: float = 0.0
