"""
OpenRouter client — OpenAI-compatible endpoint for M2 (Claude), M3 (Gemini), M4 (Llama).

OpenRouter uses the same SDK as OpenAI but with a different base_url and api_key.
Internal model IDs are mapped to OpenRouter model IDs via OPENROUTER_MODEL_MAP.
"""

from __future__ import annotations

import os
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import GenerationResult, LLMConfig

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Internal model ID → OpenRouter model ID
OPENROUTER_MODEL_MAP: dict[str, str] = {
    "claude-sonnet-4-5-20250929":           "anthropic/claude-sonnet-4-5",
    "claude-3-5-sonnet-20241022":           "anthropic/claude-3.5-sonnet",
    "claude-opus-4":                         "anthropic/claude-opus-4",
    "gemini-2.5-pro":                        "google/gemini-2.0-flash-001",  # 2.5-pro is thinking-only; use Flash for dialogue gen
    "gemini-2.0-flash":                      "google/gemini-2.0-flash-001",
    "meta-llama/Llama-3.1-70B-Instruct":    "meta-llama/llama-3.1-70b-instruct",
    "Qwen/Qwen2.5-72B-Instruct":            "qwen/qwen-2.5-72b-instruct",
    "deepseek-ai/DeepSeek-V3":              "deepseek/deepseek-chat",
}

# OpenRouter pricing ($/M tokens in/out) — approximate, updated 2026-05
OPENROUTER_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-sonnet-4-5":          (3.00, 15.00),
    "anthropic/claude-3.5-sonnet":          (3.00, 15.00),
    "anthropic/claude-opus-4":              (15.00, 75.00),
    "google/gemini-2.5-pro-preview":        (1.25, 10.00),
    "google/gemini-2.0-flash-001":          (0.10,  0.40),
    "meta-llama/llama-3.1-70b-instruct":   (0.59,  0.79),
    "qwen/qwen-2.5-72b-instruct":           (0.40,  0.40),
    "deepseek/deepseek-chat":               (0.27,  1.10),
}


def resolve_openrouter_model(internal_model_id: str) -> str:
    """Map internal model ID to OpenRouter model ID."""
    return OPENROUTER_MODEL_MAP.get(internal_model_id, internal_model_id)


def openrouter_cost(openrouter_model_id: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = OPENROUTER_PRICING.get(openrouter_model_id, (1.00, 3.00))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def call_openrouter(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None = None,
) -> GenerationResult:
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/XianghuiMeng-1020/ta-ss-comparative-study",
            "X-Title": "TA vs SS Comparative Study",
        },
    )

    or_model_id = resolve_openrouter_model(config.model_id)

    kwargs: dict[str, Any] = {
        "model": or_model_id,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    # seed not universally supported on OpenRouter, pass only for OpenAI-family
    if seed is not None and "openai" in or_model_id:
        kwargs["seed"] = seed

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = (time.time() - t0) * 1000

    choice = resp.choices[0]
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0

    return GenerationResult(
        content=choice.message.content or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=or_model_id,
        finish_reason=choice.finish_reason or "",
        cost_usd=openrouter_cost(or_model_id, input_tokens, output_tokens),
    )
