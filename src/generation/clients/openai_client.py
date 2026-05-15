"""
OpenAI client (gpt-4o family).
"""

from __future__ import annotations

import os
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import GenerationResult, LLMConfig


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def call_openai(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None = None,
) -> GenerationResult:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    kwargs: dict[str, Any] = {
        "model": config.model_id,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency_ms = (time.time() - t0) * 1000

    choice = resp.choices[0]
    input_tokens = resp.usage.prompt_tokens
    output_tokens = resp.usage.completion_tokens

    return GenerationResult(
        content=choice.message.content or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=resp.model,
        finish_reason=choice.finish_reason or "",
        cost_usd=config.cost_estimate_usd(input_tokens, output_tokens),
    )
