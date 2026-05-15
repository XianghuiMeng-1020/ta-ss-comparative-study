"""
Anthropic client (claude family).
"""

from __future__ import annotations

import os
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import GenerationResult, LLMConfig


def _split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Extract system message; Anthropic API takes it separately."""
    system_content = ""
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            filtered.append(m)
    return system_content, filtered


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def call_anthropic(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None = None,
) -> GenerationResult:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    system_content, filtered = _split_system(messages)

    kwargs: dict[str, Any] = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "messages": filtered,
        "temperature": config.temperature,
    }
    if system_content:
        kwargs["system"] = system_content

    t0 = time.time()
    resp = client.messages.create(**kwargs)
    latency_ms = (time.time() - t0) * 1000

    content = resp.content[0].text if resp.content else ""
    input_tokens = resp.usage.input_tokens
    output_tokens = resp.usage.output_tokens

    return GenerationResult(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=resp.model,
        finish_reason=resp.stop_reason or "",
        cost_usd=config.cost_estimate_usd(input_tokens, output_tokens),
    )
