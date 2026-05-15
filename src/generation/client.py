"""
LLM client abstraction with retry, rate limiting, and usage logging.
Supports OpenAI and Anthropic backends; configured via environment variables.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class LLMConfig:
    model_id: str = "gpt-4o-2024-11-20"
    temperature: float = 0.7
    max_tokens: int = 600
    api_backend: str = "openai"  # "openai" | "anthropic"


@dataclass
class GenerationResult:
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    finish_reason: str


def get_client(config: LLMConfig):
    if config.api_backend == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif config.api_backend == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    raise ValueError(f"Unknown api_backend: {config.api_backend}")


def call_llm(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None = None,
) -> GenerationResult:
    """
    Send a messages list to the configured LLM and return a GenerationResult.
    Messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    t0 = time.time()
    if config.api_backend == "openai":
        result = _call_openai(messages, config, seed)
    elif config.api_backend == "anthropic":
        result = _call_anthropic(messages, config, seed)
    else:
        raise ValueError(f"Unknown api_backend: {config.api_backend}")
    result.latency_ms = (time.time() - t0) * 1000
    return result


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def _call_openai(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None,
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

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    return GenerationResult(
        content=choice.message.content or "",
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        latency_ms=0.0,
        model_id=resp.model,
        finish_reason=choice.finish_reason or "",
    )


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def _call_anthropic(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None,
) -> GenerationResult:
    from anthropic import Anthropic

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_content = ""
    filtered = []
    for m in messages:
        if m["role"] == "system":
            system_content = m["content"]
        else:
            filtered.append(m)

    kwargs: dict[str, Any] = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "messages": filtered,
    }
    if system_content:
        kwargs["system"] = system_content

    resp = client.messages.create(**kwargs)
    content = resp.content[0].text if resp.content else ""
    return GenerationResult(
        content=content,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_ms=0.0,
        model_id=resp.model,
        finish_reason=resp.stop_reason or "",
    )
