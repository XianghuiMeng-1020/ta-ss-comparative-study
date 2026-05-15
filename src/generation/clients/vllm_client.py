"""
vLLM / Hugging Face Inference client for open-source models.

Supports:
  - HuggingFace Inference API (free tier or Pro)
  - Self-hosted vLLM server via OpenAI-compatible endpoint
  - HuggingFace Inference Endpoints

Set environment variables:
  HF_API_TOKEN   — for HF Inference API / Endpoints
  VLLM_BASE_URL  — for self-hosted vLLM (e.g. "http://localhost:8000/v1")
"""

from __future__ import annotations

import os
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import GenerationResult, LLMConfig

# Models routed via HuggingFace Inference API (serverless)
HF_SERVERLESS_MODELS = {
    "meta-llama/Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "deepseek-ai/DeepSeek-V3",
}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=120))
def call_vllm(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None = None,
) -> GenerationResult:
    """
    Route to: vLLM self-hosted (if VLLM_BASE_URL set), else HF Inference API.
    Uses the OpenAI-compatible /v1/chat/completions endpoint in both cases.
    """
    vllm_base = os.environ.get("VLLM_BASE_URL", "")
    hf_token = os.environ.get("HF_API_TOKEN", "")

    if vllm_base:
        return _call_openai_compat(messages, config, seed, base_url=vllm_base, api_key="EMPTY")
    elif hf_token:
        hf_base = "https://api-inference.huggingface.co/v1"
        return _call_openai_compat(messages, config, seed, base_url=hf_base, api_key=hf_token)
    else:
        raise EnvironmentError(
            "For open-source models, set either VLLM_BASE_URL (self-hosted) "
            "or HF_API_TOKEN (HuggingFace Inference API)."
        )


def _call_openai_compat(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None,
    base_url: str,
    api_key: str,
) -> GenerationResult:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
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
    input_tokens = resp.usage.prompt_tokens if resp.usage else 0
    output_tokens = resp.usage.completion_tokens if resp.usage else 0

    return GenerationResult(
        content=choice.message.content or "",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=config.model_id,
        finish_reason=choice.finish_reason or "",
        cost_usd=config.cost_estimate_usd(input_tokens, output_tokens),
    )
