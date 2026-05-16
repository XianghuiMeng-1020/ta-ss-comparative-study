"""
Multi-model LLM client dispatcher.

Supported backends:
  openai      — gpt-4o family (direct API)
  openrouter  — Claude / Gemini / Llama / etc. via OpenRouter (OpenAI-compatible)
  anthropic   — claude family (direct Anthropic API, optional)
  google      — gemini family (direct Google API, optional)
  vllm        — open-source models via HF Inference API or self-hosted vLLM
"""

from .base import GenerationResult, LLMConfig
from .openai_client import call_openai
from .openrouter_client import call_openrouter
from .anthropic_client import call_anthropic
from .google_client import call_google
from .vllm_client import call_vllm

# Model ID → backend routing table
# M2-M4 route through OpenRouter by default (single key, OpenAI-compatible)
MODEL_BACKEND_MAP: dict[str, str] = {
    # M1 — direct OpenAI
    "gpt-4o-2024-11-20":                 "openai",
    "gpt-4o-mini":                        "openai",
    "gpt-4o":                             "openai",
    "gpt-5":                              "openai",
    # M2 — Claude via OpenRouter
    "claude-sonnet-4-5-20250929":         "openrouter",
    "claude-opus-4":                      "openrouter",
    "claude-3-5-sonnet-20241022":         "openrouter",
    # M3 — Gemini via OpenRouter
    "gemini-2.5-pro":                     "openrouter",
    "gemini-2.0-flash":                   "openrouter",
    # M4 — Llama/open-source via OpenRouter
    "meta-llama/Llama-3.1-70B-Instruct":  "openrouter",
    "Qwen/Qwen2.5-72B-Instruct":          "openrouter",
    "deepseek-ai/DeepSeek-V3":            "openrouter",
}


def infer_backend(model_id: str) -> str:
    """Infer backend from model_id; fall back to openai for unknown models."""
    return MODEL_BACKEND_MAP.get(model_id, "openai")


def call_llm(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None = None,
) -> GenerationResult:
    """
    Unified dispatcher: route to the correct backend based on config.api_backend
    (or inferred from model_id if api_backend == 'auto').
    """
    backend = config.api_backend
    if backend == "auto":
        backend = infer_backend(config.model_id)

    if backend == "openai":
        return call_openai(messages, config, seed)
    elif backend == "openrouter":
        return call_openrouter(messages, config, seed)
    elif backend == "anthropic":
        return call_anthropic(messages, config, seed)
    elif backend == "google":
        return call_google(messages, config, seed)
    elif backend == "vllm":
        return call_vllm(messages, config, seed)
    else:
        raise ValueError(
            f"Unknown api_backend: {backend!r}. "
            f"Valid: openai, openrouter, anthropic, google, vllm, auto"
        )


__all__ = [
    "LLMConfig",
    "GenerationResult",
    "call_llm",
    "call_openai",
    "call_openrouter",
    "call_anthropic",
    "call_google",
    "call_vllm",
    "infer_backend",
    "MODEL_BACKEND_MAP",
]
