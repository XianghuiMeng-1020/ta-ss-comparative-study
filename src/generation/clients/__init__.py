"""
Multi-model LLM client dispatcher.

Supported backends:
  openai    — gpt-4o family
  anthropic — claude family
  google    — gemini family
  vllm      — open-source models via HF Inference API or self-hosted vLLM
"""

from .base import GenerationResult, LLMConfig
from .openai_client import call_openai
from .anthropic_client import call_anthropic
from .google_client import call_google
from .vllm_client import call_vllm

# Model ID → backend routing table
MODEL_BACKEND_MAP: dict[str, str] = {
    "gpt-4o-2024-11-20":                 "openai",
    "gpt-4o-mini":                        "openai",
    "gpt-4o":                             "openai",
    "gpt-5":                              "openai",
    "claude-sonnet-4-5-20250929":         "anthropic",
    "claude-opus-4":                      "anthropic",
    "claude-3-5-sonnet-20241022":         "anthropic",
    "gemini-2.5-pro":                     "google",
    "gemini-2.0-flash":                   "google",
    "meta-llama/Llama-3.1-70B-Instruct":  "vllm",
    "Qwen/Qwen2.5-72B-Instruct":          "vllm",
    "deepseek-ai/DeepSeek-V3":            "vllm",
}


def infer_backend(model_id: str) -> str:
    """Infer backend from model_id; fall back to explicit config.api_backend."""
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
    elif backend == "anthropic":
        return call_anthropic(messages, config, seed)
    elif backend == "google":
        return call_google(messages, config, seed)
    elif backend == "vllm":
        return call_vllm(messages, config, seed)
    else:
        raise ValueError(
            f"Unknown api_backend: {backend!r}. "
            f"Valid: openai, anthropic, google, vllm, auto"
        )


__all__ = [
    "LLMConfig",
    "GenerationResult",
    "call_llm",
    "call_openai",
    "call_anthropic",
    "call_google",
    "call_vllm",
    "infer_backend",
    "MODEL_BACKEND_MAP",
]
