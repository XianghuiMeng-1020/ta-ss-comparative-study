"""
Google Gemini client (gemini-2.5-pro family).
Uses the google-generativeai SDK.
"""

from __future__ import annotations

import os
import time

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import GenerationResult, LLMConfig


def _convert_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
    """Convert OpenAI-style messages to Gemini format.

    Returns (system_instruction, contents_list).
    """
    system_instruction = ""
    contents = []
    for m in messages:
        role = m["role"]
        text = m["content"]
        if role == "system":
            system_instruction = text
        elif role == "user":
            contents.append({"role": "user", "parts": [{"text": text}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": text}]})
    return system_instruction, contents


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=120))
def call_google(
    messages: list[dict[str, str]],
    config: LLMConfig,
    seed: int | None = None,
) -> GenerationResult:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
    system_instruction, contents = _convert_messages(messages)

    generation_config = {
        "temperature": config.temperature,
        "max_output_tokens": config.max_tokens,
    }
    if seed is not None:
        generation_config["seed"] = seed

    model_kwargs: dict = {"model_name": config.model_id, "generation_config": generation_config}
    if system_instruction:
        model_kwargs["system_instruction"] = system_instruction

    model = genai.GenerativeModel(**model_kwargs)

    t0 = time.time()
    response = model.generate_content(contents)
    latency_ms = (time.time() - t0) * 1000

    content = response.text if response.parts else ""
    usage = response.usage_metadata
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    return GenerationResult(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_id=config.model_id,
        finish_reason=str(response.candidates[0].finish_reason) if response.candidates else "",
        cost_usd=config.cost_estimate_usd(input_tokens, output_tokens),
    )
