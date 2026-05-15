"""
Legacy compatibility shim — redirects to src.generation.clients multi-model dispatcher.

New code should import directly from src.generation.clients.
This module is kept to avoid breaking any scripts that import from src.generation.client.
"""

from src.generation.clients import (  # noqa: F401
    LLMConfig,
    GenerationResult,
    call_llm,
)
