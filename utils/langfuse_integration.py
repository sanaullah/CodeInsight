"""
Langfuse integration for LLM observability.

This module is maintained for backward compatibility.
All functionality has been refactored into the utils.langfuse package.

For new code, prefer importing directly from utils.langfuse:
    from utils.langfuse import get_langfuse_client, create_langfuse_trace

This module re-exports all functions from utils.langfuse for backward compatibility.
"""

# Re-export all public functions from the langfuse package
from .langfuse import (
    # Client management
    get_langfuse_client,
    # LiteLLM integration
    configure_litellm_for_langfuse,
    # Connection utilities
    flush_langfuse,
    validate_langfuse_connection,
    # Diagnostics
    diagnose_langfuse_litellm_integration,
    # Tracing and observations
    trace_litellm_completion,
    atrace_litellm_completion,
    create_observation,
    create_llm_observation,
    create_langfuse_trace,
    # Prompt management
    create_langfuse_prompt,
    get_langfuse_prompt_by_name,
    normalize_label,
    # Evaluation and scoring
    score_trace,
    evaluate_prompt_effectiveness,
)

__all__ = [
    # Client
    "get_langfuse_client",
    # LiteLLM
    "configure_litellm_for_langfuse",
    # Connection
    "flush_langfuse",
    "validate_langfuse_connection",
    # Diagnostics
    "diagnose_langfuse_litellm_integration",
    # Tracing
    "trace_litellm_completion",
    "atrace_litellm_completion",
    "create_observation",
    "create_llm_observation",
    "create_langfuse_trace",
    # Prompts
    "create_langfuse_prompt",
    "get_langfuse_prompt_by_name",
    "normalize_label",
    # Evaluation
    "score_trace",
    "evaluate_prompt_effectiveness",
]
