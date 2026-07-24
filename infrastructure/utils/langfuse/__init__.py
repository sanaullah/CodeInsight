"""
Langfuse integration package for LLM observability.

This package provides modular Langfuse integration utilities:
- Client management and initialization
- LiteLLM integration (OTEL and native SDK)
- Connection utilities
- Diagnostic functions
- Tracing and observation creation
- Prompt management
- Evaluation and scoring
"""

# Client management
from .client import get_langfuse_client

# LiteLLM integration
from .litellm_config import configure_litellm_for_langfuse

# Connection utilities
from .connection import flush_langfuse, validate_langfuse_connection

# Diagnostics
from .diagnostics import diagnose_langfuse_litellm_integration

# Tracing and observations
from .tracing import (
    trace_litellm_completion,
    atrace_litellm_completion,
    create_observation,
    create_llm_observation,
    create_langfuse_trace
)

# Prompt management
from .prompts import (
    create_langfuse_prompt,
    get_langfuse_prompt_by_name,
    get_langfuse_prompt_by_metadata,
    normalize_label
)

# Evaluation and scoring
from .evaluation import (
    score_trace,
    evaluate_prompt_effectiveness
)

# Metadata schema
from .metadata_schema import (
    SwarmAnalysisMetadata,
    normalize_metadata
)

# Cost tracking
from .cost_tracking import (
    get_model_pricing,
    calculate_token_cost,
    aggregate_costs_from_traces,
    aggregate_costs_from_state
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
    "get_langfuse_prompt_by_metadata",
    "normalize_label",
    # Evaluation
    "score_trace",
    "evaluate_prompt_effectiveness",
    # Metadata schema
    "SwarmAnalysisMetadata",
    "normalize_metadata",
    # Cost tracking
    "get_model_pricing",
    "calculate_token_cost",
    "aggregate_costs_from_traces",
    "aggregate_costs_from_state",
]

