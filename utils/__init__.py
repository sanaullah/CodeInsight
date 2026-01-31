"""Utility functions for LLM framework."""

from .langfuse_integration import (
    get_langfuse_client,
    configure_litellm_for_langfuse,
    create_langfuse_trace,
    flush_langfuse,
    validate_langfuse_connection,
    diagnose_langfuse_litellm_integration,
    trace_litellm_completion,
    atrace_litellm_completion
)

from .workflows_integration import (
    get_workflows_langfuse_callback,
    configure_workflows_for_langfuse,
    create_workflows_config
)

from .config.logging_config import (
    setup_logging,
    setup_logging_from_config
)

__all__ = [
    "get_langfuse_client",
    "configure_litellm_for_langfuse",
    "create_langfuse_trace",
    "flush_langfuse",
    "validate_langfuse_connection",
    "diagnose_langfuse_litellm_integration",
    "trace_litellm_completion",
    "atrace_litellm_completion",
    "get_workflows_langfuse_callback",
    "configure_workflows_for_langfuse",
    "create_workflows_config",
    "setup_logging",
    "setup_logging_from_config"
]

