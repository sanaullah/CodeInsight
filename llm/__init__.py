"""
LLM Framework - LiteLLM + Langfuse Integration

A minimal framework for using LiteLLM with Langfuse observability.
"""

from .config import ConfigManager, ensure_env_ready

__version__ = "2.0.0"
__all__ = [
    "ConfigManager",
    "ensure_env_ready",
]

