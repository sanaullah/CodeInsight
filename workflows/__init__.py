"""Workflows package for LangGraph-based workflows."""

"""
Workflows module for building stateful, multi-step agent workflows.

This module provides utilities for creating LangGraph workflows with integration
to LiteLLM and Langfuse observability.
"""

from .graph_builder import GraphBuilder, create_graph
from .state import StateSchema, create_state_schema
from .nodes import llm_node, tool_node, conditional_node
from .checkpoints import get_checkpoint_adapter, setup_checkpoints
from .streaming import stream_graph, format_stream_event
from .human_in_loop import human_approval_node, human_feedback_node
from .integration import setup_langfuse_callbacks, LangGraphLangfuseCallback  # LangGraphLangfuseCallback is deprecated

__version__ = "1.0.0"
__all__ = [
    "GraphBuilder",
    "create_graph",
    "StateSchema",
    "create_state_schema",
    "llm_node",
    "tool_node",
    "conditional_node",
    "get_checkpoint_adapter",
    "setup_checkpoints",
    "stream_graph",
    "format_stream_event",
    "human_approval_node",
    "human_feedback_node",
    "setup_langfuse_callbacks",
    "LangGraphLangfuseCallback",  # Deprecated - use setup_langfuse_callbacks() which returns native CallbackHandler
]

