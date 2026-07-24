"""Native durable workflow primitives with lazy legacy compatibility."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .task_scheduler import NativeTaskScheduler, TaskContext, TaskResult

_LEGACY_EXPORTS = {
    "GraphBuilder": ("workflow.graph_builder", "GraphBuilder"),
    "create_graph": ("workflow.graph_builder", "create_graph"),
    "StateSchema": ("workflow.state", "StateSchema"),
    "create_state_schema": ("workflow.state", "create_state_schema"),
    "llm_node": ("workflow.nodes", "llm_node"),
    "tool_node": ("workflow.nodes", "tool_node"),
    "conditional_node": ("workflow.nodes", "conditional_node"),
    "get_checkpoint_adapter": ("workflow.checkpoints", "get_checkpoint_adapter"),
    "setup_checkpoints": ("workflow.checkpoints", "setup_checkpoints"),
    "stream_graph": ("workflow.streaming", "stream_graph"),
    "format_stream_event": ("workflow.streaming", "format_stream_event"),
    "human_approval_node": ("workflow.human_in_loop", "human_approval_node"),
    "human_feedback_node": ("workflow.human_in_loop", "human_feedback_node"),
    "setup_langfuse_callbacks": ("workflow.integration", "setup_langfuse_callbacks"),
}

__all__ = ["NativeTaskScheduler", "TaskContext", "TaskResult", *_LEGACY_EXPORTS]


def __getattr__(name: str) -> Any:
    """Load legacy LangGraph exports only when an old caller requests one."""

    target = _LEGACY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
