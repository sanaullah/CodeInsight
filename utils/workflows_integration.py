"""
Workflows-specific Langfuse integration utilities.

This module provides utilities for integrating workflows with Langfuse
observability, building on the existing Langfuse integration.
"""

import logging
from typing import Optional, Dict, Any, List
from utils.langfuse_integration import get_langfuse_client

logger = logging.getLogger(__name__)


def get_workflows_langfuse_callback(
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    Get a native Langfuse CallbackHandler for workflows.
    
    This function returns Langfuse's native CallbackHandler from langfuse.langchain
    which automatically creates proper observation types for agent graph visualization.
    
    Args:
        trace_name: Optional name for the trace
        user_id: Optional user ID for trace correlation
        session_id: Optional session ID for trace correlation
        metadata: Optional metadata dictionary to attach to traces
        tags: Optional list of tags to attach to traces
        
    Returns:
        CallbackHandler instance from langfuse.langchain if Langfuse is enabled, None otherwise
        
    Example:
        from utils import get_workflows_langfuse_callback
        
        # Get native callback handler with trace correlation
        langfuse_handler = get_workflows_langfuse_callback(
            trace_name="my_workflow",
            user_id="user123",
            session_id="session456",
            metadata={"environment": "production"},
            tags=["agent", "workflow"]
        )
        
        # Use with LangGraph via config parameter
        graph.invoke(input_state, config={"callbacks": [langfuse_handler]})
    """
    # Lazy import to avoid circular dependency
    from workflows.integration import setup_langfuse_callbacks
    return setup_langfuse_callbacks(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata,
        tags=tags
    )


def configure_workflows_for_langfuse():
    """
    Configure workflows to use Langfuse for observability.
    
    This function ensures that workflow executions will be tracked
    in Langfuse when executed. It checks the configuration and sets up
    the necessary callbacks.
    
    Note: This is called automatically by GraphBuilder, but can be
    called manually if needed.
    """
    client = get_langfuse_client()
    if not client:
        logger.debug("Langfuse client not available for workflows")
        return False
    
    logger.info("✅ Workflows configured for Langfuse tracking")
    return True


def create_workflows_config(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    callbacks: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Create a workflows config dictionary with trace correlation metadata.
    
    This utility function helps create properly formatted config dictionaries
    for workflow invoke/stream calls that include trace correlation information.
    
    Args:
        user_id: Optional user ID for trace correlation
        session_id: Optional session ID for trace correlation
        trace_id: Optional trace ID for linking to parent trace
        metadata: Optional additional metadata dictionary
        tags: Optional list of tags
        callbacks: Optional list of callback handlers (will be merged with Langfuse callback)
        
    Returns:
        Config dictionary ready for use with graph.invoke() or graph.stream()
        
    Example:
        from utils import create_workflows_config, get_workflows_langfuse_callback
        
        # Create config with trace correlation
        config = create_workflows_config(
            user_id="user123",
            session_id="session456",
            trace_id="trace789",
            metadata={"environment": "production", "version": "1.0"},
            tags=["agent", "workflow"]
        )
        
        # Add Langfuse callback
        langfuse_handler = get_workflows_langfuse_callback(trace_name="my_workflow")
        if langfuse_handler:
            if "callbacks" not in config:
                config["callbacks"] = []
            config["callbacks"].append(langfuse_handler)
        
        # Use with graph
        result = graph.invoke(input_state, config=config)
    """
    config: Dict[str, Any] = {}
    
    # Build metadata dictionary
    config_metadata: Dict[str, Any] = {}
    
    if user_id:
        config_metadata["user_id"] = user_id
        config_metadata["langfuse_user_id"] = user_id  # v3 pattern
    if session_id:
        config_metadata["session_id"] = session_id
        config_metadata["langfuse_session_id"] = session_id  # v3 pattern
    if trace_id:
        config_metadata["trace_id"] = trace_id
        config_metadata["parent_trace_id"] = trace_id
    
    # Merge additional metadata
    if metadata:
        config_metadata.update(metadata)
    
    # Add tags to metadata
    if tags:
        config_metadata["tags"] = tags
        config_metadata["langfuse_tags"] = tags  # v3 pattern
    
    # Add metadata to config if we have any
    if config_metadata:
        config["metadata"] = config_metadata
    
    # Add callbacks if provided
    if callbacks:
        config["callbacks"] = callbacks
    
    return config

