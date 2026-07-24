"""
Streaming utilities for LangGraph execution.

This module provides utilities for streaming graph execution
and handling streaming events.
"""

from typing import AsyncIterator, Dict, Any, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


async def stream_graph(
    graph: Any,
    input_state: Dict[str, Any],
    stream_mode: str = "values",
    config: Optional[Dict[str, Any]] = None
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream graph execution events.
    
    Args:
        graph: Compiled LangGraph instance
        input_state: Initial state for the graph
        stream_mode: Stream mode ("values", "updates", or "debug")
        config: Optional configuration dict
        
    Yields:
        Stream events as dictionaries
    """
    try:
        async for event in graph.astream(input_state, stream_mode=stream_mode, config=config):
            yield event
    except Exception as e:
        logger.error(f"Error streaming graph: {e}", exc_info=True)
        yield {"error": str(e)}


def format_stream_event(event: Dict[str, Any]) -> str:
    """
    Format a stream event for display.
    
    Args:
        event: Stream event dictionary
        
    Returns:
        Formatted string representation
    """
    if "error" in event:
        return f"❌ Error: {event['error']}"
    
    # Format based on event type
    if "node" in event:
        node_name = event["node"]
        if "chunk" in event:
            return f"📦 Node '{node_name}': {event['chunk']}"
        return f"🔄 Node '{node_name}' executed"
    
    return str(event)


def stream_graph_sync(
    graph: Any,
    input_state: Dict[str, Any],
    stream_mode: str = "values",
    config: Optional[Dict[str, Any]] = None
) -> Iterator[Dict[str, Any]]:
    """
    Stream graph execution events (synchronous version).
    
    Args:
        graph: Compiled LangGraph instance
        input_state: Initial state for the graph
        stream_mode: Stream mode ("values", "updates", or "debug")
        config: Optional configuration dict
        
    Yields:
        Stream events as dictionaries
    """
    try:
        for event in graph.stream(input_state, stream_mode=stream_mode, config=config):
            yield event
    except Exception as e:
        logger.error(f"Error streaming graph: {e}", exc_info=True)
        yield {"error": str(e)}

