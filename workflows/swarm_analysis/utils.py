"""
Utility functions for Swarm Analysis workflow nodes.
"""

from typing import Callable, Optional
from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.state_keys import StateKeys


def get_stream_callback(state: SwarmAnalysisState) -> Optional[Callable]:
    """
    Get stream callback from registry using ID in state.
    
    Supports both new registry-based pattern (stream_callback_id) and
    legacy direct callback pattern (stream_callback) for backward compatibility.
    
    Args:
        state: Swarm analysis state dictionary
        
    Returns:
        Callback function if found, None otherwise
    """
    # Try new registry-based pattern first
    callback_id = state.get(StateKeys.STREAM_CALLBACK_ID)
    if callback_id:
        try:
            from workflows.callback_registry import get_callback_registry
            registry = get_callback_registry()
            callback = registry.get(callback_id)
            if callback:
                return callback
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to retrieve callback from registry: {e}")
    
    # Fallback to legacy direct callback pattern (for backward compatibility)
    callback = state.get(StateKeys.STREAM_CALLBACK)
    if callback and callable(callback):
        return callback
    
    return None
