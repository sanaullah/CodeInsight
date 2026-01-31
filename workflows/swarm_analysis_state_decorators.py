"""
Validation decorators for Swarm Analysis workflow nodes.

Provides decorators to validate state before node execution.
"""

from functools import wraps
from typing import Callable, List, Optional, Dict, Any
import logging

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.state_keys import StateKeys
from workflows.swarm_analysis_state_validator import (
    validate_state,
    validate_required_fields
)

logger = logging.getLogger(__name__)


def validate_state_fields(
    required_fields: Optional[List[str]] = None,
    operation: Optional[str] = None
):
    """
    Decorator to validate state fields before node execution.
    
    Args:
        required_fields: List of required field names (e.g., ["project_path", "files"])
        operation: Operation name for error messages (defaults to function name)
        
    Example:
        @validate_state_fields(["project_path", "files"], "scan_project")
        def scan_project_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
            ...
    """
    import asyncio
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(state: SwarmAnalysisState, *args, **kwargs) -> SwarmAnalysisState:
                # Skip validation if state already has an error
                if state.get(StateKeys.ERROR):
                    return await func(state, *args, **kwargs)
                op_name = operation or func.__name__
                
                # Basic state structure validation
                if not validate_state(state):
                    error_msg = f"State validation failed for {op_name}: state must be a dictionary"
                    logger.error(error_msg)
                    updated_state = state.copy()
                    updated_state[StateKeys.ERROR] = error_msg
                    updated_state[StateKeys.ERROR_STAGE] = op_name
                    return updated_state
                
                # Validate required fields
                if required_fields:
                    try:
                        validate_required_fields(state, required_fields, op_name)
                    except ValueError as e:
                        error_msg = f"Missing required fields in {op_name}: {str(e)}"
                        logger.error(error_msg)
                        updated_state = state.copy()
                        updated_state["error"] = error_msg
                        updated_state["error_stage"] = op_name
                        return updated_state
                
                # Execute function (await it)
                try:
                    return await func(state, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {op_name}: {e}", exc_info=True)
                    updated_state = state.copy()
                    updated_state[StateKeys.ERROR] = str(e)
                    updated_state[StateKeys.ERROR_STAGE] = op_name
                    return updated_state
                    
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(state: SwarmAnalysisState, *args, **kwargs) -> SwarmAnalysisState:
                # Skip validation if state already has an error
                if state.get(StateKeys.ERROR):
                    return func(state, *args, **kwargs)
                op_name = operation or func.__name__
                
                # Basic state structure validation
                if not validate_state(state):
                    error_msg = f"State validation failed for {op_name}: state must be a dictionary"
                    logger.error(error_msg)
                    updated_state = state.copy()
                    updated_state[StateKeys.ERROR] = error_msg
                    updated_state[StateKeys.ERROR_STAGE] = op_name
                    return updated_state
                
                # Validate required fields
                if required_fields:
                    try:
                        validate_required_fields(state, required_fields, op_name)
                    except ValueError as e:
                        error_msg = f"Missing required fields in {op_name}: {str(e)}"
                        logger.error(error_msg)
                        updated_state = state.copy()
                        updated_state["error"] = error_msg
                        updated_state["error_stage"] = op_name
                        return updated_state
                
                # Execute function
                try:
                    return func(state, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in {op_name}: {e}", exc_info=True)
                    updated_state = state.copy()
                    updated_state[StateKeys.ERROR] = str(e)
                    updated_state[StateKeys.ERROR_STAGE] = op_name
                    return updated_state
            
            return wrapper
    return decorator

