"""Error context utilities for structured error handling."""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def create_error_context(
    operation: str,
    state: Dict[str, Any],
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create structured error context for logging and debugging.
    
    Args:
        operation: Name of the operation that failed
        state: Current state dictionary
        additional_info: Optional additional context
        
    Returns:
        Dictionary with error context
    """
    context = {
        "operation": operation,
        "project_path": state.get("project_path"),
        "error_stage": state.get("error_stage", operation),
    }
    
    # Add relevant state fields
    if "files" in state:
        context["file_count"] = len(state.get("files", []))
    if "selected_roles" in state:
        context["role_count"] = len(state.get("selected_roles", []))
    if "model_name" in state:
        context["model_name"] = state.get("model_name")
    
    if additional_info:
        context.update(additional_info)
    
    return context

