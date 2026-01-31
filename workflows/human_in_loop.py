"""
Human-in-the-loop node implementations for LangGraph.

This module provides nodes for human approval, feedback collection,
and interactive decision points in workflows.
"""

from typing import Dict, Any, Optional, Callable
import logging
import asyncio
from llm.config import ConfigManager

logger = logging.getLogger(__name__)


def human_approval_node(
    timeout: float = 300.0,
    default_action: str = "reject",
    approval_func: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> Callable:
    """
    Create a human approval node.
    
    Args:
        timeout: Timeout in seconds for approval
        default_action: Default action if timeout ("approve" or "reject")
        approval_func: Optional function to get approval (for testing/mocking)
                      If None, will prompt user for input
        
    Returns:
        Node function for use in LangGraph
        
    Example:
        approval = human_approval_node(timeout=60.0, default_action="reject")
    """
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Human approval node implementation."""
        updated_state = state.copy()
        
        # Get approval configuration from config if available
        config_manager = ConfigManager()
        config = config_manager.load_config()
        hitl_config = config.workflows.human_in_loop
        
        if hitl_config.enabled:
            timeout_val = timeout if timeout else hitl_config.timeout
            default = default_action if default_action else hitl_config.default_action
        else:
            # If HITL disabled, use default action immediately
            updated_state["approved"] = (default_action == "approve")
            updated_state["approval_reason"] = "HITL disabled, using default action"
            return updated_state
        
        # Get approval (use provided function or prompt user)
        if approval_func:
            approved = approval_func(state)
        else:
            # In a real implementation, this would prompt the user
            # For now, we'll use a simple input (can be overridden)
            try:
                prompt = state.get("approval_prompt", "Approve this action? (yes/no): ")
                # For async/real implementations, use proper async input handling
                # This is a simplified version
                approved = _get_user_approval(prompt, timeout_val, default)
            except Exception as e:
                logger.warning(f"Error getting approval: {e}, using default: {default}")
                approved = (default == "approve")
        
        updated_state["approved"] = approved
        updated_state["approval_reason"] = "user_approved" if approved else "user_rejected"
        
        return updated_state
    
    return node


def _get_user_approval(prompt: str, timeout: float, default: str) -> bool:
    """
    Get user approval (simplified version).
    
    In a real implementation, this would handle async input with timeout.
    For now, returns default action.
    """
    # This is a placeholder - real implementation would handle async input
    logger.info(f"Approval prompt: {prompt} (timeout: {timeout}s, default: {default})")
    return default == "approve"


def human_feedback_node(
    feedback_prompt: Optional[str] = None,
    feedback_func: Optional[Callable[[Dict[str, Any]], str]] = None
) -> Callable:
    """
    Create a human feedback collection node.
    
    Args:
        feedback_prompt: Prompt to show user for feedback
        feedback_func: Optional function to get feedback (for testing/mocking)
                      If None, will prompt user for input
        
    Returns:
        Node function for use in LangGraph
        
    Example:
        feedback = human_feedback_node(feedback_prompt="Please provide feedback:")
    """
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Human feedback node implementation."""
        updated_state = state.copy()
        
        # Get feedback (use provided function or prompt user)
        if feedback_func:
            feedback = feedback_func(state)
        else:
            # In a real implementation, this would prompt the user
            prompt = feedback_prompt or state.get("feedback_prompt", "Please provide feedback: ")
            try:
                feedback = _get_user_feedback(prompt)
            except Exception as e:
                logger.warning(f"Error getting feedback: {e}")
                feedback = ""
        
        updated_state["human_feedback"] = feedback
        return updated_state
    
    return node


def _get_user_feedback(prompt: str) -> str:
    """
    Get user feedback (simplified version).
    
    In a real implementation, this would handle async input.
    """
    # This is a placeholder - real implementation would handle async input
    logger.info(f"Feedback prompt: {prompt}")
    return ""

