"""
File selection node for agent-driven dynamic file selection.

Allows agents to dynamically select which files they need for analysis
before execution, improving relevance and reducing token usage.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from workflows.nodes import llm_node
from .file_selection_utils import (
    format_file_list_for_selection,
    parse_file_selection,
    validate_file_selection,
    filter_files_by_selection
)
from .file_selection_prompts import build_file_selection_prompt

try:
    from langfuse.decorators import observe
except ImportError:
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


@validate_state_fields(["agent_info"], "select_agent_files")
@observe(as_type="span", name="select_agent_files")
async def select_agent_files_node(
    state: SwarmAnalysisState,
    config: Optional[Dict[str, Any]] = None
) -> SwarmAnalysisState:
    """
    Ask LLM which files this agent needs for analysis.
    
    This node is called before agent execution to let the agent select
    relevant files based on its role and analysis goals.
    
    Args:
        state: State containing agent_info and all necessary context
        config: Optional LangGraph config
        
    Returns:
        Updated state with selected_files in per-role dictionaries
    """
    # Extract agent info from state (set by Send)
    agent_info = state.get(StateKeys.AGENT_INFO)
    if not agent_info:
        state[StateKeys.ERROR] = "agent_info is required for file selection"
        state[StateKeys.ERROR_STAGE] = "select_agent_files"
        logger.error("agent_info is required for file selection")
        return state
    
    role_name = agent_info.get("role", "Unknown")
    prompt = agent_info.get("prompt", "")
    
    # Get configuration from state
    files = state.get(StateKeys.FILES, [])
    model_name = state.get(StateKeys.MODEL_NAME)
    architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL)
    goal = state.get(StateKeys.GOAL)
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    logger.info(f"Selecting files for agent {role_name} from {len(files)} available files")
    
    # Start timing for selection
    selection_start_time = time.time()
    
    if stream_callback:
        stream_callback("file_selection_started", {
            "agent": role_name,
            "available_files": len(files)
        })
    
    # Track fallback usage
    used_fallback = False
    fallback_reason = None
    confidence = None
    validation_issues = []
    
    # Fallback: if no files, return all files
    if not files:
        logger.warning(f"No files available for selection, agent {role_name} will have no files")
        selected_files = []
        reasoning = "No files available in project"
        used_fallback = True
        fallback_reason = "No files available in project"
    else:
        try:
            # Format file list for LLM
            file_list = format_file_list_for_selection(files)
            
            # Build file selection prompt
            selection_prompt = build_file_selection_prompt(
                role_name=role_name,
                prompt=prompt,
                file_list=file_list,
                architecture_model=architecture_model,
                goal=goal
            )
            
            # Call LLM for file selection
            llm_state = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a {role_name} selecting files for code analysis. "
                                 f"Respond with a JSON object containing the files you need."
                    },
                    {
                        "role": "user",
                        "content": selection_prompt
                    }
                ],
                "model": model_name,
                "temperature": 0.3,  # Lower temperature for more deterministic selection
                "metadata": {
                    "role": role_name,
                    "operation": "file_selection"
                }
            }
            
            logger.debug(f"Calling LLM for file selection (agent: {role_name})")
            llm_result = await llm_node(llm_state, config=config)
            
            # Parse LLM response
            response_text = llm_result.get("last_response", "")
            parsed = parse_file_selection(response_text)
            
            selected_files = parsed.get("files", [])
            reasoning = parsed.get("reasoning", "")
            confidence = parsed.get("confidence")  # Extract confidence score
            
            # Validate selection
            validation_issues = validate_file_selection(selected_files, files)
            
            if validation_issues:
                logger.warning(
                    f"File selection validation issues for {role_name}: {validation_issues}"
                )
                # Log but don't fail - LLM may know better than our heuristics
            
            # Fallback: if no files selected or selection failed, use all files
            if not selected_files:
                logger.warning(
                    f"No files selected by LLM for {role_name}, falling back to all files"
                )
                selected_files = [f.get("relative_path", "") or f.get("path", "") 
                                 for f in files if f.get("relative_path") or f.get("path")]
                reasoning = "Fallback: using all files (LLM selection returned no files)"
                used_fallback = True
                fallback_reason = "LLM selection returned no files"
                confidence = None  # Reset confidence on fallback
            
            # Log selection results
            logger.info(
                f"Agent {role_name} selected {len(selected_files)} files "
                f"from {len(files)} available files"
            )
            
            if stream_callback:
                stream_callback("file_selection_completed", {
                    "agent": role_name,
                    "selected_count": len(selected_files),
                    "available_count": len(files),
                    "token_savings_estimate": f"{(1 - len(selected_files) / len(files)) * 100:.1f}%" if files else "0%"
                })
        
        except Exception as e:
            logger.error(
                f"File selection failed for agent {role_name}: {e}",
                exc_info=True
            )
            # Fallback to all files on error
            selected_files = [f.get("relative_path", "") or f.get("path", "") 
                             for f in files if f.get("relative_path") or f.get("path")]
            reasoning = f"Fallback: using all files (error during selection: {str(e)})"
            used_fallback = True
            fallback_reason = f"Error during selection: {str(e)}"
            confidence = None  # No confidence on error fallback
            
            if stream_callback:
                stream_callback("file_selection_error", {
                    "agent": role_name,
                    "error": str(e),
                    "fallback": "using_all_files"
                })
    
    # Calculate selection time
    selection_time_seconds = time.time() - selection_start_time
    
    # Calculate quality metrics
    available_files_count = len(files)
    selected_files_count = len(selected_files)
    
    # Selection coverage: ratio of selected to available
    selection_coverage = (
        selected_files_count / available_files_count 
        if available_files_count > 0 else 0.0
    )
    
    # Validation issues count
    validation_issues_count = len(validation_issues)
    
    # Token savings estimate: percentage reduction
    token_savings_estimate = (
        (1 - selected_files_count / available_files_count) * 100 
        if available_files_count > 0 else 0.0
    )
    
    # Update agent_info with selected files (local copy, not written back to state)
    updated_agent_info = agent_info.copy()
    updated_agent_info["selected_files"] = selected_files
    updated_agent_info["file_selection_reasoning"] = reasoning
    
    # Build complete audit trail for file_selection_results_list
    file_selection_result = {
        "role_name": role_name,
        "selected_files": selected_files,
        "reasoning": reasoning,
        "confidence": confidence,
        "available_files_count": available_files_count,
        "selected_files_count": selected_files_count,
        "selection_coverage": selection_coverage,
        "selection_time_seconds": selection_time_seconds,
        "validation_issues": validation_issues,
        "validation_issues_count": validation_issues_count,
        "token_savings_estimate": token_savings_estimate,
        "timestamp": datetime.now().isoformat(),
        "model_used": model_name,
        "temperature": 0.3,  # Fixed temperature used for file selection
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason
    }
    
    logger.debug(
        f"File selection complete for {role_name}: "
        f"{len(selected_files)} files selected"
    )
    
    # Return ONLY the fields that should be updated
    # Do NOT return the entire state copy - that causes concurrent update errors
    # The reducer (operator.add) will aggregate file_selection_results_list from all parallel nodes
    # LangGraph will merge dictionary fields (selected_files, file_selection_reasoning) automatically
    return {
        "file_selection_results_list": [file_selection_result],
        "selected_files": {role_name: selected_files},
        "file_selection_reasoning": {role_name: reasoning}
    }

