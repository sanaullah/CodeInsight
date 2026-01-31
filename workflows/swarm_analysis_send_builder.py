"""
Send payload builder utilities for Swarm Analysis workflow.

Ensures all required fields are included in Send payloads for parallel node execution.
"""

from typing import Dict, Any, Optional, List
import logging

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.state_keys import StateKeys

logger = logging.getLogger(__name__)


def build_agent_send_payload(
    state: SwarmAnalysisState,
    agent_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build complete agent state payload for execute_single_agent_node.
    
    Ensures all required fields are included for agent execution.
    
    Args:
        state: Current swarm analysis state
        agent_info: Agent information dictionary
        
    Returns:
        Complete state payload for agent execution
    """
    role_name = agent_info.get("role", "Unknown")
    
    payload = {
        StateKeys.AGENT_INFO: agent_info,
        StateKeys.ROLE_NAME: role_name,  # Explicit role_name for convenience
        "node_type": "agent_execution",  # Node type for context tracking
        StateKeys.PROJECT_PATH: state.get(StateKeys.PROJECT_PATH),
        StateKeys.FILES: state.get(StateKeys.FILES, []),
        StateKeys.ARCHITECTURE_MODEL: state.get(StateKeys.ARCHITECTURE_MODEL, {}),
        StateKeys.MODEL_NAME: state.get(StateKeys.MODEL_NAME),
        StateKeys.MAX_TOKENS_PER_CHUNK: state.get(StateKeys.MAX_TOKENS_PER_CHUNK, 100000),
        StateKeys.ENABLE_CHUNKING: state.get(StateKeys.ENABLE_CHUNKING, True),
        StateKeys.CHUNKING_STRATEGY: state.get(StateKeys.CHUNKING_STRATEGY, "STANDARD"),
        StateKeys.ENABLE_DYNAMIC_FILE_SELECTION: state.get(StateKeys.ENABLE_DYNAMIC_FILE_SELECTION, False),
        StateKeys.STREAM_CALLBACK_ID: state.get(StateKeys.STREAM_CALLBACK_ID),
        StateKeys.GOAL: state.get(StateKeys.GOAL),
        StateKeys.LANGFUSE_PROMPT_IDS: state.get(StateKeys.LANGFUSE_PROMPT_IDS, {}),
        StateKeys.LANGFUSE_PROMPT_VERSIONS: state.get(StateKeys.LANGFUSE_PROMPT_VERSIONS, {}),
        StateKeys.TRACE_IDS: state.get(StateKeys.TRACE_IDS, {}),
        StateKeys.METADATA: state.get(StateKeys.METADATA, {}),
    }
    
    # Include selected_files if present in agent_info (from file selection node)
    if "selected_files" in agent_info:
        payload[StateKeys.AGENT_INFO]["selected_files"] = agent_info["selected_files"]
    if "file_selection_reasoning" in agent_info:
        payload[StateKeys.AGENT_INFO]["file_selection_reasoning"] = agent_info["file_selection_reasoning"]
    
    # Log payload completeness for debugging
    required_fields = [
        StateKeys.AGENT_INFO, StateKeys.PROJECT_PATH, StateKeys.FILES, StateKeys.ARCHITECTURE_MODEL,
        StateKeys.MODEL_NAME, StateKeys.LANGFUSE_PROMPT_IDS
    ]
    missing_fields = [field for field in required_fields if field not in payload or payload[field] is None]
    if missing_fields:
        logger.warning(f"Agent send payload missing fields: {missing_fields} for role {role_name}")
    
    return payload


def build_prompt_send_payload(
    state: SwarmAnalysisState,
    role_obj: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build complete prompt state payload for generate_single_prompt_node.
    
    Ensures all required fields are included for prompt generation.
    
    Args:
        state: Current swarm analysis state
        role_obj: Role object dictionary
        
    Returns:
        Complete state payload for prompt generation
    """
    role_name = role_obj["name"] if isinstance(role_obj, dict) else str(role_obj)
    
    payload = {
        StateKeys.ROLE_INFO: role_obj,
        StateKeys.ROLE_NAME: role_name,  # Explicit role_name for convenience
        "node_type": "prompt_generation",  # Node type for context tracking
        StateKeys.ARCHITECTURE_MODEL: state.get(StateKeys.ARCHITECTURE_MODEL, {}),
        StateKeys.ARCHITECTURE_HASH: state.get(StateKeys.ARCHITECTURE_HASH),  # CRITICAL: Required for prompt reuse
        StateKeys.FILE_HASH: state.get(StateKeys.FILE_HASH),  # CRITICAL: Required for prompt reuse
        StateKeys.GOAL: state.get(StateKeys.GOAL),
        StateKeys.MODEL_NAME: state.get(StateKeys.MODEL_NAME),
        StateKeys.STREAM_CALLBACK_ID: state.get(StateKeys.STREAM_CALLBACK_ID),
        StateKeys.METADATA: state.get(StateKeys.METADATA, {}),
    }
    
    # Log payload completeness for debugging
    required_fields = [StateKeys.ROLE_INFO, StateKeys.ARCHITECTURE_MODEL, StateKeys.MODEL_NAME]
    missing_fields = [field for field in required_fields if field not in payload or payload[field] is None]
    if missing_fields:
        logger.warning(f"Prompt send payload missing fields: {missing_fields} for role {role_name}")
    
    # Warn if hashes are missing (they're optional but important for prompt reuse)
    if not payload.get(StateKeys.ARCHITECTURE_HASH):
        logger.warning(f"Prompt send payload missing architecture_hash for role {role_name} - prompt reuse will be disabled")
    if not payload.get(StateKeys.FILE_HASH):
        logger.warning(f"Prompt send payload missing file_hash for role {role_name} - prompt reuse will be disabled")
    
    return payload


def build_validation_send_payload(
    state: SwarmAnalysisState,
    role_name: str,
    prompt: str
) -> Dict[str, Any]:
    """
    Build complete validation state payload for validate_single_prompt_node.
    
    Ensures all required fields are included for prompt validation.
    
    Args:
        state: Current swarm analysis state
        role_name: Role name
        prompt: Prompt content to validate
        
    Returns:
        Complete state payload for prompt validation
    """
    payload = {
        StateKeys.ROLE_NAME: role_name,
        StateKeys.PROMPT: prompt,
        StateKeys.MODEL_NAME: state.get(StateKeys.MODEL_NAME),
        StateKeys.STREAM_CALLBACK_ID: state.get(StateKeys.STREAM_CALLBACK_ID),
        StateKeys.METADATA: state.get(StateKeys.METADATA, {}),
        StateKeys.LANGFUSE_PROMPT_IDS: state.get(StateKeys.LANGFUSE_PROMPT_IDS, {}),
    }
    
    # Log payload completeness for debugging
    required_fields = [StateKeys.ROLE_NAME, StateKeys.PROMPT, StateKeys.MODEL_NAME]
    missing_fields = [field for field in required_fields if field not in payload or payload[field] is None]
    if missing_fields:
        logger.warning(f"Validation send payload missing fields: {missing_fields} for role {role_name}")
    
    return payload

