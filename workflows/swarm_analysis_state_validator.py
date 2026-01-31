"""
State validation utilities for Swarm Analysis workflow.

Provides runtime validation of state structure before node execution.
"""

from typing import Dict, Any, List, Optional
import logging

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_models import SwarmAnalysisStateModel
from workflows.state_keys import StateKeys

logger = logging.getLogger(__name__)


def validate_state(state: Dict[str, Any]) -> bool:
    """
    Validate state structure before node execution.
    
    This is a lenient validation that checks structure without requiring all fields,
    since LangGraph state is built incrementally.
    
    Args:
        state: State dictionary to validate
        
    Returns:
        True if state structure is valid, False otherwise
    """
    if not isinstance(state, dict):
        logger.error(f"State must be a dictionary, got {type(state)}")
        return False
    
    # Basic structure validation - just check that it's a dict
    # We don't validate all fields because state is built incrementally
    # Individual nodes will validate their required fields
    return True


def validate_required_fields(
    state: Dict[str, Any],
    required_fields: List[str],
    operation: str = "operation"
) -> None:
    """
    Validate that required fields are present in state.
    
    Args:
        state: State dictionary
        required_fields: List of required field names
        operation: Operation name for error messages
        
    Raises:
        ValueError: If any required field is missing
    """
    missing = []
    for field in required_fields:
        # Handle nested fields (e.g., "input.project_path")
        if "." in field:
            parts = field.split(".")
            current = state
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    missing.append(field)
                    break
                current = current[part]
                if current is None:
                    missing.append(field)
                    break
        else:
            if field not in state or state[field] is None:
                missing.append(field)
    
    if missing:
        raise ValueError(
            f"{operation} requires the following fields: {', '.join(missing)}"
        )


def convert_dict_to_model(state: Dict[str, Any]) -> SwarmAnalysisStateModel:
    """
    Convert dict state to Pydantic model.
    
    Args:
        state: State dictionary
        
    Returns:
        SwarmAnalysisStateModel instance
        
    Raises:
        ValueError: If state cannot be converted
    """
    try:
        # Flatten state structure for Pydantic model
        # The model expects nested structure, but state is flat
        # We'll create a compatible structure with defaults for missing fields
        # Only project_path is required in InputState, so we provide a default if missing
        model_data = {
            "input": {
                "project_path": state.get(StateKeys.PROJECT_PATH, ""),  # Default to empty string if not present
                "goal": state.get(StateKeys.GOAL),
                "model_name": state.get(StateKeys.MODEL_NAME),
                "max_agents": state.get(StateKeys.MAX_AGENTS, 10),
                "max_tokens_per_chunk": state.get(StateKeys.MAX_TOKENS_PER_CHUNK, 100000),
                "enable_chunking": state.get(StateKeys.ENABLE_CHUNKING, True),
                "chunking_strategy": state.get(StateKeys.CHUNKING_STRATEGY, "STANDARD"),
                "auto_detect_languages": state.get(StateKeys.AUTO_DETECT_LANGUAGES),
                "file_extensions": state.get(StateKeys.FILE_EXTENSIONS),
                "selected_directories": state.get(StateKeys.SELECTED_DIRECTORIES),
                "previous_report_path": state.get(StateKeys.PREVIOUS_REPORT_PATH),
                "previous_report_id": state.get(StateKeys.PREVIOUS_REPORT_ID),
            },
            "scanning": {
                "files": state.get(StateKeys.FILES, []),
                "file_hash": state.get(StateKeys.FILE_HASH),
                "detected_languages": state.get(StateKeys.DETECTED_LANGUAGES, []),
            },
            "architecture": {
                "architecture_model": state.get(StateKeys.ARCHITECTURE_MODEL),
                "architecture_hash": state.get(StateKeys.ARCHITECTURE_HASH),
                "project_context": state.get(StateKeys.PROJECT_CONTEXT),
                "context_hash": state.get(StateKeys.CONTEXT_HASH),
            },
            "roles": {
                "selected_roles": state.get(StateKeys.SELECTED_ROLES, []),
                "role_names": state.get(StateKeys.ROLE_NAMES, []),
            },
            "prompts": {
                "generated_prompts": state.get(StateKeys.GENERATED_PROMPTS, {}),
                "generated_prompts_list": state.get(StateKeys.GENERATED_PROMPTS_LIST, []),
                "validation_results": state.get(StateKeys.VALIDATION_RESULTS, {}),
                "validation_results_list": state.get(StateKeys.VALIDATION_RESULTS_LIST, []),
                "prompt_processing_errors": state.get(StateKeys.PROMPT_PROCESSING_ERRORS, {}),
                "langfuse_prompt_ids": state.get(StateKeys.LANGFUSE_PROMPT_IDS, {}),
            },
            "agents": {
                "agents": state.get(StateKeys.AGENTS, []),
                "agent_results": state.get(StateKeys.AGENT_RESULTS, {}),
                "agent_results_list": state.get(StateKeys.AGENT_RESULTS_LIST, []),
                "agent_status": state.get(StateKeys.AGENT_STATUS, {}),
                "chunks_analyzed": state.get(StateKeys.CHUNKS_ANALYZED, 0),
                "chunks": state.get(StateKeys.CHUNKS, []),
                "chunk_results_list": state.get(StateKeys.CHUNK_RESULTS_LIST, []),
                "chunk_agent_mapping": state.get(StateKeys.CHUNK_AGENT_MAPPING, {}),
                "agent_info": state.get(StateKeys.AGENT_INFO),
                "chunk_info": state.get(StateKeys.CHUNK_INFO),
                "role_name": state.get(StateKeys.ROLE_NAME),
            },
            "synthesis": {
                "synthesized_report": state.get(StateKeys.SYNTHESIZED_REPORT),
                "individual_agent_results": state.get(StateKeys.INDIVIDUAL_AGENT_RESULTS, {}),
                "shared_findings": state.get(StateKeys.SHARED_FINDINGS, {}),
            },
            "metadata": {
                "metadata": state.get(StateKeys.METADATA, {}),
                "stream_callback": state.get(StateKeys.STREAM_CALLBACK),
                "trace_id": state.get(StateKeys.TRACE_ID),
                "trace_ids": state.get(StateKeys.TRACE_IDS, {}),
                "idempotency_key": state.get(StateKeys.IDEMPOTENCY_KEY),
                "learned_skills": state.get(StateKeys.LEARNED_SKILLS, []),
                "experience_data": state.get(StateKeys.EXPERIENCE_DATA),
            },
            "error": {
                "error": state.get(StateKeys.ERROR),
                "error_stage": state.get(StateKeys.ERROR_STAGE),
                "error_context": state.get(StateKeys.ERROR_CONTEXT),
            },
        }
        
        return SwarmAnalysisStateModel.model_validate(model_data)
    except Exception as e:
        logger.error(f"Failed to convert state to model: {e}", exc_info=True)
        raise ValueError(f"Cannot convert state to model: {e}")


def convert_model_to_dict(model: SwarmAnalysisStateModel) -> Dict[str, Any]:
    """
    Convert Pydantic model back to flat dict for LangGraph.
    
    Args:
        model: SwarmAnalysisStateModel instance
        
    Returns:
        Flat state dictionary compatible with LangGraph
    """
    state_dict = {}
    
    # Flatten nested structure back to flat dict
    if model.input:
        state_dict[StateKeys.PROJECT_PATH] = model.input.project_path
        state_dict[StateKeys.GOAL] = model.input.goal
        state_dict[StateKeys.MODEL_NAME] = model.input.model_name
        state_dict[StateKeys.MAX_AGENTS] = model.input.max_agents
        state_dict[StateKeys.MAX_TOKENS_PER_CHUNK] = model.input.max_tokens_per_chunk
        state_dict[StateKeys.ENABLE_CHUNKING] = model.input.enable_chunking
        state_dict[StateKeys.CHUNKING_STRATEGY] = model.input.chunking_strategy
        state_dict[StateKeys.AUTO_DETECT_LANGUAGES] = model.input.auto_detect_languages
        state_dict[StateKeys.FILE_EXTENSIONS] = model.input.file_extensions
        state_dict[StateKeys.SELECTED_DIRECTORIES] = model.input.selected_directories
        state_dict[StateKeys.PREVIOUS_REPORT_PATH] = model.input.previous_report_path
        state_dict[StateKeys.PREVIOUS_REPORT_ID] = model.input.previous_report_id
    
    if model.scanning:
        state_dict[StateKeys.FILES] = model.scanning.files
        state_dict[StateKeys.FILE_HASH] = model.scanning.file_hash
        state_dict[StateKeys.DETECTED_LANGUAGES] = model.scanning.detected_languages
    
    if model.architecture:
        state_dict[StateKeys.ARCHITECTURE_MODEL] = model.architecture.architecture_model
        state_dict[StateKeys.ARCHITECTURE_HASH] = model.architecture.architecture_hash
        state_dict[StateKeys.PROJECT_CONTEXT] = model.architecture.project_context
        state_dict[StateKeys.CONTEXT_HASH] = model.architecture.context_hash
    
    if model.roles:
        state_dict[StateKeys.SELECTED_ROLES] = model.roles.selected_roles
        state_dict[StateKeys.ROLE_NAMES] = model.roles.role_names
    
    if model.prompts:
        state_dict[StateKeys.GENERATED_PROMPTS] = model.prompts.generated_prompts
        state_dict[StateKeys.GENERATED_PROMPTS_LIST] = model.prompts.generated_prompts_list
        state_dict[StateKeys.VALIDATION_RESULTS] = model.prompts.validation_results
        state_dict[StateKeys.VALIDATION_RESULTS_LIST] = model.prompts.validation_results_list
        state_dict[StateKeys.PROMPT_PROCESSING_ERRORS] = model.prompts.prompt_processing_errors
        state_dict[StateKeys.LANGFUSE_PROMPT_IDS] = model.prompts.langfuse_prompt_ids
    
    if model.agents:
        state_dict[StateKeys.AGENTS] = model.agents.agents
        state_dict[StateKeys.AGENT_RESULTS] = model.agents.agent_results
        state_dict[StateKeys.AGENT_RESULTS_LIST] = model.agents.agent_results_list
        state_dict[StateKeys.AGENT_STATUS] = model.agents.agent_status
        state_dict[StateKeys.CHUNKS_ANALYZED] = model.agents.chunks_analyzed
        state_dict[StateKeys.CHUNKS] = model.agents.chunks
        state_dict[StateKeys.CHUNK_RESULTS_LIST] = model.agents.chunk_results_list
        state_dict[StateKeys.CHUNK_AGENT_MAPPING] = model.agents.chunk_agent_mapping
        state_dict[StateKeys.AGENT_INFO] = model.agents.agent_info
        state_dict[StateKeys.CHUNK_INFO] = model.agents.chunk_info
        state_dict[StateKeys.ROLE_NAME] = model.agents.role_name
    
    if model.synthesis:
        state_dict[StateKeys.SYNTHESIZED_REPORT] = model.synthesis.synthesized_report
        state_dict[StateKeys.INDIVIDUAL_AGENT_RESULTS] = model.synthesis.individual_agent_results
        state_dict[StateKeys.SHARED_FINDINGS] = model.synthesis.shared_findings
    
    if model.metadata:
        state_dict[StateKeys.METADATA] = model.metadata.metadata
        state_dict[StateKeys.STREAM_CALLBACK] = model.metadata.stream_callback
        state_dict[StateKeys.TRACE_ID] = model.metadata.trace_id
        state_dict[StateKeys.TRACE_IDS] = model.metadata.trace_ids
        state_dict[StateKeys.IDEMPOTENCY_KEY] = model.metadata.idempotency_key
        state_dict[StateKeys.LEARNED_SKILLS] = model.metadata.learned_skills
        state_dict[StateKeys.EXPERIENCE_DATA] = model.metadata.experience_data
    
    if model.error:
        state_dict[StateKeys.ERROR] = model.error.error
        state_dict[StateKeys.ERROR_STAGE] = model.error.error_stage
        state_dict[StateKeys.ERROR_CONTEXT] = model.error.error_context
    
    return state_dict

