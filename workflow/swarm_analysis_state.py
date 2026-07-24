"""
State schema for Swarm Analysis LangGraph workflow.

Defines all state fields used throughout the swarm analysis process.
"""

from typing import TypedDict, Any, Dict, List, Optional, Annotated
from typing_extensions import NotRequired
import operator


class SwarmAnalysisState(TypedDict):
    """
    State schema for Swarm Analysis workflow.
    
    Tracks all data flowing through the swarm analysis pipeline:
    - Project scanning results
    - Architecture model
    - Role selection and prompt generation
    - Agent execution results
    - Final synthesis
    """
    # Input parameters
    project_path: str
    goal: NotRequired[Optional[str]]
    model_name: NotRequired[str]
    max_agents: NotRequired[int]
    max_tokens_per_chunk: NotRequired[int]
    enable_chunking: NotRequired[bool]  # Default: True
    chunking_strategy: NotRequired[str]  # Default: "STANDARD" (NONE, STANDARD, AGGRESSIVE)
    auto_detect_languages: NotRequired[Optional[bool]]
    file_extensions: NotRequired[Optional[List[str]]]
    selected_directories: NotRequired[Optional[List[str]]]
    previous_report_path: NotRequired[Optional[str]]
    previous_report_id: NotRequired[Optional[int]]
    previous_report_text: NotRequired[Optional[str]]
    previous_report_findings: NotRequired[Optional[Dict[str, Any]]]
    # Tool calling and dependency resolution
    enable_tool_calling: NotRequired[bool]  # Default: True
    enable_static_dependency_resolution: NotRequired[bool]  # Default: True
    max_tool_calls: NotRequired[int]  # Default: 10
    # File selection (new feature)
    enable_dynamic_file_selection: NotRequired[bool]  # Default: False (backward compatible)
    
    # Project scanning
    files: NotRequired[List[Dict[str, Any]]]  # List of FileInfo as dicts
    file_hash: NotRequired[Optional[str]]
    detected_languages: NotRequired[List[str]]
    dependency_resolution: NotRequired[Optional[Dict[str, Any]]]  # Dependency resolution metadata
    
    # Architecture model
    architecture_model: NotRequired[Optional[Dict[str, Any]]]
    architecture_hash: NotRequired[Optional[str]]
    
    # Context analysis
    project_context: NotRequired[Optional[Dict[str, Any]]]
    context_hash: NotRequired[Optional[str]]
    
    # Role selection
    selected_roles: NotRequired[List[Dict[str, Any]]]  # List of role dicts with name, description, etc.
    role_names: NotRequired[List[str]]  # Extracted role names for convenience
    authorized_roles: NotRequired[List[str]]  # Whitelist of authorized roles for trust boundary enforcement
    
    # Prompt generation and validation
    generated_prompts: NotRequired[Dict[str, str]]  # role_name -> prompt_content
    generated_prompts_list: NotRequired[Annotated[List[Dict[str, Any]], operator.add]]  # For parallel aggregation
    validation_results: NotRequired[Dict[str, Dict[str, Any]]]  # role_name -> validation_data
    validation_results_list: NotRequired[Annotated[List[Dict[str, Any]], operator.add]]  # For parallel aggregation
    prompt_processing_errors: NotRequired[Dict[str, str]]  # role_name -> error_message
    
    # Agent execution
    agents: NotRequired[List[Any]]  # List of agent objects (will be serialized)
    agent_results: NotRequired[Dict[str, Dict[str, Any]]]  # role_name -> agent_result
    agent_results_list: NotRequired[Annotated[List[Dict[str, Any]], operator.add]]  # For parallel aggregation
    agent_status: NotRequired[Dict[str, str]]  # role_name -> status
    chunks_analyzed: NotRequired[int]
    # File selection (new feature)
    selected_files: NotRequired[Dict[str, List[str]]]  # role_name -> List[file_paths]
    file_selection_reasoning: NotRequired[Dict[str, str]]  # role_name -> reasoning
    file_selection_results_list: NotRequired[Annotated[List[Dict[str, Any]], operator.add]]  # For parallel aggregation
    # file_selection_results_list contains dicts with: role_name, selected_files, reasoning, confidence,
    # available_files_count, selected_files_count, selection_coverage, selection_time_seconds,
    # validation_issues, validation_issues_count, token_savings_estimate, timestamp, model_used,
    # temperature, used_fallback, fallback_reason
    # Tool calling
    tool_calls_made: NotRequired[int]  # Counter for tool calls
    tool_calls_history: NotRequired[List[Dict[str, Any]]]  # History of tool calls
    max_tool_calls: NotRequired[int]  # Limit (default: 10)
    
    # Chunk analysis (for parallel chunk processing)
    chunks: NotRequired[List[Any]]  # List of Chunk objects for an agent (temporary storage)
    chunk_results_list: NotRequired[Annotated[List[Dict[str, Any]], operator.add]]  # For parallel chunk aggregation
    chunk_agent_mapping: NotRequired[Dict[str, str]]  # chunk_id -> role_name for result correlation
    
    # Fields used in Send payloads (not in original schema but required)
    agent_info: NotRequired[Optional[Dict[str, Any]]]  # Current agent info (from Send)
    chunk_info: NotRequired[Optional[Dict[str, Any]]]  # Current chunk info (from Send)
    role_name: NotRequired[Optional[str]]  # Current role name (from Send)
    role_info: NotRequired[Optional[Dict[str, Any]]]  # Current role info (from Send)
    prompt: NotRequired[Optional[str]]  # Current prompt (from Send)
    
    # Result synthesis
    synthesized_report: NotRequired[Optional[str]]
    individual_agent_results: NotRequired[Dict[str, Dict[str, Any]]]
    shared_findings: NotRequired[Dict[str, Any]]
    
    # Metadata and tracking
    metadata: NotRequired[Dict[str, Any]]  # For trace correlation, learning data, etc.
    stream_callback: NotRequired[Optional[Any]]  # Callback function for streaming updates (deprecated, use stream_callback_id)
    stream_callback_id: NotRequired[Optional[str]]  # Callback ID for registry lookup (enables checkpoint serialization)
    trace_id: NotRequired[Optional[str]]  # Langfuse trace ID for correlation
    
    # Langfuse prompt tracking
    langfuse_prompt_ids: NotRequired[Dict[str, str]]  # role -> Langfuse prompt_id
    langfuse_prompt_versions: NotRequired[Dict[str, int]]  # role -> Langfuse prompt version
    
    # Trace tracking
    trace_ids: NotRequired[Dict[str, str]]  # role -> Langfuse trace_id
    
    # Error handling
    error: NotRequired[Optional[str]]
    error_stage: NotRequired[Optional[str]]  # Which stage failed
    error_context: NotRequired[Optional[Dict[str, Any]]]  # Error context for debugging
    
    # Idempotency
    idempotency_key: NotRequired[Optional[str]]
    
    # Learning and experience (ACE system)
    learned_skills: NotRequired[Optional[Dict[str, List[Dict[str, Any]]]]]  # skill_type -> List[skill_dicts]
    experience_data: NotRequired[Optional[Dict[str, Any]]]  # Data for experience storage
    
    # Schema versioning
    __version__: NotRequired[str]  # Semantic version of the state schema


# Current version of the state schema
CURRENT_STATE_VERSION = "2.2.0"


def migrate_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate state from older versions to current version.
    
    Args:
        state: The state dictionary to migrate
        
    Returns:
        Migrated state dictionary compliant with CURRENT_STATE_VERSION
    """
    # Get version (default to "1.0.0" if missing - pre-versioning era)
    version = state.get("__version__", "1.0.0")
    
    # If already current, return as is
    if version == CURRENT_STATE_VERSION:
        return state
    
    # Migration: v1.0.0 -> v2.0.0 (File Selection)
    if version == "1.0.0":
        # Add defaults for file selection fields introduced in v2
        state.setdefault("file_selection_results_list", [])
        state.setdefault("enable_dynamic_file_selection", False)
        version = "2.0.0"
    
    # Migration: v2.0.0 -> v2.1.0 (ACE / Learning)
    if version == "2.0.0":
        # Add defaults for learned skills and experience data
        state.setdefault("learned_skills", {})
        state.setdefault("experience_data", {})
        version = "2.1.0"
    
    # Migration: v2.1.0 -> v2.2.0 (Stream Callback Registry)
    if version == "2.1.0":
        # Add stream_callback_id field for registry-based callback storage
        state.setdefault("stream_callback_id", None)
        # Warn if legacy stream_callback is present (should use registry)
        if "stream_callback" in state and callable(state.get("stream_callback")):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("Legacy stream_callback found in state - should use stream_callback_id for checkpoint serialization")
        version = "2.2.0"
        
    # Update version in state
    state["__version__"] = version
    
    return state

