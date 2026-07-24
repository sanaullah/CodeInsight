"""
State keys constants for Swarm Analysis LangGraph workflow.

Provides centralized constants for all state keys to prevent typos,
enable IDE autocomplete, and enable safe refactoring.

Follows the same pattern as RoleEnum using StrEnum for string compatibility.
"""

from enum import StrEnum


class StateKeys(StrEnum):
    """
    Enumeration of all state keys used in SwarmAnalysisState.
    
    These constants match the field names in SwarmAnalysisState TypedDict exactly.
    Using StrEnum ensures string compatibility with LangGraph while providing
    type safety and IDE autocomplete.
    """
    
    # ============================================================================
    # Input Parameters
    # ============================================================================
    PROJECT_PATH = "project_path"
    GOAL = "goal"
    MODEL_NAME = "model_name"
    MAX_AGENTS = "max_agents"
    MAX_TOKENS_PER_CHUNK = "max_tokens_per_chunk"
    ENABLE_CHUNKING = "enable_chunking"
    CHUNKING_STRATEGY = "chunking_strategy"
    AUTO_DETECT_LANGUAGES = "auto_detect_languages"
    FILE_EXTENSIONS = "file_extensions"
    SELECTED_DIRECTORIES = "selected_directories"
    PREVIOUS_REPORT_PATH = "previous_report_path"
    PREVIOUS_REPORT_ID = "previous_report_id"
    PREVIOUS_REPORT_TEXT = "previous_report_text"
    PREVIOUS_REPORT_FINDINGS = "previous_report_findings"
    
    # Tool calling and dependency resolution
    ENABLE_TOOL_CALLING = "enable_tool_calling"
    ENABLE_STATIC_DEPENDENCY_RESOLUTION = "enable_static_dependency_resolution"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_ITERATIONS = "max_iterations"
    MAX_DUPLICATE_ITERATIONS = "max_duplicate_iterations"
    LLM_CALL_TIMEOUT = "llm_call_timeout"
    
    # File selection
    ENABLE_DYNAMIC_FILE_SELECTION = "enable_dynamic_file_selection"
    
    # ============================================================================
    # Project Scanning
    # ============================================================================
    FILES = "files"
    FILE_HASH = "file_hash"
    DETECTED_LANGUAGES = "detected_languages"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    
    # ============================================================================
    # Architecture Model
    # ============================================================================
    ARCHITECTURE_MODEL = "architecture_model"
    ARCHITECTURE_HASH = "architecture_hash"
    
    # Context analysis
    PROJECT_CONTEXT = "project_context"
    CONTEXT_HASH = "context_hash"
    
    # ============================================================================
    # Role Selection
    # ============================================================================
    SELECTED_ROLES = "selected_roles"
    ROLE_NAMES = "role_names"
    AUTHORIZED_ROLES = "authorized_roles"
    
    # ============================================================================
    # Prompt Generation and Validation
    # ============================================================================
    GENERATED_PROMPTS = "generated_prompts"
    GENERATED_PROMPTS_LIST = "generated_prompts_list"
    VALIDATION_RESULTS = "validation_results"
    VALIDATION_RESULTS_LIST = "validation_results_list"
    PROMPT_PROCESSING_ERRORS = "prompt_processing_errors"
    
    # ============================================================================
    # Agent Execution
    # ============================================================================
    AGENTS = "agents"
    AGENT_RESULTS = "agent_results"
    AGENT_RESULTS_LIST = "agent_results_list"
    AGENT_STATUS = "agent_status"
    CHUNKS_ANALYZED = "chunks_analyzed"
    
    # File selection
    SELECTED_FILES = "selected_files"
    FILE_SELECTION_REASONING = "file_selection_reasoning"
    FILE_SELECTION_RESULTS_LIST = "file_selection_results_list"
    
    # Tool calling
    TOOL_CALLS_MADE = "tool_calls_made"
    TOOL_CALLS_HISTORY = "tool_calls_history"
    FILE_CACHE = "file_cache"
    
    # Chunk analysis
    CHUNKS = "chunks"
    CHUNK_RESULTS_LIST = "chunk_results_list"
    CHUNK_AGENT_MAPPING = "chunk_agent_mapping"
    
    # Fields used in Send payloads
    AGENT_INFO = "agent_info"
    CHUNK_INFO = "chunk_info"
    ROLE_NAME = "role_name"
    ROLE_INFO = "role_info"
    PROMPT = "prompt"
    
    # ============================================================================
    # Result Synthesis
    # ============================================================================
    SYNTHESIZED_REPORT = "synthesized_report"
    INDIVIDUAL_AGENT_RESULTS = "individual_agent_results"
    SHARED_FINDINGS = "shared_findings"
    
    # ============================================================================
    # Metadata and Tracking
    # ============================================================================
    METADATA = "metadata"
    STREAM_CALLBACK = "stream_callback"  # Deprecated, use STREAM_CALLBACK_ID
    STREAM_CALLBACK_ID = "stream_callback_id"  # Callback ID for registry lookup
    TRACE_ID = "trace_id"
    
    # Langfuse prompt tracking
    LANGFUSE_PROMPT_IDS = "langfuse_prompt_ids"
    LANGFUSE_PROMPT_VERSIONS = "langfuse_prompt_versions"
    
    # Trace tracking
    TRACE_IDS = "trace_ids"
    
    # ============================================================================
    # Error Handling
    # ============================================================================
    ERROR = "error"
    ERROR_STAGE = "error_stage"
    ERROR_CONTEXT = "error_context"
    
    # ============================================================================
    # Idempotency
    # ============================================================================
    IDEMPOTENCY_KEY = "idempotency_key"
    
    # ============================================================================
    # Learning and Experience (ACE system)
    # ============================================================================
    LEARNED_SKILLS = "learned_skills"
    EXPERIENCE_DATA = "experience_data"
