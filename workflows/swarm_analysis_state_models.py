"""
Pydantic state models for Swarm Analysis workflow.

Provides runtime validation of state structure using Pydantic models.
"""

from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path


class InputState(BaseModel):
    """Input parameters for swarm analysis."""
    project_path: str = Field(default="", description="Path to project directory")  # Allow empty for incremental validation
    goal: Optional[str] = Field(None, description="Analysis goal")
    model_name: Optional[str] = Field(None, description="LLM model name")
    max_agents: int = Field(default=10, description="Maximum number of agents")
    max_tokens_per_chunk: int = Field(default=100000, description="Maximum tokens per chunk")
    enable_chunking: bool = Field(default=True, description="Enable file chunking")
    chunking_strategy: str = Field(default="STANDARD", description="Chunking strategy")
    enable_dynamic_file_selection: bool = Field(default=False, description="Enable dynamic file selection")
    auto_detect_languages: Optional[bool] = Field(None, description="Auto-detect languages")
    file_extensions: Optional[List[str]] = Field(None, description="File extensions to include")
    selected_directories: Optional[List[str]] = Field(None, description="Selected directories")
    previous_report_path: Optional[str] = Field(None, description="Previous report path")
    previous_report_id: Optional[int] = Field(None, description="Previous report ID")


class ProjectScanningState(BaseModel):
    """Project scanning results."""
    files: List[Dict[str, Any]] = Field(default_factory=list, description="List of scanned files")
    file_hash: Optional[str] = Field(None, description="Hash of file structure")
    detected_languages: List[str] = Field(default_factory=list, description="Detected languages")


class ArchitectureState(BaseModel):
    """Architecture model state."""
    architecture_model: Optional[Dict[str, Any]] = Field(None, description="Architecture model")
    architecture_hash: Optional[str] = Field(None, description="Architecture hash")
    project_context: Optional[Dict[str, Any]] = Field(None, description="Project context")
    context_hash: Optional[str] = Field(None, description="Context hash")


class RoleSelectionState(BaseModel):
    """Role selection state."""
    selected_roles: List[Dict[str, Any]] = Field(default_factory=list, description="Selected roles")
    role_names: List[str] = Field(default_factory=list, description="Role names")


class PromptGenerationState(BaseModel):
    """Prompt generation and validation state."""
    generated_prompts: Dict[str, str] = Field(default_factory=dict, description="Generated prompts")
    generated_prompts_list: List[Dict[str, Any]] = Field(default_factory=list, description="Generated prompts list")
    validation_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Validation results")
    validation_results_list: List[Dict[str, Any]] = Field(default_factory=list, description="Validation results list")
    prompt_processing_errors: Dict[str, str] = Field(default_factory=dict, description="Prompt processing errors")
    langfuse_prompt_ids: Dict[str, str] = Field(default_factory=dict, description="Langfuse prompt IDs")
    langfuse_prompt_versions: Dict[str, int] = Field(default_factory=dict, description="Langfuse prompt versions")


class AgentExecutionState(BaseModel):
    """Agent execution state."""
    agents: List[Any] = Field(default_factory=list, description="Agent objects")
    agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Agent results")
    agent_results_list: List[Dict[str, Any]] = Field(default_factory=list, description="Agent results list")
    agent_status: Dict[str, str] = Field(default_factory=dict, description="Agent status")
    chunks_analyzed: int = Field(default=0, description="Chunks analyzed")
    chunks: List[Any] = Field(default_factory=list, description="Chunks for processing")
    chunk_results_list: List[Dict[str, Any]] = Field(default_factory=list, description="Chunk results list")
    chunk_agent_mapping: Dict[str, str] = Field(default_factory=dict, description="Chunk to agent mapping")
    # File selection (new feature)
    selected_files: Dict[str, List[str]] = Field(default_factory=dict, description="Selected files by role")
    file_selection_reasoning: Dict[str, str] = Field(default_factory=dict, description="File selection reasoning by role")
    file_selection_results_list: List[Dict[str, Any]] = Field(default_factory=list, description="File selection results list")
    # Missing fields that are used but not in TypedDict
    agent_info: Optional[Dict[str, Any]] = Field(None, description="Current agent info (from Send)")
    chunk_info: Optional[Dict[str, Any]] = Field(None, description="Current chunk info (from Send)")
    role_name: Optional[str] = Field(None, description="Current role name (from Send)")


class SynthesisState(BaseModel):
    """Result synthesis state."""
    synthesized_report: Optional[str] = Field(None, description="Synthesized report")
    individual_agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Individual agent results")
    shared_findings: Dict[str, Any] = Field(default_factory=dict, description="Shared findings")


class MetadataState(BaseModel):
    """Metadata and tracking state."""
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    stream_callback: Optional[Any] = Field(None, exclude=True, description="Stream callback function (deprecated, use stream_callback_id)")
    stream_callback_id: Optional[str] = Field(None, description="Stream callback ID for registry lookup")
    trace_id: Optional[str] = Field(None, description="Langfuse trace ID")
    trace_ids: Dict[str, str] = Field(default_factory=dict, description="Trace IDs by role")
    idempotency_key: Optional[str] = Field(None, description="Idempotency key")
    learned_skills: List[Dict[str, Any]] = Field(default_factory=list, description="Learned skills")
    experience_data: Optional[Dict[str, Any]] = Field(None, description="Experience data")


class ErrorState(BaseModel):
    """Error handling state."""
    error: Optional[str] = Field(None, description="Error message")
    error_stage: Optional[str] = Field(None, description="Error stage")
    error_context: Optional[Dict[str, Any]] = Field(None, description="Error context")


class SwarmAnalysisStateModel(BaseModel):
    """
    Composite Pydantic model for Swarm Analysis state.
    
    Combines all sub-states into a single validated model.
    """
    # Input
    input: InputState = Field(default_factory=InputState)
    
    # Project scanning
    scanning: ProjectScanningState = Field(default_factory=ProjectScanningState)
    
    # Architecture
    architecture: ArchitectureState = Field(default_factory=ArchitectureState)
    
    # Role selection
    roles: RoleSelectionState = Field(default_factory=RoleSelectionState)
    
    # Prompt generation
    prompts: PromptGenerationState = Field(default_factory=PromptGenerationState)
    
    # Agent execution
    agents: AgentExecutionState = Field(default_factory=AgentExecutionState)
    
    # Synthesis
    synthesis: SynthesisState = Field(default_factory=SynthesisState)
    
    # Metadata
    metadata: MetadataState = Field(default_factory=MetadataState)
    
    # Error
    error: ErrorState = Field(default_factory=ErrorState)
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow extra fields for backward compatibility
        arbitrary_types_allowed = True  # Allow arbitrary types (e.g., callbacks)

