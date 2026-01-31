"""
Swarm Analysis Orchestrator for CodeInsight.

Wraps LangGraph workflow to provide v2-compatible interface.
All LLM calls go through LangGraph for Langfuse tracking.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime

from workflows.swarm_analysis_graph import create_swarm_analysis_graph
from workflows.swarm_analysis_state import SwarmAnalysisState, CURRENT_STATE_VERSION
from workflows.integration import setup_langfuse_callbacks
from utils.langfuse.metadata_schema import normalize_metadata, SwarmAnalysisMetadata

logger = logging.getLogger(__name__)


class SwarmAnalysisOrchestrator:
    """
    Main orchestrator for Swarm Analysis system.
    
    Uses LangGraph internally for all operations.
    Provides v2-compatible interface for backward compatibility.
    """
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        auto_detect_languages: Optional[bool] = None,
        file_extensions: Optional[List[str]] = None
    ):
        """
        Initialize swarm analysis orchestrator.
        
        Args:
            model_name: Optional model name for LLM calls
            auto_detect_languages: Optional override for auto-detect setting
            file_extensions: Optional override for file extensions
        """
        self.model_name = model_name
        self.auto_detect_languages = auto_detect_languages
        self.file_extensions = file_extensions
        
        logger.info("SwarmAnalysisOrchestrator initialized (v3 - LangGraph-based)")
    
    async def analyze(
        self,
        project_path: str,
        goal: Optional[str] = None,
        model_name: Optional[str] = None,
        max_agents: int = 10,
        stream_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        auto_detect_languages: Optional[bool] = None,
        file_extensions: Optional[List[str]] = None,
        selected_directories: Optional[List[str]] = None,
        previous_report_path: Optional[str] = None,
        previous_report_id: Optional[int] = None,
        max_tokens_per_chunk: int = 100000,
        enable_chunking: bool = True,
        chunking_strategy: str = "STANDARD",
        enable_dynamic_file_selection: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute complete swarm analysis flow using LangGraph.
        
        Args:
            project_path: Path to project directory
            goal: Optional analysis goal
            model_name: Optional model name override
            max_agents: Maximum number of agents to spawn
            stream_callback: Optional callback for streaming updates
            auto_detect_languages: Optional override for auto-detect setting
            file_extensions: Optional override for file extensions
            selected_directories: Optional list of directories to analyze
            previous_report_path: Optional path to previous analysis report
            previous_report_id: Optional scan history ID of previous analysis
            max_tokens_per_chunk: Maximum tokens per chunk (default: 100000)
            enable_chunking: Enable file chunking (default: True)
            chunking_strategy: Chunking strategy - NONE, STANDARD, or AGGRESSIVE (default: STANDARD)
            enable_dynamic_file_selection: Enable agent-driven dynamic file selection (default: False)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Load configuration for tool calling and dependency resolution
            from llm.config import ConfigManager
            import yaml
            from pathlib import Path
            
            # Get configuration values with defaults
            enable_tool_calling = kwargs.get("enable_tool_calling")
            enable_static_dependency_resolution = kwargs.get("enable_static_dependency_resolution")
            max_tool_calls = kwargs.get("max_tool_calls")
            
            # Read from config.yaml if not provided
            if enable_tool_calling is None or enable_static_dependency_resolution is None or max_tool_calls is None:
                config_path = Path("config.yaml")
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = yaml.safe_load(f)
                        swarm_config = config_data.get("workflows", {}).get("swarm_analysis", {})
                        
                        if enable_tool_calling is None:
                            enable_tool_calling = swarm_config.get("enable_tool_calling", True)
                        if enable_static_dependency_resolution is None:
                            enable_static_dependency_resolution = swarm_config.get("enable_static_dependency_resolution", True)
                        if max_tool_calls is None:
                            max_tool_calls = swarm_config.get("max_tool_calls_per_agent", 10)
                    except Exception as e:
                        logger.warning(f"Could not load swarm_analysis config: {e}, using defaults")
                        if enable_tool_calling is None:
                            enable_tool_calling = True
                        if enable_static_dependency_resolution is None:
                            enable_static_dependency_resolution = True
                        if max_tool_calls is None:
                            max_tool_calls = 10
                else:
                    # Use defaults if config file doesn't exist
                    if enable_tool_calling is None:
                        enable_tool_calling = True
                    if enable_static_dependency_resolution is None:
                        enable_static_dependency_resolution = True
                    if max_tool_calls is None:
                        max_tool_calls = 10
            
            # Register callback in registry if provided
            callback_id = None
            if stream_callback:
                try:
                    from workflows.callback_registry import get_callback_registry
                    registry = get_callback_registry()
                    callback_id = registry.register(stream_callback)
                    logger.debug(f"Registered stream callback with ID: {callback_id}")
                except Exception as reg_error:
                    logger.warning(f"Failed to register callback in registry: {reg_error}, using direct callback")
                    # Fallback to direct callback for backward compatibility
                    callback_id = None
            
            # Prepare initial state
            initial_state: SwarmAnalysisState = {
                "project_path": project_path,
                "goal": goal,
                "model_name": model_name or self.model_name,
                "max_agents": max_agents,
                "max_tokens_per_chunk": max_tokens_per_chunk,
                "enable_chunking": enable_chunking,
                "chunking_strategy": chunking_strategy,
                "enable_dynamic_file_selection": enable_dynamic_file_selection,
                "auto_detect_languages": auto_detect_languages or self.auto_detect_languages,
                "file_extensions": file_extensions or self.file_extensions,
                "selected_directories": selected_directories,
                "previous_report_path": previous_report_path,
                "previous_report_id": previous_report_id,
                "enable_tool_calling": enable_tool_calling,
                "enable_static_dependency_resolution": enable_static_dependency_resolution,
                "max_tool_calls": max_tool_calls,
                "stream_callback_id": callback_id,  # Use callback ID instead of direct callback
                "metadata": {
                    "start_time": datetime.now().isoformat(),
                    "project_path": project_path,
                    "goal": goal
                },
                "__version__": CURRENT_STATE_VERSION
            }
            
            # Normalize metadata using standardized schema
            try:
                normalized_metadata = normalize_metadata(
                    user_id=kwargs.get("user_id"),
                    session_id=kwargs.get("session_id"),
                    project_path=project_path,
                    goal=goal,
                    environment=kwargs.get("environment"),
                    run_type=kwargs.get("run_type"),
                    app_version=kwargs.get("app_version"),
                    metadata=kwargs.get("metadata", {})
                )
            except ValueError as e:
                logger.warning(f"Metadata normalization failed: {e}, using minimal metadata")
                # Fallback to minimal metadata
                normalized_metadata = SwarmAnalysisMetadata(
                    project_path=project_path,
                    goal=goal,
                    user_id=kwargs.get("user_id"),
                    session_id=kwargs.get("session_id")
                )
            
            # Extract normalized values
            user_id = normalized_metadata.user_id
            session_id = normalized_metadata.session_id
            trace_metadata = normalized_metadata.to_langfuse_metadata()
            
            # Create graph with Langfuse integration
            graph = create_swarm_analysis_graph(
                trace_name="swarm_analysis",
                user_id=user_id,
                session_id=session_id,
                metadata=trace_metadata
            )
            
            # Setup Langfuse callbacks for this execution
            langfuse_handler = setup_langfuse_callbacks(
                trace_name="swarm_analysis",
                enabled=True,
                user_id=user_id,
                session_id=session_id,
                metadata=trace_metadata,
                tags=["swarm_analysis", "code_analysis"]
            )
            
            # Prepare config for LangGraph (includes Langfuse callbacks)
            config = {
                "configurable": {
                    "model": model_name or self.model_name
                }
            }
            
            # Add Langfuse callbacks to config - LangGraph will propagate them to nodes
            if langfuse_handler:
                if "callbacks" not in config:
                    config["callbacks"] = []
                config["callbacks"].append(langfuse_handler)
            
            # Add normalized metadata to config for trace correlation (v3 pattern: langfuse_* prefix)
            if "metadata" not in config:
                config["metadata"] = {}
            
            # Add all normalized metadata fields
            config["metadata"].update(trace_metadata)
            
            # Ensure langfuse_* prefixes are present for v3 compatibility
            if user_id:
                config["metadata"]["langfuse_user_id"] = user_id
            if session_id:
                config["metadata"]["langfuse_session_id"] = session_id
            
            # Invoke graph
            logger.info(f"Starting swarm analysis for {project_path}")
            final_state = await graph.ainvoke(initial_state, config=config)
            
            # Flush Langfuse after graph execution to ensure traces are sent
            try:
                from utils import flush_langfuse
                flush_langfuse()
            except Exception as flush_error:
                logger.warning(f"Failed to flush Langfuse after graph execution: {flush_error}")
            
            # Capture main trace ID from callback handler for token usage tracking
            main_trace_id = None
            if langfuse_handler:
                try:
                    # Try to get trace ID from callback handler
                    if hasattr(langfuse_handler, 'last_trace_id'):
                        main_trace_id = langfuse_handler.last_trace_id
                    elif hasattr(langfuse_handler, 'trace_id'):
                        main_trace_id = langfuse_handler.trace_id
                    elif hasattr(langfuse_handler, 'run_id'):
                        # Sometimes run_id is the trace ID
                        main_trace_id = langfuse_handler.run_id
                except Exception as trace_id_error:
                    logger.debug(f"Could not get trace ID from callback handler: {trace_id_error}")
            
            # Also try to get trace ID from Langfuse client
            if not main_trace_id:
                try:
                    from utils.langfuse.client import get_langfuse_client
                    from langfuse import get_client
                    langfuse_client = get_langfuse_client() or get_client()
                    if langfuse_client and hasattr(langfuse_client, 'get_current_trace_id'):
                        main_trace_id = langfuse_client.get_current_trace_id()
                except Exception as client_trace_error:
                    logger.debug(f"Could not get trace ID from Langfuse client: {client_trace_error}")
            
            # Store trace ID in final_state for synthesis node
            if main_trace_id:
                final_state["trace_id"] = main_trace_id
                logger.debug(f"Captured main LangGraph trace ID: {main_trace_id}")
            else:
                logger.debug("Could not capture main trace ID, will rely on state trace_id if available")
            
            # Check for errors
            if final_state.get("error"):
                error = final_state.get("error")
                error_stage = final_state.get("error_stage", "unknown")
                error_context = final_state.get("error_context", {})
                error_msg = (
                    f"Swarm analysis failed at stage '{error_stage}': {error}. "
                    f"Project: {final_state.get('project_path', 'unknown')}. "
                    f"Check the error context for additional details."
                )
                logger.error(error_msg, extra=error_context)
                raise Exception(error_msg)
            
            # Extract results in v2-compatible format
            result = self._extract_results(final_state)
            
            logger.info(f"Swarm analysis completed: {len(result.get('roles_selected', []))} roles")
            
            # Unregister callback after workflow completes
            if callback_id:
                try:
                    from workflows.callback_registry import get_callback_registry
                    registry = get_callback_registry()
                    registry.unregister(callback_id)
                    logger.debug(f"Unregistered stream callback: {callback_id}")
                except Exception as unreg_error:
                    logger.warning(f"Failed to unregister callback: {unreg_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in swarm analysis: {e}", exc_info=True)
            
            # Try to get callback for error reporting
            error_callback = None
            if callback_id:
                try:
                    from workflows.callback_registry import get_callback_registry
                    registry = get_callback_registry()
                    error_callback = registry.get(callback_id)
                except Exception:
                    pass
            
            # Fallback to direct callback if registry lookup failed
            if not error_callback:
                error_callback = stream_callback
            
            if error_callback:
                try:
                    error_callback("swarm_analysis_error", {
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as callback_error:
                    logger.warning(f"Failed to invoke error callback: {callback_error}")
            
            # Unregister callback even on error
            if callback_id:
                try:
                    from workflows.callback_registry import get_callback_registry
                    registry = get_callback_registry()
                    registry.unregister(callback_id)
                except Exception:
                    pass
            
            raise
    
    def _extract_results(self, state: SwarmAnalysisState) -> Dict[str, Any]:
        """
        Extract results from final state in v2-compatible format.
        
        Args:
            state: Final swarm analysis state
            
        Returns:
            Dictionary with analysis results in v2 format
        """
        # Extract role names
        role_names = state.get("role_names", [])
        if not role_names:
            selected_roles = state.get("selected_roles", [])
            role_names = [r["name"] if isinstance(r, dict) else str(r) for r in selected_roles]
        
        # Extract agent results
        agent_results = state.get("agent_results", {})
        individual_agent_results = state.get("individual_agent_results", agent_results)
        
        # Extract validation metadata
        validation_results = state.get("validation_results", {})
        validation_metadata = {
            "total_agents": len(role_names),
            "successful_agents": len([r for r in individual_agent_results.values() if "error" not in r]),
            "prompt_validations": validation_results
        }
        
        # Build result dictionary
        result = {
            "project_path": state.get("project_path"),
            "agent_name": "Swarm Analysis",
            "model_used": state.get("model_name", "unknown"),
            "goal": state.get("goal"),
            "architecture_model": state.get("architecture_model", {}),
            "roles_selected": role_names,
            "individual_agent_results": individual_agent_results,
            "synthesized_report": state.get("synthesized_report", ""),
            "files_scanned": len(state.get("files", [])),
            "chunks_analyzed": state.get("chunks_analyzed", 0),
            "validation_metadata": validation_metadata,
            "processing_errors": state.get("prompt_processing_errors", {})
        }
        
        return result

