"""
Chunking nodes for agent execution in Swarm Analysis workflow.

DEPRECATED: These nodes were part of a parallel chunk routing implementation that was never fully
integrated and has been removed. Chunks are now processed sequentially within execute_single_agent_node.

This file is kept for reference only. The synthesize_chunk_results function is still used by
the sequential chunk processing implementation.
"""

import logging
from typing import Dict, Any, Optional

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from workflows.nodes import llm_node
from workflows.swarm_analysis.tools import get_tool_definitions
from workflows.swarm_analysis.agent_execution.nodes import _execute_with_tool_calling
from utils.model_capabilities import check_tool_calling_support
from .content_utils import (
    prepare_chunk_content,
    validate_chunk_content,
    build_chunk_analysis_message,
    synthesize_chunk_results
)
from .nodes import build_agent_system_prompt

logger = logging.getLogger(__name__)


@validate_state_fields(["agent_info", "chunks"], "dispatch_chunks")
def dispatch_chunks_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    DEPRECATED: This node is no longer used. Chunks are processed sequentially within execute_single_agent_node.
    
    Dispatch chunks for parallel analysis.
    
    Creates chunk metadata and mapping for result correlation.
    Send objects are created in the conditional edge function.
    
    Args:
        state: State containing agent_info, chunks, and configuration
        
    Returns:
        State with chunk_agent_mapping for result correlation
    """
    agent_info = state.get(StateKeys.AGENT_INFO)
    chunks = state.get(StateKeys.CHUNKS, [])
    
    if not agent_info or not chunks:
        state[StateKeys.ERROR] = "agent_info and chunks are required for chunk dispatch"
        state[StateKeys.ERROR_STAGE] = "dispatch_chunks"
        return state
    
    role_name = agent_info.get("role", "Unknown")
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    if stream_callback:
        stream_callback("chunks_dispatched", {
            "agent": role_name,
            "chunk_count": len(chunks)
        })
    
    try:
        # Create chunk_agent_mapping for result correlation
        chunk_agent_mapping = {}
        for chunk_idx in range(len(chunks)):
            chunk_id = f"{role_name}_chunk_{chunk_idx}"
            chunk_agent_mapping[chunk_id] = role_name
        
        # Update state with mapping
        updated_state = state.copy()
        updated_state[StateKeys.CHUNK_AGENT_MAPPING] = chunk_agent_mapping
        
        logger.info(f"Dispatched {len(chunks)} chunks for agent {role_name}")
        return updated_state
        
    except Exception as e:
        logger.error(f"Error dispatching chunks: {e}", exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = str(e)
        updated_state[StateKeys.ERROR_STAGE] = "dispatch_chunks"
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": str(e)})
        return updated_state


@validate_state_fields(["chunk_info"], "analyze_single_chunk")
def analyze_single_chunk_node(state: SwarmAnalysisState, config: Optional[Dict[str, Any]] = None) -> SwarmAnalysisState:
    """
    DEPRECATED: This node is no longer used. Chunks are processed sequentially within execute_single_agent_node.
    
    Analyze a single chunk.
    
    Designed to be called in parallel via Send from dispatch_chunks_node.
    
    Args:
        state: State containing chunk_info and all necessary context
        config: Optional LangGraph config
        
    Returns:
        Updated state with chunk result in chunk_results_list
    """
    # Extract chunk info from state (set by Send)
    chunk_info = state.get(StateKeys.CHUNK_INFO)
    if not chunk_info:
        # Return error result in chunk_results_list
        error_result = {
            "chunk_id": "unknown",
            "role_name": "Unknown",
            "error": "chunk_info is required",
            "status": "error"
        }
        return {
            "chunk_results_list": [error_result]
        }
    
    chunk = chunk_info.get("chunk")
    chunk_id = chunk_info.get("chunk_id", "unknown")
    chunk_index = chunk_info.get("chunk_index", 0)
    chunk_num = chunk_info.get("chunk_num", 1)
    total_chunks = chunk_info.get("total_chunks", 1)
    
    # Get agent and configuration info
    agent_info = state.get(StateKeys.AGENT_INFO)
    role_name = state.get(StateKeys.ROLE_NAME) or (agent_info.get("role", "Unknown") if agent_info else "Unknown")
    prompt = agent_info.get("prompt", "") if agent_info else ""
    project_path = state.get(StateKeys.PROJECT_PATH)
    files = state.get(StateKeys.FILES, [])
    model_name = state.get(StateKeys.MODEL_NAME)
    architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL)
    architecture_hash = state.get(StateKeys.ARCHITECTURE_HASH)
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    # Get Langfuse prompt ID for this role
    langfuse_prompt_ids = state.get(StateKeys.LANGFUSE_PROMPT_IDS, {})
    langfuse_prompt_id = langfuse_prompt_ids.get(role_name)
    
    if stream_callback:
        stream_callback("chunk_analysis", {
            "agent": role_name,
            "chunk_num": chunk_num,
            "total_chunks": total_chunks
        })
    
    try:
        # Prepare chunk content
        chunk_content = prepare_chunk_content(chunk, files)
        
        # Validate chunk content and log metrics
        metrics = validate_chunk_content(chunk_content, chunk_num, total_chunks, role_name)
        logger.debug(f"Chunk metrics for {role_name}: {metrics}")
        
        # Check if tool calling is enabled and model supports it
        enable_tool_calling = state.get(StateKeys.ENABLE_TOOL_CALLING, True)
        max_tool_calls = state.get(StateKeys.MAX_TOOL_CALLS, 10)
        tools_available = False
        tools = None
        
        if enable_tool_calling:
            tools_available = check_tool_calling_support(model_name)
            if tools_available:
                tools = get_tool_definitions()
                logger.debug(f"Tool calling enabled for chunk analysis (model: {model_name})")
        
        # Build analysis message
        user_message = build_chunk_analysis_message(
            chunk_content=chunk_content,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
            prompt=prompt,
            architecture_model=architecture_model,
            tools_available=tools_available
        )
        
        # Use llm_node for chunk analysis
        llm_state = {
            "messages": [
                {"role": "system", "content": build_agent_system_prompt(role_name, architecture_hash)},
                {"role": "user", "content": user_message}
            ],
            "model": model_name,
            "temperature": 0.7,
            "metadata": {
                "role": role_name,
                "chunk_num": chunk_num,
                "total_chunks": total_chunks,
                "langfuse_prompt_id": langfuse_prompt_id  # Always include, even if None
            }
        }
        
        # Add tools if available
        if tools_available and tools:
            llm_state["tools"] = tools
            llm_state["tool_choice"] = "auto"
        
        # Call llm_node with tool calling support
        if tools_available and tools:
            llm_result = _execute_with_tool_calling(
                llm_state=llm_state,
                project_path=project_path,
                state=state,
                config=config,
                max_tool_calls=max_tool_calls
            )
        else:
            llm_result = llm_node(llm_state, config=config)
        
        # Extract analysis result
        analysis_text = llm_result.get("last_response", "")
        
        # Get chunk metadata
        files_in_chunk = chunk.total_files if hasattr(chunk, 'total_files') else len(chunk.files) if hasattr(chunk, 'files') else 0
        tokens_in_chunk = chunk.total_tokens if hasattr(chunk, 'total_tokens') else 0
        
        chunk_result = {
            "chunk_id": chunk_id,
            "role_name": role_name,
            "chunk_index": chunk_index,
            "chunk_num": chunk_num,
            "total_chunks": total_chunks,
            "files_in_chunk": files_in_chunk,
            "tokens_in_chunk": tokens_in_chunk,
            "analysis": analysis_text
        }
        
        logger.info(f"Completed chunk analysis: {chunk_id} for {role_name}")
        
        # Return ONLY the fields that should be updated (chunk_results_list)
        # The reducer (operator.add) will aggregate chunk_results_list from all parallel nodes
        return {
            "chunk_results_list": [chunk_result]
        }
        
    except Exception as e:
        logger.error(f"Error analyzing chunk {chunk_id} for {role_name}: {e}", exc_info=True)
        # Return error result in chunk_results_list
        error_result = {
            "chunk_id": chunk_id,
            "role_name": role_name,
            "chunk_index": chunk_index,
            "chunk_num": chunk_num,
            "total_chunks": total_chunks,
            "error": str(e),
            "status": "error"
        }
        return {
            "chunk_results_list": [error_result]
        }


async def collect_chunk_results_node(state: SwarmAnalysisState, config: Optional[Dict[str, Any]] = None) -> SwarmAnalysisState:
    """
    DEPRECATED: This node is no longer used. Chunks are processed sequentially within execute_single_agent_node.
    
    Collect chunk results for an agent and synthesize.
    
    Aggregates chunk_results_list, groups by role_name, synthesizes chunks,
    and returns agent result in agent_results_list.
    
    Args:
        state: State with chunk_results_list and chunk_agent_mapping
        config: Optional LangGraph config
        
    Returns:
        Updated state with agent result in agent_results_list
    """
    chunk_results_list = state.get(StateKeys.CHUNK_RESULTS_LIST, [])
    chunk_agent_mapping = state.get(StateKeys.CHUNK_AGENT_MAPPING, {})
    agent_info = state.get(StateKeys.AGENT_INFO)
    
    if not chunk_results_list:
        # No chunk results - this shouldn't happen, but handle gracefully
        role_name = agent_info.get("role", "Unknown") if agent_info else "Unknown"
        error_result = {
            "role_name": role_name,
            "error": "No chunk results collected",
            "status": "error"
        }
        return {
            "agent_results_list": [error_result]
        }
    
    # Get role_name from first chunk result or agent_info
    role_name = None
    if chunk_results_list:
        role_name = chunk_results_list[0].get("role_name")
    
    if not role_name and agent_info:
        role_name = agent_info.get("role", "Unknown")
    
    if not role_name:
        role_name = "Unknown"
    
    # Get configuration
    prompt = agent_info.get("prompt", "") if agent_info else ""
    model_name = state.get(StateKeys.MODEL_NAME)
    architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL)
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    # Get Langfuse prompt ID for this role
    langfuse_prompt_ids = state.get(StateKeys.LANGFUSE_PROMPT_IDS, {})
    langfuse_prompt_id = langfuse_prompt_ids.get(role_name)
    
    try:
        # Filter chunk results for this agent (by role_name)
        agent_chunk_results = [
            r for r in chunk_results_list
            if r.get("role_name") == role_name and r.get("status") != "error"
        ]
        
        # Check for errors
        error_chunks = [
            r for r in chunk_results_list
            if r.get("role_name") == role_name and r.get("status") == "error"
        ]
        
        if not agent_chunk_results:
            # All chunks failed or no results
            error_messages = [r.get("error", "Unknown error") for r in error_chunks]
            error_result = {
                "role_name": role_name,
                "error": f"All chunks failed: {', '.join(error_messages)}",
                "status": "error"
            }
            return {
                "agent_results_list": [error_result]
            }
        
        # Sort chunks by chunk_num for proper ordering
        agent_chunk_results.sort(key=lambda x: x.get("chunk_num", 0))
        
        # Synthesize chunk results
        if len(agent_chunk_results) > 1:
            if stream_callback:
                stream_callback("synthesizing_chunks", {
                    "agent": role_name,
                    "chunk_count": len(agent_chunk_results)
                })
            
            synthesized_report = await synthesize_chunk_results(
                chunk_results=agent_chunk_results,
                role_name=role_name,
                prompt=prompt,
                model_name=model_name,
                architecture_model=architecture_model,
                langfuse_prompt_id=langfuse_prompt_id,
                config=config
            )
        else:
            # Single chunk - use analysis directly
            synthesized_report = agent_chunk_results[0].get("analysis", "No analysis available")
        
        # Calculate total chunks analyzed
        chunks_analyzed = len(agent_chunk_results)
        
        # Create agent result
        agent_result = {
            "role_name": role_name,
            "synthesized_report": synthesized_report,
            "status": "completed",
            "chunks_analyzed": chunks_analyzed,
            "chunk_results": agent_chunk_results
        }
        
        if stream_callback:
            stream_callback("chunks_collected", {
                "agent": role_name,
                "chunk_count": chunks_analyzed
            })
        
        logger.info(f"Collected and synthesized {chunks_analyzed} chunks for agent {role_name}")
        
        # Return ONLY the fields that should be updated (agent_results_list)
        # The reducer (operator.add) will aggregate agent_results_list from all parallel nodes
        return {
            "agent_results_list": [agent_result]
        }
        
    except Exception as e:
        logger.error(f"Error collecting chunk results for {role_name}: {e}", exc_info=True)
        # Return error result in agent_results_list
        error_result = {
            "role_name": role_name,
            "error": str(e),
            "status": "error"
        }
        return {
            "agent_results_list": [error_result]
        }



