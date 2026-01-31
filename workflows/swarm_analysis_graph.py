"""
Swarm Analysis LangGraph workflow.

Connects all swarm analysis nodes into a complete workflow.
"""

import logging
from typing import Dict, Any, Optional

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.state_keys import StateKeys
from workflows.swarm_analysis_nodes import (
    scan_project_node,
    build_architecture_node,
    select_roles_node,
    dispatch_prompt_generation_node,
    generate_single_prompt_node,
    collect_generated_prompts_node,
    dispatch_prompt_validation_node,
    validate_single_prompt_node,
    collect_validation_results_node,
    store_prompts_node,
    spawn_agents_node,
    dispatch_agents_node,
    execute_single_agent_node,
    collect_agent_results_node,
    synthesize_results_node,
)
from workflows.swarm_analysis.agent_execution.file_selection import select_agent_files_node
from workflows.swarm_analysis.previous_reports import load_previous_reports_node
from workflows.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)


def create_swarm_analysis_graph(
    trace_name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Create and compile the Swarm Analysis LangGraph.
    
    Args:
        trace_name: Optional name for Langfuse trace
        user_id: Optional user ID for trace correlation
        session_id: Optional session ID for trace correlation
        metadata: Optional metadata for trace correlation
        
    Returns:
        Compiled LangGraph instance
    """
    # Define all nodes
    nodes = {
        "scan_project": scan_project_node,
        "build_architecture": build_architecture_node,
        "load_previous_reports": load_previous_reports_node,
        "select_roles": select_roles_node,
        "dispatch_prompt_generation": dispatch_prompt_generation_node,
        "generate_single_prompt": generate_single_prompt_node,
        "collect_generated_prompts": collect_generated_prompts_node,
        "dispatch_prompt_validation": dispatch_prompt_validation_node,
        "validate_single_prompt": validate_single_prompt_node,
        "collect_validation_results": collect_validation_results_node,
        "store_prompts": store_prompts_node,
        "spawn_agents": spawn_agents_node,
        "dispatch_agents": dispatch_agents_node,
        "select_agent_files": select_agent_files_node,
        "execute_single_agent": execute_single_agent_node,
        "collect_results": collect_agent_results_node,
        "synthesize_results": synthesize_results_node,
    }
    
    # Define edges (parallel execution flow)
    # Note: dispatch_prompt_generation -> generate_single_prompt, dispatch_prompt_validation -> validate_single_prompt,
    # dispatch_agents -> execute_single_agent are handled via conditional edges with Send
    edges = [
        ("scan_project", "build_architecture"),
        ("build_architecture", "load_previous_reports"),
        ("load_previous_reports", "select_roles"),
        ("select_roles", "dispatch_prompt_generation"),
        ("collect_generated_prompts", "dispatch_prompt_validation"),
        ("collect_validation_results", "store_prompts"),
        ("store_prompts", "spawn_agents"),
        ("spawn_agents", "dispatch_agents"),
        # execute_single_agent -> collect_results (conditional)
        ("collect_results", "synthesize_results"),
        ("synthesize_results", "END"),
    ]
    
    # Create graph builder
    graph_builder = GraphBuilder()
    
    # Create graph with Langfuse integration
    # Note: We need to manually add the conditional edge for dispatch_agents -> execute_single_agent
    # because it uses Send for parallel execution
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        try:
            from langgraph_core.graph import StateGraph, END
        except ImportError:
            raise ImportError("LangGraph not installed. Install with: pip install langgraph>=0.2.0")
    
    # Create state graph
    graph = StateGraph(SwarmAnalysisState)
    
    # Add all nodes
    for node_name, node_func in nodes.items():
        graph.add_node(node_name, node_func)
    
    # Set entry point
    graph.set_entry_point("scan_project")
    
    # Add regular edges
    for edge in edges:
        if len(edge) == 2:
            from_node, to_node = edge
            if to_node == "END":
                graph.add_edge(from_node, END)
            else:
                graph.add_edge(from_node, to_node)
    
    # Add conditional edge for dispatch_prompt_generation -> generate_single_prompt (uses Send)
    def route_after_dispatch_prompts(state: SwarmAnalysisState):
        """Route after dispatch - creates Send objects for parallel prompt generation."""
        # Check for errors first
        if state.get(StateKeys.ERROR):
            return "END"
        
        # Get roles from state (set by select_roles_node)
        selected_roles = state.get(StateKeys.SELECTED_ROLES, [])
        if not selected_roles:
            return "END"
        
        # Create Send objects for each role
        from langgraph.types import Send
        
        send_commands = []
        from workflows.swarm_analysis_send_builder import build_prompt_send_payload
        
        for role_obj in selected_roles:
            # Use builder function to ensure all required fields are included
            prompt_state = build_prompt_send_payload(state, role_obj)
            
            # Create Send command targeting generate_single_prompt_node
            send_commands.append(Send("generate_single_prompt", prompt_state))
        
        # Guard against empty list - return END if no valid commands
        if not send_commands:
            return "END"
        
        # Return list of Send objects for parallel execution
        return send_commands
    
    graph.add_conditional_edges(
        "dispatch_prompt_generation",
        route_after_dispatch_prompts,  # Returns List[Send] for parallel execution or "END" for errors
        {
            "generate_single_prompt": "generate_single_prompt",  # Route Send targets here
            "END": END  # Route errors to end
        }
    )
    
    # Add conditional edge for generate_single_prompt -> collect_generated_prompts
    def route_after_prompt_generation(state: SwarmAnalysisState):
        """Route after prompt generation - check for errors."""
        # Check for errors
        if state.get(StateKeys.ERROR):
            return "END"
        return "collect_generated_prompts"
        
    graph.add_conditional_edges(
        "generate_single_prompt",
        route_after_prompt_generation,
        {
            "collect_generated_prompts": "collect_generated_prompts",
            "END": END
        }
    )
    
    # Add conditional edge for dispatch_prompt_validation -> validate_single_prompt (uses Send)
    def route_after_dispatch_validation(state: SwarmAnalysisState):
        """Route after dispatch - creates Send objects for parallel prompt validation."""
        # Check for errors first
        if state.get(StateKeys.ERROR):
            return "END"
        
        # Get prompts from state (set by collect_generated_prompts_node)
        generated_prompts = state.get(StateKeys.GENERATED_PROMPTS, {})
        if not generated_prompts:
            return "END"
        
        # Create Send objects for each prompt
        from langgraph.types import Send
        
        send_commands = []
        from workflows.swarm_analysis_send_builder import build_validation_send_payload
        
        for role_name, prompt in generated_prompts.items():
            # Skip error prompts
            if prompt.startswith("[ERROR:"):
                continue
            
            # Use builder function to ensure all required fields are included
            validation_state = build_validation_send_payload(state, role_name, prompt)
            
            # Create Send command targeting validate_single_prompt_node
            send_commands.append(Send("validate_single_prompt", validation_state))
        
        # Guard against empty list - return END if no valid prompts to validate
        if not send_commands:
            return "END"
        
        # Return list of Send objects for parallel execution
        return send_commands
    
    graph.add_conditional_edges(
        "dispatch_prompt_validation",
        route_after_dispatch_validation,  # Returns List[Send] for parallel execution or "END" for errors
        {
            "validate_single_prompt": "validate_single_prompt",  # Route Send targets here
            "END": END  # Route errors to end
        }
    )
    
    # Add conditional edge for validate_single_prompt -> collect_validation_results
    def route_after_prompt_validation(state: SwarmAnalysisState):
        """Route after prompt validation - check for errors."""
        # Check for errors
        if state.get(StateKeys.ERROR):
            return "END"
        return "collect_validation_results"

    graph.add_conditional_edges(
        "validate_single_prompt",
        route_after_prompt_validation,
        {
            "collect_validation_results": "collect_validation_results",
            "END": END
        }
    )
    
    # Add conditional edge for dispatch_agents -> select_agent_files or execute_single_agent (uses Send)
    # The conditional edge function creates Send objects based on state from dispatch_agents_node
    def route_after_dispatch(state: SwarmAnalysisState):
        """Route after dispatch - creates Send objects for parallel execution."""
        # Check for errors first
        if state.get(StateKeys.ERROR):
            return "END"
        
        # Get agents from state (set by dispatch_agents_node)
        agents = state.get(StateKeys.AGENTS, [])
        if not agents:
            return "END"
        
        # Check if dynamic file selection is enabled
        enable_dynamic_file_selection = state.get(StateKeys.ENABLE_DYNAMIC_FILE_SELECTION, False)
        
        # Create Send objects for each agent
        from langgraph.types import Send
        
        send_commands = []
        from workflows.swarm_analysis_send_builder import build_agent_send_payload
        
        for agent_info in agents:
            # Use builder function to ensure all required fields are included
            agent_state = build_agent_send_payload(state, agent_info)
            
            if enable_dynamic_file_selection:
                # Route to file selection first
                send_commands.append(Send("select_agent_files", agent_state))
            else:
                # Route directly to execution (current behavior)
                send_commands.append(Send("execute_single_agent", agent_state))
        
        # Guard against empty list - return END if no valid agents to execute
        if not send_commands:
            return "END"
        
        # Return list of Send objects for parallel execution
        return send_commands
    
    graph.add_conditional_edges(
        "dispatch_agents",
        route_after_dispatch,  # Returns List[Send] for parallel execution or "END" for errors
        {
            "select_agent_files": "select_agent_files",  # Route Send targets here if file selection enabled
            "execute_single_agent": "execute_single_agent",  # Route Send targets here if file selection disabled
            "END": END  # Route errors to end
        }
    )
    
    # Add edge from select_agent_files to execute_single_agent
    # All parallel file selection nodes converge to execute_single_agent
    graph.add_edge("select_agent_files", "execute_single_agent")
    
    # Add conditional edge for execute_single_agent -> collect_results
    def route_after_agent_execution(state: SwarmAnalysisState):
        """Route after agent execution - always go to collect_results."""
        # Check for errors first
        if state.get(StateKeys.ERROR):
            return "END"
        
        # Chunks are processed sequentially within execute_single_agent_node
        # Always route to collect_results
        return "collect_results"
    
    graph.add_conditional_edges(
        "execute_single_agent",
        route_after_agent_execution,
        {
            "collect_results": "collect_results",
            "END": END
        }
    )
    
    # Compile graph using graph builder's compilation logic
    if graph_builder.checkpoint_adapter:
        compiled_graph = graph.compile(checkpointer=graph_builder.checkpoint_adapter)
    else:
        compiled_graph = graph.compile()
    
    # Return compiled graph directly - callbacks are passed in config when invoking
    logger.info("Swarm Analysis graph created successfully with parallel execution")
    logger.info("Langfuse callbacks should be passed in config when invoking graph")
    return compiled_graph


def should_continue(state: SwarmAnalysisState) -> str:
    """
    Conditional edge function to determine next step.
    
    Checks for errors and routes accordingly.
    
    Args:
        state: Current state
        
    Returns:
        Next node name or "END"
    """
    if state.get(StateKeys.ERROR):
        logger.error(f"Error in swarm analysis: {state.get(StateKeys.ERROR)} at stage {state.get(StateKeys.ERROR_STAGE)}")
        return "END"
    
    # Default: continue to next node
    return "continue"


def error_handler_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Handle errors in the workflow.
    
    Args:
        state: Current state with error
        
    Returns:
        Updated state with error information
    """
    error = state.get(StateKeys.ERROR)
    error_stage = state.get(StateKeys.ERROR_STAGE, "unknown")
    
    logger.error(f"Swarm analysis error at {error_stage}: {error}")
    
    # Stream callback for error
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback("swarm_analysis_error", {
            "error": error,
            "stage": error_stage
        })
    
    return state

