"""
Langfuse helpers for Swarm Analysis.

Provides utilities for creating metadata and enriching traces
for swarm analysis workflows.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


# Phase mapping for all workflow nodes
SWARM_ANALYSIS_PHASES = {
    # Discovery Phase
    "scan_project": "discovery",
    "build_architecture": "discovery",
    "load_previous_reports": "discovery",
    
    # Planning Phase
    "select_roles": "planning",
    "dispatch_prompt_generation": "planning",
    "generate_single_prompt": "planning",
    "collect_generated_prompts": "planning",
    "dispatch_prompt_validation": "planning",
    "validate_single_prompt": "planning",
    "collect_validation_results": "planning",
    "store_prompts": "planning",
    
    # Execution Phase
    "spawn_agents": "execution",
    "dispatch_agents": "execution",
    "select_agent_files": "execution",
    "execute_single_agent": "execution",
    "collect_results": "execution",
    
    # Reporting Phase
    "synthesize_results": "reporting"
}


def get_node_phase(node_name: str) -> str:
    """
    Get the analysis phase for a given node name.
    
    Args:
        node_name: Name of the workflow node
        
    Returns:
        Phase name: "discovery", "planning", "execution", "reporting", or "processing" (fallback)
    """
    return SWARM_ANALYSIS_PHASES.get(node_name, "processing")


def create_swarm_trace_metadata(
    project_path: str,
    goal: Optional[str] = None,
    architecture_model: Optional[Dict[str, Any]] = None,
    roles_selected: Optional[List[str]] = None,
    max_agents: int = 10
) -> Dict[str, Any]:
    """
    Create metadata dictionary for swarm analysis traces.
    
    Args:
        project_path: Path to analyzed project
        goal: Optional analysis goal
        architecture_model: Optional architecture model dictionary
        roles_selected: Optional list of selected roles
        max_agents: Maximum number of agents
        
    Returns:
        Metadata dictionary for Langfuse trace
    """
    metadata = {
        "workflow_type": "swarm_analysis",
        "project_path": project_path,
        "max_agents": max_agents,
        "timestamp": datetime.now().isoformat()
    }
    
    if goal:
        metadata["goal"] = goal
    
    if architecture_model:
        metadata["architecture_system_name"] = architecture_model.get("system_name", "Unknown")
        metadata["architecture_system_type"] = architecture_model.get("system_type", "Unknown")
        metadata["architecture_pattern"] = architecture_model.get("architecture_pattern", "Unknown")
    
    if roles_selected:
        metadata["roles_selected"] = roles_selected
        metadata["role_count"] = len(roles_selected)
    
    return metadata


def enrich_node_metadata(
    node_name: str,
    state: Dict[str, Any],
    base_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enrich metadata for a specific node execution.
    
    Args:
        node_name: Name of the node
        state: Current state dictionary
        base_metadata: Optional base metadata to merge
        
    Returns:
        Enriched metadata dictionary
    """
    metadata = base_metadata.copy() if base_metadata else {}
    
    # Add node-specific metadata
    metadata["node_name"] = node_name
    metadata["stage"] = node_name
    metadata["phase"] = get_node_phase(node_name)  # Add phase tag
    
    # Add state information
    if "project_path" in state:
        metadata["project_path"] = state[StateKeys.PROJECT_PATH]
    
    if "files" in state:
        metadata["file_count"] = len(state[StateKeys.FILES])
    
    if "architecture_model" in state:
        arch = state[StateKeys.ARCHITECTURE_MODEL]
        if arch:
            metadata["architecture_type"] = arch.get("system_type", "Unknown")
    
    if "role_names" in state:
        metadata["roles_count"] = len(state[StateKeys.ROLE_NAMES])
    
    if "agent_results" in state:
        agent_results = state[StateKeys.AGENT_RESULTS]
        successful = len([r for r in agent_results.values() if "error" not in r])
        metadata["successful_agents"] = successful
        metadata["total_agents"] = len(agent_results)
    
    return metadata


def create_learning_observation(
    operation_name: str,
    experience_id: Optional[str] = None,
    learnings: Optional[Dict[str, Any]] = None,
    parent_trace: Optional[Any] = None
) -> Optional[Any]:
    """
    Create a Langfuse observation for learning events.
    
    Args:
        operation_name: Name of the learning operation
        experience_id: Optional experience ID
        learnings: Optional learnings dictionary
        parent_trace: Optional parent trace
        
    Returns:
        Langfuse observation or None
    """
    try:
        from utils.langfuse_integration import create_observation
        
        input_data = {}
        if experience_id:
            input_data["experience_id"] = experience_id
        if learnings:
            input_data["learnings"] = learnings
        
        metadata = {
            "operation": "learning",
            "operation_name": operation_name
        }
        
        observation = create_observation(
            name=f"learning_{operation_name}",
            parent_trace=parent_trace,
            input_data=input_data,
            metadata=metadata
        )
        
        return observation
        
    except Exception as e:
        logger.warning(f"Failed to create learning observation: {e}")
        return None

