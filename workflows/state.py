"""
State schema definitions for LangGraph.

This module provides utilities for defining state schemas using TypedDict,
which is the standard for LangGraph state management.
"""

from typing import TypedDict, Any, Dict, List, Optional
from typing_extensions import NotRequired


class StateSchema(TypedDict):
    """
    Base state schema for LangGraph workflows.
    
    Extend this class to create custom state schemas for your workflows.
    """
    messages: NotRequired[List[Dict[str, Any]]]
    next: NotRequired[str]
    error: NotRequired[Optional[str]]


class ProjectAnalysisState(TypedDict):
    """
    State schema for project analysis workflows.
    
    Extends base StateSchema with project analysis specific fields.
    """
    # Base fields
    messages: NotRequired[List[Dict[str, Any]]]
    next: NotRequired[str]
    error: NotRequired[Optional[str]]
    
    # Project analysis fields
    project_path: NotRequired[str]
    files: NotRequired[List[Dict[str, Any]]]  # List of FileInfo as dicts
    chunks: NotRequired[List[Dict[str, Any]]]  # List of Chunk as dicts
    analysis_results: NotRequired[List[Dict[str, Any]]]  # List of analysis results
    synthesized_report: NotRequired[str]
    current_chunk_index: NotRequired[int]
    total_chunks: NotRequired[int]
    model: NotRequired[str]
    agent_name: NotRequired[str]


def create_state_schema(**fields: Any) -> type:
    """
    Create a custom state schema from keyword arguments.
    
    Args:
        **fields: Field definitions for the state schema
        
    Returns:
        A TypedDict class for the state schema
        
    Example:
        MyState = create_state_schema(
            user_input=str,
            result=NotRequired[str],
            step_count=int
        )
    """
    return TypedDict("StateSchema", fields)

