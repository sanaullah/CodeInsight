"""
Metadata schema standardization for Langfuse traces.

Provides consistent metadata structure for Swarm Analysis traces to enable
Dynatrace-like filtering and querying in Langfuse.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SwarmAnalysisMetadata:
    """
    Standardized metadata schema for Swarm Analysis traces.
    
    Ensures all traces have consistent, queryable fields for filtering
    and correlation in Langfuse.
    """
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    project_path: str = ""
    goal: Optional[str] = None
    environment: str = "development"
    run_type: str = "swarm_analysis"
    app_version: Optional[str] = None
    start_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for Langfuse."""
        result = {
            "project_path": self.project_path,
            "environment": self.environment,
            "run_type": self.run_type,
        }
        
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.goal:
            result["goal"] = self.goal
        if self.app_version:
            result["app_version"] = self.app_version
        if self.start_time:
            result["start_time"] = self.start_time
            
        return result
    
    def to_langfuse_metadata(self) -> Dict[str, Any]:
        """
        Convert to Langfuse metadata format with langfuse_* prefixes.
        
        Returns metadata dict suitable for config["metadata"] in LangGraph.
        """
        result = self.to_dict()
        
        # Add langfuse_* prefixes for v3 compatibility
        if self.user_id:
            result["langfuse_user_id"] = self.user_id
        if self.session_id:
            result["langfuse_session_id"] = self.session_id
            
        return result


def normalize_metadata(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    project_path: Optional[str] = None,
    goal: Optional[str] = None,
    environment: Optional[str] = None,
    run_type: Optional[str] = None,
    app_version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> SwarmAnalysisMetadata:
    """
    Normalize metadata to ensure all required fields are present.
    
    Extracts values from kwargs and metadata dict, applies defaults,
    and returns standardized SwarmAnalysisMetadata object.
    
    Args:
        user_id: Optional user ID
        session_id: Optional session ID
        project_path: Project path (required if not in metadata)
        goal: Optional analysis goal
        environment: Environment name (default: "development")
        run_type: Run type (default: "swarm_analysis")
        app_version: Optional app version
        metadata: Optional metadata dictionary to extract values from
        
    Returns:
        SwarmAnalysisMetadata object with normalized values
        
    Raises:
        ValueError: If project_path is missing and cannot be extracted
    """
    # Extract from metadata dict if provided
    if metadata:
        user_id = user_id or metadata.get("user_id")
        session_id = session_id or metadata.get("session_id")
        project_path = project_path or metadata.get("project_path")
        goal = goal or metadata.get("goal")
        environment = environment or metadata.get("environment")
        run_type = run_type or metadata.get("run_type")
        app_version = app_version or metadata.get("app_version")
    
    # Apply defaults
    if not environment:
        environment = "development"
    if not run_type:
        run_type = "swarm_analysis"
    
    # Validate required fields
    if not project_path:
        raise ValueError("project_path is required for SwarmAnalysisMetadata")
    
    # Get start time
    start_time = datetime.now().isoformat()
    
    return SwarmAnalysisMetadata(
        user_id=user_id,
        session_id=session_id,
        project_path=project_path,
        goal=goal,
        environment=environment,
        run_type=run_type,
        app_version=app_version,
        start_time=start_time
    )

