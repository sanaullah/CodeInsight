"""Input validation utilities for workflow nodes."""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_project_path(project_path: Optional[str]) -> Path:
    """
    Validate and normalize project path.
    
    Args:
        project_path: Project path string
        
    Returns:
        Path object
        
    Raises:
        ValueError: If path is invalid or doesn't exist
    """
    if not project_path:
        raise ValueError("project_path is required")
    
    path = Path(project_path)
    if not path.exists():
        raise ValueError(f"Project path does not exist: {project_path}")
    if not path.is_dir():
        raise ValueError(f"Project path is not a directory: {project_path}")
    
    return path


def validate_state_fields(
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
    missing = [field for field in required_fields if field not in state or state[field] is None]
    if missing:
        raise ValueError(
            f"{operation} requires the following fields: {', '.join(missing)}"
        )


def validate_file_extensions(file_extensions: Optional[List[str]]) -> List[str]:
    """
    Validate and normalize file extensions.
    
    Args:
        file_extensions: List of file extensions (with or without leading dot)
        
    Returns:
        Normalized list of file extensions with leading dots
    """
    if not file_extensions:
        return []
    
    normalized = []
    for ext in file_extensions:
        if not ext.startswith('.'):
            ext = f'.{ext}'
        normalized.append(ext.lower())
    
    return normalized

