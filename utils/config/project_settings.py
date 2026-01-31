"""
Project-specific settings management for CodeInsight.

Provides helper functions for storing and retrieving project-specific settings,
such as selected directories for swarm analysis.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from .settings_db import get_setting, set_setting

logger = logging.getLogger(__name__)


def get_project_key(project_path: str) -> str:
    """
    Generate a normalized key for project-specific settings.
    
    Uses MD5 hash of the resolved absolute path to create a stable,
    project-unique key that works across different path representations.
    
    Args:
        project_path: Path to the project directory (absolute or relative)
    
    Returns:
        Normalized key string in format: swarm_selected_dirs_{16_char_hash}
    """
    try:
        # Resolve to absolute path and normalize
        normalized_path = str(Path(project_path).resolve())
        
        # Create MD5 hash (16 characters for reasonable uniqueness)
        path_hash = hashlib.md5(normalized_path.encode()).hexdigest()[:16]
        
        key = f"swarm_selected_dirs_{path_hash}"
        logger.debug(f"Generated project key for {project_path}: {key}")
        return key
        
    except Exception as e:
        logger.error(f"Error generating project key for {project_path}: {e}", exc_info=True)
        # Fallback to a simple hash if resolution fails
        path_hash = hashlib.md5(str(project_path).encode()).hexdigest()[:16]
        return f"swarm_selected_dirs_{path_hash}"


def get_project_selected_directories(user_id: str, project_path: str) -> List[str]:
    """
    Retrieve saved selected directories for a project.
    
    Args:
        user_id: User identifier
        project_path: Path to the project directory
    
    Returns:
        List of selected directory paths (relative to project_root)
        Returns empty list if no settings found or on error
    """
    try:
        key = get_project_key(project_path)
        directories = get_setting(user_id, key, default=[])
        
        # Ensure we return a list
        if directories is None:
            return []
        
        if isinstance(directories, list):
            # Validate that all items are strings
            return [str(d) for d in directories if d]
        else:
            logger.warning(f"Invalid directory format for project {project_path}: {type(directories)}")
            return []
            
    except Exception as e:
        logger.error(f"Error getting selected directories for project {project_path}: {e}", exc_info=True)
        return []


def save_project_selected_directories(
    user_id: str, 
    project_path: str, 
    directories: List[str]
) -> bool:
    """
    Save selected directories for a project.
    
    Args:
        user_id: User identifier
        project_path: Path to the project directory
        directories: List of directory paths (relative to project_root)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Normalize directories: ensure they're strings and filter empty
        normalized_dirs = [str(d).strip() for d in directories if d and str(d).strip()]
        
        # Normalize path separators to forward slashes for consistency
        normalized_dirs = [d.replace("\\", "/") for d in normalized_dirs]
        
        key = get_project_key(project_path)
        success = set_setting(user_id, key, normalized_dirs)
        
        if success:
            logger.info(f"Saved {len(normalized_dirs)} selected directories for project {project_path}")
        else:
            logger.warning(f"Failed to save selected directories for project {project_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving selected directories for project {project_path}: {e}", exc_info=True)
        return False

