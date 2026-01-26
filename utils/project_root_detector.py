"""
Project Root Detection Utility.

Detects the project root directory from any given path by looking for
common project indicators like .git, setup.py, pyproject.toml, etc.
"""

import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


# Common project root indicators
PROJECT_ROOT_INDICATORS = [
    ".git",                    # Git repository
    "setup.py",                # Python setup file
    "pyproject.toml",          # Modern Python project config
    "requirements.txt",        # Python dependencies
    "Pipfile",                 # Pipenv dependencies
    "poetry.lock",             # Poetry lock file
    "package.json",            # Node.js (for mixed projects)
    "README.md",               # Common documentation
    "README.rst",              # Alternative documentation
    ".gitignore",              # Git ignore file
    "setup.cfg",               # Setup configuration
    "MANIFEST.in",             # Python manifest
]


def find_project_indicators(path: Path) -> List[str]:
    """
    Find project root indicators in a directory.
    
    Args:
        path: Path to check
        
    Returns:
        List of found indicator names
    """
    if not path.exists() or not path.is_dir():
        return []
    
    found_indicators = []
    for indicator in PROJECT_ROOT_INDICATORS:
        indicator_path = path / indicator
        if indicator_path.exists():
            found_indicators.append(indicator)
    
    return found_indicators


def is_project_root(path: Path) -> bool:
    """
    Check if a path is a project root by looking for indicators.
    
    Args:
        path: Path to check
        
    Returns:
        True if path appears to be a project root
    """
    if not path.exists() or not path.is_dir():
        return False
    
    indicators = find_project_indicators(path)
    # Consider it a project root if at least one strong indicator is found
    strong_indicators = [".git", "setup.py", "pyproject.toml", "requirements.txt", "Pipfile"]
    has_strong_indicator = any(ind in indicators for ind in strong_indicators)
    
    # Or if multiple indicators are found (even if not "strong")
    return has_strong_indicator or len(indicators) >= 2


def detect_project_root(path: str) -> Optional[Path]:
    """
    Detect project root directory from any given path.
    
    Walks up the directory tree from the given path until it finds
    a directory that appears to be a project root, or reaches the
    filesystem root.
    
    Args:
        path: Starting path (file or directory)
        
    Returns:
        Path to project root, or None if not found
    """
    try:
        start_path = Path(path).resolve()
        
        # If it's a file, start from its parent directory
        if start_path.is_file():
            current = start_path.parent
        elif start_path.is_dir():
            current = start_path
        else:
            logger.warning(f"Path does not exist: {path}")
            return None
        
        # Walk up the directory tree
        max_depth = 20  # Prevent infinite loops
        depth = 0
        
        while current != current.parent and depth < max_depth:
            if is_project_root(current):
                logger.info(f"Detected project root: {current}")
                return current
            
            current = current.parent
            depth += 1
        
        # If we reached max depth or filesystem root without finding indicators,
        # return the original directory if it exists
        if start_path.is_dir() and start_path.exists():
            logger.warning(f"No project root indicators found, using provided path: {start_path}")
            return start_path
        
        logger.warning(f"Could not detect project root from: {path}")
        return None
        
    except Exception as e:
        logger.error(f"Error detecting project root: {e}", exc_info=True)
        return None

