"""
File hash computation utility for consistent caching across systems.

Provides shared file hash computation logic used by both Swarm Analysis
and Architecture Model Management UI to ensure consistent cache keys.
"""

import json
import hashlib
import logging
from typing import List, Union

from scanners.project_scanner import FileInfo

logger = logging.getLogger(__name__)


def compute_file_hash(files: List[FileInfo]) -> str:
    """
    Compute hash of file list for caching.
    
    Creates a stable hash based on file paths, relative paths, sizes, and line counts.
    This hash is used as part of the cache key to detect when project files have changed.
    
    Args:
        files: List of FileInfo objects from project scan
    
    Returns:
        SHA256 hash as hexadecimal string
    """
    if not files:
        logger.warning("Empty file list provided for hash computation")
        return hashlib.sha256(b"").hexdigest()
    
    # Create stable representation using available FileInfo attributes
    file_data = [
        {
            "path": str(f.path),
            "relative_path": f.relative_path,
            "size": f.size,
            "line_count": f.line_count
        }
        for f in files
    ]
    
    # Sort by path for consistent hashing regardless of scan order
    file_str = json.dumps(file_data, sort_keys=True)
    return hashlib.sha256(file_str.encode()).hexdigest()

