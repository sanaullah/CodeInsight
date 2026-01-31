"""
File access tools for agents.

Implements the actual tool functions that agents can call.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


async def read_file_tool(file_path: str, project_path: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the content of a file.
    
    Args:
        file_path: Relative path to the file from project root
        project_path: Absolute path to project root
        state: Current workflow state (for caching)
    
    Returns:
        Dictionary with 'content' (file content) or 'error' (error message)
    """
    import asyncio
    
    try:
        # Validate and resolve path
        validated_path = _validate_path(file_path, project_path)
        if not validated_path:
            return {"error": f"Invalid file path: {file_path}. Path must be within project root."}
        
        # Check cache
        file_cache = state.get("_file_cache", {})
        cache_key = str(validated_path)
        if cache_key in file_cache:
            logger.debug(f"File cache hit: {file_path}")
            return {"content": file_cache[cache_key]}
        
        # Read file
        if not validated_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if not validated_path.is_file():
            return {"error": f"Path is not a file: {file_path}"}
        
        # Read with encoding detection (blocking I/O in thread)
        from scanners.file_reader import read_file_with_encoding
        content, encoding = await asyncio.to_thread(read_file_with_encoding, validated_path)
        
        # Cache result
        if "_file_cache" not in state:
            state["_file_cache"] = {}
        state["_file_cache"][cache_key] = content
        
        logger.info(f"Read file: {file_path} ({len(content)} chars, {encoding})")
        return {"content": content, "encoding": encoding}
        
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
        return {"error": f"Error reading file {file_path}: {str(e)}"}


async def list_directory_tool(directory_path: str, project_path: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    List files in a directory.
    
    Args:
        directory_path: Relative path to the directory from project root
        project_path: Absolute path to project root
        state: Current workflow state
    
    Returns:
        Dictionary with 'files' (list of file paths) or 'error' (error message)
    """
    import asyncio

    try:
        # Validate and resolve path
        validated_path = _validate_path(directory_path, project_path)
        if not validated_path:
            return {"error": f"Invalid directory path: {directory_path}. Path must be within project root."}
        
        if not validated_path.exists():
            return {"error": f"Directory not found: {directory_path}"}
        
        if not validated_path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}
        
        # List files (blocking I/O in thread)
        def _list_files_sync(path_obj: Path, root_path: Path):
            files_list = []
            try:
                for item in path_obj.iterdir():
                    try:
                        relative_path = str(item.relative_to(root_path))
                    except ValueError:
                        # Should not happen given logic, but safe fallback
                        relative_path = item.name
                        
                    if item.is_file():
                        files_list.append({
                            "path": relative_path,
                            "type": "file",
                            "name": item.name
                        })
                    elif item.is_dir():
                        files_list.append({
                            "path": relative_path,
                            "type": "directory",
                            "name": item.name
                        })
                return files_list
            except PermissionError as e:
                 raise e
        
        project_root = Path(project_path)
        try:
            files = await asyncio.to_thread(_list_files_sync, validated_path, project_root)
        except PermissionError as e:
            return {"error": f"Permission denied accessing directory {directory_path}: {str(e)}"}
        
        # Sort files
        files.sort(key=lambda x: (x["type"] == "directory", x["name"]))
        
        logger.info(f"Listed directory: {directory_path} ({len(files)} items)")
        return {"files": files}
        
    except Exception as e:
        logger.error(f"Error listing directory {directory_path}: {e}", exc_info=True)
        return {"error": f"Error listing directory {directory_path}: {str(e)}"}


async def get_file_info_tool(file_path: str, project_path: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get metadata about a file.
    
    Args:
        file_path: Relative path to the file from project root
        project_path: Absolute path to project root
        state: Current workflow state
    
    Returns:
        Dictionary with file metadata or 'error' (error message)
    """
    import asyncio

    try:
        # Validate and resolve path
        validated_path = _validate_path(file_path, project_path)
        if not validated_path:
            return {"error": f"Invalid file path: {file_path}. Path must be within project root."}
        
        if not validated_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if not validated_path.is_file():
            return {"error": f"Path is not a file: {file_path}"}
        
        # Get file info (blocking I/O in thread)
        def _get_info_sync(path_obj: Path):
            stat = path_obj.stat()
            line_count = None
            try:
                with open(path_obj, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                pass  # Binary file or encoding issue
            return stat.st_size, line_count

        size, line_count = await asyncio.to_thread(_get_info_sync, validated_path)
        
        info = {
            "path": file_path,
            "size": size,
            "line_count": line_count,
            "exists": True
        }
        
        logger.info(f"Got file info: {file_path} ({size} bytes, {line_count} lines)")
        return info
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}", exc_info=True)
        return {"error": f"Error getting file info for {file_path}: {str(e)}"}


def _validate_path(path: str, project_path: str) -> Optional[Path]:
    """
    Validate and resolve a file path, ensuring it's within project root.
    
    Args:
        path: Relative path from project root
        project_path: Absolute path to project root
    
    Returns:
        Resolved Path object or None if invalid
    """
    try:
        project_root = Path(project_path).resolve()
        path_obj = Path(path)
        
        # Handle absolute paths
        if path_obj.is_absolute():
            # Check if it's within project root
            try:
                relative = path_obj.relative_to(project_root)
                resolved = project_root / relative
            except ValueError:
                return None  # Path outside project root
        else:
            # Resolve relative path
            resolved = (project_root / path_obj).resolve()
        
        # Ensure resolved path is within project root (prevent path traversal)
        try:
            resolved.relative_to(project_root)
        except ValueError:
            return None  # Path traversal attempt
        
        # Normalize path (resolve .. and .)
        resolved = resolved.resolve()
        
        # Final check: ensure still within project root after normalization
        try:
            resolved.relative_to(project_root)
            return resolved
        except ValueError:
            return None
            
    except Exception as e:
        logger.warning(f"Path validation error for {path}: {e}")
        return None

