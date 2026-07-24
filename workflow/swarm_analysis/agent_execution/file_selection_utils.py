"""
File selection utilities for agent-driven dynamic file selection.

Provides functions for formatting file lists, parsing LLM responses,
validating selections, and filtering files.
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


def format_file_list_for_selection(files: List[Dict[str, Any]]) -> str:
    """
    Format file list for LLM file selection.
    
    Formats files with paths, sizes, line counts, and groups by directory
    for readability. Includes file type hints (config, test, source, etc.).
    
    Args:
        files: List of file dictionaries with path, relative_path, size, line_count, etc.
        
    Returns:
        Formatted string with file list
    """
    if not files:
        return "No files available."
    
    # Group files by directory
    files_by_dir = defaultdict(list)
    for f in files:
        relative_path = f.get("relative_path", "") or f.get("path", "")
        if relative_path:
            dir_path = str(Path(relative_path).parent)
            if dir_path == ".":
                dir_path = "root"
            files_by_dir[dir_path].append(f)
    
    # Build formatted output
    parts = []
    parts.append(f"## Available Files ({len(files)} total)\n")
    
    # Sort directories for consistent output
    for dir_path in sorted(files_by_dir.keys()):
        dir_files = files_by_dir[dir_path]
        parts.append(f"\n### {dir_path}/ ({len(dir_files)} files)\n")
        
        for f in sorted(dir_files, key=lambda x: x.get("relative_path", "")):
            relative_path = f.get("relative_path", "") or f.get("path", "")
            size = f.get("size", 0)
            line_count = f.get("line_count", 0)
            
            # Determine file type hint
            file_type = _get_file_type_hint(relative_path)
            
            # Format file entry
            parts.append(f"- `{relative_path}`")
            if size > 0:
                parts.append(f" ({size:,} bytes")
                if line_count > 0:
                    parts.append(f", {line_count:,} lines")
                parts.append(")")
            if file_type:
                parts.append(f" [{file_type}]")
            parts.append("\n")
    
    return "".join(parts)


def _get_file_type_hint(file_path: str) -> str:
    """Get file type hint based on path patterns."""
    path_lower = file_path.lower()
    
    # Config files
    if any(pattern in path_lower for pattern in ["config", "settings", ".env", ".yaml", ".yml", ".toml", ".ini"]):
        return "config"
    
    # Test files
    if any(pattern in path_lower for pattern in ["test", "spec", "__test__", "__spec__"]):
        return "test"
    
    # Documentation
    if any(pattern in path_lower for pattern in ["readme", "docs", ".md", ".rst", ".txt"]):
        return "docs"
    
    # Source code
    if any(pattern in path_lower for pattern in [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c"]):
        return "source"
    
    # Database
    if any(pattern in path_lower for pattern in [".sql", ".db", "migration", "schema"]):
        return "database"
    
    # Build/scripts
    if any(pattern in path_lower for pattern in ["build", "script", "setup", "makefile"]):
        return "build"
    
    return ""


def parse_file_selection(llm_response: str) -> Dict[str, Any]:
    """
    Parse JSON response from LLM file selection.
    
    Args:
        llm_response: LLM response text (may contain JSON or markdown code blocks)
        
    Returns:
        Dictionary with 'files' list, 'reasoning' string, and optional 'confidence' float
    """
    if not llm_response:
        return {"files": [], "reasoning": "No response from LLM", "confidence": None}
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*?"files".*?\}', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = llm_response.strip()
    
    try:
        parsed = json.loads(json_str)
        
        # Validate structure
        if not isinstance(parsed, dict):
            raise ValueError("Response is not a JSON object")
        
        files = parsed.get("files", [])
        if not isinstance(files, list):
            raise ValueError("'files' must be a list")
        
        # Ensure all file paths are strings
        files = [str(f) for f in files if f]
        
        reasoning = parsed.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning) if reasoning else ""
        
        # Extract confidence score (optional)
        confidence = None
        if "confidence" in parsed:
            conf_value = parsed.get("confidence")
            if conf_value is not None:
                try:
                    confidence = float(conf_value)
                    # Validate confidence is in range [0.0, 1.0]
                    if confidence < 0.0 or confidence > 1.0:
                        logger.warning(
                            f"Confidence value {confidence} out of range [0.0, 1.0], "
                            "setting to None"
                        )
                        confidence = None
                except (ValueError, TypeError):
                    logger.warning(f"Invalid confidence value type: {type(conf_value)}, setting to None")
                    confidence = None
        
        return {
            "files": files,
            "reasoning": reasoning,
            "confidence": confidence
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON from LLM response: {e}")
        logger.debug(f"Response text: {llm_response[:500]}")
        return {"files": [], "reasoning": f"JSON parsing error: {e}", "confidence": None}
    except Exception as e:
        logger.warning(f"Error parsing file selection response: {e}")
        return {"files": [], "reasoning": f"Parsing error: {e}", "confidence": None}


def validate_file_selection(
    selected_files: List[str],
    available_files: List[Dict[str, Any]]
) -> List[str]:
    """
    Validate file selection.
    
    Checks that all selected files exist and validates for suspicious patterns.
    
    Args:
        selected_files: List of selected file paths
        available_files: List of available file dictionaries
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if not selected_files:
        issues.append("No files selected")
        return issues
    
    # Create lookup for available files
    available_paths = set()
    available_by_relative = {}
    available_by_filename = {}
    
    for f in available_files:
        relative_path = f.get("relative_path", "")
        path = f.get("path", "")
        filename = Path(relative_path or path).name
        
        if relative_path:
            available_paths.add(relative_path)
            available_by_relative[relative_path] = f
        if path:
            available_paths.add(path)
        if filename:
            available_by_filename[filename] = f
    
    # Check each selected file
    missing_files = []
    for file_path in selected_files:
        # Try multiple matching strategies
        found = (
            file_path in available_paths or
            file_path in available_by_relative or
            Path(file_path).name in available_by_filename
        )
        
        if not found:
            missing_files.append(file_path)
    
    if missing_files:
        issues.append(f"Selected files not found: {', '.join(missing_files[:5])}")
        if len(missing_files) > 5:
            issues.append(f"... and {len(missing_files) - 5} more")
    
    # Check for suspicious patterns
    total_available = len(available_files)
    selected_count = len(selected_files)
    
    # Too many files (>80% of project)
    if total_available > 0 and selected_count > total_available * 0.8:
        issues.append(
            f"Selected too many files ({selected_count}/{total_available}, "
            f"{selected_count/total_available*100:.1f}%) - may indicate poor selection"
        )
    
    # Too few files (<1% of project, but only if project is large)
    if total_available > 50 and selected_count < total_available * 0.01:
        issues.append(
            f"Selected very few files ({selected_count}/{total_available}, "
            f"{selected_count/total_available*100:.1f}%) - may miss important files"
        )
    
    return issues


def filter_files_by_selection(
    files: List[Dict[str, Any]],
    selected_paths: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter files list to only selected paths.
    
    Handles path matching using multiple strategies:
    - Exact relative_path match
    - Exact path match
    - Filename match (as fallback)
    
    Args:
        files: List of file dictionaries
        selected_paths: List of selected file paths
        
    Returns:
        Filtered list of file dictionaries
    """
    if not selected_paths:
        return []
    
    # Create lookup sets
    selected_paths_set = set(selected_paths)
    selected_filenames = {Path(p).name for p in selected_paths}
    
    filtered = []
    matched_paths = set()
    
    for f in files:
        relative_path = f.get("relative_path", "")
        path = f.get("path", "")
        filename = Path(relative_path or path).name if (relative_path or path) else ""
        
        # Try multiple matching strategies
        matched = False
        
        # Strategy 1: Exact relative_path match
        if relative_path and relative_path in selected_paths_set:
            matched = True
            matched_paths.add(relative_path)
        
        # Strategy 2: Exact path match
        elif path and path in selected_paths_set:
            matched = True
            matched_paths.add(path)
        
        # Strategy 3: Filename match (fallback, but only if not already matched by another file)
        elif filename and filename in selected_filenames:
            # Check if this filename is unique in selected_paths
            matching_selected = [p for p in selected_paths if Path(p).name == filename]
            if len(matching_selected) == 1:
                matched = True
                matched_paths.add(matching_selected[0])
        
        if matched:
            filtered.append(f)
    
    # Log any unmatched selected paths
    unmatched = selected_paths_set - matched_paths
    if unmatched:
        logger.warning(
            f"Could not match {len(unmatched)} selected file paths: "
            f"{', '.join(list(unmatched)[:5])}"
        )
    
    return filtered

