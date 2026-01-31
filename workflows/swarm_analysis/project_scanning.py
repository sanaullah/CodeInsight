"""
Project scanning node for Swarm Analysis workflow.

Scans project files and detects programming languages.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields as validate_state_fields_decorator
from workflows.state_keys import StateKeys
from utils.validation import validate_project_path, validate_file_extensions
from utils.error_context import create_error_context

logger = logging.getLogger(__name__)


@validate_state_fields_decorator(["project_path"], "scan_project")
def scan_project_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Scan project files and detect languages.
    
    Args:
        state: Current swarm analysis state
        
    Returns:
        Updated state with files and detected_languages
    """
    try:
        # Validate and normalize project path
        project_path_str: Optional[str] = state.get(StateKeys.PROJECT_PATH)
        project_path: Path = validate_project_path(project_path_str)
        
        # Validate file extensions if provided
        file_extensions: Optional[List[str]] = state.get(StateKeys.FILE_EXTENSIONS)
        if file_extensions:
            normalized_extensions = validate_file_extensions(file_extensions)
            state[StateKeys.FILE_EXTENSIONS] = normalized_extensions
    except ValueError as e:
        error_msg = (
            f"project_path is required for project scanning. "
            f"Please provide a valid project directory path. "
            f"Error: {str(e)}"
        )
        state[StateKeys.ERROR] = error_msg
        state[StateKeys.ERROR_STAGE] = "scan_project"
        logger.error(error_msg)
        return state
    
    # Stream callback
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback("project_scanning_start", {
            "message": "Scanning project files...",
            "project_path": project_path
        })
    
    try:
        # Import and use actual project scanner
        logger.info(f"Scanning project: {project_path}")
        
        from scanners import create_project_scanner
        from scanners.language_detector import LanguageDetector
        from utils.langfuse_integration import create_observation
        
        # Get configuration from state before creating observation
        auto_detect: Optional[bool] = state.get(StateKeys.AUTO_DETECT_LANGUAGES)
        file_extensions: Optional[List[str]] = state.get(StateKeys.FILE_EXTENSIONS)
        
        # Create Langfuse observation for project scanning
        scan_obs = create_observation(
            name="project_scanning",
            input_data={
                "project_path": str(project_path),
                "file_extensions": file_extensions,
                "auto_detect_languages": auto_detect,
                "selected_directories": state.get(StateKeys.SELECTED_DIRECTORIES)
            },
            metadata={
                "operation": "scan_project",
                "node_type": "project_scanning"
            }
        )
        
        try:
            # Create scanner with configuration from state
            scanner = create_project_scanner(
                auto_detect=auto_detect,
                file_extensions=file_extensions
            )
            
            # Check if specific directories are selected
            selected_dirs: Optional[List[str]] = state.get(StateKeys.SELECTED_DIRECTORIES)
            enable_static_dependency_resolution = state.get(StateKeys.ENABLE_STATIC_DEPENDENCY_RESOLUTION, True)
            
            # Ensure selected_dirs is a valid list if provided, or None
            if selected_dirs is not None:
                if not isinstance(selected_dirs, list):
                    selected_dirs = None
                elif len(selected_dirs) == 0:
                    # Empty list is treated as None (no specific directories selected)
                    selected_dirs = None
            
            if selected_dirs and len(selected_dirs) > 0:
                # Directory selection mode: scan selected directories first
                logger.info(f"Scanning {len(selected_dirs)} selected directories: {selected_dirs}")
                
                # Defensive check: ensure selected_dirs is not None before calling scan_directories
                if selected_dirs is None:
                    raise ValueError("selected_dirs cannot be None when scanning directories")
                
                initial_files = scanner.scan_directories(
                    directories=selected_dirs,
                    project_root=project_path,
                    file_extensions=state.get(StateKeys.FILE_EXTENSIONS)
                )
                logger.info(f"Scanned {len(selected_dirs)} selected directories, found {len(initial_files)} files")
                
                # Static dependency resolution
                dependency_resolution = None
                all_files = initial_files.copy()
                
                if enable_static_dependency_resolution:
                    try:
                        from services.dependency_resolver import DependencyResolver
                        from scanners.project_scanner import FileInfo
                        
                        # Convert dict files to FileInfo objects for dependency resolver
                        file_objects = []
                        for f in initial_files:
                            if isinstance(f, dict):
                                file_objects.append(FileInfo(
                                    path=Path(f["path"]),
                                    relative_path=f["relative_path"],
                                    content=f.get("content", ""),
                                    size=f.get("size", 0),
                                    line_count=f.get("line_count", 0),
                                    encoding=f.get("encoding", "utf-8")
                                ))
                            else:
                                file_objects.append(f)
                        
                        # Resolve dependencies
                        resolver = DependencyResolver(max_depth=3)
                        all_dirs, auto_included_dirs = resolver.resolve_dependencies(
                            selected_dirs,
                            str(project_path),
                            file_objects
                        )
                        
                        logger.info(f"Dependency resolution: {len(selected_dirs)} selected, {len(auto_included_dirs)} auto-included")
                        
                        # Scan auto-included directories
                        if auto_included_dirs:
                            dependency_files = scanner.scan_directories(
                                directories=auto_included_dirs,
                                project_root=project_path,
                                file_extensions=state.get(StateKeys.FILE_EXTENSIONS)
                            )
                            logger.info(f"Scanned {len(auto_included_dirs)} dependency directories, found {len(dependency_files)} files")
                            all_files.extend(dependency_files)
                        
                        # Store dependency resolution metadata
                        dependency_resolution = {
                            "selected_dirs": selected_dirs,
                            "auto_included_dirs": auto_included_dirs,
                            "all_dirs": all_dirs
                        }
                        
                        if stream_callback:
                            stream_callback("dependencies_detected", {
                                "selected_directories": selected_dirs,
                                "auto_included_directories": auto_included_dirs,
                                "total_directories": all_dirs
                            })
                    except Exception as e:
                        logger.warning(f"Error during dependency resolution: {e}", exc_info=True)
                        # Continue without dependency resolution
                
                files = all_files
            else:
                # Full project scan (existing behavior - backward compatible)
                files = scanner.scan_directory(project_path)
                logger.info(f"Scanned entire project directory, found {len(files)} files")
                dependency_resolution = None
            
            # Detect languages if auto-detect is enabled
            detected_languages: List[str] = []
            if state.get(StateKeys.AUTO_DETECT_LANGUAGES, False):
                try:
                    detector = LanguageDetector()
                    # Language detection works on project root regardless of selected directories
                    detected_languages_dict = detector.detect_languages(project_path)
                    detected_languages = list(detected_languages_dict.keys())
                except Exception as e:
                    logger.warning(f"Error detecting languages: {e}")
            
            # Compute file hash using centralized utility
            file_hash: Optional[str] = None
            if files:
                from utils.io.file_hash import compute_file_hash
                # Convert dict files back to FileInfo objects if needed
                if files and isinstance(files[0], dict):
                    from scanners.project_scanner import FileInfo
                    file_objects = [
                        FileInfo(
                            path=Path(f["path"]),
                            relative_path=f["relative_path"],
                            content=f.get("content", ""),
                            size=f.get("size", 0),
                            line_count=f.get("line_count", 0),
                            encoding=f.get("encoding", "utf-8")
                        ) for f in files
                    ]
                    file_hash = compute_file_hash(file_objects)
                else:
                    file_hash = compute_file_hash(files)
            
            # Convert FileInfo objects to dictionaries for state
            # Tag files as primary or dependency
            selected_dirs_set = set(selected_dirs) if selected_dirs else set()
            auto_included_dirs_set = set(dependency_resolution.get("auto_included_dirs", [])) if dependency_resolution else set()
            
            files_dict: List[Dict[str, Any]] = []
            for f in files:
                file_dict = {
                    "path": str(f.path) if hasattr(f, 'path') else f.get("path", ""),
                    "relative_path": f.relative_path if hasattr(f, 'relative_path') else f.get("relative_path", ""),
                    "content": f.content if hasattr(f, 'content') else f.get("content", ""),
                    "size": f.size if hasattr(f, 'size') else f.get("size", 0),
                    "line_count": f.line_count if hasattr(f, 'line_count') else f.get("line_count", 0),
                    "encoding": f.encoding if hasattr(f, 'encoding') else f.get("encoding", "utf-8")
                }
                
                # Determine file type
                relative_path = file_dict["relative_path"]
                file_path_obj = Path(relative_path)
                file_dir = str(file_path_obj.parent) if file_path_obj.parent != Path(".") else ""
                
                if selected_dirs and file_dir in selected_dirs_set:
                    file_dict["file_type"] = "primary"
                elif dependency_resolution and file_dir in auto_included_dirs_set:
                    file_dict["file_type"] = "dependency"
                else:
                    file_dict["file_type"] = "primary"  # Default for full project scans
                
                files_dict.append(file_dict)
            
            # Debug information available via logger.debug() if needed
            files_with_content = sum(1 for f in files_dict if f.get('content'))
            logger.debug(f"Scanned {len(files_dict)} files, {files_with_content} with content")
        
            # Update observation with results
            if scan_obs:
                try:
                    scan_obs.update(
                        output={
                            "file_count": len(files_dict),
                            "files_with_content": files_with_content,
                            "detected_languages": detected_languages,
                            "file_hash": file_hash
                        },
                        metadata={
                            "files_scanned": len(files_dict),
                            "languages_detected": len(detected_languages),
                            "dependency_resolution_enabled": dependency_resolution is not None
                        }
                    )
                except AttributeError:
                    # Observation is a context manager, update not supported directly
                    pass
            
            # Update state
            updated_state = state.copy()
            updated_state[StateKeys.FILES] = files_dict
            updated_state[StateKeys.FILE_HASH] = file_hash
            updated_state[StateKeys.DETECTED_LANGUAGES] = detected_languages
            if dependency_resolution:
                updated_state[StateKeys.DEPENDENCY_RESOLUTION] = dependency_resolution
            # Track workflow start time for execution time calculation (not in StateKeys - temporary field)
            updated_state["workflow_start_time"] = datetime.now().timestamp()
            
            if stream_callback:
                stream_callback("project_scanning_complete", {
                    "file_count": len(files),
                    "detected_languages": detected_languages
                })
            
            logger.info(f"Scanned {len(files)} files")
            return updated_state
            
        finally:
            # Close observation
            if scan_obs:
                try:
                    scan_obs.end()
                except Exception:
                    pass
        
    except Exception as e:
        error_context = create_error_context("scan_project", state, {
            "scanner_type": type(scanner).__name__ if 'scanner' in locals() else "unknown",
            "selected_dirs": state.get(StateKeys.SELECTED_DIRECTORIES),
            "files_scanned": len(files) if 'files' in locals() else 0
        })
        error_msg = (
            f"Failed to scan project at '{project_path_str if 'project_path_str' in locals() else state.get(StateKeys.PROJECT_PATH)}': {str(e)}. "
            f"Scanned {len(files) if 'files' in locals() else 0} files before error. "
            f"Check that the project path is accessible and contains valid files."
        )
        logger.error(error_msg, extra=error_context, exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = error_msg
        updated_state[StateKeys.ERROR_STAGE] = "scan_project"
        updated_state[StateKeys.ERROR_CONTEXT] = error_context
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": error_msg})
        return updated_state

