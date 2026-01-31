"""
Directory selector component for CodeInsight.

Provides a tree view UI component for selecting directories for analysis.
"""

import streamlit as st
import logging
import hashlib
import fnmatch
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

from scanners.project_scanner import ProjectScanner

logger = logging.getLogger(__name__)


def _should_ignore_directory(dir_name: str, ignore_patterns: Set[str]) -> bool:
    """
    Check if a directory should be ignored based on ignore patterns.
    
    Args:
        dir_name: Name of the directory
        ignore_patterns: Set of patterns to ignore
    
    Returns:
        True if directory should be ignored
    """
    # Check for exact matches first
    if dir_name in ignore_patterns:
        return True
    
    # Check for wildcard patterns using fnmatch for proper pattern matching
    for pattern in ignore_patterns:
        if "*" in pattern:
            # Use fnmatch for proper wildcard matching
            if fnmatch.fnmatch(dir_name, pattern):
                return True
    
    return False


def build_directory_tree(
    root_path: Path, 
    max_depth: int = 10, 
    show_hidden: bool = False,
    current_depth: int = 0,
    ignore_patterns: Optional[Set[str]] = None,
    original_root: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Recursively build a tree structure of directories.
    
    Args:
        root_path: Current directory path being processed
        max_depth: Maximum depth to traverse
        show_hidden: Whether to show hidden directories (starting with .)
        current_depth: Current depth in recursion
        ignore_patterns: Set of directory patterns to ignore
        original_root: Original root path (for calculating relative paths)
    
    Returns:
        Dictionary with structure: {
            "name": str,
            "path": str (relative path),
            "children": Dict[str, Dict],
            "is_dir": bool
        }
    """
    if ignore_patterns is None:
        ignore_patterns = ProjectScanner.DEFAULT_IGNORE_PATTERNS
    
    # Track original root for relative path calculation
    if original_root is None:
        original_root = root_path
    
    tree = {
        "name": root_path.name if root_path.name else str(root_path),
        "path": ".",  # Root is represented as "."
        "children": {},
        "is_dir": True
    }
    
    # Stop if max depth reached
    if current_depth >= max_depth:
        return tree
    
    try:
        if not root_path.exists() or not root_path.is_dir():
            return tree
        
        # Get all items in directory
        items = sorted(root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        
        for item in items:
            # Skip files, only process directories
            if not item.is_dir():
                continue
            
            dir_name = item.name
            
            # Skip hidden directories if not showing them
            if not show_hidden and dir_name.startswith("."):
                continue
            
            # Skip ignored patterns
            if _should_ignore_directory(dir_name, ignore_patterns):
                continue
            
            try:
                # Calculate relative path from original root
                relative_path = str(item.relative_to(original_root)).replace("\\", "/")
                
                # Recursively build subtree
                subtree = build_directory_tree(
                    item,
                    max_depth=max_depth,
                    show_hidden=show_hidden,
                    current_depth=current_depth + 1,
                    ignore_patterns=ignore_patterns,
                    original_root=original_root
                )
                
                # Set the relative path in the subtree
                subtree["path"] = relative_path
                
                tree["children"][dir_name] = subtree
                
            except PermissionError:
                logger.debug(f"Permission denied accessing {item}")
                continue
            except Exception as e:
                logger.debug(f"Error processing directory {item}: {e}")
                continue
                
    except PermissionError:
        logger.debug(f"Permission denied accessing {root_path}")
    except Exception as e:
        logger.warning(f"Error building directory tree for {root_path}: {e}")
    
    return tree


def _render_tree_node(
    node: Dict[str, Any],
    project_root: Path,
    selected_paths_key: str,
    key_prefix: str,
    indent_level: int = 0,
    node_counter_key: str = ""
) -> None:
    """
    Recursively render a tree node and its children.
    
    Args:
        node: Tree node dictionary
        project_root: Root path of the project
        selected_paths_key: Session state key for selected paths set
        key_prefix: Prefix for Streamlit keys
        indent_level: Current indentation level
        node_counter_key: Session state key for node counter
    """
    node_path = node["path"]
    node_name = node["name"]
    children = node.get("children", {})
    
    # Create unique key using counter to ensure absolute uniqueness
    # This prevents collisions even if paths are duplicated
    if node_path == ".":
        current_key = f"{key_prefix}_root"
    else:
        # Increment counter for each node to ensure uniqueness
        st.session_state[node_counter_key] = st.session_state.get(node_counter_key, 0) + 1
        node_id = st.session_state[node_counter_key]
        current_key = f"{key_prefix}_node_{node_id}"
    
    # Get current selections from session state
    selected_paths = st.session_state.get(selected_paths_key, set())
    
    # Determine if this node is selected
    is_selected = node_path in selected_paths if node_path != "." else False
    
    # Render checkbox for this directory (skip root)
    if node_path != ".":
        # Use indentation for visual hierarchy
        indent = "  " * indent_level
        checkbox_label = f"{indent}📁 {node_name}"
        
        # Checkbox state - read from session state
        checkbox_key = f"{current_key}_checkbox"
        checked = st.checkbox(
            checkbox_label,
            value=is_selected,
            key=checkbox_key
        )
        
        # Update session state based on checkbox state
        if checked and node_path not in selected_paths:
            selected_paths.add(node_path)
            st.session_state[selected_paths_key] = selected_paths
        elif not checked and node_path in selected_paths:
            selected_paths.discard(node_path)
            st.session_state[selected_paths_key] = selected_paths
    
    # Render children in expander if there are any
    if children:
        if node_path == ".":
            # Root level - show all children directly
            for child_name, child_node in sorted(children.items()):
                _render_tree_node(
                    child_node,
                    project_root,
                    selected_paths_key,
                    key_prefix,
                    indent_level=indent_level,
                    node_counter_key=node_counter_key
                )
        else:
            # Non-root - use expander
            with st.expander(f"📁 {node_name} ({len(children)} subdirectories)", expanded=False):
                for child_name, child_node in sorted(children.items()):
                    _render_tree_node(
                        child_node,
                        project_root,
                        selected_paths_key,
                        key_prefix,
                        indent_level=indent_level + 1,
                        node_counter_key=node_counter_key
                    )


def render_directory_tree_selector(
    project_root: str,
    selected_directories: List[str],
    key_prefix: str = "dir_selector",
    max_depth: int = 10,
    show_hidden: bool = False
) -> List[str]:
    """
    Render an interactive directory tree selector.
    
    Args:
        project_root: Root path of the project
        selected_directories: Currently selected directories (relative paths)
        key_prefix: Prefix for Streamlit session state keys
        max_depth: Maximum depth to show in tree
        show_hidden: Whether to show hidden folders (starting with .)
    
    Returns:
        List of selected directory paths (relative to project_root)
    """
    try:
        project_root_path = Path(project_root).resolve()
        
        if not project_root_path.exists():
            st.error(f"Project root does not exist: {project_root}")
            return []
        
        if not project_root_path.is_dir():
            st.error(f"Project root is not a directory: {project_root}")
            return []
        
        # Initialize session state for selections
        state_key = f"{key_prefix}_selected"
        if state_key not in st.session_state:
            # Convert list to set for efficient lookups
            st.session_state[state_key] = set(selected_directories) if selected_directories else set()
            logger.debug(f"Initialized {state_key} with {len(selected_directories)} directories: {selected_directories}")
        
        # Initialize counter for unique node IDs
        counter_key = f"{key_prefix}_counter"
        if counter_key not in st.session_state:
            st.session_state[counter_key] = 0
        
        # Get current selections from session state
        selected_paths = st.session_state[state_key].copy()
        
        # Build directory tree
        with st.spinner("Building directory tree..."):
            tree = build_directory_tree(
                project_root_path,
                max_depth=max_depth,
                show_hidden=show_hidden
            )
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Select All", key=f"{key_prefix}_select_all"):
                # Select all directories in tree
                def collect_all_paths(node: Dict[str, Any], paths: Set[str]):
                    if node["path"] != ".":
                        paths.add(node["path"])
                    for child in node.get("children", {}).values():
                        collect_all_paths(child, paths)
                
                all_paths = set()
                collect_all_paths(tree, all_paths)
                st.session_state[state_key] = all_paths
                st.rerun()
        
        with col2:
            if st.button("Deselect All", key=f"{key_prefix}_deselect_all"):
                st.session_state[state_key] = set()
                st.rerun()
        
        with col3:
            st.caption(f"Selected: {len(selected_paths)} folder(s)")
        
        st.divider()
        
        # Reset counter before rendering (ensures consistent numbering)
        st.session_state[counter_key] = 0
        
        # Render tree (will update session state directly)
        _render_tree_node(
            tree,
            project_root_path,
            state_key,
            key_prefix,
            indent_level=0,
            node_counter_key=counter_key
        )
        
        # Get updated selections from session state
        selected_paths = st.session_state.get(state_key, set())
        
        # Return as sorted list
        return sorted(list(selected_paths))
        
    except Exception as e:
        logger.error(f"Error rendering directory selector: {e}", exc_info=True)
        st.error(f"Error building directory tree: {str(e)}")
        return []

