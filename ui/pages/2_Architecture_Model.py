"""
Architecture Model Management page.

Provides UI for building, managing, and deleting architecture models.
Ensures models are always built from project root for complete coverage.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import logging
from datetime import datetime
from utils.sidebar import render_sidebar
from utils.config.settings_db import init_settings_db
from utils.project_root_detector import detect_project_root
from ui.components.architecture_model_ui import (
    render_architecture_model_status,
    render_build_controls,
    get_all_stored_architecture_models,
    format_model_for_dropdown
)

# Import shared functions from main app
try:
    from ui.app import initialize_global_settings, get_help_text
except ImportError:
    # Fallback: define functions locally if import fails
    def initialize_global_settings():
        """Initialize global UI settings in session state."""
        defaults = {
            "ui_layout_mode": "wide",
            "ui_display_density": "spacious",
            "ui_sidebar_default": "expanded",
            "ui_animations_enabled": True,
            "ui_auto_expand_sections": False,
            "ui_show_help_text": True,
            "ui_default_results_tab": "Summary",
            "ui_progress_style": "detailed",
            "ui_items_per_page": 10,
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def get_help_text(text: str):
        """Get help text if enabled in global settings."""
        return text if st.session_state.get("ui_show_help_text", True) else None

logger = logging.getLogger(__name__)

# Initialize settings
init_settings_db()

# Initialize global settings
initialize_global_settings()

# Apply layout mode
layout = st.session_state.get("ui_layout_mode", "wide")
st.set_page_config(
    page_title="Architecture Model - CodeInsight",
    page_icon="🏗️",
    layout=layout,
    initial_sidebar_state="expanded"
)

# Render sidebar
render_sidebar()

# Page header
from utils.version import VERSION_STRING

st.title(f"🏗️ Architecture Model Management")
st.markdown(f"""
**Architecture Design & Modeling** | `{VERSION_STRING}`

Build and manage architecture models for your projects. Architecture models provide 
a high-level understanding of your application's structure, which helps all agents 
generate more accurate and context-aware analyses.

**Key Features:**
- Build models from project root for complete coverage
- Rebuild models when your codebase changes
- Delete models when no longer needed
- All agents automatically use cached models
""")
st.divider()

# Initialize session state
if "arch_model_project_path" not in st.session_state:
    st.session_state.arch_model_project_path = ""
if "arch_model_project_root" not in st.session_state:
    st.session_state.arch_model_project_root = None
if "arch_model_building" not in st.session_state:
    st.session_state.arch_model_building = False
if "arch_model_input_method" not in st.session_state:
    st.session_state.arch_model_input_method = None
if "arch_model_list_cache" not in st.session_state:
    st.session_state.arch_model_list_cache = None
if "arch_model_list_cache_time" not in st.session_state:
    st.session_state.arch_model_list_cache_time = None
if "arch_model_dropdown_selection" not in st.session_state:
    st.session_state.arch_model_dropdown_selection = "Select a stored model..."
if "arch_model_selected_file_hash" not in st.session_state:
    st.session_state.arch_model_selected_file_hash = None
if "arch_model_selected_knowledge_id" not in st.session_state:
    st.session_state.arch_model_selected_knowledge_id = None

# Project Configuration Section
st.markdown("### 📁 Project Configuration")

# Cache model list in session state (refresh on page load or button click)
cache_age = (datetime.now() - st.session_state.arch_model_list_cache_time).total_seconds() \
    if st.session_state.arch_model_list_cache_time else 999

if cache_age > 30 or st.session_state.arch_model_list_cache is None:
    st.session_state.arch_model_list_cache = get_all_stored_architecture_models()
    st.session_state.arch_model_list_cache_time = datetime.now()

stored_models = st.session_state.arch_model_list_cache

# Two-column layout: Dropdown | Text Input
col1, col2 = st.columns([1, 1])

with col1:
    # Dropdown for stored models
    if stored_models:
        # Format options for dropdown
        model_options = [format_model_for_dropdown(m) for m in stored_models]
        # Add "Select a model..." as first option
        model_options.insert(0, "Select a stored model...")
        
        # Create mapping: display_text -> model_dict
        model_map = {opt: stored_models[i-1] for i, opt in enumerate(model_options) if i > 0}
        
        # Get current selection (default to placeholder or previously selected)
        default_index = 0
        if st.session_state.arch_model_input_method == "dropdown" and st.session_state.arch_model_dropdown_selection in model_options:
            try:
                default_index = model_options.index(st.session_state.arch_model_dropdown_selection)
            except ValueError:
                default_index = 0
        
        selected_option = st.selectbox(
            "Select from stored models",
            options=model_options,
            index=default_index,
            key="arch_model_dropdown",
            help="Choose an existing architecture model from the knowledge base"
        )
        
        # Check if selection actually changed
        previous_selection = st.session_state.get("arch_model_dropdown_selection", "Select a stored model...")
        selection_changed = selected_option != previous_selection
        
        # Update session state with selection
        st.session_state.arch_model_dropdown_selection = selected_option
        
        # If user selected a model (not the placeholder) AND selection changed, update project_path, file_hash, and knowledge_id
        if selected_option and selected_option != "Select a stored model..." and selection_changed:
            selected_model = model_map[selected_option]
            selected_path = selected_model.get('project_path')
            selected_file_hash = selected_model.get('file_hash')
            selected_knowledge_id = selected_model.get('knowledge_id')
            if selected_path:
                st.session_state.arch_model_project_path = selected_path
                st.session_state.arch_model_selected_file_hash = selected_file_hash
                st.session_state.arch_model_selected_knowledge_id = selected_knowledge_id
                st.session_state.arch_model_input_method = "dropdown"
                # Only rerun when selection actually changed
                st.rerun()
    else:
        st.info("No stored models found. Build your first model below!")
        st.session_state.arch_model_dropdown_selection = "Select a stored model..."

with col2:
    # Project path input
    project_path = st.text_input(
        "Project Path",
        value=st.session_state.get("arch_model_project_path", ""),
        help=get_help_text("Enter the path to your project directory (file or subdirectory). The system will automatically detect the project root."),
        key="arch_model_path_input"
    )
    
    # If text input changed (and it wasn't from dropdown), clear dropdown selection and selected file_hash/knowledge_id
    previous_path = st.session_state.get("arch_model_project_path", "")
    if project_path != previous_path and st.session_state.get("arch_model_input_method") != "dropdown":
        st.session_state.arch_model_input_method = "manual"
        st.session_state.arch_model_dropdown_selection = "Select a stored model..."
        st.session_state.arch_model_selected_file_hash = None
        st.session_state.arch_model_selected_knowledge_id = None
    
    # Only update session state if not set by dropdown (to preserve dropdown selection)
    if st.session_state.get("arch_model_input_method") != "dropdown":
        st.session_state.arch_model_project_path = project_path
    else:
        # Ensure project_path variable reflects session state when set by dropdown
        project_path = st.session_state.arch_model_project_path

# Add refresh button for manual model list refresh
if st.button("🔄 Refresh Model List", help="Refresh the list of stored architecture models", key="refresh_models"):
    st.session_state.arch_model_list_cache = None
    st.session_state.arch_model_list_cache_time = None
    st.rerun()

# Auto-detect project root
detected_root = None
if project_path:
    try:
        # Validate path exists first
        path_obj = Path(project_path)
        if not path_obj.exists():
            st.error(f"❌ Path does not exist: `{project_path}`")
            st.session_state.arch_model_project_root = None
        else:
            detected_root = detect_project_root(project_path)
            if detected_root:
                st.session_state.arch_model_project_root = str(detected_root)
                st.success(f"✅ Detected project root: `{detected_root}`")
            else:
                st.session_state.arch_model_project_root = None
                st.warning("⚠️ Could not detect project root. Please verify the path is correct.")
    except Exception as e:
        logger.error(f"Error detecting project root: {e}", exc_info=True)
        st.error(f"❌ Error detecting project root: {str(e)}")
        st.session_state.arch_model_project_root = None

# Use detected root or fallback to provided path
project_root = st.session_state.arch_model_project_root or project_path

st.divider()

# Architecture Model Status Section
st.markdown("### 📊 Architecture Model Status")

if project_path:
    # The enhanced status component now handles all metadata display, export, and details
    # Use selected knowledge_id if available (from dropdown selection), otherwise use file_hash or compute from files
    selected_knowledge_id = st.session_state.get("arch_model_selected_knowledge_id")
    selected_file_hash = st.session_state.get("arch_model_selected_file_hash")
    model = render_architecture_model_status(
        project_path, 
        file_hash=selected_file_hash,
        knowledge_id=selected_knowledge_id
    )
else:
    st.info("Enter a project path above to check for architecture models.")
    model = None

st.divider()

# Build Controls Section
st.markdown("### 🛠️ Model Actions")

if project_path and project_root:
    render_build_controls(project_path, project_root)
    
    # Show building status
    if st.session_state.arch_model_building:
        st.info("🔄 Building architecture model... This may take a few minutes.")
else:
    st.info("Enter a project path above to enable model actions.")

st.divider()

# Information Section
with st.expander("ℹ️ About Architecture Models", expanded=False):
    st.markdown("""
    **What are Architecture Models?**
    
    Architecture models are structured representations of your application's architecture,
    including:
    - System structure and organization
    - Modules and components
    - Dependencies and relationships
    - Design patterns and anti-patterns
    - Technology stack
    - Security and performance characteristics
    
    **Why Build Architecture Models?**
    
    - **Better Analysis**: Agents use architecture models to generate more accurate,
      context-aware analyses
    - **Efficiency**: Models are built once and reused by all agents
    - **Consistency**: All agents work with the same understanding of your system
    - **Token Savings**: Using models instead of raw files reduces token usage by ~70%
    
    **How It Works**
    
    1. Enter your project path (can be any file or directory in the project)
    2. The system automatically detects the project root
    3. Click "Build Model" to create an architecture model from the entire project
    4. All agents automatically use the cached model for their analyses
    5. Rebuild the model when your codebase changes significantly
    
    **Best Practices**
    
    - Always build models from the project root (automatically detected)
    - Rebuild models after major architectural changes
    - Delete old models if you're no longer working on a project
    - Models are cached in the Knowledge Base for fast access
    """)

