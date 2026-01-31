"""
Settings page for CodeInsight.
"""

import streamlit as st
from llm.config import ConfigManager
import yaml
from utils.sidebar import render_sidebar
from utils.config.settings_db import (
    init_settings_db, get_user_id, set_setting, reset_user_settings,
    verify_setting_sync
)
from utils.config.settings_validator import validate_setting, get_all_defaults
import logging

logger = logging.getLogger(__name__)


def initialize_global_settings() -> None:
    """
    Initialize global UI settings in session state.
    
    Loads settings from database, merges with defaults.
    Handles errors gracefully by falling back to hardcoded defaults.
    """
    try:
        # Initialize database with error handling
        try:
            init_settings_db()
        except Exception as db_init_error:
            logger.warning(f"Could not initialize settings database: {db_init_error}. Using defaults only.")
        
        # Get user ID with fallback
        try:
            user_id = get_user_id()
        except Exception as user_id_error:
            logger.warning(f"Could not get user ID: {user_id_error}. Using default user.")
            user_id = "default_user"
        
        # Get defaults from validator (config.yaml + hardcoded)
        try:
            defaults = get_all_defaults()
        except Exception as defaults_error:
            logger.warning(f"Could not load defaults from validator: {defaults_error}. Using hardcoded defaults.")
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
        
        # Load settings from database with error handling
        db_settings = {}
        try:
            from utils.config.settings_db import get_all_settings
            db_settings = get_all_settings(user_id)
        except Exception as db_load_error:
            logger.warning(f"Could not load settings from database: {db_load_error}. Using defaults only.")
        
        # Merge: defaults first, then DB settings override
        merged_settings = defaults.copy()
        merged_settings.update(db_settings)
        
        # Validate and populate session state
        for key, value in merged_settings.items():
            if key not in st.session_state:
                st.session_state[key] = value
        
    except Exception as e:
        logger.error(f"Error initializing global settings: {e}", exc_info=True)
        # Set minimal defaults
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


def main():
    """Settings page."""
    # Initialize settings if not already done (loads from DB)
    initialize_global_settings()
    
    # Initialize database and get user ID
    try:
        init_settings_db()
        user_id = get_user_id()
    except Exception as e:
        logger.error(f"Error initializing settings DB: {e}", exc_info=True)
        user_id = None
        st.warning("⚠️ Settings database unavailable. Changes may not persist.")
    
    # Apply layout mode
    layout = st.session_state.get("ui_layout_mode", "wide")
    st.set_page_config(
        page_title="Settings - CodeInsight",
        page_icon="⚙️",
        layout=layout,
        initial_sidebar_state="expanded"
    )
    
    # Shared sidebar navigation and status
    render_sidebar()
    
    from utils.version import VERSION_STRING
    st.title("⚙️ Settings")
    st.markdown(f"Configure CodeInsight preferences and options | `{VERSION_STRING}`")
    st.divider()
    
    # Create tabs for different setting categories
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎨 Display & UI",
        "⚙️ Analysis Options", 
        "🧠 Model Configuration",
        "🔧 Advanced"
    ])
    
    # Tab 1: Display & UI Settings
    with tab1:
        st.header("Display Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            layout_mode = st.radio(
                "Layout Mode",
                options=["wide", "centered"],
                index=0 if st.session_state.get("ui_layout_mode", "wide") == "wide" else 1,
                help="Wide: Full width layout | Centered: Narrow centered layout"
            )
            if layout_mode != st.session_state.get("ui_layout_mode"):
                is_valid, error_msg = validate_setting("ui_layout_mode", layout_mode)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_layout_mode", layout_mode):
                        if verify_setting_sync(user_id, "ui_layout_mode", layout_mode):
                            st.session_state.ui_layout_mode = layout_mode
                            st.success("✅ Setting saved!")
                            st.rerun()
                        else:
                            st.warning("⚠️ Setting saved but sync verification failed")
                    else:
                        st.error("❌ Failed to save setting to database")
                elif not is_valid:
                    st.error(f"❌ {error_msg}")
                else:
                    st.session_state.ui_layout_mode = layout_mode
            
            display_density = st.radio(
                "Display Density",
                options=["compact", "spacious"],
                index=0 if st.session_state.get("ui_display_density", "spacious") == "compact" else 1,
                help="Compact: Dense layout | Spacious: More breathing room"
            )
            if display_density != st.session_state.get("ui_display_density"):
                is_valid, error_msg = validate_setting("ui_display_density", display_density)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_display_density", display_density):
                        st.session_state.ui_display_density = display_density
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                else:
                    st.session_state.ui_display_density = display_density
        
        with col2:
            progress_style = st.radio(
                "Progress Style",
                options=["minimal", "detailed"],
                index=0 if st.session_state.get("ui_progress_style", "detailed") == "minimal" else 1,
                help="Minimal: Simple progress bar | Detailed: Step-by-step progress"
            )
            if progress_style != st.session_state.get("ui_progress_style"):
                is_valid, error_msg = validate_setting("ui_progress_style", progress_style)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_progress_style", progress_style):
                        st.session_state.ui_progress_style = progress_style
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                else:
                    st.session_state.ui_progress_style = progress_style
        
        st.divider()
        
        st.header("Behavior Preferences")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            animations_enabled = st.checkbox(
                "Enable Animations",
                value=st.session_state.get("ui_animations_enabled", True),
                help="Show success animations on completion"
            )
            if animations_enabled != st.session_state.get("ui_animations_enabled"):
                is_valid, error_msg = validate_setting("ui_animations_enabled", animations_enabled)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_animations_enabled", animations_enabled):
                        st.session_state.ui_animations_enabled = animations_enabled
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                else:
                    st.session_state.ui_animations_enabled = animations_enabled
        
        with col2:
            auto_expand = st.checkbox(
                "Auto-expand Sections",
                value=st.session_state.get("ui_auto_expand_sections", False),
                help="Automatically expand collapsible sections"
            )
            if auto_expand != st.session_state.get("ui_auto_expand_sections"):
                is_valid, error_msg = validate_setting("ui_auto_expand_sections", auto_expand)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_auto_expand_sections", auto_expand):
                        st.session_state.ui_auto_expand_sections = auto_expand
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                else:
                    st.session_state.ui_auto_expand_sections = auto_expand
        
        with col3:
            show_help = st.checkbox(
                "Show Help Text",
                value=st.session_state.get("ui_show_help_text", True),
                help="Display help text and descriptions for inputs"
            )
            if show_help != st.session_state.get("ui_show_help_text"):
                is_valid, error_msg = validate_setting("ui_show_help_text", show_help)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_show_help_text", show_help):
                        st.session_state.ui_show_help_text = show_help
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                else:
                    st.session_state.ui_show_help_text = show_help
        
        st.divider()
        
        st.header("Results Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            default_tab = st.selectbox(
                "Default Results Tab",
                options=["Summary", "Detailed", "Recommendations", "Export"],
                index=["Summary", "Detailed", "Recommendations", "Export"].index(
                    st.session_state.get("ui_default_results_tab", "Summary")
                ) if st.session_state.get("ui_default_results_tab", "Summary") in ["Summary", "Detailed", "Recommendations", "Export"] else 0,
                help="Default tab to show when results are displayed"
            )
            if default_tab != st.session_state.get("ui_default_results_tab"):
                is_valid, error_msg = validate_setting("ui_default_results_tab", default_tab)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_default_results_tab", default_tab):
                        st.session_state.ui_default_results_tab = default_tab
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                else:
                    st.session_state.ui_default_results_tab = default_tab
        
        with col2:
            items_per_page = st.number_input(
                "Items Per Page",
                min_value=5,
                max_value=50,
                value=st.session_state.get("ui_items_per_page", 10),
                step=5,
                help="Number of items to show per page in results"
            )
            if items_per_page != st.session_state.get("ui_items_per_page"):
                is_valid, error_msg = validate_setting("ui_items_per_page", items_per_page)
                if is_valid and user_id:
                    if set_setting(user_id, "ui_items_per_page", items_per_page):
                        st.session_state.ui_items_per_page = items_per_page
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                else:
                    st.session_state.ui_items_per_page = items_per_page
    
    # Tab 2: Analysis Options
    with tab2:
        st.header("Analysis Configuration")
        
        max_tokens = st.number_input(
            "Max Tokens per Chunk",
            min_value=10000,
            max_value=500000,
            value=st.session_state.get("max_tokens_per_chunk", 100000),
            step=10000,
            help="Maximum tokens per analysis chunk. Larger values = fewer chunks but more processing per chunk."
        )
        if max_tokens != st.session_state.get("max_tokens_per_chunk"):
            if user_id:
                if set_setting(user_id, "max_tokens_per_chunk", max_tokens):
                    st.session_state.max_tokens_per_chunk = max_tokens
                    st.success("✅ Setting saved!")
                    st.rerun()
                else:
                    st.error("❌ Failed to save setting")
            else:
                st.session_state.max_tokens_per_chunk = max_tokens
        
        st.info("💡 Recommended: 100K-200K for most projects")
    
    # Tab 3: Model Configuration
    with tab3:
        st.header("Model Settings")
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        available_models = config_manager.get_available_models()
        
        if available_models:
            # Get current selected model from session state or use default
            current_model = st.session_state.get("selected_model", getattr(config, 'default_model', None) or available_models[0])
            
            # Find index of current model
            try:
                current_index = available_models.index(current_model) if current_model in available_models else 0
            except (ValueError, AttributeError):
                current_index = 0
            
            selected_model = st.selectbox(
                "Default Model",
                options=available_models,
                index=current_index,
                help="Select the default AI model for analysis"
            )
            if selected_model != st.session_state.get("selected_model"):
                is_valid, error_msg = validate_setting("selected_model", selected_model)
                if is_valid and user_id:
                    if set_setting(user_id, "selected_model", selected_model):
                        st.session_state.selected_model = selected_model
                        st.success("✅ Setting saved!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save setting")
                elif not is_valid:
                    st.error(f"❌ {error_msg}")
                else:
                    st.session_state.selected_model = selected_model
        else:
            st.warning("No models available. Check your config.yaml file.")
            if hasattr(config, 'default_model'):
                st.session_state.selected_model = config.default_model
    
    # Tab 4: Advanced
    with tab4:
        st.header("Advanced Settings")
        
        st.subheader("Reset Settings")
        
        if st.button("🔄 Reset All Settings to Defaults", type="secondary"):
            if user_id:
                if reset_user_settings(user_id):
                    st.success("✅ All settings reset to defaults!")
                    # Reload defaults into session state
                    defaults = get_all_defaults()
                    for key, value in defaults.items():
                        st.session_state[key] = value
                    st.rerun()
                else:
                    st.error("❌ Failed to reset settings")
            else:
                st.warning("⚠️ Cannot reset settings: user ID not available")


if __name__ == "__main__":
    main()

