"""
Streamlit application for CodeInsight.

Main UI for project analysis with agent selection and results display.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables from .env file if it exists
from utils.config.env_loader import load_env
load_env()

import streamlit as st
import logging
import pandas as pd
from typing import Dict, Any, Optional

from utils.config.logging_config import setup_logging
from utils.sidebar import render_sidebar
from utils.config.settings_db import (
    init_settings_db, get_user_id, get_all_settings
)
from utils.config.settings_validator import get_all_defaults
from utils.scan_history import save_scan, get_all_scans, get_statistics
from ui.components.dashboard_widgets import stat_card, activity_feed
from ui.components.scan_history_ui import render_scan_history_tab
from scanners.language_config import get_supported_languages, get_language_metadata
from utils.version import VERSION_STRING

# Configure logging
setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Global UI Settings System
# ============================================================================

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


# ============================================================================
# Main Application
# ============================================================================

def main() -> None:
    """
    Main Streamlit application entry point.
    
    Initializes settings, renders UI, and handles navigation.
    Includes graceful degradation for optional services.
    """
    logger.info("Starting CodeInsight application")
    
    # Initialize global settings
    try:
        initialize_global_settings()
        logger.debug("Global settings initialized")
    except Exception as settings_error:
        logger.error(f"Error initializing global settings: {settings_error}", exc_info=True)
        # Continue with defaults
    
    # Apply layout mode with fallback
    layout = st.session_state.get("ui_layout_mode", "wide")
    logger.debug(f"Using layout mode: {layout}")
    
    try:
        st.set_page_config(
            page_title="CodeInsight",
            page_icon="💡",
            layout=layout,
            initial_sidebar_state="expanded"
        )
    except Exception as config_error:
        logger.warning(f"Could not set page config (may already be set): {config_error}")
    
    # Shared sidebar navigation and status
    try:
        render_sidebar()
        logger.debug("Sidebar rendered")
    except Exception as sidebar_error:
        logger.error(f"Error rendering sidebar: {sidebar_error}", exc_info=True)
        from utils.error_formatting import format_user_error
        st.error(f"❌ Error loading sidebar: {format_user_error(sidebar_error)}")
    
    # Enhanced Header
    header_container = st.container()
    with header_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title(f"💡 CodeInsight")
            st.markdown(f"**AI-powered project analysis platform** | `Version: {VERSION_STRING}`")
        with col2:
            # Status indicator
            try:
                analysis_in_progress = st.session_state.get("analysis_in_progress", False)
                analysis_result = st.session_state.get("analysis_result")
                
                if analysis_in_progress:
                    st.status("🔄 Analysis in Progress", state="running")
                elif analysis_result:
                    st.status("✅ Analysis Complete", state="complete")
                else:
                    st.status("⚪ Ready", state="complete")
            except AttributeError:
                # Fallback for older Streamlit versions
                analysis_in_progress = st.session_state.get("analysis_in_progress", False)
                analysis_result = st.session_state.get("analysis_result")
                
                if analysis_in_progress:
                    st.info("🔄 Analysis in Progress")
                elif analysis_result:
                    st.success("✅ Analysis Complete")
                else:
                    st.info("⚪ Ready")
    
    st.divider()
    
    # Show welcome screen
    try:
        logger.debug("Rendering welcome screen")
        show_welcome()
    except Exception as welcome_error:
        logger.error(f"Error rendering welcome screen: {welcome_error}", exc_info=True)
        from utils.error_formatting import format_user_error
        st.error(f"❌ Error loading welcome screen: {format_user_error(welcome_error)}")


# ============================================================================
# Welcome Screen / Dashboard
# ============================================================================

def _format_timestamp(timestamp: Any, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Safely format a timestamp for display.
    
    Args:
        timestamp: Can be a datetime object, string, or None
        format_str: Format string for datetime objects (default: ISO-like format)
    
    Returns:
        Formatted timestamp string, or 'Unknown' if timestamp is missing/invalid
    """
    if timestamp is None:
        return 'Unknown'
    
    if isinstance(timestamp, datetime):
        return timestamp.strftime(format_str)
    elif isinstance(timestamp, str):
        return timestamp
    else:
        try:
            return str(timestamp)
        except Exception:
            return 'Unknown'


def show_welcome() -> None:
    """
    Render the main dashboard.
    """
    st.markdown("## 📊 Dashboard")
    
    # Get statistics
    try:
        stats = get_statistics()
    except Exception as e:
        logger.warning(f"Could not load statistics: {e}")
        stats = {}
    
    # Stats Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_scans = stats.get("total_scans", 0)
        stat_card("Total Scans", str(total_scans))
    with col2:
        recent_scans = stats.get("recent_scans", 0)
        stat_card("Recent Scans (7d)", str(recent_scans))
    with col3:
        avg_files = stats.get("average_files_scanned", 0)
        stat_card("Avg Files Scanned", str(avg_files))
    with col4:
        most_used = stats.get("most_used_agent", "N/A")
        stat_card("Most Used Agent", most_used if most_used else "N/A")
    
    st.divider()
    
    # Supported Languages Section
    try:
        st.subheader("🌐 Supported Languages")
        languages = get_supported_languages()
        total_languages = len(languages)
        st.markdown(f"**{total_languages} languages supported**")
        
        # Display languages in a grid format (5 columns)
        sorted_languages = sorted(languages)
        cols = st.columns(5)
        for idx, lang in enumerate(sorted_languages):
            # Get display name from metadata if available, otherwise capitalize
            metadata = get_language_metadata(lang)
            display_name = metadata.name if metadata else lang.title()
            
            with cols[idx % 5]:
                st.markdown(f"• {display_name}")
        
        # Debug: Verify BoxLang and ColdFusion are in the list
        logger.debug(f"Languages displayed: {len(sorted_languages)} total")
        logger.debug(f"BoxLang in list: {'boxlang' in sorted_languages}")
        logger.debug(f"ColdFusion in list: {'coldfusion' in sorted_languages}")
    except Exception as e:
        logger.warning(f"Could not load supported languages: {e}", exc_info=True)
        st.warning("Could not load supported languages list.")
    
    st.divider()
    
    # Main Content Area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🚀 Quick Actions")
        c1, c2 = st.columns(2)
        with c1:
            with st.container():
                st.info("Start swarm analysis")
                if st.button("Swarm Analysis", width='stretch'):
                    st.switch_page("ui/pages/swarm_analysis.py")
        with c2:
            with st.container():
                st.info("View settings")
                if st.button("Settings", width='stretch'):
                    st.switch_page("ui/pages/1_Settings.py")
        
        st.subheader("📈 Recent Scans")
        try:
            recent_scans = get_all_scans(limit=10)
            if recent_scans:
                # Prepare data for display
                scan_data = []
                for scan in recent_scans[:10]:
                    timestamp_str = _format_timestamp(scan.get('timestamp'))
                    # Extract date portion (first 10 characters) if it's a full datetime string
                    date_str = timestamp_str[:10] if len(timestamp_str) >= 10 else timestamp_str
                    scan_data.append({
                        "Project": Path(scan.get('project_path', 'Unknown')).name,
                        "Date": date_str,
                        "Status": scan.get('status', 'unknown'),
                        "Agent": scan.get('agent_name', 'Unknown'),
                        "Files": scan.get('files_scanned', 0)
                    })
                
                # Create DataFrame with proper column types
                df = pd.DataFrame(scan_data)
                
                # Enhanced dataframe with column configuration
                st.dataframe(
                    df,
                    column_config={
                        "Project": st.column_config.TextColumn("Project", width="medium"),
                        "Date": st.column_config.TextColumn("Date", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                        "Agent": st.column_config.TextColumn("Agent", width="medium"),
                        "Files": st.column_config.NumberColumn("Files", width="small", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("No scans yet. Run an analysis to get started!")
        except Exception as e:
            logger.error(f"Error loading recent scans: {e}", exc_info=True)
            st.warning("Could not load recent scans.")
    
    with col_right:
        # Activity feed
        activity_items = [
            {"action": "Welcome to CodeInsight", "timestamp": datetime.now().strftime("%H:%M")}
        ]
        
        # Add recent activity from scans if available
        try:
            recent_scans = get_all_scans(limit=5)
            for scan in recent_scans[:5]:
                timestamp_str = _format_timestamp(scan.get('timestamp'), format_str="%Y-%m-%d %H:%M")
                # Extract first 16 characters for datetime display (YYYY-MM-DD HH:MM)
                timestamp_display = timestamp_str[:16] if len(timestamp_str) >= 16 else timestamp_str
                activity_items.append({
                    "action": f"Scan completed: {Path(scan.get('project_path', 'Unknown')).name}",
                    "timestamp": timestamp_display
                })
        except Exception:
            pass
        
        activity_feed(activity_items)
    
    st.divider()
    
    # Scan History Tab
    render_scan_history_tab(
        key_prefix="dashboard",
        show_statistics=False,  # Already shown above
        show_filters=True,
        items_per_page=5
    )


if __name__ == "__main__":
    main()

