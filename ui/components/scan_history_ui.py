"""
Scan History UI components for CodeInsight.

This module provides a reusable UI component for displaying scan history
with statistics, filtering, pagination, and CRUD operations.
"""

import streamlit as st
import logging
from typing import Optional, Any
from pathlib import Path
from datetime import datetime

from utils.scan_history import (
    init_database,
    get_all_scans,
    get_scan_by_id,
    delete_scan,
    get_statistics as get_history_statistics
)
from ui.analysis_utils import display_analysis_results
from reports.report_generator import generate_markdown_report, generate_json_report

logger = logging.getLogger(__name__)


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


def render_scan_history_tab(
    key_prefix: str = "scan_history",
    agent_filter: Optional[str] = None,
    show_statistics: bool = True,
    show_filters: bool = True,
    items_per_page: int = 10
) -> None:
    """
    Render the Scan History tab component.
    
    Args:
        key_prefix: Unique prefix for Streamlit widget keys to avoid conflicts
        agent_filter: Optional agent name to pre-filter scans (e.g., "Swarm Analysis")
        show_statistics: Whether to display statistics dashboard
        show_filters: Whether to display filter controls
        items_per_page: Number of items per page for pagination
    """
    st.subheader("📜 Scan History")
    st.markdown("View, download, and manage your analysis history")
    st.divider()
    
    # Initialize database
    try:
        init_database()
    except Exception as e:
        from utils.error_formatting import format_user_error
        logger.error(f"Error initializing database: {e}", exc_info=True)
        st.error(f"Error initializing database: {format_user_error(e)}")
        return
    
    # Get statistics
    if show_statistics:
        try:
            stats = get_history_statistics()
        except Exception as e:
            st.warning(f"Could not load statistics: {str(e)}")
            stats = {}
        
        # Statistics section
        st.markdown("### 📊 History Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Scans", stats.get("total_scans", 0))
        with col2:
            most_used = stats.get("most_used_agent", "N/A")
            st.metric("Most Used Agent", most_used if most_used else "N/A")
        with col3:
            st.metric("Avg Files Scanned", stats.get("average_files_scanned", 0))
        with col4:
            st.metric("Recent Scans (7d)", stats.get("recent_scans", 0))
        
        st.divider()
    
    # Filters
    if show_filters:
        st.markdown("### 🔍 Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_agent = st.selectbox(
                "Filter by Agent",
                options=["All"] + [agent["agent_name"] for agent in get_all_scans(limit=1000) if agent.get("agent_name")],
                key=f"{key_prefix}_filter_agent"
            )
        
        with col2:
            filter_status = st.selectbox(
                "Filter by Status",
                options=["All", "completed", "error"],
                key=f"{key_prefix}_filter_status"
            )
        
        st.divider()
    else:
        # If filters are hidden, set defaults
        filter_agent = "All"
        filter_status = "All"
    
    # Get scans
    try:
        all_scans = get_all_scans()
        
        # Apply agent_filter if provided (pre-filter)
        if agent_filter:
            all_scans = [s for s in all_scans if s.get("agent_name") == agent_filter]
        
        # Apply filters
        filtered_scans = all_scans
        if filter_agent != "All":
            filtered_scans = [s for s in filtered_scans if s.get("agent_name") == filter_agent]
        if filter_status != "All":
            filtered_scans = [s for s in filtered_scans if s.get("status") == filter_status]
        
        if not filtered_scans:
            st.info("No scans found. Run an analysis to create history.")
            return
        
        # Display scans in a table
        st.markdown(f"### 📋 Scans ({len(filtered_scans)})")
        
        # Pagination
        pagination_size = items_per_page
        total_items = len(filtered_scans)
        
        if total_items > pagination_size:
            page_num = st.number_input(
                "Page",
                min_value=1,
                max_value=(total_items + pagination_size - 1) // pagination_size,
                value=1,
                step=1,
                key=f"{key_prefix}_page"
            )
            start_idx = (page_num - 1) * pagination_size
            end_idx = min(start_idx + pagination_size, total_items)
            page_scans = filtered_scans[start_idx:end_idx]
            st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_items} scans")
        else:
            page_scans = filtered_scans
        
        # Display each scan
        for scan in page_scans:
            timestamp_str = _format_timestamp(scan.get('timestamp'))
            # Extract date portion (first 10 characters) if it's a full datetime string
            date_str = timestamp_str[:10] if len(timestamp_str) >= 10 else timestamp_str
            with st.expander(
                f"📊 {Path(scan['project_path']).name} - {scan['agent_name']} ({date_str})",
                expanded=False
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Project:** `{scan['project_path']}`")
                    st.markdown(f"**Agent:** {scan['agent_name']}")
                    st.markdown(f"**Model:** {scan['model_used']}")
                    st.markdown(f"**Files Scanned:** {scan['files_scanned']}")
                    st.markdown(f"**Chunks Analyzed:** {scan['chunks_analyzed']}")
                    timestamp_display = _format_timestamp(scan.get('timestamp'))
                    st.markdown(f"**Timestamp:** {timestamp_display}")
                    
                    # Status
                    if scan.get("status") == "completed":
                        st.success("✓ Completed")
                    elif scan.get("status") == "error":
                        st.error("✗ Error")
                    else:
                        st.info(f"Status: {scan.get('status', 'unknown')}")
                
                with col2:
                    scan_id = scan["id"]
                    
                    # View Details button
                    if st.button("👁️ View", key=f"{key_prefix}_view_{scan_id}", width='stretch'):
                        st.session_state[f"{key_prefix}_view_scan_{scan_id}"] = True
                        # Set a flag to prevent immediate auto-close on first render
                        st.session_state[f"{key_prefix}_view_scan_{scan_id}_just_opened"] = True
                    
                    # Download buttons
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        if st.button("📥 MD", key=f"{key_prefix}_dl_md_{scan_id}", width='stretch'):
                            st.session_state[f"{key_prefix}_download_md_{scan_id}"] = True
                    with col_dl2:
                        if st.button("📥 JSON", key=f"{key_prefix}_dl_json_{scan_id}", width='stretch'):
                            st.session_state[f"{key_prefix}_download_json_{scan_id}"] = True
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"{key_prefix}_delete_{scan_id}", width='stretch', type="secondary"):
                        st.session_state[f"{key_prefix}_delete_scan_{scan_id}"] = True
                
                # Handle view details
                if st.session_state.get(f"{key_prefix}_view_scan_{scan_id}", False):
                    st.divider()
                    full_scan = get_scan_by_id(scan_id)
                    download_key_base = f"{key_prefix}_{scan_id}"
                    
                    # Display results FIRST - this allows button clicks to set triggers
                    if full_scan and full_scan.get("result"):
                        display_analysis_results(full_scan["result"], full_scan["project_path"], unique_id=f"{key_prefix}_{scan_id}")
                    
                    # Check for active download triggers AFTER calling display_analysis_results
                    # This ensures we see triggers that were set during button clicks
                    has_active_downloads = False
                    
                    # Check for synthesized report download triggers
                    synth_md_trigger = st.session_state.get(f"{download_key_base}_dl_synth_md_trigger", False)
                    synth_json_trigger = st.session_state.get(f"{download_key_base}_dl_synth_json_trigger", False)
                    if synth_md_trigger or synth_json_trigger:
                        has_active_downloads = True
                        logger.info(f"Active download detected: synthesized report for scan {scan_id}")
                    
                    # Check for "download all" triggers
                    all_md_trigger_key = f"{download_key_base}_dl_all_md_trigger"
                    all_json_trigger_key = f"{download_key_base}_dl_all_json_trigger"
                    all_md_trigger = st.session_state.get(all_md_trigger_key, False)
                    all_json_trigger = st.session_state.get(all_json_trigger_key, False)
                    if all_md_trigger or all_json_trigger:
                        has_active_downloads = True
                        logger.info(f"Active download detected: all agents for scan {scan_id}")
                    
                    # Check for individual agent download triggers
                    # We check all session state keys that start with our download key base
                    if not has_active_downloads:
                        for key in st.session_state.keys():
                            if (key.startswith(f"{download_key_base}_dl_agent_") and 
                                key.endswith("_trigger") and 
                                st.session_state.get(key, False)):
                                has_active_downloads = True
                                logger.info(f"Active download detected: individual agent for scan {scan_id}, key: {key}")
                                break
                    
                    # Add close button if there are active downloads
                    if has_active_downloads:
                        st.divider()
                        if st.button("❌ Close View", key=f"{key_prefix}_close_view_{scan_id}", type="secondary"):
                            logger.info(f"User manually closed view for scan {scan_id}")
                            # Reset all download triggers for this scan
                            for key in list(st.session_state.keys()):
                                if key.startswith(f"{download_key_base}_dl_") and key.endswith("_trigger"):
                                    st.session_state[key] = False
                                    logger.info(f"Reset download trigger: {key}")
                                if key.startswith(f"{download_key_base}_dl_") and key.endswith("_trigger_time"):
                                    del st.session_state[key]
                            st.session_state[f"{key_prefix}_view_scan_{scan_id}"] = False
                            st.rerun()
                    
                    # Only auto-close the view if there are no active downloads and user hasn't manually closed it
                    # Don't auto-close on the same render cycle that the view was just opened
                    # This allows button clicks to be processed before the view closes
                    just_opened_key = f"{key_prefix}_view_scan_{scan_id}_just_opened"
                    just_opened = st.session_state.get(just_opened_key, False)
                    if just_opened:
                        # Clear the flag - view is now open and can be closed on next render if needed
                        st.session_state[just_opened_key] = False
                        logger.info(f"View just opened for scan {scan_id}, skipping auto-close on this render")
                    
                    # Don't auto-close if view was just opened (give button clicks a chance to be processed)
                    # or if there are active downloads
                    if not has_active_downloads and not just_opened and st.session_state.get(f"{key_prefix}_view_scan_{scan_id}", False):
                        logger.info(f"Auto-closing view for scan {scan_id} - no active downloads")
                        st.session_state[f"{key_prefix}_view_scan_{scan_id}"] = False
                
                # Handle downloads
                if st.session_state.get(f"{key_prefix}_download_md_{scan_id}", False):
                    full_scan = get_scan_by_id(scan_id)
                    if full_scan and full_scan.get("result"):
                        project_name = Path(full_scan["project_path"]).name
                        markdown_report = generate_markdown_report(full_scan["result"], project_name)
                        agent_name = full_scan["result"].get("agent_name", "Code Reviewer")
                        timestamp_str = _format_timestamp(full_scan.get("timestamp"))
                        timestamp = timestamp_str.replace(":", "-")
                        st.download_button(
                            label="📥 Download Markdown Report",
                            data=markdown_report,
                            file_name=f"{agent_name.lower().replace(' ', '_')}_{project_name}_{timestamp}.md",
                            mime="text/markdown",
                            key=f"{key_prefix}_dl_md_file_{scan_id}"
                        )
                    st.session_state[f"{key_prefix}_download_md_{scan_id}"] = False
                
                if st.session_state.get(f"{key_prefix}_download_json_{scan_id}", False):
                    full_scan = get_scan_by_id(scan_id)
                    if full_scan and full_scan.get("result"):
                        project_name = Path(full_scan["project_path"]).name
                        json_report = generate_json_report(full_scan["result"], project_name)
                        agent_name = full_scan["result"].get("agent_name", "Code Reviewer")
                        timestamp_str = _format_timestamp(full_scan.get("timestamp"))
                        timestamp = timestamp_str.replace(":", "-")
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=json_report,
                            file_name=f"{agent_name.lower().replace(' ', '_')}_{project_name}_{timestamp}.json",
                            mime="application/json",
                            key=f"{key_prefix}_dl_json_file_{scan_id}"
                        )
                    st.session_state[f"{key_prefix}_download_json_{scan_id}"] = False
                
                # Handle delete
                if st.session_state.get(f"{key_prefix}_delete_scan_{scan_id}", False):
                    if delete_scan(scan_id):
                        st.success(f"✅ Scan {scan_id} deleted successfully")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to delete scan {scan_id}")
                    st.session_state[f"{key_prefix}_delete_scan_{scan_id}"] = False
                
    except Exception as e:
        from utils.error_formatting import format_user_error
        logger.error(f"Error loading scan history: {e}", exc_info=True)
        st.error(f"Error loading scan history: {format_user_error(e)}")

