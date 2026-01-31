"""
Health Status Page for CodeInsight.

Displays health status of all services with both aggregated and individual views.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import altair as alt
import logging
from datetime import datetime
from typing import Dict, Any

from services.health_monitor import get_health_monitor
from services.health_check import HealthStatus
from ui.app import initialize_global_settings
from utils.sidebar import render_sidebar

logger = logging.getLogger(__name__)


def get_status_color(status: HealthStatus) -> str:
    """Get color for status badge."""
    color_map = {
        HealthStatus.HEALTHY: "🟢",
        HealthStatus.DEGRADED: "🟡",
        HealthStatus.UNHEALTHY: "🔴",
        HealthStatus.UNKNOWN: "⚪"
    }
    return color_map.get(status, "⚪")


def get_status_badge_color(status: HealthStatus) -> str:
    """Get Streamlit badge color for status."""
    color_map = {
        HealthStatus.HEALTHY: "normal",
        HealthStatus.DEGRADED: "warning",
        HealthStatus.UNHEALTHY: "error",
        HealthStatus.UNKNOWN: "off"
    }
    return color_map.get(status, "off")


def format_response_time(response_time_ms: float) -> str:
    """Format response time in milliseconds."""
    if response_time_ms is None:
        return "N/A"
    if response_time_ms < 1000:
        return f"{response_time_ms:.1f} ms"
    else:
        return f"{response_time_ms / 1000:.2f} s"


def main():
    """Health Status page."""
    # Initialize settings
    initialize_global_settings()
    
    # Apply layout mode
    layout = st.session_state.get("ui_layout_mode", "wide")
    st.set_page_config(
        page_title="Health Status - CodeInsight",
        page_icon="🏥",
        layout=layout,
        initial_sidebar_state="expanded"
    )
    
    # Render sidebar
    render_sidebar()
    
    # Page header
    from utils.version import VERSION_STRING
    st.title("🏥 System Health Status")
    st.markdown(f"""
    **Service Health & Monitoring** | `{VERSION_STRING}`
    
    Monitor the health status of all services used by CodeInsight.
    """)
    
    # Initialize health monitor
    try:
        health_monitor = get_health_monitor()
    except Exception as e:
        logger.error(f"Error initializing health monitor: {e}", exc_info=True)
        st.error(f"Error initializing health monitor: {e}")
        return
    
    # Refresh button and auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        # Visual Polish: Modern Toggle
        auto_refresh = st.toggle("Auto-refresh (30s)", value=False, help="Automatically refresh health status every 30 seconds")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    
    # Get health summary
    try:
        summary = health_monitor.get_health_summary()
        overall_status = HealthStatus(summary["overall_status"])
        all_services = summary["services"]
    except Exception as e:
        logger.error(f"Error getting health summary: {e}", exc_info=True)
        st.error(f"Error getting health summary: {e}")
        return

    # Visual Polish: Toast Notification for issues
    if overall_status != HealthStatus.HEALTHY:
        st.toast(f"System Health Alert: {overall_status.value.upper()}", icon="⚠️")
    
    # Aggregated Status Section
    st.divider()
    
    # Visual Polish: Summary Card
    with st.container(border=True):
        st.subheader("📊 Overall System Health")
        
        # Overall status badge
        status_emoji = get_status_color(overall_status)
        status_badge_color = get_status_badge_color(overall_status)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Overall Status",
                f"{status_emoji} {overall_status.value.upper()}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Healthy Services",
                summary["healthy_count"],
                delta=None
            )
        
        with col3:
            st.metric(
                "Degraded Services",
                summary["degraded_count"],
                delta=None
            )
        
        with col4:
            st.metric(
                "Unhealthy Services",
                summary["unhealthy_count"],
                delta=None
            )
        
        # Visual Polish: Native Altair Chart instead of HTML
        total_services = summary["service_count"]
        if total_services > 0:
            st.caption("Service Health Distribution")
            status_data = pd.DataFrame({
                'Status': ['Healthy', 'Degraded', 'Unhealthy', 'Unknown'],
                'Count': [summary['healthy_count'], summary['degraded_count'], summary['unhealthy_count'], summary['unknown_count']],
                'Color': ['#28a745', '#ffc107', '#dc3545', '#6c757d']
            })
            
            # Filter zero counts to make chart cleaner
            status_data = status_data[status_data['Count'] > 0]
            
            chart = alt.Chart(status_data).mark_bar(cornerRadius=4).encode(
                x=alt.X('sum(Count)', stack='normalize', axis=None, title=None),
                color=alt.Color('Status', scale=alt.Scale(
                    domain=['Healthy', 'Degraded', 'Unhealthy', 'Unknown'], 
                    range=['#28a745', '#ffc107', '#dc3545', '#6c757d']
                ), legend=None),
                tooltip=['Status', 'Count']
            ).properties(height=30)
            
            st.altair_chart(chart, use_container_width=True)
            
            # Simple Legend
            legend_cols = st.columns(4)
            if summary['healthy_count'] > 0: legend_cols[0].caption(f"🟢 Healthy: {summary['healthy_count']}")
            if summary['degraded_count'] > 0: legend_cols[1].caption(f"🟡 Degraded: {summary['degraded_count']}")
            if summary['unhealthy_count'] > 0: legend_cols[2].caption(f"🔴 Unhealthy: {summary['unhealthy_count']}")
            if summary['unknown_count'] > 0: legend_cols[3].caption(f"⚪ Unknown: {summary['unknown_count']}")
    
    # Individual Services Section
    st.divider()
    st.subheader("🔍 Individual Service Status")
    
    # Visual Polish: Dataframe instead of manual loop
    service_rows = []
    for service_name, service_data in all_services.items():
        service_status = service_data.get("status", "unknown")
        # Add emoji for filtration/sorting visual
        
        service_rows.append({
            "Service": service_name,
            "Status": service_status.upper(),
            "Response Time (ms)": service_data.get("response_time_ms", None),
            "Message": service_data.get("message", ""),
            "Last Update": service_data.get("timestamp", "")
        })
    
    if service_rows:
        # Sort by Service name to ensure stable indices across reruns
        service_rows.sort(key=lambda x: x["Service"])
        df_services = pd.DataFrame(service_rows)
        
        # Display dataframe for overview (no selection)
        st.dataframe(
            df_services,
            width='stretch',
            hide_index=True,
            column_config={
                "Service": st.column_config.TextColumn("Service", help="Service Name", width="medium"),
                "Status": st.column_config.TextColumn(
                    "Status",
                    help="Current Health Status",
                    width="small"
                ),
                "Response Time (ms)": st.column_config.NumberColumn(
                    "Response Time", 
                    format="%.1f ms",
                    help="Latency in milliseconds"
                ),
                "Message": st.column_config.TextColumn("Message", width="large"),
                "Last Update": st.column_config.DatetimeColumn("Last Check", format="DD MMM HH:mm:ss")
            },
            key="health_service_table_display"
        )
        
        # Master-Detail View with Selectbox
        st.divider()
        st.caption("🔍 View Service Details")
        
        service_names = [row["Service"] for row in service_rows]
        selected_service_name = st.selectbox(
            "Select a service to view full details:",
            options=["-- Select Service --"] + service_names,
            key="health_service_selector"
        )
        
        if selected_service_name and selected_service_name != "-- Select Service --":
            service_data = all_services.get(selected_service_name, {})
            details = service_data.get("details", {})
            
            with st.container(border=True):
                st.subheader(f"ℹ️ Details: {selected_service_name}")
                
                # Main metrics row
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.write(f"**Status:** {service_data.get('status', 'unknown').upper()}")
                with m_col2:
                    rt = service_data.get('response_time_ms')
                    st.write(f"**Response:** {f'{rt:.1f} ms' if rt else 'N/A'}")
                with m_col3:
                    ts = service_data.get('timestamp', '')
                    if ts:
                        try:
                            parsed_ts = ts.replace('T', ' ').split('.')[0]
                            st.write(f"**Updated:** {parsed_ts}")
                        except:
                            st.write(f"**Updated:** {ts}")

                # Error Highlight
                if service_data.get("status") in ["unhealthy", "degraded"]:
                    if "error" in details:
                        st.error(f"Error: {details['error']}", icon="🚨")
                
                # Detailed KV pairs
                if details:
                    st.markdown("##### Configuration & Metadata")
                    # Filter out error as it is already shown
                    display_details = {k: v for k, v in details.items() if k != "error"}
                    
                    if display_details:
                        d_cols = st.columns(min(3, len(display_details)))
                        for idx, (k, v) in enumerate(display_details.items()):
                            col_idx = idx % 3
                            with d_cols[col_idx]:
                                if isinstance(v, (dict, list)):
                                    st.write(f"**{k}:**")
                                    st.json(v, expanded=False)
                                else:
                                    st.write(f"**{k}:**")
                                    st.code(str(v), language=None)
                else:
                    st.caption("No additional details available for this service.")

    else:
        st.info("No services found.")
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()

