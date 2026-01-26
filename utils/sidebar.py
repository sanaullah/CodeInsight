"""
Reusable sidebar component for CodeInsight.

Provides consistent navigation and status display across all pages.
"""

import streamlit as st
from datetime import datetime, timezone


def render_sidebar():
    """
    Render the shared sidebar with navigation and status.
    
    This function:
    - Displays sidebar title
    - Uses native multipage navigation
    - Shows current date/time in a status block
    - Includes a refresh button
    """
    # Sidebar title
    st.sidebar.title("💡 CodeInsight")
    
    # Native multipage navigation
    st.sidebar.markdown("### 🧭 Navigation")
    st.sidebar.page_link("app.py", label="Dashboard", icon="📊")
    
    st.sidebar.markdown("#### Analysis")
    st.sidebar.page_link("pages/swarm_analysis.py", label="Swarm Analysis", icon="🐝")
    st.sidebar.page_link("pages/4_Approvals.py", label="Approvals", icon="⏸️")
    
    st.sidebar.markdown("#### Management")
    st.sidebar.page_link("pages/2_Architecture_Model.py", label="Architecture Model", icon="🏗️")
    st.sidebar.page_link("pages/3_Health_Status.py", label="Health Status", icon="🏥")
    st.sidebar.page_link("pages/1_Settings.py", label="Settings", icon="⚙️")
    
    st.sidebar.divider()
    
    # Status block with current date/time
    now_local = datetime.now(timezone.utc).astimezone()
    st.sidebar.info(f"🕐 {now_local.strftime('%Y-%m-%d %H:%M')}")
    
    st.sidebar.divider()
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh page", use_container_width=True):
        st.rerun()

