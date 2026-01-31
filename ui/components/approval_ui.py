"""
Approval UI components for CodeInsight.

This module provides UI components for displaying approval requests.
Note: Full approval management system will be integrated in future phases.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime


def display_approval_request(approval: Dict[str, Any]):
    """
    Display an approval request with action buttons.
    
    Args:
        approval: Approval request dictionary
    """
    st.subheader(f"⏸️ Approval Required: {approval.get('workflow_name', 'Unknown')}")
    
    # Approval details
    col1, col2, col3 = st.columns(3)
    with col1:
        approval_id = approval.get('approval_id', 'N/A')
        st.metric("Approval ID", approval_id[:8] + "..." if len(approval_id) > 8 else approval_id)
    with col2:
        st.metric("Node", approval.get('node_name', 'N/A'))
    with col3:
        status = approval.get('status', 'pending')
        status_emoji = "⏳" if status == "pending" else "✅" if status == "approved" else "❌"
        st.metric("Status", f"{status_emoji} {status.title()}")
    
    st.divider()
    
    # Request information
    st.markdown("### 📋 Request Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Workflow:** {approval.get('workflow_name', 'N/A')}")
        st.markdown(f"**Thread ID:** {approval.get('thread_id', 'N/A')[:8]}...")
    with col2:
        created_at = approval.get('created_at', datetime.now())
        if isinstance(created_at, str):
            st.markdown(f"**Created:** {created_at}")
        else:
            st.markdown(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        expires_at = approval.get('expires_at')
        if expires_at:
            if isinstance(expires_at, str):
                expires_dt = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            else:
                expires_dt = expires_at
            time_remaining = (expires_dt - datetime.now()).total_seconds()
            if time_remaining > 0:
                st.markdown(f"**Expires in:** {int(time_remaining // 60)} minutes")
            else:
                st.markdown("**Status:** Expired")
    
    context = approval.get('context', {})
    if context:
        with st.expander("🔍 Context Details", expanded=False):
            st.json(context)
    
    st.divider()
    
    # Action buttons (only if pending)
    if status == "pending":
        st.markdown("### ⚡ Actions")
        st.info("💡 Approval management system is being integrated. For now, approvals are handled automatically.")
    else:
        # Show response if already resolved
        response = approval.get('response')
        if response:
            st.info(f"**Response:** {response.get('reason', 'No reason provided')}")


def display_approval_history(history: List[Dict[str, Any]], limit: int = 20):
    """
    Display approval request history.
    
    Args:
        history: List of approval request dictionaries
        limit: Maximum number of entries to display
    """
    st.subheader("📜 Approval History")
    
    if not history:
        st.info("No approval requests in history.")
        return
    
    # Display history
    for approval in history[:limit]:
        with st.expander(
            f"{approval.get('workflow_name', 'Unknown')} - {approval.get('node_name', 'Unknown')} - {approval.get('status', 'unknown')}",
            expanded=False
        ):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Approval ID:** `{approval.get('approval_id', 'N/A')[:16]}...`")
                st.markdown(f"**Workflow:** {approval.get('workflow_name', 'N/A')}")
                st.markdown(f"**Node:** {approval.get('node_name', 'N/A')}")
            with col2:
                st.markdown(f"**Status:** {approval.get('status', 'N/A')}")
                created_at = approval.get('created_at', datetime.now())
                if isinstance(created_at, str):
                    st.markdown(f"**Created:** {created_at}")
                else:
                    st.markdown(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if approval.get('context'):
                with st.expander("Context"):
                    st.json(approval['context'])
            
            if approval.get('response'):
                st.markdown(f"**Response:** {approval['response'].get('reason', 'No reason provided')}")


def display_pending_approvals(pending: List[Dict[str, Any]]):
    """
    Display list of pending approval requests.
    
    Args:
        pending: List of pending approval request dictionaries
    """
    st.subheader("⏳ Pending Approvals")
    
    if not pending:
        st.success("✅ No pending approvals.")
        return
    
    st.warning(f"⚠️ You have {len(pending)} pending approval(s) requiring your attention.")
    
    for idx, approval in enumerate(pending):
        if idx > 0:
            st.divider()
        display_approval_request(approval)


def render_approval_tab():
    """
    Render a complete approval management tab.
    """
    st.header("⏸️ Approval Management")
    
    st.info("💡 Approval management system is being integrated. Full functionality will be available in a future update.")
    
    # Placeholder for future implementation
    st.markdown("### ⏳ Pending Approvals")
    st.info("No pending approvals at this time.")
    
    st.markdown("### 📜 History")
    st.info("No approval history available.")

