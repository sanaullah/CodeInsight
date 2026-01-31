"""
Approvals page for CodeInsight.
"""

import streamlit as st
from utils.sidebar import render_sidebar
from ui.components.approval_ui import render_approval_tab


def main():
    """Approvals page."""
    st.set_page_config(
        page_title="Approvals - CodeInsight",
        page_icon="⏸️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    render_sidebar()
    
    from utils.version import VERSION_STRING
    st.title("⏸️ Pending Approvals")
    st.markdown(f"**Human-in-the-loop validation** | `{VERSION_STRING}`")
    st.divider()
    
    render_approval_tab()


if __name__ == "__main__":
    main()

