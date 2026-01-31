"""
Agent cards component for CodeInsight.
"""

import streamlit as st

def render_agent_card(agent_data):
    """
    Renders a card for a single agent.
    """
    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                margin-bottom: 20px;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h4 style="margin: 0; color: #333;">{agent_data['name']}</h4>
                    <span style="
                        background-color: {'#e6f4ea' if agent_data['status'] == 'Active' else '#fce8e6'};
                        color: {'#1e8e3e' if agent_data['status'] == 'Active' else '#c5221f'};
                        padding: 2px 8px;
                        border-radius: 12px;
                        font-size: 12px;
                        font-weight: 500;
                    ">{agent_data['status']}</span>
                </div>
                <p style="color: #666; font-size: 14px; margin-bottom: 15px;">{agent_data['role']}</p>
                <div style="border-top: 1px solid #eee; padding-top: 10px;">
                    <p style="font-size: 12px; color: #888; margin: 0;">Last Active: {agent_data['last_active']}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

