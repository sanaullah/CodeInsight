"""
Dashboard widgets for CodeInsight.
"""

import streamlit as st

def stat_card(label, value, delta=None):
    """
    Displays a statistic in a card-like format.
    """
    with st.container():
        st.markdown(
            f"""
            <div style="
                background-color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 10px;
            ">
                <p style="color: #666; font-size: 14px; margin-bottom: 5px;">{label}</p>
                <h3 style="color: #0F52BA; margin: 0;">{value}</h3>
                {f'<p style="color: {"green" if delta and delta > 0 else "red"}; font-size: 12px; margin-top: 5px;">{"+" if delta and delta > 0 else ""}{delta}%</p>' if delta is not None else ""}
            </div>
            """,
            unsafe_allow_html=True
        )

def activity_feed(items):
    """
    Displays a list of recent activities.
    """
    st.subheader("Recent Activity")
    for item in items:
        with st.container():
            st.markdown(
                f"""
                <div style="
                    padding: 10px;
                    border-bottom: 1px solid #eee;
                    display: flex;
                    align_items: center;
                ">
                    <div style="
                        width: 8px;
                        height: 8px;
                        background-color: #0F52BA;
                        border-radius: 50%;
                        margin-right: 10px;
                    "></div>
                    <div>
                        <p style="margin: 0; font-weight: 500;">{item['action']}</p>
                        <p style="margin: 0; font-size: 12px; color: #888;">{item['timestamp']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

