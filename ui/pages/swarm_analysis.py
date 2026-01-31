"""
Swarm Analysis Page for CodeInsight.

Allows users to run autonomous swarm analysis using LangGraph-based orchestrator.
"""

import streamlit as st
from utils.version import VERSION_STRING
import logging
import queue
import threading
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd

from agents.swarm_analysis_orchestrator import SwarmAnalysisOrchestrator
from utils.sidebar import render_sidebar
from utils.scan_history import save_scan
from ui.analysis_utils import display_swarm_results
from ui.app import initialize_global_settings
from llm.config import ConfigManager
from utils.config.settings_db import init_settings_db, get_user_id, get_setting
from workflows.integration import setup_langfuse_callbacks_for_streamlit
from ui.components.prompt_library_ui import render_prompt_library_tab
from ui.components.scan_history_ui import render_scan_history_tab
from ui.components.directory_selector import render_directory_tree_selector
from utils.config.project_settings import (
    get_project_selected_directories,
    save_project_selected_directories,
)
from utils.error_formatting import format_user_error

logger = logging.getLogger(__name__)


def render_swarm_dashboard(placeholder, state):
    """Render the high-fidelity swarm analysis dashboard."""
    with placeholder.container():
        # Overall Stats Header
        st.markdown("### 🐝 Swarm Analysis Dashboard")
        
        # Split into two panels
        left_col, right_col = st.columns([1, 2])
        
        with left_col:
            with st.container(border=True):
                st.markdown("#### 📊 Overall Progress")
                
                # Goal & Project
                if state.get("goal"):
                    st.caption(f"🎯 **Goal:** {state['goal']}")
                st.caption(f"📂 **Project:** {state.get('project_path', 'Unknown')}")
                
                # Progress Stats
                completed_chunks = state.get("completed_chunks", 0)
                total_chunks = state.get("total_chunks_estimated", 0)
                
                # Calculate percentage
                progress_pct = 0
                if total_chunks > 0:
                    progress_pct = min(completed_chunks / total_chunks, 1.0)
                
                # Display Circular-like progress using metric + progress bar
                st.metric("Chunks Processed", f"{completed_chunks}/{total_chunks}Chunks")
                st.progress(progress_pct)
                
                # Active Agents Count
                active_count = len(state.get("active_agents", {}))
                st.metric("⚡ Active Agents", active_count)
                
                # Time Elapsed
                start_time = state.get("start_time")
                if start_time:
                    elapsed = datetime.now() - start_time
                    st.caption(f"⏱️ Time: {str(elapsed).split('.')[0]}")

        with right_col:
            st.markdown("#### ⚡ Active Agents")
            
            active_agents = state.get("active_agents", {})
            if active_agents:
                for agent_name, agent_data in active_agents.items():
                    with st.container(border=True):
                        col_icon, col_details = st.columns([1, 10])
                        with col_icon:
                            st.markdown("🤖")
                        with col_details:
                            st.markdown(f"**{agent_name}**")
                            # Progress bar for this agent
                            chunk_idx = agent_data.get("chunk_index", 0)
                            total_chunks = agent_data.get("total_chunks", 1)
                            agent_pct = 0
                            if total_chunks > 0:
                                agent_pct = min(chunk_idx / total_chunks, 1.0)
                            
                            st.progress(agent_pct)
                            
                            # Status text
                            file_name = agent_data.get("file", "Starting...")
                            st.caption(f"📄 {file_name}")
                            st.caption(f"Status: Chunk {chunk_idx}/{total_chunks}")
            else:
                st.info("Waiting for agents to start...")

            # Recently Completed
            completed_agents = state.get("completed_agents", [])
            if completed_agents:
                st.markdown("#### ✅ Recently Completed")
                for agent in completed_agents[-3:]: # Show last 3
                    with st.container(border=True):
                        st.markdown(f"✅ **{agent['name']}**")
                        st.caption(f"Processed {agent.get('chunks', 0)} chunks")


def main():
    """Swarm Analysis page."""
    # Initialize settings
    initialize_global_settings()

    # Apply layout mode
    layout = st.session_state.get("ui_layout_mode", "wide")
    st.set_page_config(
        page_title="Swarm Analysis - CodeInsight",
        page_icon="🐝",
        layout=layout,
        initial_sidebar_state="expanded",
    )

    # Render sidebar
    render_sidebar()

    # Page header
    st.title("🐝 Swarm Analysis")
    st.markdown(f"**Autonomous Swarm Analysis** | `Version: {VERSION_STRING}`")
    st.markdown("""
    Uses LLM to intelligently determine analysis roles,
    generate context-specific prompts, validate quality, and execute agents in parallel
    for comprehensive project analysis.
    """)

    # Configuration (needed for Analysis tab)
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        available_models = config_manager.get_available_models()
    except Exception as e:
        logger.error(f"Error loading model configuration: {e}", exc_info=True)
        available_models = ["qwen/qwen3-coder", "gpt-4o-mini", "gpt-4"]
        config = None

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Analysis", "📚 Prompt Library", "📜 Scan History"])

    with tab1:
        # Analysis tab content
        _render_analysis_tab(config, available_models)

    with tab2:
        render_prompt_library_tab()

    with tab3:
        render_scan_history_tab(
            key_prefix="swarm_analysis_page", agent_filter="Swarm Analysis"
        )


def _render_analysis_tab(config, available_models):
    """Render the Analysis tab content."""

    with st.container(border=True):
        st.subheader("⚙️ Swarm Configuration")
        col1, col2 = st.columns(2)

        with col1:
            # Project root input
            project_root = st.text_input(
                "Project Root",
                value=st.session_state.get("swarm_project_root", ""),
                help="Absolute path to the project root directory",
            )
            st.session_state.swarm_project_root = project_root

        # Directory Selection
        if project_root:
            with st.expander("📁 Directory Selection", expanded=False):
                st.info(
                    "Select specific folders to analyze. Leave empty to scan entire project."
                )

                try:
                    # Initialize settings and get user_id
                    init_settings_db()
                    user_id = get_user_id()

                    # Load saved selections for this project
                    saved_dirs = get_project_selected_directories(user_id, project_root)

                    # Render directory tree selector
                    selected_dirs = render_directory_tree_selector(
                        project_root=project_root,
                        selected_directories=saved_dirs,
                        key_prefix="swarm_dir_selector",
                    )

                    # Save selections when changed
                    if selected_dirs != saved_dirs:
                        save_project_selected_directories(
                            user_id, project_root, selected_dirs
                        )
                        st.session_state.swarm_selected_directories = selected_dirs

                    # Store in session state for use in analysis
                    if "swarm_selected_directories" not in st.session_state:
                        st.session_state.swarm_selected_directories = selected_dirs
                    else:
                        st.session_state.swarm_selected_directories = selected_dirs

                    # Show preview
                    if selected_dirs:
                        st.success(
                            f"Selected {len(selected_dirs)} folder(s) for analysis"
                        )
                        st.success(
                            f"Selected {len(selected_dirs)} folder(s) for analysis"
                        )

                        # Use DataFrame for professional display
                        df_selected = pd.DataFrame({"Selected Paths": selected_dirs})
                        st.dataframe(
                            df_selected,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Selected Paths": st.column_config.TextColumn(
                                    "📁 Analysis Targets",
                                    help="Folders included in this scan",
                                    disabled=True,
                                )
                            },
                        )
                    else:
                        st.info("No folders selected - entire project will be scanned")

                except Exception as e:
                    logger.error(f"Error in directory selection: {e}", exc_info=True)
                    st.warning(f"Error loading directory selector: {str(e)}")
                    st.session_state.swarm_selected_directories = []
        else:
            st.session_state.swarm_selected_directories = []

        with col1:
            max_agents = st.slider(
                "Max Agents",
                min_value=1,
                max_value=100,
                value=10,
                help="Maximum number of agents to spawn",
            )

            goal = st.text_area(
                "Analysis Goal (Optional)",
                height=100,
                help="Optional goal to guide role selection and prompt generation",
                placeholder="Example: Focus on security vulnerabilities and performance bottlenecks",
            )

        with col2:
            # Get default model
            default_model = None
            try:
                init_settings_db()
                user_id = get_user_id()
                db_model = get_setting(user_id, "selected_model", None)
                if db_model and db_model in available_models:
                    default_model = db_model
            except Exception:
                pass

            if not default_model:
                default_model = st.session_state.get("selected_model")

            if not default_model or default_model not in available_models:
                default_model = (
                    getattr(config, "default_model", None) if config else None
                )
                if not default_model or default_model not in available_models:
                    default_model = (
                        available_models[0] if available_models else "qwen/qwen3-coder"
                    )

            try:
                default_index = (
                    available_models.index(default_model)
                    if default_model in available_models
                    else 0
                )
            except (ValueError, AttributeError):
                default_index = 0

            model_name = st.selectbox(
                "Model",
                options=available_models,
                index=default_index,
                help="AI model to use for analysis",
            )

            max_tokens = st.number_input(
                "Max Tokens per Chunk",
                min_value=10000,
                max_value=500000,
                value=100000,
                step=10000,
                help="Maximum tokens per chunk for analysis",
            )

    # Advanced Configuration (Chunking & Language)
    with st.expander("🛠️ Advanced Settings", expanded=False):
        st.caption("Fine-tune how files are processed and detected.")

        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            enable_chunking = st.toggle(
                "Enable File Chunking",
                value=True,
                help="Enable file chunking for context-aware analysis.",
                key="swarm_enable_chunking",
            )

            if enable_chunking:
                chunking_strategy = st.selectbox(
                    "Chunking Strategy",
                    options=["STANDARD", "AGGRESSIVE"],
                    index=0,
                    help="STANDARD (100K tokens), AGGRESSIVE (50K tokens)",
                    key="swarm_chunking_strategy",
                )
            else:
                chunking_strategy = "STANDARD"

        with col_adv2:
            auto_detect = st.toggle(
                "Auto-detect Languages",
                value=True,
                help="Automatically detect programming languages in the project",
                key="swarm_auto_detect_languages",
            )

            if not auto_detect:
                st.warning("Manual selection coming soon")

    # Previous Report Comparison
    with st.expander("📊 Compare with Previous Analysis (Optional)", expanded=False):
        from utils.scan_history import search_scans, get_latest_swarm_analysis
        from utils.report_loader import extract_synthesized_report_from_result
        from agents.swarm_analysis_learning import get_previous_report_from_experiences
        import tempfile
        import os

        previous_report_option = st.segmented_control(
            "Previous Report Source",
            options=[
                "None",
                "Upload File",
                "Select from History",
                "Auto-select Latest",
                "Auto-select Similar",
            ],
            selection_mode="single",
            default="None",
            help="Compare current analysis with a previous report to track progress and identify resolved issues",
        )

        previous_report_path = None
        previous_report_id = None

        if previous_report_option == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Previous Report",
                type=["md", "txt"],
                help="Upload a previous swarm analysis report (.md or .txt)",
            )
            if uploaded_file:
                try:
                    content = uploaded_file.read().decode("utf-8")
                    with tempfile.NamedTemporaryFile(
                        mode="w", delete=False, suffix=".md", encoding="utf-8"
                    ) as tmp:
                        tmp.write(content)
                        previous_report_path = tmp.name
                    st.success(
                        f"Loaded report: {uploaded_file.name} ({len(content)} chars)"
                    )
                except Exception as exc:
                    st.error(f"Error loading file: {exc}")
                    logger.error("Error loading uploaded report", exc_info=True)

        elif previous_report_option == "Select from History":
            try:
                scans = search_scans(agent_name="Swarm Analysis", limit=50)
                if scans:
                    scan_options: Dict[str, int] = {}
                    for scan in scans:
                        timestamp = scan.get(
                            "timestamp", scan.get("created_at", "Unknown")
                        )
                        project = scan.get("project_path", "Unknown")
                        scan_id = scan.get("id")
                        if scan_id is not None:
                            display_name = f"{timestamp} - {project}"
                            scan_options[display_name] = scan_id

                    if scan_options:
                        selected_scan_display = st.selectbox(
                            "Select Previous Analysis",
                            options=list(scan_options.keys()),
                            help="Choose a previous Swarm Analysis from scan history",
                        )
                        if selected_scan_display:
                            previous_report_id = scan_options[selected_scan_display]
                            selected_scan = next(
                                s for s in scans if s.get("id") == previous_report_id
                            )
                            st.info(
                                f"Selected: {selected_scan.get('timestamp', 'Unknown')} - "
                                f"{selected_scan.get('project_path', 'Unknown')}"
                            )
                    else:
                        st.warning("No Swarm Analysis scans found in history")
                else:
                    st.info("No previous Swarm Analysis scans found")
            except Exception as exc:
                st.error(f"Error loading scan history: {exc}")
                logger.error("Error loading scan history", exc_info=True)

        elif previous_report_option == "Auto-select Latest":
            if project_root:
                try:
                    latest_scan = get_latest_swarm_analysis(project_root)
                    if latest_scan:
                        previous_report_id = latest_scan.get("id")
                        timestamp = latest_scan.get(
                            "timestamp", latest_scan.get("created_at", "Unknown")
                        )
                        st.success(f"Auto-selected latest analysis: {timestamp}")
                    else:
                        st.info(f"No previous Swarm Analysis found for: {project_root}")
                except Exception as exc:
                    st.error(f"Error finding latest analysis: {exc}")
                    logger.error("Error finding latest analysis", exc_info=True)
            else:
                st.warning(
                    "Please enter a project path first to auto-select latest analysis"
                )

        elif previous_report_option == "Auto-select Similar":
            # Use experience learning to find a similar previous report
            try:
                # We do not yet know the architecture_type here, so pass None
                learning_candidate = get_previous_report_from_experiences(
                    architecture_type=None,
                    goal=goal if goal and goal.strip() else None,
                    project_path=project_root if project_root else None,
                    limit=1,
                )
                if learning_candidate and learning_candidate.get("report_text"):
                    # Store the report in a temp file so the orchestrator can load it
                    report_text = learning_candidate["report_text"]
                    with tempfile.NamedTemporaryFile(
                        mode="w", delete=False, suffix=".md", encoding="utf-8"
                    ) as tmp:
                        tmp.write(report_text)
                        previous_report_path = tmp.name
                    st.success(
                        "Auto-selected a similar previous analysis from experience learning "
                        f"(experience_id={learning_candidate.get('experience_id', 'unknown')})"
                    )
                else:
                    st.info("No suitable previous analysis found in experience storage")
            except Exception as exc:
                st.error(f"Error auto-selecting similar analysis: {exc}")
                logger.error("Error in Auto-select Similar", exc_info=True)

        # Store in session state for use when running analysis
        st.session_state.swarm_previous_report_path = previous_report_path
        st.session_state.swarm_previous_report_id = previous_report_id

    # Run Button
    button_enabled = bool(project_root)

    if st.button(
        "🚀 Launch Swarm Analysis",
        type="primary",
        disabled=not button_enabled,
        width="stretch",
    ):
        # Get selected directories from session state (empty list if not set)
        selected_dirs = st.session_state.get("swarm_selected_directories", [])
        # Convert empty list to None for backward compatibility
        selected_dirs = selected_dirs if selected_dirs else None

        _run_swarm_analysis(
            project_path=project_root,
            goal=goal if goal.strip() else None,
            model_name=model_name,
            max_agents=max_agents,
            max_tokens=max_tokens,
            enable_chunking=enable_chunking,
            chunking_strategy=chunking_strategy,
            auto_detect_languages=auto_detect,
            selected_directories=selected_dirs,
            previous_report_path=st.session_state.get("swarm_previous_report_path"),
            previous_report_id=st.session_state.get("swarm_previous_report_id"),
        )

    # Display Results
    if st.session_state.get("swarm_results"):
        st.divider()
        display_swarm_results(st.session_state.swarm_results)


def _run_swarm_analysis(
    project_path: str,
    goal: Optional[str],
    model_name: str,
    max_agents: int,
    max_tokens: int,
    enable_chunking: bool = True,
    chunking_strategy: str = "STANDARD",
    auto_detect_languages: Optional[bool] = None,
    selected_directories: Optional[List[str]] = None,
    previous_report_path: Optional[str] = None,
    previous_report_id: Optional[int] = None,
):
    """Execute the swarm analysis using SwarmAnalysisOrchestrator."""

    # Initialize session state for tracking
    if "swarm_agent_statuses" not in st.session_state:
        st.session_state.swarm_agent_statuses = {}
    if "swarm_activity_feed" not in st.session_state:
        st.session_state.swarm_activity_feed = []
    if "swarm_stage_outputs" not in st.session_state:
        st.session_state.swarm_stage_outputs = {
            "project_scanning": None,
            "architecture_building": None,
            "role_selection": None,
            "prompt_generation": None,
            "agent_execution": None,
            "result_synthesis": None,
        }

    # Reset tracking state
    st.session_state.swarm_agent_statuses = {}
    st.session_state.swarm_activity_feed = []
    st.session_state.swarm_stage_outputs = {
        "project_scanning": None,
        "architecture_building": None,
        "role_selection": None,
        "prompt_generation": None,
        "agent_execution": None,
        "result_synthesis": None,
    }

    # Create progress bar
    progress_bar = st.progress(0, text="Starting analysis...")

    # Create event queue for streaming
    event_queue = queue.Queue(maxsize=1000)
    analysis_complete = threading.Event()

    # Stream callback function
    def stream_callback(event_type: str, data: Dict[str, Any]) -> None:
        """Put streaming events into thread-safe queue."""
        try:
            event_queue.put(
                {
                    "type": event_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                },
                timeout=1.0,
            )
        except queue.Full:
            logger.warning(f"Event queue full, dropping event: {event_type}")
        except Exception as e:
            logger.warning(f"Error putting event in queue: {e}")

    try:
        # Initialize dashboard state
        dashboard_state = {
            "goal": goal,
            "project_path": project_path,
            "start_time": datetime.now(),
            "active_agents": {},
            "completed_agents": [],
            "completed_chunks": 0,
            "total_chunks_estimated": 0, # Will update as we discover chunks
        }
        
        # Create dashboard placeholder
        dashboard_placeholder = st.empty()
        
        # Render initial empty dashboard
        render_swarm_dashboard(dashboard_placeholder, dashboard_state)

        # Initialize orchestrator
        st.toast("🚀 Starting Swarm Analysis...", icon="🚀")
        orchestrator = SwarmAnalysisOrchestrator(
            model_name=model_name, auto_detect_languages=auto_detect_languages
        )

        # Setup Langfuse callbacks with Streamlit context
        langfuse_handler = setup_langfuse_callbacks_for_streamlit(
            trace_name="swarm_analysis",
            additional_metadata={
                "project_path": project_path,
                "goal": goal,
                "max_agents": max_agents,
            },
        )

        # Run analysis with status updates
        with st.status("🚀 Swarm Analysis in Progress...", expanded=True) as status:
            st.write("Initializing orchestrator...")

            result_queue = queue.Queue()

            def run_analysis_thread():
                try:
                    # Create new event loop for this thread
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Extract user_id and session_id from Streamlit for trace correlation
                        try:
                            init_settings_db()
                            user_id = get_user_id()
                        except Exception:
                            user_id = None

                        # Get session_id from Streamlit
                        session_id = None
                        try:
                            import streamlit as st

                            if hasattr(st.session_state, "_session_id"):
                                session_id = str(st.session_state._session_id)
                        except Exception:
                            pass

                        res = loop.run_until_complete(
                            orchestrator.analyze(
                                project_path=project_path,
                                goal=goal,
                                model_name=model_name,
                                max_agents=max_agents,
                                stream_callback=stream_callback,
                                max_tokens_per_chunk=max_tokens,
                                enable_chunking=enable_chunking,
                                chunking_strategy=chunking_strategy,
                                auto_detect_languages=auto_detect_languages,
                                selected_directories=selected_directories,
                                previous_report_path=previous_report_path,
                                previous_report_id=previous_report_id,
                                user_id=user_id,
                                session_id=session_id,
                                metadata={
                                    "project_path": project_path,
                                    "goal": goal,
                                    "max_agents": max_agents,
                                },
                            )
                        )
                        result_queue.put({"status": "success", "result": res})
                    finally:
                        loop.close()
                except Exception as e:
                    result_queue.put({"status": "error", "error": e})
                finally:
                    analysis_complete.set()

            analysis_thread = threading.Thread(target=run_analysis_thread, daemon=True)
            analysis_thread.start()

            # Create placeholders for UI updates
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            # Main loop to update UI while analysis runs
            while not analysis_complete.is_set() or not event_queue.empty():
                # Process events from queue
                while not event_queue.empty():
                    try:
                        event = event_queue.get_nowait()
                        event_type = event["type"]
                        event_data = event["data"]

                        # Update status based on event
                        if event_type == "swarm_analysis_started":
                            status.write("🚀 Analysis started...")
                            progress_bar.progress(5, text="Analysis started")
                            
                            # Update dashboard
                            render_swarm_dashboard(dashboard_placeholder, dashboard_state)
                            
                        elif event_type == "project_scanning_start":
                            status.write("📁 Scanning project files...")
                            progress_bar.progress(10, text="Scanning project files...")
                        elif event_type == "project_scanning_complete":
                            file_count = event_data.get("file_count", 0)
                            detected_languages = event_data.get(
                                "detected_languages", []
                            )
                            lang_info = (
                                f" ({', '.join(detected_languages[:3])})"
                                if detected_languages
                                else ""
                            )
                            status.write(f"✅ Scanned {file_count} files{lang_info}")
                            progress_bar.progress(
                                15, text=f"Scanned {file_count} files"
                            )
                        elif event_type == "architecture_building_start":
                            status.write("🏗️ Building architecture model...")
                            progress_bar.progress(
                                20, text="Building architecture model..."
                            )
                        elif event_type == "architecture_status_update":
                            # More granular status updates
                            message = event_data.get("message", "")
                            status_code = event_data.get("status", "")
                            
                            status.write(message)
                            
                            # Adjust progress bar slightly based on sub-step
                            if status_code == "checking_cache":
                                progress_bar.progress(18, text="Checking cache...")
                            elif status_code == "building_llm":
                                progress_bar.progress(20, text="Analysing codebase with LLM...")
                            elif status_code == "cache_hit":
                                progress_bar.progress(25, text="Using cached architecture")
                                
                        elif event_type == "architecture_building_complete":
                            system_name = event_data.get("system_name", "Unknown")
                            status.write(f"✅ Architecture model built: {system_name}")
                            progress_bar.progress(
                                25, text=f"Architecture: {system_name}"
                            )
                        elif event_type == "role_selection":
                            status.write("🎯 Determining analysis roles...")
                            progress_bar.progress(
                                30, text="Determining analysis roles..."
                            )
                        elif event_type == "roles_selected":
                            roles = event_data.get("roles", [])
                            status.write(f"✅ Selected {len(roles)} roles")
                            progress_bar.progress(
                                40, text=f"Selected {len(roles)} roles"
                            )
                        elif event_type == "prompt_generation_start":
                            status.write("📝 Generating prompts...")
                            progress_bar.progress(50, text="Generating prompts...")
                        elif event_type == "prompt_generation":
                            # Individual prompt progress - update without changing main status
                            role = event_data.get("role", "Unknown")
                            index = event_data.get("index", 0)
                            total = event_data.get("total", 0)
                            # Keep main status, just update progress
                            progress_bar.progress(
                                50 + int((index / total) * 5),
                                text=f"Generating prompt {index}/{total}: {role}",
                            )
                            
                        elif event_type == "prompt_source_detected":
                            # Show toast for prompt source
                            role_name = event_data.get("role", "Unknown Role")
                            source = event_data.get("source", "Unknown Source")
                            
                            # Distinct icons/messages based on source
                            if source == "CodeLumen DB":
                                st.toast(f"⚡ Found {role_name} prompt in CodeLumen DB", icon="💾")
                            elif source == "Cache":
                                st.toast(f"⚡ Using cached prompt for {role_name}", icon="🚀")
                            elif source == "Langfuse":
                                st.toast(f"⚡ Retrieved {role_name} prompt from Langfuse", icon="🔗")
                            # We generally don't toast for LLM generation as it's the default/slow path
                            
                        elif event_type == "prompt_generation_complete":
                            successful = event_data.get("successful", 0)
                            total = event_data.get("total", 0)
                            status.write(f"✅ Generated {successful}/{total} prompts")
                            progress_bar.progress(
                                55, text=f"Generated {successful} prompts"
                            )
                        elif event_type == "prompt_validation_start":
                            status.write("🔍 Validating prompts...")
                            progress_bar.progress(60, text="Validating prompts...")
                        elif event_type == "prompt_validation":
                            # Individual validation progress - update without changing main status
                            role = event_data.get("role", "Unknown")
                            # Keep main status, just update progress
                            progress_bar.progress(60, text=f"Validating: {role}")
                        elif event_type == "prompt_validation_complete":
                            successful = event_data.get("successful", 0)
                            total = event_data.get("total", 0)
                            status.write(f"✅ Validated {successful}/{total} prompts")
                            progress_bar.progress(
                                65, text=f"Validated {successful} prompts"
                            )
                        elif event_type == "agent_execution_start":
                            agent_count = event_data.get("agent_count", 0)
                            status.write(f"🤖 Executing {agent_count} agents...")
                            progress_bar.progress(
                                70, text=f"Executing {agent_count} agents..."
                            )

                        elif event_type == "agent_chunk_start":
                            agent = event_data.get("agent", "Unknown")
                            chunk_idx = event_data.get("chunk_index", 0)
                            total_chunks = event_data.get("total_chunks", 1)
                            file = event_data.get("file", "")
                            
                            # Update global stats
                            # Note: This is a rough estimate as we process events
                            dashboard_state["total_chunks_estimated"] = max(
                                dashboard_state["total_chunks_estimated"], 
                                dashboard_state["completed_chunks"] + 1
                            )
                            
                            # Update active agent state
                            dashboard_state["active_agents"][agent] = {
                                "chunk_index": chunk_idx,
                                "total_chunks": total_chunks,
                                "file": file,
                                "status": "processing"
                            }
                            
                            # Update dashboard
                            render_swarm_dashboard(dashboard_placeholder, dashboard_state)
                            
                            status.write(
                                f"🔍 {agent}: Processing chunk {chunk_idx}/{total_chunks} ({file})"
                            )
                            progress_bar.progress(
                                70, text=f"{agent}: chunk {chunk_idx}/{total_chunks}"
                            )

                        elif event_type == "agent_chunk_complete":
                            # Use this to increment global progress?
                            dashboard_state["completed_chunks"] += 1
                            render_swarm_dashboard(dashboard_placeholder, dashboard_state)

                        elif event_type == "agent_execution_complete":
                            agent = event_data.get("agent", "Unknown")
                            total_chunks = event_data.get("total_chunks", 0)
                            
                            # Move from active to completed
                            if agent in dashboard_state["active_agents"]:
                                del dashboard_state["active_agents"][agent]
                                
                            dashboard_state["completed_agents"].append({
                                "name": agent,
                                "chunks": total_chunks
                            })
                            
                            render_swarm_dashboard(dashboard_placeholder, dashboard_state)
                            
                            status.write(
                                f"✅ Agent {agent} completed ({total_chunks} chunks)"
                            )
                            
                        elif event_type == "swarm_analysis_completed":
                            # Final update
                            render_swarm_dashboard(dashboard_placeholder, dashboard_state)
                            
                            successful = event_data.get("successful", 0) # This might be wrong key for this event type based on logs, but let's stick to safe fallback
                            status.update(
                                label="✅ Swarm Analysis Complete!",
                                state="complete",
                                expanded=False,
                            )
                            progress_bar.progress(100, text="Analysis complete!")

                        elif event_type == "result_synthesis_start":
                            status.write("📊 Synthesizing results...")
                            progress_bar.progress(90, text="Synthesizing results...")
                        elif event_type == "result_synthesis_complete":
                            agents_count = event_data.get("agents_count", 0)
                            status.write(
                                f"✅ Synthesized results from {agents_count} agents"
                            )
                            progress_bar.progress(95, text="Results synthesized")
                        elif event_type == "swarm_analysis_completed":
                            status.update(
                                label="✅ Swarm Analysis Complete!",
                                state="complete",
                                expanded=False,
                            )
                            progress_bar.progress(100, text="Analysis complete!")
                        elif event_type == "swarm_analysis_error":
                            status.update(
                                label="❌ Swarm Analysis Failed", state="error"
                            )
                            st.error(
                                f"Error: {event_data.get('error', 'Unknown error')}"
                            )

                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.warning(f"Error processing event: {e}")

                # Small sleep to prevent busy loop
                import time

                time.sleep(0.1)

            # Get result
            try:
                thread_result = result_queue.get(timeout=1.0)
                if thread_result["status"] == "error":
                    raise thread_result["error"]
                analysis_result = thread_result["result"]
            except queue.Empty:
                st.error("Analysis timed out or did not complete")
                return
            except Exception as e:
                logger.error(f"Swarm analysis failed: {e}", exc_info=True)
                st.error(format_user_error(e))
                return

        # Save scan to history
        save_scan(
            {
                "project_path": project_path,
                "agent_name": "Swarm Analysis",
                "model_used": model_name,
                "files_scanned": analysis_result.get("files_scanned", 0),
                "chunks_analyzed": analysis_result.get("chunks_analyzed", 0),
                "result": analysis_result,
                "status": "completed",
            }
        )

        # Store in session state
        st.session_state.swarm_results = analysis_result
        st.rerun()

    except Exception as e:
        analysis_complete.set()
        logger.error(f"Swarm analysis failed: {e}", exc_info=True)
        st.error(f"❌ Swarm Analysis Failed: {format_user_error(e)}")
        st.toast("Analysis failed!", icon="❌")


if __name__ == "__main__":
    main()
