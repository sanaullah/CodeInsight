"""
Analysis utilities for CodeInsight.

Shared functions for displaying analysis results.
"""

import streamlit as st
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from reports.report_generator import (
    generate_markdown_report,
    generate_json_report,
    generate_agent_markdown_report,
    generate_agent_json_report,
    generate_all_agents_markdown_report,
    generate_all_agents_json_report,
    _sanitize_filename
)

logger = logging.getLogger(__name__)

# Timeout for download triggers (30 seconds)
DOWNLOAD_TRIGGER_TIMEOUT = timedelta(seconds=30)


def is_swarm_analysis_result(result: Dict[str, Any]) -> bool:
    """
    Check if result is from swarm analysis.
    
    Args:
        result: Analysis result dictionary
        
    Returns:
        True if result is from swarm analysis
    """
    # Check for swarm analysis indicators
    if "synthesized_report" in result:
        return True
    if "roles_selected" in result:
        return True
    if "individual_agent_results" in result:
        return True
    if "architecture_model" in result and "validation_metadata" in result:
        return True
    
    return False


def display_swarm_results(result: Dict[str, Any], unique_id: Optional[str] = None):
    """
    Display swarm analysis results with rich tabbed interface.
    
    Args:
        result: Swarm analysis result dictionary
        unique_id: Optional unique identifier for generating unique widget keys
    """
    st.subheader("📊 Swarm Analysis Results")
    
    # Get project name for file naming
    project_path = result.get("project_path", "Unknown Project")
    project_name = Path(project_path).name if project_path else "Unknown Project"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    key_prefix = unique_id or "swarm_results"
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Roles Selected", len(result.get("roles_selected", [])))
    with col2:
        st.metric("Agents Executed", result.get("validation_metadata", {}).get("total_agents", 0))
    with col3:
        st.metric("Files Scanned", result.get("files_scanned", 0))
    with col4:
        st.metric("Chunks Analyzed", result.get("chunks_analyzed", 0))
    
    # Tabs for detailed results
    tab_overview, tab_arch, tab_roles, tab_agents = st.tabs([
        "📋 Overview", 
        "🏗️ Architecture", 
        "🎯 Roles & Prompts", 
        "🤖 Agent Results"
    ])
    
    with tab_overview:
        if "synthesized_report" in result:
            st.markdown("### Synthesized Report")
            st.markdown(result["synthesized_report"])
            
            # Download buttons for synthesized report
            st.divider()
            st.markdown("### Download Synthesized Report")
            col_md, col_json = st.columns(2)
            
            with col_md:
                if st.button("📥 Download Markdown", key=f"{key_prefix}_dl_synth_md"):
                    st.session_state[f"{key_prefix}_dl_synth_md_trigger"] = True
                    st.session_state[f"{key_prefix}_dl_synth_md_trigger_time"] = datetime.now()
                    logger.info(f"Download trigger {key_prefix}_dl_synth_md_trigger set")
            
            with col_json:
                if st.button("📥 Download JSON", key=f"{key_prefix}_dl_synth_json"):
                    st.session_state[f"{key_prefix}_dl_synth_json_trigger"] = True
                    st.session_state[f"{key_prefix}_dl_synth_json_trigger_time"] = datetime.now()
                    logger.info(f"Download trigger {key_prefix}_dl_synth_json_trigger set")
            
            # Handle download triggers with timeout-based reset
            if st.session_state.get(f"{key_prefix}_dl_synth_md_trigger", False):
                try:
                    # Check timeout
                    trigger_time_key = f"{key_prefix}_dl_synth_md_trigger_time"
                    if trigger_time_key not in st.session_state:
                        st.session_state[trigger_time_key] = datetime.now()
                    
                    trigger_time = st.session_state[trigger_time_key]
                    if datetime.now() - trigger_time > DOWNLOAD_TRIGGER_TIMEOUT:
                        logger.info(f"Download trigger {key_prefix}_dl_synth_md_trigger timed out, resetting")
                        st.session_state[f"{key_prefix}_dl_synth_md_trigger"] = False
                        del st.session_state[trigger_time_key]
                    else:
                        markdown_report = generate_markdown_report(result, project_name)
                        agent_name = result.get("agent_name", "Swarm Analysis")
                        sanitized_agent = _sanitize_filename(agent_name)
                        sanitized_project = _sanitize_filename(project_name)
                        file_name = f"{sanitized_agent}_{sanitized_project}_{timestamp_str}.md"
                        st.download_button(
                            label="📥 Download Markdown Report",
                            data=markdown_report,
                            file_name=file_name,
                            mime="text/markdown",
                            key=f"{key_prefix}_dl_synth_md_file"
                        )
                except Exception as e:
                    logger.error(f"Error generating markdown report: {e}", exc_info=True)
                    from utils.error_formatting import format_user_error
                    st.error(f"Error generating markdown report: {format_user_error(e)}")
                    st.session_state[f"{key_prefix}_dl_synth_md_trigger"] = False
                    if f"{key_prefix}_dl_synth_md_trigger_time" in st.session_state:
                        del st.session_state[f"{key_prefix}_dl_synth_md_trigger_time"]
            
            if st.session_state.get(f"{key_prefix}_dl_synth_json_trigger", False):
                try:
                    # Check timeout
                    trigger_time_key = f"{key_prefix}_dl_synth_json_trigger_time"
                    if trigger_time_key not in st.session_state:
                        st.session_state[trigger_time_key] = datetime.now()
                    
                    trigger_time = st.session_state[trigger_time_key]
                    if datetime.now() - trigger_time > DOWNLOAD_TRIGGER_TIMEOUT:
                        logger.info(f"Download trigger {key_prefix}_dl_synth_json_trigger timed out, resetting")
                        st.session_state[f"{key_prefix}_dl_synth_json_trigger"] = False
                        del st.session_state[trigger_time_key]
                    else:
                        json_report = generate_json_report(result, project_name)
                        agent_name = result.get("agent_name", "Swarm Analysis")
                        sanitized_agent = _sanitize_filename(agent_name)
                        sanitized_project = _sanitize_filename(project_name)
                        file_name = f"{sanitized_agent}_{sanitized_project}_{timestamp_str}.json"
                        st.download_button(
                            label="📥 Download JSON Report",
                            data=json_report,
                            file_name=file_name,
                            mime="application/json",
                            key=f"{key_prefix}_dl_synth_json_file"
                        )
                except Exception as e:
                    logger.error(f"Error generating JSON report: {e}", exc_info=True)
                    from utils.error_formatting import format_user_error
                    st.error(f"Error generating JSON report: {format_user_error(e)}")
                    st.session_state[f"{key_prefix}_dl_synth_json_trigger"] = False
                    if f"{key_prefix}_dl_synth_json_trigger_time" in st.session_state:
                        del st.session_state[f"{key_prefix}_dl_synth_json_trigger_time"]
        
        # Goal (if provided)
        if result.get("goal"):
            st.info(f"**Analysis Goal**: {result['goal']}")

    with tab_arch:
        if "architecture_model" in result:
            arch = result["architecture_model"]
            st.markdown(f"**System**: {arch.get('system_name', 'N/A')}")
            st.markdown(f"**Type**: {arch.get('system_type', 'N/A')}")
            st.markdown(f"**Pattern**: {arch.get('architecture_pattern', 'N/A')}")
            
            st.markdown("#### Key Components")
            for comp in arch.get("components", [])[:10]:  # Show first 10
                st.text(f"- {comp}")
            if len(arch.get("components", [])) > 10:
                st.text(f"...and {len(arch.get('components', [])) - 10} more")

    with tab_roles:
        if "roles_selected" in result:
            st.markdown("### Selected Roles")
            roles = result["roles_selected"]
            for i, role in enumerate(roles, 1):
                st.markdown(f"{i}. **{role}**")
            
            st.divider()
            st.markdown("### Generated Prompts")
            # Show validation metadata
            if "validation_metadata" in result:
                validation = result["validation_metadata"]
                if "prompt_validations" in validation:
                    for role, val_data in validation["prompt_validations"].items():
                        with st.expander(f"Prompt Validation: {role}"):
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Confidence", f"{val_data.get('confidence', 0):.2f}")
                            c2.metric("Clarity", f"{val_data.get('clarity', 0):.2f}")
                            c3.metric("Completeness", f"{val_data.get('completeness', 0):.2f}")
                            c4.metric("Relevance", f"{val_data.get('relevance', 0):.2f}")

    with tab_agents:
        if "individual_agent_results" in result:
            agent_results = result["individual_agent_results"]
            
            if not agent_results:
                st.info("No agent results available.")
            else:
                # Download All Agents section
                st.markdown("### Download All Agents")
                col_all_md, col_all_json = st.columns(2)
                
                with col_all_md:
                    if st.button("📥 Download All (MD)", key=f"{key_prefix}_dl_all_md"):
                        st.session_state[f"{key_prefix}_dl_all_md_trigger"] = True
                        st.session_state[f"{key_prefix}_dl_all_md_trigger_time"] = datetime.now()
                        logger.info(f"Download trigger {key_prefix}_dl_all_md_trigger set")
                
                with col_all_json:
                    if st.button("📥 Download All (JSON)", key=f"{key_prefix}_dl_all_json"):
                        st.session_state[f"{key_prefix}_dl_all_json_trigger"] = True
                        st.session_state[f"{key_prefix}_dl_all_json_trigger_time"] = datetime.now()
                        logger.info(f"Download trigger {key_prefix}_dl_all_json_trigger set")
                
                # Handle download all triggers with timeout
                if st.session_state.get(f"{key_prefix}_dl_all_md_trigger", False):
                    try:
                        # Check timeout
                        trigger_time_key = f"{key_prefix}_dl_all_md_trigger_time"
                        if trigger_time_key not in st.session_state:
                            st.session_state[trigger_time_key] = datetime.now()
                        
                        trigger_time = st.session_state[trigger_time_key]
                        if datetime.now() - trigger_time > DOWNLOAD_TRIGGER_TIMEOUT:
                            logger.info(f"Download trigger {key_prefix}_dl_all_md_trigger timed out, resetting")
                            st.session_state[f"{key_prefix}_dl_all_md_trigger"] = False
                            del st.session_state[trigger_time_key]
                        else:
                            all_agents_md = generate_all_agents_markdown_report(
                                agent_results, project_name, result
                            )
                            sanitized_project = _sanitize_filename(project_name)
                            file_name = f"all_agents_{sanitized_project}_{timestamp_str}.md"
                            st.download_button(
                                label="📥 Download All Agents (MD)",
                                data=all_agents_md,
                                file_name=file_name,
                                mime="text/markdown",
                                key=f"{key_prefix}_dl_all_md_file"
                            )
                    except Exception as e:
                        logger.error(f"Error generating all agents markdown report: {e}", exc_info=True)
                        from utils.error_formatting import format_user_error
                        st.error(f"Error generating all agents markdown report: {format_user_error(e)}")
                        st.session_state[f"{key_prefix}_dl_all_md_trigger"] = False
                        if f"{key_prefix}_dl_all_md_trigger_time" in st.session_state:
                            del st.session_state[f"{key_prefix}_dl_all_md_trigger_time"]
                
                if st.session_state.get(f"{key_prefix}_dl_all_json_trigger", False):
                    try:
                        # Check timeout
                        trigger_time_key = f"{key_prefix}_dl_all_json_trigger_time"
                        if trigger_time_key not in st.session_state:
                            st.session_state[trigger_time_key] = datetime.now()
                        
                        trigger_time = st.session_state[trigger_time_key]
                        if datetime.now() - trigger_time > DOWNLOAD_TRIGGER_TIMEOUT:
                            logger.info(f"Download trigger {key_prefix}_dl_all_json_trigger timed out, resetting")
                            st.session_state[f"{key_prefix}_dl_all_json_trigger"] = False
                            del st.session_state[trigger_time_key]
                        else:
                            all_agents_json = generate_all_agents_json_report(
                                agent_results, project_name, result
                            )
                            sanitized_project = _sanitize_filename(project_name)
                            file_name = f"all_agents_{sanitized_project}_{timestamp_str}.json"
                            st.download_button(
                                label="📥 Download All Agents (JSON)",
                                data=all_agents_json,
                                file_name=file_name,
                                mime="application/json",
                                key=f"{key_prefix}_dl_all_json_file"
                            )
                    except Exception as e:
                        logger.error(f"Error generating all agents JSON report: {e}", exc_info=True)
                        from utils.error_formatting import format_user_error
                        st.error(f"Error generating all agents JSON report: {format_user_error(e)}")
                        st.session_state[f"{key_prefix}_dl_all_json_trigger"] = False
                        if f"{key_prefix}_dl_all_json_trigger_time" in st.session_state:
                            del st.session_state[f"{key_prefix}_dl_all_json_trigger_time"]
                
                st.divider()
                st.markdown("### Individual Agent Results")
                
                # Individual agent results with download buttons
                for role, agent_result in agent_results.items():
                    sanitized_role = _sanitize_filename(role)
                    with st.expander(f"🤖 {role}", expanded=False):
                        # Content column and download buttons column
                        content_col, dl_col = st.columns([3, 1])
                        
                        with content_col:
                            if "error" in agent_result:
                                st.error(f"**Error**: {agent_result['error']}")
                            else:
                                if "synthesized_report" in agent_result:
                                    st.markdown(agent_result["synthesized_report"])
                                else:
                                    st.info("No report available for this agent.")
                                
                                # Show metrics
                                if "chunks_analyzed" in agent_result:
                                    st.caption(f"Chunks Analyzed: {agent_result['chunks_analyzed']}")
                                
                                # Display individual chunk results if available
                                if "chunk_results" in agent_result and agent_result.get("chunk_results"):
                                    chunk_results = agent_result["chunk_results"]
                                    if chunk_results:  # Ensure it's not empty
                                        st.divider()
                                        st.markdown("### Individual Chunk Results")
                                        
                                        for chunk_result in chunk_results:
                                            if not isinstance(chunk_result, dict):
                                                continue
                                                
                                            chunk_num = chunk_result.get("chunk_num", "?")
                                            total_chunks = chunk_result.get("total_chunks", "?")
                                            files_count = chunk_result.get("files_in_chunk", 0)
                                            tokens_count = chunk_result.get("tokens_in_chunk", 0)
                                            
                                            # Create expander title with metadata
                                            expander_title = f"📄 Chunk {chunk_num} of {total_chunks}"
                                            
                                            with st.expander(expander_title, expanded=False):
                                                # Show chunk metadata
                                                metadata_parts = []
                                                if files_count:
                                                    metadata_parts.append(f"Files: {files_count}")
                                                if tokens_count:
                                                    metadata_parts.append(f"Tokens: ~{tokens_count:,}")
                                                
                                                if metadata_parts:
                                                    st.caption(" | ".join(metadata_parts))
                                                
                                                # Display chunk analysis
                                                if "analysis" in chunk_result and chunk_result["analysis"]:
                                                    st.markdown(chunk_result["analysis"])
                                                elif "result" in chunk_result:
                                                    st.markdown(chunk_result["result"])
                                                else:
                                                    st.info("No analysis available for this chunk.")
                        
                        with dl_col:
                            st.markdown("**Download**")
                            
                            # Download MD button
                            if st.button("📥 MD", key=f"{key_prefix}_dl_agent_md_{sanitized_role}"):
                                st.session_state[f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger"] = True
                                st.session_state[f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger_time"] = datetime.now()
                                logger.info(f"Download trigger {key_prefix}_dl_agent_md_{sanitized_role}_trigger set")
                            
                            # Download JSON button
                            if st.button("📥 JSON", key=f"{key_prefix}_dl_agent_json_{sanitized_role}"):
                                st.session_state[f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger"] = True
                                st.session_state[f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger_time"] = datetime.now()
                                logger.info(f"Download trigger {key_prefix}_dl_agent_json_{sanitized_role}_trigger set")
                            
                            # Handle download triggers with timeout
                            if st.session_state.get(f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger", False):
                                try:
                                    # Check timeout
                                    trigger_time_key = f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger_time"
                                    if trigger_time_key not in st.session_state:
                                        st.session_state[trigger_time_key] = datetime.now()
                                    
                                    trigger_time = st.session_state[trigger_time_key]
                                    if datetime.now() - trigger_time > DOWNLOAD_TRIGGER_TIMEOUT:
                                        logger.info(f"Download trigger {key_prefix}_dl_agent_md_{sanitized_role}_trigger timed out, resetting")
                                        st.session_state[f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger"] = False
                                        del st.session_state[trigger_time_key]
                                    else:
                                        agent_md = generate_agent_markdown_report(
                                            agent_result, role, project_name, result
                                        )
                                        sanitized_project = _sanitize_filename(project_name)
                                        file_name = f"{sanitized_role}_{sanitized_project}_{timestamp_str}.md"
                                        st.download_button(
                                            label="📥 Download MD",
                                            data=agent_md,
                                            file_name=file_name,
                                            mime="text/markdown",
                                            key=f"{key_prefix}_dl_agent_md_file_{sanitized_role}"
                                        )
                                except Exception as e:
                                    logger.error(f"Error generating markdown report for {role}: {e}", exc_info=True)
                                    from utils.error_formatting import format_user_error
                                    st.error(f"Error generating markdown report for {role}: {format_user_error(e)}")
                                    st.session_state[f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger"] = False
                                    if f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger_time" in st.session_state:
                                        del st.session_state[f"{key_prefix}_dl_agent_md_{sanitized_role}_trigger_time"]
                            
                            if st.session_state.get(f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger", False):
                                try:
                                    # Check timeout
                                    trigger_time_key = f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger_time"
                                    if trigger_time_key not in st.session_state:
                                        st.session_state[trigger_time_key] = datetime.now()
                                    
                                    trigger_time = st.session_state[trigger_time_key]
                                    if datetime.now() - trigger_time > DOWNLOAD_TRIGGER_TIMEOUT:
                                        logger.info(f"Download trigger {key_prefix}_dl_agent_json_{sanitized_role}_trigger timed out, resetting")
                                        st.session_state[f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger"] = False
                                        del st.session_state[trigger_time_key]
                                    else:
                                        agent_json = generate_agent_json_report(
                                            agent_result, role, project_name, result
                                        )
                                        sanitized_project = _sanitize_filename(project_name)
                                        file_name = f"{sanitized_role}_{sanitized_project}_{timestamp_str}.json"
                                        st.download_button(
                                            label="📥 Download JSON",
                                            data=agent_json,
                                            file_name=file_name,
                                            mime="application/json",
                                            key=f"{key_prefix}_dl_agent_json_file_{sanitized_role}"
                                        )
                                except Exception as e:
                                    logger.error(f"Error generating JSON report for {role}: {e}", exc_info=True)
                                    from utils.error_formatting import format_user_error
                                    st.error(f"Error generating JSON report for {role}: {format_user_error(e)}")
                                    st.session_state[f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger"] = False
                                    if f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger_time" in st.session_state:
                                        del st.session_state[f"{key_prefix}_dl_agent_json_{sanitized_role}_trigger_time"]
        else:
            st.info("No individual agent results available.")


def display_analysis_results(result: Dict[str, Any], project_path: str, unique_id: Optional[str] = None):
    """
    Display analysis results with enhanced formatting.
    
    Args:
        result: Analysis result dictionary
        project_path: Path to analyzed project
        unique_id: Optional unique identifier for generating unique widget keys (e.g., scan_id)
    """
    # Header
    st.header("📊 Analysis Results")
    
    # Check for errors
    if "error" in result:
        st.error(f"❌ **Error:** {result['error']}")
        return
    
    # Detect result type and route to appropriate display function
    if is_swarm_analysis_result(result):
        # Swarm Analysis result
        display_swarm_results(result, unique_id=unique_id)
        return
    
    # Only swarm analysis results are supported now
    st.warning("⚠️ This result type is not supported. Only Swarm Analysis results are available.")
    st.info("💡 Use Swarm Analysis page for project analysis.")

