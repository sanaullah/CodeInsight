"""
Markdown and JSON report generator for CodeInsight analysis results.
"""

import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_markdown_report(analysis_result: Dict[str, Any], project_name: Optional[str] = None) -> str:
    """
    Generate a Markdown report from analysis results.
    
    Args:
        analysis_result: Analysis result dictionary
        project_name: Optional project name
        
    Returns:
        Markdown report as string
    """
    try:
        if "error" in analysis_result:
            return f"# Analysis Error\n\nError: {analysis_result['error']}\n"
        
        project_name = project_name or analysis_result.get("project_path", "Unknown Project")
        agent_name = analysis_result.get("agent_name", "Analysis Agent")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# {agent_name} Report

**Project:** {project_name}  
**Generated:** {timestamp}  
**Model Used:** {analysis_result.get('model_used', 'Unknown')}

## Executive Summary

"""
        
        # Add synthesized report if available
        if "synthesized_report" in analysis_result:
            report += f"{analysis_result['synthesized_report']}\n\n"
        
        # Add metrics
        report += "## Metrics\n\n"
        report += f"- Files Scanned: {analysis_result.get('files_scanned', 0)}\n"
        report += f"- Chunks Analyzed: {analysis_result.get('chunks_analyzed', 0)}\n"
        
        # Add detailed results if available
        if "analysis_results" in analysis_result:
            report += "\n## Detailed Analysis\n\n"
            for i, chunk in enumerate(analysis_result["analysis_results"][:10], 1):  # Limit to first 10
                report += f"### Chunk {i}\n\n"
                if "analysis" in chunk:
                    report += f"{chunk['analysis']}\n\n"
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating markdown report: {e}", exc_info=True)
        return f"# Report Generation Error\n\nError: {str(e)}\n"


def generate_json_report(analysis_result: Dict[str, Any], project_name: Optional[str] = None) -> str:
    """
    Generate a JSON report from analysis results.
    
    Args:
        analysis_result: Analysis result dictionary
        project_name: Optional project name
        
    Returns:
        JSON report as string
    """
    try:
        project_name = project_name or analysis_result.get("project_path", "Unknown Project")
        timestamp = datetime.now().isoformat()
        
        report_data = {
            "project_name": project_name,
            "generated_at": timestamp,
            "agent_name": analysis_result.get("agent_name", "Analysis Agent"),
            "model_used": analysis_result.get("model_used", "Unknown"),
            "files_scanned": analysis_result.get("files_scanned", 0),
            "chunks_analyzed": analysis_result.get("chunks_analyzed", 0),
            "synthesized_report": analysis_result.get("synthesized_report", ""),
            "analysis_results": analysis_result.get("analysis_results", [])
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error generating JSON report: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, indent=2)


def _sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use in filenames.
    
    Args:
        name: String to sanitize
        
    Returns:
        Sanitized string safe for filenames
    """
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove special characters except underscores and hyphens
    name = re.sub(r'[^\w\-_]', '', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    return name.lower() if name else "unknown"


def generate_agent_markdown_report(
    agent_result: Dict[str, Any],
    role_name: str,
    project_name: str,
    parent_result: Dict[str, Any]
) -> str:
    """
    Generate a Markdown report for a single agent's results.
    
    Args:
        agent_result: Individual agent result dictionary
        role_name: Name of the agent role
        project_name: Project name
        parent_result: Parent swarm analysis result for context
        
    Returns:
        Markdown report as string
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        project_path = parent_result.get("project_path", project_name)
        model_used = parent_result.get("model_used", "Unknown")
        goal = parent_result.get("goal")
        
        report = f"""# {role_name} Report

**Project:** {project_name}  
**Project Path:** {project_path}  
**Generated:** {timestamp}  
**Model Used:** {model_used}
"""
        
        if goal:
            report += f"**Analysis Goal:** {goal}\n"
        
        report += "\n"
        
        # Check for error status
        if agent_result.get("status") == "error" or "error" in agent_result:
            report += "## Error Status\n\n"
            report += f"**Error:** {agent_result.get('error', 'Unknown error')}\n\n"
            return report
        
        # Add agent's synthesized report
        if "synthesized_report" in agent_result and agent_result["synthesized_report"]:
            report += "## Analysis Report\n\n"
            report += f"{agent_result['synthesized_report']}\n\n"
        else:
            report += "## Analysis Report\n\n"
            report += "*No report available for this agent.*\n\n"
        
        # Add metrics
        report += "## Metrics\n\n"
        report += f"- Status: {agent_result.get('status', 'unknown')}\n"
        report += f"- Chunks Analyzed: {agent_result.get('chunks_analyzed', 0)}\n"
        
        # Add chunk results if available
        if "chunk_results" in agent_result and agent_result["chunk_results"]:
            report += "\n## Chunk Results\n\n"
            for i, chunk_result in enumerate(agent_result["chunk_results"][:10], 1):  # Limit to first 10
                report += f"### Chunk {i}\n\n"
                if isinstance(chunk_result, dict):
                    if "analysis" in chunk_result:
                        report += f"{chunk_result['analysis']}\n\n"
                    elif "result" in chunk_result:
                        report += f"{chunk_result['result']}\n\n"
                else:
                    report += f"{chunk_result}\n\n"
            
            if len(agent_result["chunk_results"]) > 10:
                report += f"*...and {len(agent_result['chunk_results']) - 10} more chunks*\n\n"
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating agent markdown report: {e}", exc_info=True)
        return f"# Report Generation Error\n\nError: {str(e)}\n"


def generate_agent_json_report(
    agent_result: Dict[str, Any],
    role_name: str,
    project_name: str,
    parent_result: Dict[str, Any]
) -> str:
    """
    Generate a JSON report for a single agent's results.
    
    Args:
        agent_result: Individual agent result dictionary
        role_name: Name of the agent role
        project_name: Project name
        parent_result: Parent swarm analysis result for context
        
    Returns:
        JSON report as string
    """
    try:
        timestamp = datetime.now().isoformat()
        project_path = parent_result.get("project_path", project_name)
        
        report_data = {
            "role_name": role_name,
            "project_name": project_name,
            "project_path": project_path,
            "generated_at": timestamp,
            "model_used": parent_result.get("model_used", "Unknown"),
            "goal": parent_result.get("goal"),
            "status": agent_result.get("status", "unknown"),
            "chunks_analyzed": agent_result.get("chunks_analyzed", 0),
            "synthesized_report": agent_result.get("synthesized_report", ""),
            "chunk_results": agent_result.get("chunk_results", [])
        }
        
        # Include error if present
        if "error" in agent_result:
            report_data["error"] = agent_result["error"]
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error generating agent JSON report: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, indent=2)


def generate_all_agents_markdown_report(
    individual_agent_results: Dict[str, Dict[str, Any]],
    project_name: str,
    parent_result: Dict[str, Any]
) -> str:
    """
    Generate a combined Markdown report with all agents' results.
    
    Args:
        individual_agent_results: Dictionary mapping role names to agent results
        project_name: Project name
        parent_result: Parent swarm analysis result for context
        
    Returns:
        Combined markdown report as string
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        project_path = parent_result.get("project_path", project_name)
        model_used = parent_result.get("model_used", "Unknown")
        goal = parent_result.get("goal")
        
        report = f"""# All Agents Report

**Project:** {project_name}  
**Project Path:** {project_path}  
**Generated:** {timestamp}  
**Model Used:** {model_used}
"""
        
        if goal:
            report += f"**Analysis Goal:** {goal}\n"
        
        report += f"**Total Agents:** {len(individual_agent_results)}\n\n"
        
        # Table of contents
        report += "## Table of Contents\n\n"
        for i, role_name in enumerate(individual_agent_results.keys(), 1):
            sanitized_role = _sanitize_filename(role_name)
            report += f"{i}. [{role_name}](#{sanitized_role.replace('_', '-')})\n"
        report += "\n---\n\n"
        
        # Add each agent's report
        for role_name, agent_result in individual_agent_results.items():
            sanitized_role = _sanitize_filename(role_name)
            report += f"## {role_name} {{#{sanitized_role.replace('_', '-')}}}\n\n"
            
            # Check for error status
            if agent_result.get("status") == "error" or "error" in agent_result:
                report += f"**Status:** Error\n\n"
                report += f"**Error:** {agent_result.get('error', 'Unknown error')}\n\n"
                report += "---\n\n"
                continue
            
            # Add agent's synthesized report
            if "synthesized_report" in agent_result and agent_result["synthesized_report"]:
                report += f"{agent_result['synthesized_report']}\n\n"
            else:
                report += "*No report available for this agent.*\n\n"
            
            # Add metrics
            report += f"**Status:** {agent_result.get('status', 'unknown')}  \n"
            report += f"**Chunks Analyzed:** {agent_result.get('chunks_analyzed', 0)}\n\n"
            
            # Add chunk results summary if available
            if "chunk_results" in agent_result and agent_result["chunk_results"]:
                report += f"**Chunks Processed:** {len(agent_result['chunk_results'])}\n\n"
            
            report += "---\n\n"
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating all agents markdown report: {e}", exc_info=True)
        return f"# Report Generation Error\n\nError: {str(e)}\n"


def generate_all_agents_json_report(
    individual_agent_results: Dict[str, Dict[str, Any]],
    project_name: str,
    parent_result: Dict[str, Any]
) -> str:
    """
    Generate a combined JSON report with all agents' results.
    
    Args:
        individual_agent_results: Dictionary mapping role names to agent results
        project_name: Project name
        parent_result: Parent swarm analysis result for context
        
    Returns:
        Combined JSON report as string
    """
    try:
        timestamp = datetime.now().isoformat()
        project_path = parent_result.get("project_path", project_name)
        
        report_data = {
            "project_name": project_name,
            "project_path": project_path,
            "generated_at": timestamp,
            "model_used": parent_result.get("model_used", "Unknown"),
            "goal": parent_result.get("goal"),
            "total_agents": len(individual_agent_results),
            "agents": {}
        }
        
        # Add each agent's data
        for role_name, agent_result in individual_agent_results.items():
            agent_data = {
                "status": agent_result.get("status", "unknown"),
                "chunks_analyzed": agent_result.get("chunks_analyzed", 0),
                "synthesized_report": agent_result.get("synthesized_report", ""),
                "chunk_results": agent_result.get("chunk_results", [])
            }
            
            # Include error if present
            if "error" in agent_result:
                agent_data["error"] = agent_result["error"]
            
            report_data["agents"][role_name] = agent_data
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"Error generating all agents JSON report: {e}", exc_info=True)
        return json.dumps({"error": str(e)}, indent=2)

