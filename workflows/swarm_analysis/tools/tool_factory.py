"""
Tool factory for creating LangChain tools with MCP or embedded execution.

Handles MCP connection, tool creation, and cleanup.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.tools import BaseTool

# from workflows.swarm_analysis.mcp.filesystem_client import FilesystemMCPClient # Removed: MCP archived
from workflows.swarm_analysis.tools.langchain_tools import (
    create_read_file_tool,
    create_list_directory_tool,
    create_get_file_info_tool
)

logger = logging.getLogger(__name__)


async def create_langchain_tools(
    project_path: str,
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    use_mcp: bool = False # Deprecated, kept for compat
) -> Tuple[List[BaseTool], Any]: # Removed FilesystemMCPClient type hint
    """
    Create LangChain tools for agent execution.
    
    Attempts MCP connection if enabled, falls back to embedded tools.
    
    Args:
        project_path: Absolute path to project root
        state: Current workflow state
        config: Optional LangGraph config (for trace context)
        use_mcp: Whether to attempt MCP connection (default: True)
    
    Returns:
        Tuple of (list of tools, MCP client or None)
    """
    mcp_client = None
    tools: List[BaseTool] = []
    
    # MCP filesystem server support removed (archived feature)
    mcp_client = None
    tools: List[BaseTool] = []
    
    # Use embedded tools if MCP not available or failed
    if not tools:
        logger.info("Using embedded file tools")
        tools = [
            create_read_file_tool(project_path, state, None),
            create_list_directory_tool(project_path, state, None),
            create_get_file_info_tool(project_path, state, None)
        ]
    
    logger.debug(f"Created {len(tools)} LangChain tools")
    return tools, mcp_client

