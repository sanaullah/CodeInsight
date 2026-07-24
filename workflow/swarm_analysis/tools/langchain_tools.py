"""
LangChain-compatible tool wrappers for agent tool calling.

Wraps existing tool functions to work with LangGraph's create_react_agent.
Supports both MCP and embedded tool execution.
"""

import logging
from typing import Dict, Any, Optional, Callable
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ReadFileInput(BaseModel):
    """Input schema for read_file tool."""
    file_path: str = Field(description="The relative path to the file from the project root (e.g., 'utils/experience_storage.py')")


class ListDirectoryInput(BaseModel):
    """Input schema for list_directory tool."""
    directory_path: str = Field(description="The relative path to the directory from the project root (e.g., 'utils' or 'services/storage')")


class GetFileInfoInput(BaseModel):
    """Input schema for get_file_info tool."""
    file_path: str = Field(description="The relative path to the file from the project root")


async def _execute_read_file_async(
    file_path: str,
    project_path: str,
    state: Dict[str, Any],
    mcp_client: Optional[Any] = None
) -> str:
    """Execute read_file tool (async version)."""
    try:
        if mcp_client:
            # Use MCP client (async)
            try:
                result = await mcp_client.execute_tool("read_file", {"file_path": file_path})
                if isinstance(result, dict):
                    return result.get("content", result.get("error", str(result)))
                return str(result)
            except Exception as e:
                logger.warning(f"MCP read_file failed: {e}, using embedded tool")
                from .file_tools import read_file_tool
                result = read_file_tool(file_path, project_path, state)
        else:
            # Use embedded tool (sync, but we're in async context)
            from .file_tools import read_file_tool
            result = read_file_tool(file_path, project_path, state)
        
        # Extract content or error
        if isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            return result.get("content", str(result))
        return str(result)
    except Exception as e:
        logger.error(f"Error in read_file tool: {e}", exc_info=True)
        return f"Error reading file: {str(e)}"


def create_read_file_tool(
    project_path: str,
    state: Dict[str, Any],
    mcp_client: Optional[Any] = None
) -> BaseTool:
    """
    Create a LangChain tool for reading files.
    
    Args:
        project_path: Absolute path to project root
        state: Current workflow state (for caching)
        mcp_client: Optional MCP client for MCP tool execution
    
    Returns:
        LangChain BaseTool instance
    """
    # Create a sync wrapper that handles async MCP calls
    def _execute_read_file_sync(file_path: str) -> str:
        """Execute read_file tool (sync wrapper)."""
        import asyncio
        try:
            # Try to get running loop
            loop = asyncio.get_running_loop()
            # We're in async context - create a task
            # But since this is a sync function, we need to handle it differently
            # For now, use embedded tools if in async context
            from .file_tools import read_file_tool
            result = read_file_tool(file_path, project_path, state)
            if isinstance(result, dict):
                if "error" in result:
                    return f"Error: {result['error']}"
                return result.get("content", str(result))
            return str(result)
        except RuntimeError:
            # No running loop - can use asyncio.run
            if mcp_client:
                try:
                    result = asyncio.run(mcp_client.execute_tool("read_file", {"file_path": file_path}))
                    if isinstance(result, dict):
                        return result.get("content", result.get("error", str(result)))
                    return str(result)
                except Exception as e:
                    logger.warning(f"MCP read_file failed: {e}, using embedded tool")
                    from .file_tools import read_file_tool
                    result = read_file_tool(file_path, project_path, state)
            else:
                from .file_tools import read_file_tool
                result = read_file_tool(file_path, project_path, state)
            
            if isinstance(result, dict):
                if "error" in result:
                    return f"Error: {result['error']}"
                return result.get("content", str(result))
            return str(result)
        except Exception as e:
            logger.error(f"Error in read_file tool: {e}", exc_info=True)
            return f"Error reading file: {str(e)}"
    
    return StructuredTool.from_function(
        func=_execute_read_file_sync,
        name="read_file",
        description="Read the content of a file by its relative path from project root. Use this when you need to see the implementation of an imported module or referenced file.",
        args_schema=ReadFileInput
    )


def create_list_directory_tool(
    project_path: str,
    state: Dict[str, Any],
    mcp_client: Optional[Any] = None
) -> BaseTool:
    """
    Create a LangChain tool for listing directories.
    
    Args:
        project_path: Absolute path to project root
        state: Current workflow state
        mcp_client: Optional MCP client for MCP tool execution
    
    Returns:
        LangChain BaseTool instance
    """
    def _execute_list_directory_sync(directory_path: str) -> str:
        """Execute list_directory tool (sync wrapper)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # In async context - use embedded tools
            from .file_tools import list_directory_tool
            result = list_directory_tool(directory_path, project_path, state)
        except RuntimeError:
            # No running loop - can use asyncio.run
            if mcp_client:
                try:
                    result = asyncio.run(mcp_client.execute_tool("list_directory", {"directory_path": directory_path}))
                    if isinstance(result, dict):
                        files = result.get("files", [])
                        if files:
                            return "\n".join([f"{f.get('type', 'unknown')}: {f.get('path', f.get('name', ''))}" for f in files])
                        return result.get("error", str(result))
                    return str(result)
                except Exception as e:
                    logger.warning(f"MCP list_directory failed: {e}, using embedded tool")
                    from .file_tools import list_directory_tool
                    result = list_directory_tool(directory_path, project_path, state)
            else:
                from .file_tools import list_directory_tool
                result = list_directory_tool(directory_path, project_path, state)
        
        # Format result
        if isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            files = result.get("files", [])
            if files:
                return "\n".join([f"{f.get('type', 'unknown')}: {f.get('path', f.get('name', ''))}" for f in files])
            return str(result)
        return str(result)
    
    return StructuredTool.from_function(
        func=_execute_list_directory_sync,
        name="list_directory",
        description="List files in a directory. Use this to explore the project structure or find files.",
        args_schema=ListDirectoryInput
    )


def create_get_file_info_tool(
    project_path: str,
    state: Dict[str, Any],
    mcp_client: Optional[Any] = None
) -> BaseTool:
    """
    Create a LangChain tool for getting file info.
    
    Args:
        project_path: Absolute path to project root
        state: Current workflow state
        mcp_client: Optional MCP client for MCP tool execution
    
    Returns:
        LangChain BaseTool instance
    """
    def _execute_get_file_info_sync(file_path: str) -> str:
        """Execute get_file_info tool (sync wrapper)."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # In async context - use embedded tools
            from .file_tools import get_file_info_tool
            result = get_file_info_tool(file_path, project_path, state)
        except RuntimeError:
            # No running loop - can use asyncio.run
            if mcp_client:
                try:
                    result = asyncio.run(mcp_client.execute_tool("get_file_info", {"file_path": file_path}))
                    if isinstance(result, dict):
                        return f"Path: {result.get('path', file_path)}, Size: {result.get('size', 'unknown')} bytes, Lines: {result.get('line_count', 'unknown')}"
                    return str(result)
                except Exception as e:
                    logger.warning(f"MCP get_file_info failed: {e}, using embedded tool")
                    from .file_tools import get_file_info_tool
                    result = get_file_info_tool(file_path, project_path, state)
            else:
                from .file_tools import get_file_info_tool
                result = get_file_info_tool(file_path, project_path, state)
        
        # Format result
        if isinstance(result, dict):
            if "error" in result:
                return f"Error: {result['error']}"
            return f"Path: {result.get('path', file_path)}, Size: {result.get('size', 'unknown')} bytes, Lines: {result.get('line_count', 'unknown')}"
        return str(result)
    
    return StructuredTool.from_function(
        func=_execute_get_file_info_sync,
        name="get_file_info",
        description="Get metadata about a file (size, line count, encoding). Use this to check if a file exists or get file statistics.",
        args_schema=GetFileInfoInput
    )

