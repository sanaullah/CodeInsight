"""
Tool executor for agent tool calls.

Handles tool execution, validation, and security checks.
"""

import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

from .file_tools import read_file_tool, list_directory_tool, get_file_info_tool

logger = logging.getLogger(__name__)

# Tool registry
TOOL_REGISTRY = {
    "read_file": read_file_tool,
    "list_directory": list_directory_tool,
    "get_file_info": get_file_info_tool,
}


async def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    project_path: str,
    state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute a tool call.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments (from LLM)
        project_path: Absolute path to project root
        state: Current workflow state
    
    Returns:
        Dictionary with tool result or error
    """
    # Validate tool name
    if tool_name not in TOOL_REGISTRY:
        logger.warning(f"Unknown tool: {tool_name}")
        return {"error": f"Unknown tool: {tool_name}"}
    
    # Get tool function
    tool_func = TOOL_REGISTRY[tool_name]
    
    try:
        # Execute tool
        if tool_name == "read_file":
            file_path = arguments.get("file_path")
            if not file_path:
                return {"error": "Missing required argument: file_path"}
            result = await tool_func(file_path, project_path, state)
        elif tool_name == "list_directory":
            directory_path = arguments.get("directory_path")
            if not directory_path:
                return {"error": "Missing required argument: directory_path"}
            result = await tool_func(directory_path, project_path, state)
        elif tool_name == "get_file_info":
            file_path = arguments.get("file_path")
            if not file_path:
                return {"error": "Missing required argument: file_path"}
            result = await tool_func(file_path, project_path, state)
        else:
            return {"error": f"Unsupported tool: {tool_name}"}
        
        # Log tool execution
        logger.info(f"Executed tool: {tool_name} with arguments: {arguments}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
        return {"error": f"Error executing tool {tool_name}: {str(e)}"}


def parse_tool_call(tool_call: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    Parse a tool call from LLM response.
    
    Args:
        tool_call: Tool call dictionary from LLM
    
    Returns:
        Tuple of (tool_name, arguments)
    """
    try:
        if "function" in tool_call:
            func = tool_call["function"]
            tool_name = func.get("name", "")
            arguments_str = func.get("arguments", "{}")
            
            # Parse JSON arguments
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str
            
            return tool_name, arguments
        else:
            logger.warning(f"Invalid tool call format: {tool_call}")
            return "", {}
    except Exception as e:
        logger.error(f"Error parsing tool call: {e}", exc_info=True)
        return "", {}

