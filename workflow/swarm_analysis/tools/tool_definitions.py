"""
Tool definitions for agent tool calling.

Defines OpenAI function calling format tool schemas.
"""

from typing import List, Dict, Any


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get all available tool definitions in OpenAI function calling format.
    
    Returns:
        List of tool definition dictionaries
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a file by its relative path from project root. Use this when you need to see the implementation of an imported module or referenced file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The relative path to the file from the project root (e.g., 'utils/experience_storage.py')"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files in a directory. Use this to explore the project structure or find files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "The relative path to the directory from the project root (e.g., 'utils' or 'services/storage')"
                        }
                    },
                    "required": ["directory_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_file_info",
                "description": "Get metadata about a file (size, line count, encoding). Use this to check if a file exists or get file statistics.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The relative path to the file from the project root"
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    ]

