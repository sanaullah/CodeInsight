"""
Async I/O utilities for wrapping blocking operations.

Provides async wrappers for blocking I/O and CPU-bound operations
to prevent event loop blocking in async contexts.
"""

import asyncio
import ast
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


async def async_ast_parse(content: str, filename: str = "<unknown>") -> ast.AST:
    """
    Parse Python code using AST asynchronously.
    
    Args:
        content: Python source code to parse
        filename: Optional filename for error messages
        
    Returns:
        AST node representing the parsed code
        
    Raises:
        SyntaxError: If the code cannot be parsed
    """
    try:
        return await asyncio.to_thread(ast.parse, content, filename)
    except SyntaxError as e:
        logger.debug(f"Syntax error parsing {filename}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing AST from {filename}: {e}", exc_info=True)
        raise

