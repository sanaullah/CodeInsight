"""
Unified factory for creating ProjectScanner instances with consistent language configuration.

This module provides a single source of truth for ProjectScanner initialization,
ensuring all components use the same language-agnostic configuration logic.
"""

import logging
from typing import Optional, List, Set

from .project_scanner import ProjectScanner
from llm.config import ConfigManager

logger = logging.getLogger(__name__)


def create_project_scanner(
    auto_detect: Optional[bool] = None,
    file_extensions: Optional[List[str]] = None,
    ignore_patterns: Optional[Set[str]] = None
) -> ProjectScanner:
    """
    Unified factory function to create a ProjectScanner with consistent language configuration.
    
    This ensures all parts of the application use the same language configuration logic,
    supporting both config-based defaults and UI-based overrides.
    
    Args:
        auto_detect: Override auto-detect setting (None = use config default)
        file_extensions: Override file extensions (None = use config default)
        ignore_patterns: Additional ignore patterns to merge with defaults
    
    Returns:
        Configured ProjectScanner instance with language-agnostic settings
    
    Example:
        >>> # Use config defaults
        >>> scanner = create_project_scanner()
        
        >>> # Override auto-detect from UI
        >>> scanner = create_project_scanner(auto_detect=True)
        
        >>> # Override with specific extensions
        >>> scanner = create_project_scanner(file_extensions=[".py", ".js", ".ts"])
        
        >>> # Both overrides
        >>> scanner = create_project_scanner(
        ...     auto_detect=False,
        ...     file_extensions=[".java", ".kt"]
        ... )
    """
    config_manager = ConfigManager()
    
    # Determine auto-detect setting
    if auto_detect is None:
        # Default to True if not specified
        # In v3, this could be read from config.yaml if needed
        try:
            config = config_manager.load_config()
            # Check if there's a language config in the config
            # For now, default to True
            auto_detect = True
        except Exception:
            auto_detect = True
        logger.debug(f"Using config default for auto_detect: {auto_detect}")
    else:
        logger.debug(f"Using override for auto_detect: {auto_detect}")
    
    # Determine file extensions
    if file_extensions is None:
        # Default to common extensions if not specified
        # In v3, this could be read from config.yaml if needed
        try:
            from .language_config import get_extensions_for_languages, get_supported_languages
            # Default to Python, JavaScript, TypeScript for now
            default_languages = ["python", "javascript", "typescript"]
            file_extensions = get_extensions_for_languages(default_languages)
        except Exception:
            # Fallback to Python only
            file_extensions = [".py"]
        logger.debug(f"Using config default extensions: {file_extensions}")
    else:
        logger.debug(f"Using override extensions: {file_extensions}")
    
    # Create and return configured scanner
    scanner = ProjectScanner(
        file_extensions=file_extensions,
        auto_detect_languages=auto_detect,
        ignore_patterns=ignore_patterns
    )
    
    logger.info(
        f"Created ProjectScanner: auto_detect={auto_detect}, "
        f"extensions={file_extensions}, ignore_patterns={ignore_patterns is not None}"
    )
    
    return scanner

