"""
Environment variable loader using python-dotenv.

Loads .env file from project root if it exists.
This module should be imported early in application startup.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_env(env_file: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Optional path to .env file. If not provided, looks for .env
                  in the project root (parent of utils/ directory).
    
    Returns:
        True if .env file was found and loaded, False otherwise.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning(
            "python-dotenv not installed. Install with: pip install python-dotenv"
        )
        return False
    
    # Determine project root (parent of utils/ directory)
    if env_file is None:
        # This file is in utils/, so parent is project root
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
    else:
        env_file = Path(env_file)
    
    # Check if .env file exists
    if not env_file.exists():
        logger.debug(f".env file not found at {env_file}. Using system environment variables.")
        return False
    
    # Load .env file
    try:
        load_dotenv(env_file, override=False)  # Don't override existing env vars
        logger.info(f"Loaded environment variables from {env_file}")
        return True
    except Exception as e:
        logger.warning(f"Failed to load .env file from {env_file}: {e}")
        return False


def ensure_env_loaded() -> None:
    """
    Ensure .env file is loaded (idempotent).
    
    This function can be called multiple times safely.
    It will only load the .env file once per process.
    """
    if not hasattr(ensure_env_loaded, '_loaded'):
        load_env()
        ensure_env_loaded._loaded = True

