"""
File reading utilities with encoding detection.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def read_file_with_encoding(file_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Read file content with automatic encoding detection.
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        Tuple of (content, encoding) or (None, None) if read failed
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()
            return content, encoding
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error reading file {file_path} with encoding {encoding}: {e}")
            return None, None
    
    logger.warning(f"Could not read file {file_path} with any encoding")
    return None, None

