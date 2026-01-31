"""
Langfuse connection utilities.
"""

import logging
from .client import get_langfuse_client

logger = logging.getLogger(__name__)


def flush_langfuse():
    """
    Flush the Langfuse client to ensure all pending traces are sent to the server.
    This is useful to ensure data is sent even if the application exits quickly.
    """
    client = get_langfuse_client()
    if client is None:
        logger.debug("Cannot flush Langfuse: client not initialized")
        return False
    
    try:
        client.flush()
        logger.debug("Langfuse client flushed successfully")
        return True
    except Exception as e:
        logger.error(f"Error flushing Langfuse client: {e}", exc_info=True)
        return False


def validate_langfuse_connection() -> bool:
    """
    Validate that Langfuse credentials are correct and the server is reachable.
    
    Returns:
        True if connection is valid, False otherwise
    """
    client = get_langfuse_client()
    if client is None:
        logger.debug("Cannot validate Langfuse connection: client not initialized")
        return False
    
    try:
        # Try to flush the client to verify connection
        client.flush()
        logger.debug("Langfuse connection validation successful (flush completed)")
        return True
    except Exception as e:
        logger.warning(f"Langfuse connection validation failed: {e}")
        return False

