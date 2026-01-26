"""
Langfuse client management and singleton initialization.
"""

import os
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Global Langfuse client instance (thread-safe singleton)
_langfuse_client = None
_client_lock = threading.Lock()


def get_langfuse_client():
    """
    Get or create Langfuse client instance using ConfigManager.
    Returns None if Langfuse is disabled or not configured.
    """
    global _langfuse_client
    
    with _client_lock:
        if _langfuse_client is not None:
            return _langfuse_client
        
        try:
            from llm.config import ConfigManager
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            langfuse_config = config.langfuse
            
            # Check if Langfuse is enabled
            if not langfuse_config.enabled:
                logger.debug("Langfuse is disabled in config.yaml")
                return None
            
            # Check if credentials are provided (check config first, then env vars)
            public_key = langfuse_config.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
            secret_key = langfuse_config.secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
            
            if not public_key or not secret_key:
                logger.info(
                    "Langfuse credentials not found. Skipping Langfuse initialization. "
                    "To enable, set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file."
                )
                return None
            
            # Ensure config object has the keys (in case we got them from env)
            langfuse_config.public_key = public_key
            langfuse_config.secret_key = secret_key
            
            # Import Langfuse v3
            try:
                from langfuse import get_client, Langfuse
            except ImportError:
                logger.warning("Langfuse not installed. Install with: pip install langfuse>=3.0.0")
                return None
            
            # Set environment variables for v3 SDK (get_client() reads from env vars)
            logger.debug(f"Setting Langfuse environment variables for v3 SDK")
            os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_config.public_key
            os.environ["LANGFUSE_SECRET_KEY"] = langfuse_config.secret_key
            os.environ["LANGFUSE_HOST"] = langfuse_config.host
            os.environ["LANGFUSE_BASE_URL"] = langfuse_config.host
            # Enable debug mode for better diagnostics (v3 uses environment variable)
            os.environ["LANGFUSE_DEBUG"] = "True"
            
            # Use get_client() for v3 (reads from environment variables)
            # This is the recommended approach for v3
            logger.debug(f"Creating Langfuse v3 client with host: {langfuse_config.host}")
            _langfuse_client = get_client()
            
            # If get_client() returns None or we need explicit config, use Langfuse() constructor
            if _langfuse_client is None:
                logger.debug("get_client() returned None, using Langfuse() constructor with explicit config")
                _langfuse_client = Langfuse(
                    public_key=langfuse_config.public_key,
                    secret_key=langfuse_config.secret_key,
                    host=langfuse_config.host,
                    timeout=langfuse_config.timeout,
                    flush_interval=langfuse_config.flush_interval if langfuse_config.flush_interval > 0 else None
                    # Note: debug parameter removed in v3, use LANGFUSE_DEBUG env var instead
                )
            
            logger.info(f"✅ Langfuse client initialized (host: {langfuse_config.host}, flush_interval: {langfuse_config.flush_interval}s)")
            logger.debug(f"Langfuse client created with public_key: {langfuse_config.public_key[:10]}...")
            return _langfuse_client
            
        except Exception as e:
            logger.error(f"Error initializing Langfuse client: {e}", exc_info=True)
            return None

