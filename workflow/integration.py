"""
Langfuse integration for LangGraph observability.

This module provides callback handlers and utilities for tracking
LangGraph execution in Langfuse using the native Langfuse LangChain integration.
"""

from typing import Optional, Dict, Any, List
import logging
from infrastructure.utils.langfuse_integration import get_langfuse_client

logger = logging.getLogger(__name__)


def setup_langfuse_callbacks(
    trace_name: Optional[str] = None,
    enabled: bool = True,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    Setup native Langfuse callbacks for LangGraph using Langfuse's LangChain integration.
    
    This function returns Langfuse's native CallbackHandler which automatically:
    - Creates proper observation types (tool, llm, chain) for agent graph visualization
    - Tracks graph runs, node executions, and state transitions
    - Integrates seamlessly with LangGraph's callback system
    - Supports trace correlation via user_id, session_id, and metadata
    
    Args:
        trace_name: Optional name for the trace (passed to CallbackHandler)
        enabled: Whether to enable callbacks (checks config if True)
        user_id: Optional user ID for trace correlation
        session_id: Optional session ID for trace correlation
        metadata: Optional metadata dictionary to attach to traces
        tags: Optional list of tags to attach to traces
        
    Returns:
        CallbackHandler instance from langfuse.langchain if enabled, None otherwise
        
    Example:
        from workflow.integration import setup_langfuse_callbacks
        
        # Get native callback handler with trace correlation
        langfuse_handler = setup_langfuse_callbacks(
            trace_name="my_workflow",
            user_id="user123",
            session_id="session456",
            metadata={"environment": "production"},
            tags=["agent", "workflow"]
        )
        
        # Use with LangGraph via config parameter
        graph.invoke(input_state, config={"callbacks": [langfuse_handler]})
    """
    # Check if callbacks should be enabled
    if enabled:
        try:
            from infrastructure.llm.config import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.load_config()
            if not config.langfuse.enabled:
                logger.debug("Langfuse is disabled in configuration")
                return None
        except ImportError as e:
            logger.warning(f"ConfigManager unavailable, disabling Langfuse: {e}")
            return None  # Fail secure: disable if config system unavailable
        except (AttributeError, KeyError) as e:
            logger.warning(f"Langfuse config missing, disabling: {e}")
            return None  # Fail secure: disable if config malformed
        except Exception as e:
            logger.error(f"Unexpected error, disabling Langfuse: {e}")
            return None  # Fail secure: disable on any unexpected error
    
    # Verify Langfuse client is available
    client = get_langfuse_client()
    if not client:
        logger.debug("Langfuse client not available")
        return None
    
    # Import and create native CallbackHandler
    try:
        from langfuse.langchain import CallbackHandler
        
        # CallbackHandler only accepts public_key and update_trace in constructor
        # Trace metadata (trace_name, user_id, session_id, metadata) should be
        # passed through LangGraph config metadata, not through CallbackHandler constructor
        handler_kwargs = {}
        
        # Get public_key from config if available (optional, usually from env)
        try:
            from infrastructure.llm.config import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.load_config()
            if hasattr(config.langfuse, 'public_key') and config.langfuse.public_key:
                handler_kwargs['public_key'] = config.langfuse.public_key
        except Exception:
            pass  # Use default from environment
        
        # Create callback handler with only supported parameters
        handler = CallbackHandler(**handler_kwargs)
        
        # Store trace metadata as attributes for use in config metadata (v3 pattern)
        # In v3, trace attributes are passed via config metadata using langfuse_* prefix
        # These will be picked up by LangGraph and passed to Langfuse via config
        handler._trace_name = trace_name
        handler._user_id = user_id
        handler._session_id = session_id
        
        # Merge metadata and tags - prepare for v3 metadata pattern
        handler_metadata = metadata.copy() if metadata else {}
        if tags:
            handler_metadata['tags'] = tags
        
        # Store metadata that will be passed via config metadata with langfuse_* prefix
        # This follows v3 pattern: langfuse_user_id, langfuse_session_id, langfuse_tags
        handler._metadata = handler_metadata if handler_metadata else {}
        handler._langfuse_user_id = user_id
        handler._langfuse_session_id = session_id
        handler._langfuse_tags = tags
        
        logger.debug(f"Native Langfuse v3 CallbackHandler created for LangGraph (trace_name={trace_name}, user_id={user_id}, session_id={session_id})")
        return handler
        
    except ImportError:
        logger.warning(
            "langfuse.langchain.CallbackHandler not available. "
            "Run `uv sync --locked` to install Langfuse."
        )
        return None
    except Exception as e:
        logger.error(f"Error creating Langfuse CallbackHandler: {e}", exc_info=True)
        return None

