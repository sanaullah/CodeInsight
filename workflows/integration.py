"""
Langfuse integration for LangGraph observability.

This module provides callback handlers and utilities for tracking
LangGraph execution in Langfuse using the native Langfuse LangChain integration.
"""

from typing import Optional, Dict, Any, List
import logging
from utils.langfuse_integration import get_langfuse_client

logger = logging.getLogger(__name__)


def extract_streamlit_context() -> Dict[str, Any]:
    """
    Extract context from Streamlit session state for Langfuse trace correlation.
    
    Returns:
        Dictionary with user_id, session_id, and metadata from Streamlit
    """
    try:
        import streamlit as st
        
        context = {
            "user_id": None,
            "session_id": None,
            "metadata": {}
        }
        
        # Extract user_id from session state
        if "user_id" in st.session_state:
            context["user_id"] = st.session_state["user_id"]
        elif "user_id" in st.session_state.get("metadata", {}):
            context["user_id"] = st.session_state["metadata"]["user_id"]
        
        # Extract session_id (Streamlit session ID)
        if hasattr(st, 'session_state') and hasattr(st.session_state, '_session_id'):
            context["session_id"] = str(st.session_state._session_id)
        
        # Extract metadata from session state
        metadata = {}
        
        # Project path if available
        if "project_path" in st.session_state:
            metadata["project_path"] = st.session_state["project_path"]
        
        # Analysis goal if available
        if "analysis_goal" in st.session_state:
            metadata["goal"] = st.session_state["analysis_goal"]
        
        # Model used
        if "selected_model" in st.session_state:
            metadata["model"] = st.session_state["selected_model"]
        
        # UI settings
        if "ui_layout_mode" in st.session_state:
            metadata["ui_layout_mode"] = st.session_state["ui_layout_mode"]
        
        context["metadata"] = metadata
        
        return context
        
    except ImportError:
        # Streamlit not available (e.g., in non-UI context)
        logger.debug("Streamlit not available, returning empty context")
        return {
            "user_id": None,
            "session_id": None,
            "metadata": {}
        }
    except Exception as e:
        logger.warning(f"Error extracting Streamlit context: {e}")
        return {
            "user_id": None,
            "session_id": None,
            "metadata": {}
        }


def setup_langfuse_callbacks_for_streamlit(
    trace_name: Optional[str] = None,
    enabled: bool = True,
    additional_metadata: Optional[Dict[str, Any]] = None
):
    """
    Setup Langfuse callbacks with Streamlit context extraction.
    
    Automatically extracts user_id, session_id, and metadata from Streamlit
    session state for trace correlation.
    
    Args:
        trace_name: Optional name for the trace
        enabled: Whether to enable callbacks
        additional_metadata: Optional additional metadata to merge
        
    Returns:
        CallbackHandler instance or None
    """
    # Extract Streamlit context
    streamlit_context = extract_streamlit_context()
    
    # Merge additional metadata
    metadata = streamlit_context["metadata"].copy()
    if additional_metadata:
        metadata.update(additional_metadata)
    
    # Setup callbacks with extracted context
    return setup_langfuse_callbacks(
        trace_name=trace_name,
        enabled=enabled,
        user_id=streamlit_context["user_id"],
        session_id=streamlit_context["session_id"],
        metadata=metadata if metadata else None
    )


# Legacy class kept for backward compatibility but deprecated
class LangGraphLangfuseCallback:
    """
    DEPRECATED: Legacy custom callback handler.
    
    This class is kept for backward compatibility but is no longer used.
    The framework now uses Langfuse's native CallbackHandler from langfuse.langchain
    which automatically creates proper observation types for agent graph visualization.
    
    Use setup_langfuse_callbacks() instead, which returns the native CallbackHandler.
    """
    
    def __init__(self, trace_name: Optional[str] = None):
        """Initialize legacy callback (deprecated)."""
        logger.warning(
            "LangGraphLangfuseCallback is deprecated. "
            "Use setup_langfuse_callbacks() which returns the native CallbackHandler."
        )
        self.trace_name = trace_name or "workflow_run"
        self.langfuse_client = get_langfuse_client()
        self.trace = None
        self.node_spans = {}
    
    def on_graph_start(self, input_state, metadata=None):
        """Legacy method (deprecated)."""
        pass
    
    def on_node_start(self, node_name, state):
        """Legacy method (deprecated)."""
        pass
    
    def on_node_end(self, node_name, output_state, error=None):
        """Legacy method (deprecated)."""
        pass
    
    def on_graph_end(self, final_state, error=None):
        """Legacy method (deprecated)."""
        pass


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
        from workflows.integration import setup_langfuse_callbacks
        
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
            from llm.config import ConfigManager
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
            from llm.config import ConfigManager
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
            "Install langfuse with: pip install langfuse>=3.0.0"
        )
        return None
    except Exception as e:
        logger.error(f"Error creating Langfuse CallbackHandler: {e}", exc_info=True)
        return None

