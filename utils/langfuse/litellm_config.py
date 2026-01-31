"""
LiteLLM integration configuration for Langfuse.
"""

import os
import logging
import threading

logger = logging.getLogger(__name__)

# Global flag to ensure LiteLLM is configured only once
_litellm_configured = False
_litellm_config_lock = threading.Lock()


def configure_litellm_for_langfuse():
    """
    Configure environment for Langfuse tracking.
    
    This acts as a bridge to ensure Langfuse credentials from config.yaml 
    are available as environment variables for both:
    1. LangChain's LangfuseCallbackHandler (if used)
    2. LiteLLM's native integration (fallback)
    """
    global _litellm_configured
    
    # Check if already configured (thread-safe)
    with _litellm_config_lock:
        if _litellm_configured:
            return
        
    try:
        from llm.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        langfuse_config = config.langfuse
        
        # Check if Langfuse is enabled
        if not langfuse_config.enabled:
            return
        
        # Check if credentials are provided
        if not langfuse_config.public_key or not langfuse_config.secret_key:
            logger.warning("Langfuse is enabled but credentials are missing.")
            return
            
        # Set environment variables for Langfuse SDK
        os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_config.public_key
        os.environ["LANGFUSE_SECRET_KEY"] = langfuse_config.secret_key
        os.environ["LANGFUSE_HOST"] = langfuse_config.host
        
        # Configure LiteLLM native callback if enabled for it
        # This is useful even when using ChatLiteLLM as it catches any direct litellm calls
        if langfuse_config.enabled_for_litellm:
            try:
                import litellm
                # Check if callback is already configured
                current_callbacks = getattr(litellm, 'callbacks', [])
                if "langfuse_otel" not in current_callbacks:
                    litellm.callbacks = ["langfuse_otel"]
                    logger.info("✅ LiteLLM configured with 'langfuse_otel' callback (OTEL)")
            except ImportError:
                pass
        
        # Mark as configured
        with _litellm_config_lock:
            _litellm_configured = True
            
    except Exception as e:
        logger.error(f"Error configuring Langfuse: {e}", exc_info=True)
