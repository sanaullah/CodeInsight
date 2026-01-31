"""
LLM Factory Module.

This module provides a centralized way to instantiate LLM models using LiteLLM.
 It abstracts away the details of configuration and environment setup.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from langchain_litellm import ChatLiteLLM
from langchain_core.language_models import BaseChatModel

from llm.config import ConfigManager, ensure_env_ready

logger = logging.getLogger(__name__)

# Track if environment has been initialized to avoid repeated overhead
_env_initialized = False

def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
    callbacks: Optional[List[Any]] = None,
    **kwargs
) -> BaseChatModel:
    """
    Get a configured ChatLiteLLM instance.
    
    Args:
        model_name: Name of the model to use (default: from config)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: None)
        streaming: Whether to enable streaming (default: False)
        callbacks: List of LangChain callbacks (default: None). 
                   Note: Langfuse callback is often automatically added if enabled in config.
        **kwargs: Additional arguments to pass to ChatLiteLLM
        
    Returns:
        Configured ChatLiteLLM instance
    """
    global _env_initialized
    
    # 1. Ensure basic environment is ready (API keys, etc.)
    if not _env_initialized:
        ensure_env_ready()
        _env_initialized = True
        
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # 2. Determine model name
    final_model = model_name or config.default_model
    
    # Prefix with openai/ if not present, as this is standard for OpenAI-compatible endpoints
    if not final_model.startswith("openai/") and "openai" not in final_model and "/" in final_model:
        # If it has a slash (e.g. nvidia/MODEL) but isn't explicitly openai/...,
        # and we are using a custom base, prefix with openai/ to force OpenAI-compatible client.
        if config.api_base and "openai.com" not in config.api_base:
             final_model = f"openai/{final_model}"
    
    # 3. Prepare parameters
    params = {
        "model": final_model,
        "temperature": temperature if temperature is not None else 0.7,
        "streaming": streaming,
        "callbacks": callbacks or [],
    }
    
    if max_tokens:
        params["max_tokens"] = max_tokens
        
    # Add any extra kwargs
    params.update(kwargs)
    
    # 4. Instantiate ChatLiteLLM
    try:
        llm = ChatLiteLLM(**params)
        return llm
    except ImportError:
        logger.error("ChatLiteLLM execution failed. Ensure langchain-litellm is installed.")
        raise
    except Exception as e:
        logger.error(f"Failed to create ChatLiteLLM instance: {e}")
        raise
