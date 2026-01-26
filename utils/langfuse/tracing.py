"""
Langfuse tracing and observation creation utilities.
"""

import logging
from typing import Optional, Dict, Any, List
from contextlib import nullcontext
from datetime import datetime
from .client import get_langfuse_client

logger = logging.getLogger(__name__)


def trace_litellm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    trace_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    prompt_id: Optional[str] = None,
    return_trace_info: bool = False,
    **litellm_kwargs
):
    """
    Manually trace a LiteLLM completion call with Langfuse using span API.
    
    This function wraps litellm.completion() with explicit Langfuse tracing,
    ensuring traces are captured even if LiteLLM's automatic callbacks fail.
    
    Supports trace correlation via user_id, session_id, and trace_id to link
    LLM calls with parent LangGraph traces.
    
    Supports prompt-to-trace linking via prompt_id to link LLM calls with
    Langfuse prompts for better observability.
    
    Automatically extracts and includes token usage (prompt_tokens, completion_tokens,
    total_tokens) from the LiteLLM response to ensure consistency with other
    integration methods.
    
    Args:
        model: Model name to use
        messages: List of message dictionaries
        trace_name: Optional name for the trace (defaults to "litellm_completion")
        metadata: Optional metadata to attach to the trace
        user_id: Optional user ID for trace correlation
        session_id: Optional session ID for trace correlation
        trace_id: Optional trace ID for linking to parent trace
        prompt_id: Optional Langfuse prompt ID for linking to prompt
        return_trace_info: If True, returns tuple (response, trace_info_dict) instead of just response
        **litellm_kwargs: Additional arguments to pass to litellm.completion()
    
    Returns:
        The response from litellm.completion() if return_trace_info=False
        Tuple of (response, trace_info_dict) if return_trace_info=True
        trace_info_dict contains: {"trace_id": str, "generation": object}
    """
    import litellm
    
    client = get_langfuse_client()
    trace = None
    
    # If Langfuse is not available, just call LiteLLM directly
    if client is None:
        logger.debug("Langfuse client not available, calling LiteLLM without manual tracing")
        return litellm.completion(model=model, messages=messages, **litellm_kwargs)
    
    try:
        from langfuse import propagate_attributes
        
        # Prepare trace name and metadata
        trace_name_str = trace_name or f"litellm_completion_{model}"
        trace_metadata = metadata.copy() if metadata else {}
        trace_metadata.update({
            "model": model,
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat(),
            "input": str(messages)[:500]  # First 500 chars of input
        })
        
        # Extract prompt_id from metadata if not provided directly (for backward compatibility)
        if not prompt_id and metadata:
            prompt_id = metadata.get("langfuse_prompt_id")
        
        # Add prompt_id to metadata for backward compatibility
        if prompt_id:
            trace_metadata["langfuse_prompt_id"] = prompt_id
        
        # Use v3 context manager with propagate_attributes for trace correlation
        logger.debug(f"Creating Langfuse v3 trace: {trace_name_str}")
        
        # Build observation kwargs
        obs_kwargs = {
            "as_type": "generation",
            "name": trace_name_str,
            "model": model,
            "input": messages,
            "metadata": trace_metadata
        }
        
        # Add prompt_id if available
        if prompt_id:
            obs_kwargs["prompt_id"] = prompt_id
        
        # Create generation observation using v3 context manager
        # If prompt_id parameter is not supported, catch TypeError and retry without it
        try:
            observation_cm = client.start_as_current_observation(**obs_kwargs)
        except TypeError as e:
            # If prompt_id was provided and we got TypeError, assume it's unsupported parameter
            # Retry without prompt_id (metadata still contains it for backward compatibility)
            if prompt_id and "prompt_id" in obs_kwargs:
                logger.debug(f"prompt_id parameter not supported by SDK, using metadata-only for prompt linking")
                # Remove prompt_id and retry
                obs_kwargs.pop("prompt_id", None)
                observation_cm = client.start_as_current_observation(**obs_kwargs)
            else:
                # Re-raise if it's a different TypeError (not related to prompt_id)
                raise
        
        with observation_cm as generation:
            
            # Use propagate_attributes for trace correlation (v3 pattern)
            with propagate_attributes(
                user_id=user_id,
                session_id=session_id
            ) if (user_id or session_id) else nullcontext():
                
                # Call LiteLLM
                logger.debug(f"Calling LiteLLM with model: {model} (traced)")
                start_time = datetime.now()
                response = litellm.completion(model=model, messages=messages, **litellm_kwargs)
                end_time = datetime.now()
                
                # Extract response content
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    output_content = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else str(response.choices[0].message)
                else:
                    output_content = str(response)
                
                # Extract token usage from LiteLLM response (if available)
                token_usage_metadata = {}
                if hasattr(response, 'usage') and response.usage:
                    token_usage_metadata = {
                        "token_usage": {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0)
                        }
                    }
                    logger.debug(f"Extracted token usage: {token_usage_metadata['token_usage']}")
                
                # Update generation with output and token usage
                generation.update(
                    output=output_content[:1000],  # First 1000 chars of output
                    metadata={
                        **trace_metadata,
                        **token_usage_metadata,
                        "duration_ms": (end_time - start_time).total_seconds() * 1000,
                        "output_length": len(output_content)
                    }
                )
                
                # Explicitly set trace input/output for LLM-as-a-judge compatibility
                generation.update_trace(
                    input=messages,
                    output=output_content[:1000]
                )
                
                # Extract trace ID from generation object (for return_trace_info)
                # Must extract inside context manager before it exits
                extracted_trace_id = None
                if return_trace_info:
                    # Try multiple ways to get trace ID from generation object
                    try:
                        # Method 1: Direct trace_id attribute
                        if hasattr(generation, 'trace_id'):
                            extracted_trace_id = getattr(generation, 'trace_id', None)
                        # Method 2: Get from trace object
                        elif hasattr(generation, 'trace'):
                            trace_obj = getattr(generation, 'trace', None)
                            if trace_obj and hasattr(trace_obj, 'id'):
                                extracted_trace_id = getattr(trace_obj, 'id', None)
                        # Method 3: Try get_trace_id method
                        elif hasattr(generation, 'get_trace_id'):
                            try:
                                extracted_trace_id = generation.get_trace_id()
                            except Exception:
                                pass
                        # Method 4: ID attribute (may be observation ID, but better than nothing)
                        elif hasattr(generation, 'id'):
                            extracted_trace_id = getattr(generation, 'id', None)
                            logger.debug("Using generation.id as trace ID (may be observation ID)")
                        
                        if extracted_trace_id:
                            logger.debug(f"Extracted trace ID from generation: {extracted_trace_id}")
                        else:
                            logger.debug("Could not extract trace ID from generation object, will use provided trace_id")
                    except Exception as extract_error:
                        logger.debug(f"Error extracting trace ID from generation: {extract_error}")
        
        logger.debug(f"Langfuse trace completed: {trace_name_str}")
        
        # Flush to ensure trace is sent
        try:
            client.flush()
            logger.info(f"✅ Langfuse trace sent and flushed: {trace_name_str}")
        except Exception as flush_error:
            logger.warning(f"Failed to flush Langfuse: {flush_error}")
        
        # Return response with optional trace info
        # Note: generation object is not returned as it's only valid within context manager
        if return_trace_info:
            trace_info = {
                "trace_id": extracted_trace_id or trace_id,  # Use extracted or fallback to provided
                "generation": None  # Cannot return generation object outside context
            }
            return response, trace_info
        else:
            return response
        
    except Exception as e:
        logger.error(f"Error in manual Langfuse tracing: {e}", exc_info=True)
        
        logger.warning("Falling back to direct LiteLLM call without tracing")
        return litellm.completion(model=model, messages=messages, **litellm_kwargs)


async def atrace_litellm_completion(
    model: str,
    messages: List[Dict[str, Any]],
    trace_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    prompt_id: Optional[str] = None,
    return_trace_info: bool = False,
    **litellm_kwargs
):
    """
    Async version of trace_litellm_completion.
    
    Manually trace an async LiteLLM completion call with Langfuse using span API.
    Wraps litellm.acompletion() with explicit Langfuse tracing.
    
    Supports prompt-to-trace linking via prompt_id to link LLM calls with
    Langfuse prompts for better observability.
    
    Args:
        model: Model name to use
        messages: List of message dictionaries
        trace_name: Optional name for the trace (defaults to "litellm_acompletion")
        metadata: Optional metadata to attach to the trace
        user_id: Optional user ID for trace correlation
        session_id: Optional session ID for trace correlation
        trace_id: Optional trace ID for linking to parent trace
        prompt_id: Optional Langfuse prompt ID for linking to prompt
        return_trace_info: If True, returns tuple (response, trace_info_dict) instead of just response
        **litellm_kwargs: Additional arguments to pass to litellm.acompletion()
    
    Returns:
        The response from litellm.acompletion() if return_trace_info=False
        Tuple of (response, trace_info_dict) if return_trace_info=True
        trace_info_dict contains: {"trace_id": str, "generation": object}
    """
    import litellm
    
    client = get_langfuse_client()
    trace = None
    
    # If Langfuse is not available, just call LiteLLM directly
    if client is None:
        logger.debug("Langfuse client not available, calling LiteLLM without manual tracing")
        return await litellm.acompletion(model=model, messages=messages, **litellm_kwargs)
    
    try:
        from langfuse import propagate_attributes
        
        # Prepare trace name and metadata
        trace_name_str = trace_name or f"litellm_acompletion_{model}"
        trace_metadata = metadata.copy() if metadata else {}
        trace_metadata.update({
            "model": model,
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat(),
            "input": str(messages)[:500]  # First 500 chars of input
        })
        
        # Extract prompt_id from metadata if not provided directly (for backward compatibility)
        if not prompt_id and metadata:
            prompt_id = metadata.get("langfuse_prompt_id")
        
        # Add prompt_id to metadata for backward compatibility
        if prompt_id:
            trace_metadata["langfuse_prompt_id"] = prompt_id
        
        # Use v3 context manager with propagate_attributes for trace correlation
        logger.debug(f"Creating Langfuse v3 trace (async): {trace_name_str}")
        
        # Build observation kwargs
        obs_kwargs = {
            "as_type": "generation",
            "name": trace_name_str,
            "model": model,
            "input": messages,
            "metadata": trace_metadata
        }
        
        # Add prompt_id if available
        if prompt_id:
            obs_kwargs["prompt_id"] = prompt_id
        
        # Create generation observation using v3 context manager
        # Note: We use the context manager synchronously as the client itself is synchronous wrapper
        # The Langfuse SDK handles async IO in background threads
        # If prompt_id parameter is not supported, catch TypeError and retry without it
        try:
            observation_cm = client.start_as_current_observation(**obs_kwargs)
        except TypeError as e:
            # If prompt_id was provided and we got TypeError, assume it's unsupported parameter
            # Retry without prompt_id (metadata still contains it for backward compatibility)
            if prompt_id and "prompt_id" in obs_kwargs:
                logger.debug(f"prompt_id parameter not supported by SDK, using metadata-only for prompt linking")
                # Remove prompt_id and retry
                obs_kwargs.pop("prompt_id", None)
                observation_cm = client.start_as_current_observation(**obs_kwargs)
            else:
                # Re-raise if it's a different TypeError (not related to prompt_id)
                raise
        
        with observation_cm as generation:
            
            # Use propagate_attributes for trace correlation (v3 pattern)
            with propagate_attributes(
                user_id=user_id,
                session_id=session_id
            ) if (user_id or session_id) else nullcontext():
                
                # Call LiteLLM Async
                logger.debug(f"Calling LiteLLM (async) with model: {model} (traced)")
                start_time = datetime.now()
                response = await litellm.acompletion(model=model, messages=messages, **litellm_kwargs)
                end_time = datetime.now()
                
                # Extract response content
                if response and hasattr(response, 'choices') and len(response.choices) > 0:
                    output_content = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else str(response.choices[0].message)
                else:
                    output_content = str(response)
                
                # Extract token usage from LiteLLM response (if available)
                token_usage_metadata = {}
                if hasattr(response, 'usage') and response.usage:
                    token_usage_metadata = {
                        "token_usage": {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0)
                        }
                    }
                
                # Update generation with output and token usage
                generation.update(
                    output=output_content[:1000],
                    metadata={
                        **trace_metadata,
                        **token_usage_metadata,
                        "duration_ms": (end_time - start_time).total_seconds() * 1000,
                        "output_length": len(output_content)
                    }
                )
                
                generation.update_trace(
                    input=messages,
                    output=output_content[:1000]
                )
                
                # Extract trace ID
                extracted_trace_id = None
                if return_trace_info:
                    try:
                        if hasattr(generation, 'trace_id'):
                            extracted_trace_id = getattr(generation, 'trace_id', None)
                        elif hasattr(generation, 'trace'):
                            trace_obj = getattr(generation, 'trace', None)
                            if trace_obj and hasattr(trace_obj, 'id'):
                                extracted_trace_id = getattr(trace_obj, 'id', None)
                        elif hasattr(generation, 'id'):
                            extracted_trace_id = getattr(generation, 'id', None)
                    except Exception:
                        pass
        
        # Flush to ensure trace is sent
        try:
            client.flush()
        except Exception:
            pass
        
        if return_trace_info:
            trace_info = {
                "trace_id": extracted_trace_id or trace_id,
                "generation": None
            }
            return response, trace_info
        else:
            return response
            
    except Exception as e:
        logger.error(f"Error in manual Langfuse tracing (async): {e}", exc_info=True)
        return await litellm.acompletion(model=model, messages=messages, **litellm_kwargs)


def create_observation(
    name: str,
    parent_trace: Optional[Any] = None,
    trace_id: Optional[str] = None,
    input_data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    level: str = "DEFAULT"
) -> Optional[Any]:
    """
    Create a Langfuse observation using v3 API.
    
    This is a v3-compatible wrapper for creating observations with proper context management.
    
    Args:
        name: Observation name
        parent_trace: Optional parent trace object
        trace_id: Optional trace ID for linking
        input_data: Optional input data
        metadata: Optional metadata dictionary
        level: Observation level (DEFAULT, DEBUG, WARNING, ERROR)
        
    Returns:
        Langfuse observation context manager or None if disabled/not configured
    """
    client = get_langfuse_client()
    if client is None:
        logger.debug(f"Cannot create Langfuse observation '{name}': client not initialized")
        return None
    
    try:
        from langfuse import propagate_attributes
        
        # Prepare metadata
        obs_metadata = metadata.copy() if metadata else {}
        if input_data:
            obs_metadata["input_data"] = input_data
        
        # Create observation using v3 context manager
        observation = client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
            metadata=obs_metadata if obs_metadata else None
        )
        
        # Set level if provided
        if level != "DEFAULT":
            try:
                observation.update(level=level)
            except Exception:
                pass
        
        logger.debug(f"Langfuse observation '{name}' created successfully")
        return observation
        
    except Exception as e:
        logger.error(f"Error creating Langfuse observation: {e}", exc_info=True)
        return None


def create_llm_observation(
    operation_name: str,
    model: Optional[str] = None,
    parent_trace: Optional[Any] = None,
    trace_id: Optional[str] = None,
    input_data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Any]:
    """
    Create a Langfuse observation for LLM operations using v3 API.
    
    This creates a generation-type observation for LLM calls.
    
    Args:
        operation_name: Name of the LLM operation (e.g., "role_selection", "prompt_generation")
        model: Model name used for the operation
        parent_trace: Parent trace object
        trace_id: Parent trace ID string
        input_data: Input data (e.g., messages, prompt)
        metadata: Additional metadata
        
    Returns:
        Langfuse generation observation context manager or None
    """
    client = get_langfuse_client()
    if client is None:
        logger.debug(f"Cannot create LLM observation '{operation_name}': client not initialized")
        return None
    
    try:
        obs_metadata = metadata.copy() if metadata else {}
        obs_metadata['observation_type'] = 'llm_operation'
        obs_metadata['llm_operation'] = operation_name
        if model:
            obs_metadata['model'] = model
        
        # Create generation observation using v3 context manager
        observation = client.start_as_current_observation(
            as_type="generation",
            name=f"llm_{operation_name}",
            model=model,
            input=input_data,
            metadata=obs_metadata if obs_metadata else None
        )
        
        logger.debug(f"Langfuse LLM observation '{operation_name}' created successfully")
        return observation
        
    except Exception as e:
        logger.error(f"Error creating Langfuse LLM observation: {e}", exc_info=True)
        return None


def create_langfuse_trace(
    name: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    Create a Langfuse trace for manual tracking using v3 API.
    
    Returns a context manager that should be used with 'with' statement.
    
    Args:
        name: Trace name
        user_id: Optional user ID
        session_id: Optional session ID
        metadata: Optional metadata dictionary
        tags: Optional list of tags
        
    Returns:
        Langfuse span context manager or None if disabled/not configured
        
    Example:
        with create_langfuse_trace("my_trace", user_id="user123") as trace:
            # Your code here
            trace.update(output="result")
    """
    client = get_langfuse_client()
    if client is None:
        logger.debug(f"Cannot create Langfuse trace '{name}': client not initialized")
        return None
    
    try:
        from langfuse import propagate_attributes
        from llm.config import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        langfuse_config = config.langfuse
        
        # Merge default tags with provided tags
        all_tags = list(langfuse_config.default_tags)
        if tags:
            all_tags.extend(tags)
        
        # Prepare metadata
        span_metadata = metadata.copy() if metadata else {}
        if all_tags:
            span_metadata['tags'] = all_tags
        
        logger.debug(f"Creating Langfuse v3 trace: {name}")
        
        # Create root span using v3 context manager
        # This becomes the trace root observation
        span = client.start_as_current_observation(
            as_type="span",
            name=name,
            metadata=span_metadata if span_metadata else None
        )
        
        # Use propagate_attributes for trace attributes (v3 pattern)
        if user_id or session_id or all_tags:
            # Set trace attributes via propagate_attributes or update_trace
            try:
                # For v3, we can set trace attributes on the span
                attr_dict = {}
                if user_id:
                    attr_dict['user_id'] = user_id
                if session_id:
                    attr_dict['session_id'] = session_id
                if all_tags:
                    attr_dict['tags'] = all_tags
                
                if attr_dict:
                    span.update_trace(**attr_dict)
            except Exception as e:
                logger.debug(f"Could not set trace attributes directly: {e}, will use propagate_attributes")
                # Fallback: wrap with propagate_attributes context manager
                # Note: This requires the caller to use the context manager properly
                pass
        
        logger.debug(f"Langfuse trace '{name}' created successfully")
        return span
        
    except Exception as e:
        logger.error(f"Error creating Langfuse trace: {e}", exc_info=True)
        return None


# Thread-local storage for trace context (for async operations)
import threading
_trace_context_local = threading.local()


def set_trace_context(trace_id: Optional[str] = None, user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """
    Set trace context in thread-local storage.
    
    Useful for async operations where LangGraph config is not directly available.
    
    Args:
        trace_id: Optional trace ID
        user_id: Optional user ID
        session_id: Optional session ID
    """
    _trace_context_local.trace_id = trace_id
    _trace_context_local.user_id = user_id
    _trace_context_local.session_id = session_id


def get_current_trace_context(
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Optional[str]]:
    """
    Get current trace context from LangGraph, state, or thread-local storage.
    
    Tries multiple sources in order:
    1. LangGraph config metadata (if provided)
    2. State dictionary (if provided)
    3. Thread-local storage (for async operations)
    4. Returns None values if not found
    
    Args:
        state: Optional state dictionary from LangGraph node
        config: Optional config dictionary from LangGraph node
        
    Returns:
        Dictionary with trace_id, user_id, session_id (may be None)
    """
    trace_id = None
    user_id = None
    session_id = None
    
    # Try to get from config metadata (highest priority)
    if config:
        metadata = config.get("metadata", {})
        if isinstance(metadata, dict):
            trace_id = metadata.get("trace_id") or trace_id
            user_id = metadata.get("user_id") or user_id
            session_id = metadata.get("session_id") or session_id
    
    # Try to get from state
    if state:
        trace_id = state.get("trace_id") or trace_id
        user_id = state.get("user_id") or user_id
        session_id = state.get("session_id") or session_id
        
        # Also check metadata in state
        state_metadata = state.get("metadata", {})
        if isinstance(state_metadata, dict):
            trace_id = state_metadata.get("trace_id") or trace_id
            user_id = state_metadata.get("user_id") or user_id
            session_id = state_metadata.get("session_id") or session_id
    
    # Try to get from thread-local storage (for async operations)
    try:
        if hasattr(_trace_context_local, 'trace_id'):
            trace_id = _trace_context_local.trace_id or trace_id
        if hasattr(_trace_context_local, 'user_id'):
            user_id = _trace_context_local.user_id or user_id
        if hasattr(_trace_context_local, 'session_id'):
            session_id = _trace_context_local.session_id or session_id
    except AttributeError:
        pass
    
    return {
        "trace_id": trace_id,
        "user_id": user_id,
        "session_id": session_id
    }
