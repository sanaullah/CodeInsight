"""
Cost tracking utilities for Langfuse traces.

Calculates token costs and aggregates cost information from LLM calls.
"""

import logging
from typing import Dict, Any, List, Optional
from llm.config import ConfigManager

logger = logging.getLogger(__name__)


def get_model_pricing(model_name: str) -> Optional[Dict[str, float]]:
    """
    Get pricing information for a model from config.
    
    Args:
        model_name: Model name (may include custom_openai/ prefix)
        
    Returns:
        Dictionary with input_price_per_1m and output_price_per_1m, or None if not found
    """
    try:
        # Remove custom_openai/ prefix if present
        clean_model = model_name.replace("custom_openai/", "")
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Check if model pricing is in config
        if hasattr(config, 'models') and config.models:
            model_config = config.models.get(clean_model)
            if model_config:
                if isinstance(model_config, dict):
                    input_price = model_config.get("input_price_per_1m", 0.0)
                    output_price = model_config.get("output_price_per_1m", 0.0)
                else:
                    # Pydantic model
                    input_price = getattr(model_config, "input_price_per_1m", 0.0)
                    output_price = getattr(model_config, "output_price_per_1m", 0.0)
                
                if input_price > 0 or output_price > 0:
                    return {
                        "input_price_per_1m": float(input_price),
                        "output_price_per_1m": float(output_price)
                    }
        
        # Check cost_tracking.model_pricing if available
        if hasattr(config, 'langfuse') and hasattr(config.langfuse, 'cost_tracking'):
            cost_tracking = config.langfuse.cost_tracking
            if hasattr(cost_tracking, 'model_pricing'):
                model_pricing = cost_tracking.model_pricing
                if isinstance(model_pricing, dict):
                    pricing = model_pricing.get(clean_model)
                    if pricing:
                        return {
                            "input_price_per_1m": float(pricing),
                            "output_price_per_1m": float(pricing)  # Assume same for both
                        }
        
        logger.debug(f"No pricing information found for model: {clean_model}")
        return None
        
    except Exception as e:
        logger.debug(f"Error getting model pricing: {e}")
        return None


def calculate_token_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> float:
    """
    Calculate cost for token usage.
    
    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model name
        
    Returns:
        Total cost in USD, or 0.0 if pricing not available
    """
    pricing = get_model_pricing(model)
    if not pricing:
        return 0.0
    
    input_price = pricing.get("input_price_per_1m", 0.0)
    output_price = pricing.get("output_price_per_1m", 0.0)
    
    # Calculate cost: (tokens / 1_000_000) * price_per_1m
    input_cost = (prompt_tokens / 1_000_000.0) * input_price
    output_cost = (completion_tokens / 1_000_000.0) * output_price
    
    total_cost = input_cost + output_cost
    
    return total_cost


def aggregate_costs_from_traces(
    trace_ids: List[str],
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Aggregate costs from Langfuse traces by querying the Langfuse API.
    
    If a single trace ID is provided (the main LangGraph trace), it will
    aggregate all token usage from all nested observations in that trace.
    
    Args:
        trace_ids: List of trace IDs (typically just the main LangGraph trace)
        model: Optional model name (if all traces use same model)
        
    Returns:
        Dictionary with total_tokens, total_cost_usd, and breakdown
    """
    from utils.langfuse.client import get_langfuse_client
    
    client = get_langfuse_client()
    if not client:
        logger.warning("Langfuse client not available, cannot query traces")
        return {
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
            "model": model,
            "trace_count": len(trace_ids),
            "error": "Langfuse client not available"
        }
    
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    successful_traces = 0
    failed_traces = []
    
    def collect_observations(obs_list) -> Dict[str, int]:
        """
        Recursively collect token usage from all observations.
        
        Returns:
            Dictionary with 'prompt' and 'completion' token counts
        """
        tokens = {"prompt": 0, "completion": 0}
        
        if not obs_list:
            return tokens
        
        for obs in obs_list:
            try:
                # Try to get token usage from observation
                # Langfuse v3 uses usageDetails with input/output fields
                usage_data = None
                
                # Method 1: Check usageDetails (v2 API format)
                if hasattr(obs, 'usageDetails') and obs.usageDetails:
                    usage_data = obs.usageDetails
                    if isinstance(usage_data, dict):
                        prompt_tokens = usage_data.get('input', 0) or usage_data.get('prompt_tokens', 0) or 0
                        completion_tokens = usage_data.get('output', 0) or usage_data.get('completion_tokens', 0) or 0
                    else:
                        # usageDetails might be an object
                        prompt_tokens = getattr(usage_data, 'input', 0) or getattr(usage_data, 'prompt_tokens', 0) or 0
                        completion_tokens = getattr(usage_data, 'output', 0) or getattr(usage_data, 'completion_tokens', 0) or 0
                
                # Method 2: Check usage attribute (v1 API format)
                elif hasattr(obs, 'usage') and obs.usage:
                    usage_data = obs.usage
                    if isinstance(usage_data, dict):
                        prompt_tokens = usage_data.get('prompt_tokens', 0) or usage_data.get('input', 0) or 0
                        completion_tokens = usage_data.get('completion_tokens', 0) or usage_data.get('output', 0) or 0
                    else:
                        prompt_tokens = getattr(usage_data, 'prompt_tokens', 0) or getattr(usage_data, 'input', 0) or 0
                        completion_tokens = getattr(usage_data, 'completion_tokens', 0) or getattr(usage_data, 'output', 0) or 0
                
                # Method 3: Check metadata for token usage (fallback)
                elif hasattr(obs, 'metadata') and obs.metadata:
                    metadata = obs.metadata
                    if isinstance(metadata, dict) and 'token_usage' in metadata:
                        usage = metadata['token_usage']
                        if isinstance(usage, dict):
                            prompt_tokens = usage.get('prompt_tokens', 0) or usage.get('input', 0) or 0
                            completion_tokens = usage.get('completion_tokens', 0) or usage.get('output', 0) or 0
                        else:
                            prompt_tokens = 0
                            completion_tokens = 0
                    else:
                        prompt_tokens = 0
                        completion_tokens = 0
                else:
                    prompt_tokens = 0
                    completion_tokens = 0
                
                tokens["prompt"] += prompt_tokens
                tokens["completion"] += completion_tokens
                
            except Exception as obs_error:
                logger.debug(f"Error extracting token usage from observation: {obs_error}")
            
            # Recursively process nested observations
            try:
                if hasattr(obs, 'observations') and obs.observations:
                    nested_tokens = collect_observations(obs.observations)
                    tokens["prompt"] += nested_tokens["prompt"]
                    tokens["completion"] += nested_tokens["completion"]
            except Exception as nested_error:
                logger.debug(f"Error processing nested observations: {nested_error}")
        
        return tokens
    
    # Process each trace
    for trace_id in trace_ids:
        try:
            # Fetch trace from Langfuse using v3 API
            trace = None
            if hasattr(client, 'api') and hasattr(client.api, 'trace'):
                # Try v3 API format
                trace = client.api.trace.get(trace_id)
            elif hasattr(client, 'get_trace'):
                # Try direct method
                trace = client.get_trace(trace_id)
            elif hasattr(client, 'fetch_trace'):
                # Try fetch method
                trace = client.fetch_trace(trace_id)
            
            if not trace:
                logger.debug(f"Trace {trace_id} not found in Langfuse")
                failed_traces.append(trace_id)
                continue
            
            # Get all observations from trace
            observations = []
            if isinstance(trace, dict):
                observations = trace.get('observations', [])
            elif hasattr(trace, 'observations'):
                observations = trace.observations
            elif hasattr(trace, 'get') and callable(getattr(trace, 'get')):
                # Try dict-like access
                observations = trace.get('observations', [])
            
            if not observations:
                logger.debug(f"Trace {trace_id} has no observations")
                failed_traces.append(trace_id)
                continue
            
            # Recursively collect token usage from all observations
            trace_tokens = collect_observations(observations)
            
            total_prompt_tokens += trace_tokens["prompt"]
            total_completion_tokens += trace_tokens["completion"]
            
            # Calculate cost if model is available
            if model and (trace_tokens["prompt"] > 0 or trace_tokens["completion"] > 0):
                cost = calculate_token_cost(
                    trace_tokens["prompt"],
                    trace_tokens["completion"],
                    model
                )
                total_cost_usd += cost
            
            successful_traces += 1
            logger.debug(
                f"Collected {trace_tokens['prompt'] + trace_tokens['completion']} tokens "
                f"({trace_tokens['prompt']} prompt + {trace_tokens['completion']} completion) "
                f"from trace {trace_id}"
            )
            
        except Exception as e:
            logger.warning(f"Error fetching trace {trace_id}: {e}")
            failed_traces.append(trace_id)
    
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    return {
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost_usd": total_cost_usd,
        "model": model,
        "trace_count": len(trace_ids),
        "successful_traces": successful_traces,
        "failed_traces": failed_traces
    }


def aggregate_costs_from_state(
    state: Dict[str, Any],
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Aggregate costs from state metadata (if token usage is stored there).
    
    Args:
        state: State dictionary that may contain token usage information
        model: Optional model name
        
    Returns:
        Dictionary with total_tokens, total_cost_usd, and breakdown
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    
    # Try to extract from metadata
    metadata = state.get("metadata", {})
    
    # Check for token usage in metadata
    if "token_usage" in metadata:
        usage = metadata["token_usage"]
        if isinstance(usage, dict):
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
    
    # Calculate cost if model is provided
    total_cost = 0.0
    if model and total_tokens > 0:
        total_cost = calculate_token_cost(
            total_prompt_tokens,
            total_completion_tokens,
            model
        )
    
    return {
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_cost_usd": total_cost,
        "model": model
    }

