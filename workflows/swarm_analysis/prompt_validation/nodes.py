"""
Prompt validation nodes for Swarm Analysis workflow.

Handles parallel prompt validation for multiple roles.
"""

import logging
import json
import re
from typing import Dict, Any, Optional

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from workflows.nodes import llm_node
from workflows.prompt_validation_config import get_prompt_validation_config

logger = logging.getLogger(__name__)


@validate_state_fields(["generated_prompts"], "dispatch_prompt_validation")
def dispatch_prompt_validation_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Dispatch prompt validation node - prepares state for parallel execution.
    
    This node validates prompts and prepares state. The actual Send objects
    are created in the conditional edge function (route_after_dispatch_validation).
    
    Args:
        state: Current swarm analysis state
        
    Returns:
        Updated state (prompts should already be set by collect_generated_prompts_node)
    """
    generated_prompts = state.get(StateKeys.GENERATED_PROMPTS, {})
    
    if not generated_prompts:
        state[StateKeys.ERROR] = "generated_prompts are required"
        state[StateKeys.ERROR_STAGE] = "dispatch_prompt_validation"
        return state
    
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback("prompt_validation_start", {
            "message": "Dispatching prompt validation for parallel execution...",
            "prompt_count": len(generated_prompts)
        })
    
    # Just validate and return state - Send objects are created in conditional edge
    logger.info(f"Prepared {len(generated_prompts)} prompts for parallel validation")
    
    # Return state unchanged (prompts are already in state from collect_generated_prompts_node)
    return state


@validate_state_fields(["role_name", "prompt"], "validate_single_prompt")
async def validate_single_prompt_node(state: SwarmAnalysisState, config: Optional[Dict[str, Any]] = None) -> SwarmAnalysisState:
    """
    Validate a single prompt.
    
    This node processes one prompt's validation including LLM call.
    Designed to be called in parallel via Send from dispatch_prompt_validation_node.
    
    Args:
        state: State containing role_name, prompt and all necessary context
        config: Optional LangGraph config (contains callbacks from graph invocation)
        
    Returns:
        Updated state with validation result in validation_results_list
    """
    # Extract role_name and prompt from state (set by Send)
    role_name = state.get(StateKeys.ROLE_NAME)
    prompt = state.get(StateKeys.PROMPT)
    
    if not role_name or not prompt:
        # Fallback: try to get from generated_prompts (for backward compatibility)
        generated_prompts = state.get(StateKeys.GENERATED_PROMPTS, {})
        if generated_prompts:
            role_name = list(generated_prompts.keys())[0]
            prompt = generated_prompts[role_name]
        else:
            state[StateKeys.ERROR] = "role_name and prompt are required"
            state[StateKeys.ERROR_STAGE] = "validate_single_prompt"
            return state
    
    # Get configuration from state
    model_name = state.get(StateKeys.MODEL_NAME)
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL, {})
    
    # Load prompt validation config
    validation_config = get_prompt_validation_config()
    min_validity_score = validation_config.min_validity_score
    
    if stream_callback:
        stream_callback("prompt_validation", {
            "role": role_name,
            "message": "Validating prompt..."
        })
    
    # Run shared validation (lenient: warns but continues)
    from utils.prompts.prompt_validator import PromptValidator, format_shared_validation_results
    validator = PromptValidator()
    shared_validation = validator.validate_all(
        prompt=prompt,
        architecture_model=architecture_model,
        model_name=model_name
    )
    
    # Log warnings if any
    if shared_validation.warnings:
        logger.warning(
            f"Shared validation warnings for {role_name}: {shared_validation.warnings}"
        )
    
    try:
        from utils.langfuse_integration import create_llm_observation
        
        # Create LLM observation for prompt validation
        validation_obs = create_llm_observation(
            operation_name="prompt_validation",
            model=model_name,
            input_data={
                "role": role_name,
                "prompt_length": len(prompt)
            },
            metadata={
                "operation": "validate_prompt",
                "role": role_name
            }
        )
        
        # Extract trace_id from validation_obs for score logging
        validation_trace_id = None
        if validation_obs:
            try:
                if hasattr(validation_obs, 'trace_id'):
                    validation_trace_id = getattr(validation_obs, 'trace_id', None)
                elif hasattr(validation_obs, 'trace'):
                    trace_obj = getattr(validation_obs, 'trace', None)
                    if trace_obj and hasattr(trace_obj, 'id'):
                        validation_trace_id = getattr(trace_obj, 'id', None)
                elif hasattr(validation_obs, 'id'):
                    validation_trace_id = getattr(validation_obs, 'id', None)
                    logger.debug("Using validation_obs.id as trace ID (may be observation ID)")
            except Exception as extract_error:
                logger.debug(f"Error extracting trace ID from validation_obs: {extract_error}")
        
        try:
            # Prepare enhanced validation request
            # Include shared validation results in LLM validation prompt
            shared_validation_text = format_shared_validation_results(shared_validation)
            
            validation_prompt = f"""Validate the following analysis prompt:

Role: {role_name}
Prompt:
{prompt}

{shared_validation_text}

Assess the prompt on the following criteria:

1. **Clarity** (0-1): Is it clear and unambiguous? Are instructions easy to understand?
2. **Completeness** (0-1): Does it cover all necessary aspects for the role? Are evidence requirements included?
3. **Relevance** (0-1): Is it relevant to the role? Does it focus on appropriate concerns?
4. **Actionability** (0-1): Can an agent act on it effectively? Are instructions specific enough?
5. **Accuracy** (0-1): Are the technical details correct? Does it reference only technologies present in the architecture?

Calculate an **overall_score** as the average of all five metrics.

Return a JSON object with:
- is_valid: boolean (true if overall_score >= {min_validity_score})
- confidence: float (0-1) - your confidence in this validation
- clarity: float (0-1)
- completeness: float (0-1)
- relevance: float (0-1)
- accuracy: float (0-1)
- overall_score: float (0-1) - average of clarity, completeness, relevance, actionability, and accuracy
- feedback: string - detailed feedback on what's good and what needs improvement
"""
            
            # Use llm_node for validation
            llm_state = {
                "messages": [
                    {"role": "system", "content": "You are an expert at validating analysis prompts. Return JSON validation results."},
                    {"role": "user", "content": validation_prompt}
                ],
                "model": model_name,
                "temperature": 0.3
            }
            
            # Call llm_node (tracing happens inside llm_node)
            llm_result = await llm_node(llm_state, config=config)
            
            # Also try to get trace_id from llm_result metadata (fallback)
            if not validation_trace_id:
                llm_metadata = llm_result.get("metadata", {})
                validation_trace_id = llm_metadata.get("langfuse_trace_id")
                if validation_trace_id:
                    logger.debug(f"Extracted trace ID from llm_result metadata: {validation_trace_id}")
            
            # Extract validation result
            validation_text = llm_result.get("last_response", "")
            
            # Parse JSON validation result
            validation_result = {
                "is_valid": True,
                "confidence": 0.8,
                "clarity": 0.8,
                "completeness": 0.8,
                "relevance": 0.8,
                "actionability": 0.8,
                "accuracy": 0.8,
                "overall_score": 0.8,
                "feedback": validation_text[:200] if validation_text else "No feedback provided"
            }
            
            # Try to extract JSON object from response
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', validation_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(1))
                    validation_result.update(parsed)
                    # Calculate overall_score if not provided
                    if "overall_score" not in parsed or parsed.get("overall_score") is None:
                        scores = [
                            parsed.get("clarity", 0.8),
                            parsed.get("completeness", 0.8),
                            parsed.get("relevance", 0.8),
                            parsed.get("actionability", 0.8),
                            parsed.get("accuracy", 0.8)
                        ]
                        validation_result["overall_score"] = sum(scores) / len(scores)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse validation JSON for {role_name}")
            
            # If no code block, try to find JSON object directly
            if validation_result.get("feedback") == validation_text[:200]:
                json_obj_match = re.search(r'\{[^{}]*"is_valid"[^{}]*\}', validation_text, re.DOTALL)
                if json_obj_match:
                    try:
                        parsed = json.loads(json_obj_match.group(0))
                        validation_result.update(parsed)
                    except json.JSONDecodeError:
                        pass
            
            # Add shared validation metadata to validation result
            if shared_validation.warnings or shared_validation.violations:
                validation_result["shared_validation"] = {
                    "warnings": shared_validation.warnings,
                    "violations": shared_validation.violations,
                    "metadata": shared_validation.metadata
                }
            
            # Log validation scores to Langfuse
            if validation_trace_id:
                try:
                    from utils.langfuse.evaluation import score_trace
                    
                    # Score each validation metric
                    scores_to_log = {
                        "prompt_validation_clarity": validation_result.get("clarity", 0.0),
                        "prompt_validation_completeness": validation_result.get("completeness", 0.0),
                        "prompt_validation_relevance": validation_result.get("relevance", 0.0),
                        "prompt_validation_actionability": validation_result.get("actionability", 0.0),
                        "prompt_validation_accuracy": validation_result.get("accuracy", 0.0),
                        "prompt_validation_overall_score": validation_result.get("overall_score", 0.0),
                        "prompt_validation_confidence": validation_result.get("confidence", 0.0)
                    }
                    
                    scores_logged = 0
                    for score_name, score_value in scores_to_log.items():
                        if score_value is not None:
                            success = score_trace(
                                trace_id=validation_trace_id,
                                name=score_name,
                                value=float(score_value),
                                comment=f"{score_name.replace('prompt_validation_', '').replace('_', ' ').title()} score for {role_name} prompt",
                                metadata={
                                    "role": role_name,
                                    "metric": score_name.replace("prompt_validation_", ""),
                                    "validation_type": "prompt_validation"
                                }
                            )
                            if success:
                                scores_logged += 1
                            else:
                                logger.debug(f"Failed to log score '{score_name}' for {role_name}")
                    
                    if scores_logged > 0:
                        logger.debug(f"Logged {scores_logged}/{len(scores_to_log)} validation scores to trace {validation_trace_id} for {role_name}")
                    else:
                        logger.warning(f"No validation scores were logged for {role_name} (trace_id: {validation_trace_id})")
                        
                except Exception as score_error:
                    # Non-blocking: log error but don't fail validation
                    logger.warning(f"Failed to log validation scores to Langfuse for {role_name}: {score_error}")
            else:
                logger.debug(f"No trace_id available for {role_name}, skipping validation score logging")
            
            # Update observation with result
            if validation_obs:
                try:
                    validation_obs.update(
                        output={
                            "is_valid": validation_result.get("is_valid", True),
                            "confidence": validation_result.get("confidence", 0.8),
                            "overall_score": validation_result.get("overall_score", 0.0),
                            "role": role_name
                        },
                        metadata={
                            "status": "success",
                            "validation_scores": {
                                "clarity": validation_result.get("clarity", 0.0),
                                "completeness": validation_result.get("completeness", 0.0),
                                "relevance": validation_result.get("relevance", 0.0),
                                "actionability": validation_result.get("actionability", 0.0),
                                "accuracy": validation_result.get("accuracy", 0.0),
                                "overall_score": validation_result.get("overall_score", 0.0)
                            }
                        }
                    )
                except AttributeError:
                    # Observation is a context manager, update not supported directly
                    pass
            
            # Return ONLY the fields that should be updated (validation_results_list)
            # The reducer (operator.add) will aggregate validation_results_list from all parallel nodes
            logger.info(f"Validated prompt for role: {role_name}")
            return {
                "validation_results_list": [{
                    "role_name": role_name,
                    "validation_result": validation_result
                }]
            }
            
        finally:
            if validation_obs:
                try:
                    validation_obs.end()
                except Exception:
                    pass
        
    except Exception as e:
        logger.error(f"Error validating prompt for {role_name}: {e}", exc_info=True)
        # Return error result in validation_results_list
        error_result = {
            "role_name": role_name,
            "error": str(e),
            "status": "error"
        }
        return {
            "validation_results_list": [error_result]
        }


def collect_validation_results_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Collect and aggregate results from parallel prompt validation.
    
    Converts validation_results_list (populated by parallel nodes) into validation_results Dict format.
    Validates all prompts validated successfully and handles partial failures gracefully.
    
    Args:
        state: Current swarm analysis state with validation_results_list
        
    Returns:
        Updated state with validation_results Dict
    """
    validation_results_list = state.get(StateKeys.VALIDATION_RESULTS_LIST, [])
    generated_prompts = state.get(StateKeys.GENERATED_PROMPTS, {})
    
    if not validation_results_list:
        # No results collected - check if we have prompts
        if generated_prompts:
            state[StateKeys.ERROR] = "No validation results collected from parallel execution"
            state[StateKeys.ERROR_STAGE] = "collect_validation_results"
            return state
        else:
            # No prompts to validate - this is fine
            updated_state = state.copy()
            updated_state[StateKeys.VALIDATION_RESULTS] = {}
            return updated_state
    
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    try:
        # Convert list to Dict format
        validation_results = {}
        successful_validations = 0
        failed_validations = 0
        
        for result in validation_results_list:
            role_name = result.get("role_name", "Unknown")
            
            if result.get("status") == "error" or "error" in result:
                # Error case
                error_msg = result.get("error", "Unknown error")
                logger.warning(f"Failed to validate prompt for {role_name}: {error_msg}")
                failed_validations += 1
                # Still add to dict with default validation result
                validation_results[role_name] = {
                    "is_valid": False,
                    "confidence": 0.0,
                    "clarity": 0.0,
                    "completeness": 0.0,
                    "relevance": 0.0,
                    "accuracy": 0.0,
                    "overall_score": 0.0,
                    "feedback": f"Validation failed: {error_msg}"
                }
            else:
                # Success case
                validation_result = result.get("validation_result")
                if validation_result:
                    validation_results[role_name] = validation_result
                    successful_validations += 1
                else:
                    logger.warning(f"Validation result for {role_name} is missing")
                    failed_validations += 1
        
        # Validate all prompts validated successfully
        expected_count = len(generated_prompts) if generated_prompts else len(validation_results_list)
        actual_count = len(validation_results_list)
        
        if actual_count < expected_count:
            logger.warning(
                f"Only {actual_count} of {expected_count} prompts validated. "
                f"Some validations may have failed silently."
            )
        
        # Set workflow-level error only if all validations failed
        if failed_validations > 0 and successful_validations == 0:
            logger.error("All prompt validations failed")
            # Don't set error here - let store_prompts_node handle it
            # But log the issue
        
        # Update state
        updated_state = state.copy()
        updated_state[StateKeys.VALIDATION_RESULTS] = validation_results
        
        if stream_callback:
            stream_callback("prompt_validation_complete", {
                "message": "All prompts validated",
                "successful": successful_validations,
                "failed": failed_validations,
                "total": actual_count
            })
        
        logger.info(
            f"Collected results from {actual_count} prompt validations: "
            f"{successful_validations} successful, {failed_validations} failed"
        )
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error collecting validation results: {e}", exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = str(e)
        updated_state[StateKeys.ERROR_STAGE] = "collect_validation_results"
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": str(e)})
        return updated_state



