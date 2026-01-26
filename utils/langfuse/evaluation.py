"""
Langfuse evaluation and scoring utilities.
"""

import logging
from typing import Optional, Dict, Any, Union
from .client import get_langfuse_client

logger = logging.getLogger(__name__)


def score_trace(
    trace_id: Optional[str] = None,
    name: str = None,
    value: float = None,
    comment: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    generation: Optional[Any] = None
) -> bool:
    """
    Add a score to a Langfuse trace.
    
    Supports two modes:
    1. Generation object mode (preferred): Pass generation object from start_as_current_observation()
    2. Trace ID mode (fallback): Pass trace_id string when generation object not available
    
    Args:
        trace_id: Langfuse trace ID (required if generation not provided)
        name: Score name (e.g., "prompt_quality", "analysis_effectiveness")
        value: Score value (typically 0.0 to 1.0)
        comment: Optional comment explaining the score
        metadata: Optional metadata dictionary
        generation: Optional Langfuse generation/observation object (preferred method)
        
    Returns:
        True if score was added successfully, False otherwise
    """
    # Validate required parameters
    if not name or value is None:
        logger.warning("score_trace requires name and value parameters")
        return False
    
    # Prefer generation object if available (Langfuse v3 best practice)
    if generation is not None:
        try:
            # Langfuse v3: generation objects have .score() method
            if hasattr(generation, 'score'):
                generation.score(
                    name=name,
                    value=value,
                    comment=comment
                )
                logger.debug(f"Added score '{name}' (value: {value}) via generation object")
                return True
            elif hasattr(generation, 'score_trace'):
                # Alternative: score the entire trace
                generation.score_trace(
                    name=name,
                    value=value,
                    comment=comment
                )
                logger.debug(f"Added score '{name}' (value: {value}) via generation.score_trace()")
                return True
            else:
                logger.warning("Generation object does not have score() or score_trace() method")
        except Exception as e:
            logger.warning(f"Error adding score via generation object: {e}")
            # Fall through to trace_id method
    
    # Fallback: Use trace_id if generation object not available or failed
    if not trace_id:
        logger.warning("Cannot score trace: neither generation object nor trace_id provided")
        return False
    
    client = get_langfuse_client()
    if client is None:
        logger.debug(f"Cannot score trace '{trace_id}': client not initialized")
        return False
    
    try:
        # Langfuse v3: Try create_score() method on client
        if hasattr(client, 'create_score'):
            client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
                metadata=metadata
            )
            logger.debug(f"Added score '{name}' (value: {value}) to trace {trace_id} via create_score()")
            return True
        else:
            logger.warning(f"Langfuse client does not have create_score() method for trace_id {trace_id}")
            return False
            
    except Exception as e:
        logger.warning(f"Error adding score to trace {trace_id}: {e}")
        return False


def evaluate_prompt_effectiveness(
    trace_id: str,
    role_name: str,
    validation_results: Optional[Dict[str, Any]] = None,
    agent_result: Optional[Dict[str, Any]] = None,
    prompt_id: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate prompt effectiveness and attach scores to trace.
    
    Calculates multiple scores:
    - Validation score: Based on validation results (confidence, clarity, etc.)
    - Quality score: Based on analysis report quality
    - Success score: Based on agent completion status
    
    Args:
        trace_id: Langfuse trace ID
        role_name: Role name for context
        validation_results: Optional validation results dictionary
        agent_result: Optional agent result dictionary
        prompt_id: Optional Langfuse prompt ID
        
    Returns:
        Dictionary with score names and values
    """
    scores = {}
    client = get_langfuse_client()
    
    # Note: We can still calculate scores even if client is not available
    # We just won't be able to attach them to traces
    
    try:
        # Calculate validation score
        validation_score = 0.0
        if validation_results:
            confidence = validation_results.get("confidence", 0.0)
            clarity = validation_results.get("clarity", 0.0)
            completeness = validation_results.get("completeness", 0.0)
            relevance = validation_results.get("relevance", 0.0)
            accuracy = validation_results.get("accuracy", 0.0)
            
            # Average of all validation metrics
            validation_score = (confidence + clarity + completeness + relevance + accuracy) / 5.0
        
        # Calculate quality score
        quality_score = 0.0
        if agent_result:
            report = agent_result.get("synthesized_report", "")
            if report:
                # Simple quality metrics
                report_length = len(report)
                has_structure = any(marker in report for marker in ["##", "###", "**", "-"])
                has_evidence = any(marker in report for marker in ["file:", "line", "path"])
                
                # Normalize scores (0.0 to 1.0)
                length_score = min(report_length / 1000.0, 1.0)  # 1000 chars = full score
                structure_score = 1.0 if has_structure else 0.5
                evidence_score = 1.0 if has_evidence else 0.5
                
                quality_score = (length_score * 0.4 + structure_score * 0.3 + evidence_score * 0.3)
        
        # Calculate success score
        success_score = 1.0
        if agent_result:
            if "error" in agent_result:
                success_score = 0.0
            elif agent_result.get("status") != "completed":
                success_score = 0.5
        
        # Calculate overall effectiveness score
        overall_score = (validation_score * 0.4 + quality_score * 0.4 + success_score * 0.2)
        
        # Store scores
        scores = {
            "validation_score": validation_score,
            "quality_score": quality_score,
            "success_score": success_score,
            "overall_effectiveness": overall_score
        }
        
        # Attach scores to trace
        score_metadata = {
            "role": role_name,
            "evaluation_type": "prompt_effectiveness"
        }
        if prompt_id:
            score_metadata["prompt_id"] = prompt_id
        
        # Add individual scores
        scores_attached = 0
        
        # Skip scoring if trace_id is "unknown" (indicates trace ID was not captured)
        if trace_id == "unknown" or not trace_id:
            logger.debug(f"Skipping score attachment for {role_name}: trace_id not available")
        else:
            for score_name, score_value in scores.items():
                success = score_trace(
                    trace_id=trace_id,
                    name=score_name,
                    value=score_value,
                    comment=f"Prompt effectiveness evaluation for {role_name}",
                    metadata={**score_metadata, "score_type": score_name}
                )
                if success:
                    scores_attached += 1
                else:
                    logger.warning(f"Failed to attach score '{score_name}' to trace {trace_id}")
        
        if scores_attached > 0:
            logger.info(f"Attached {scores_attached}/{len(scores)} scores for {role_name}: {scores}")
        elif trace_id and trace_id != "unknown":
            logger.warning(f"No scores were attached to trace {trace_id} for {role_name}")
        
        return scores
        
    except Exception as e:
        logger.error(f"Error evaluating prompt effectiveness: {e}", exc_info=True)
        return scores

