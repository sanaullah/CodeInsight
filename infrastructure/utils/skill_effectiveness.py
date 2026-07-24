"""
Skill effectiveness calculation utilities.

Calculates effectiveness scores for skills based on analysis quality metrics.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def calculate_skill_effectiveness(
    skill_type: str,
    skill_category: str,
    evaluation_results: Optional[Dict[str, Dict[str, float]]] = None,
    validation_results: Optional[Dict[str, Dict[str, Any]]] = None,
    agent_results: Optional[Dict[str, Dict[str, Any]]] = None,
    synthesized_report: Optional[str] = None,
    role_names: Optional[List[str]] = None
) -> float:
    """
    Calculate effectiveness score for a skill based on its type.
    
    Args:
        skill_type: Type of skill ("role_selection", "prompt_generation", "synthesis")
        skill_category: Category of skill ("helpful", "harmful", "neutral")
        evaluation_results: Dictionary of role_name -> evaluation scores
        validation_results: Dictionary of role_name -> validation data
        agent_results: Dictionary of role_name -> agent result
        synthesized_report: Final synthesized report
        role_names: List of role names that were selected
        
    Returns:
        Effectiveness score (0.0 to 1.0)
    """
    if skill_category == "harmful":
        # For harmful skills, effectiveness is inverse (lower is better)
        # We'll calculate normal effectiveness then invert it
        base_effectiveness = _calculate_base_effectiveness(
            skill_type, evaluation_results, validation_results,
            agent_results, synthesized_report, role_names
        )
        # Invert: if base is high (bad), effectiveness is low
        return 1.0 - base_effectiveness
    
    return _calculate_base_effectiveness(
        skill_type, evaluation_results, validation_results,
        agent_results, synthesized_report, role_names
    )


def _calculate_base_effectiveness(
    skill_type: str,
    evaluation_results: Optional[Dict[str, Dict[str, float]]],
    validation_results: Optional[Dict[str, Dict[str, Any]]],
    agent_results: Optional[Dict[str, Dict[str, Any]]],
    synthesized_report: Optional[str],
    role_names: Optional[List[str]]
) -> float:
    """Calculate base effectiveness (before inversion for harmful skills)."""
    
    if skill_type == "role_selection":
        return _calculate_role_selection_effectiveness(
            role_names, agent_results, synthesized_report
        )
    
    elif skill_type == "prompt_generation":
        return _calculate_prompt_generation_effectiveness(
            evaluation_results, validation_results, agent_results
        )
    
    elif skill_type == "synthesis":
        return _calculate_synthesis_effectiveness(
            synthesized_report, evaluation_results, agent_results
        )
    
    else:
        # Unknown skill type - use overall analysis success
        return _calculate_overall_effectiveness(
            agent_results, synthesized_report
        )


def _calculate_role_selection_effectiveness(
    role_names: Optional[List[str]],
    agent_results: Optional[Dict[str, Dict[str, Any]]],
    synthesized_report: Optional[str]
) -> float:
    """Calculate effectiveness for role selection skills."""
    if not role_names or not agent_results:
        return 0.5  # Default neutral score
    
    # Calculate success rate: how many agents completed successfully
    successful_agents = sum(
        1 for role in role_names
        if role in agent_results and "error" not in agent_results[role]
    )
    success_rate = successful_agents / len(role_names) if role_names else 0.0
    
    # Factor in report quality
    report_quality = 1.0 if synthesized_report and len(synthesized_report) > 100 else 0.5
    
    # Weighted average: 70% success rate, 30% report quality
    return (success_rate * 0.7 + report_quality * 0.3)


def _calculate_prompt_generation_effectiveness(
    evaluation_results: Optional[Dict[str, Dict[str, float]]],
    validation_results: Optional[Dict[str, Dict[str, Any]]],
    agent_results: Optional[Dict[str, Dict[str, Any]]]
) -> float:
    """Calculate effectiveness for prompt generation skills."""
    if not evaluation_results and not validation_results:
        return 0.5  # Default neutral score
    
    scores = []
    
    # Use evaluation results if available (more comprehensive)
    if evaluation_results:
        for role_scores in evaluation_results.values():
            overall = role_scores.get("overall_effectiveness", 0.0)
            if overall > 0:  # Only count if calculated
                scores.append(overall)
    
    # Fallback to validation results
    if not scores and validation_results:
        for validation in validation_results.values():
            confidence = validation.get("confidence", 0.0)
            if confidence > 0:
                scores.append(confidence)
    
    # Average all scores
    if scores:
        return sum(scores) / len(scores)
    
    return 0.5  # Default neutral score


def _calculate_synthesis_effectiveness(
    synthesized_report: Optional[str],
    evaluation_results: Optional[Dict[str, Dict[str, float]]],
    agent_results: Optional[Dict[str, Dict[str, Any]]]
) -> float:
    """Calculate effectiveness for synthesis skills."""
    if not synthesized_report:
        return 0.0  # No report = no effectiveness
    
    # Report quality metrics
    report_length = len(synthesized_report)
    has_structure = any(marker in synthesized_report for marker in ["##", "###", "**", "-"])
    has_evidence = any(marker in synthesized_report for marker in ["file:", "line", "path"])
    
    # Normalize scores
    length_score = min(report_length / 2000.0, 1.0)  # 2000 chars = full score
    structure_score = 1.0 if has_structure else 0.5
    evidence_score = 1.0 if has_evidence else 0.5
    
    # Weighted report quality
    report_quality = (length_score * 0.4 + structure_score * 0.3 + evidence_score * 0.3)
    
    # Factor in agent results quality (if available)
    agent_quality = 0.5  # Default
    if agent_results:
        successful = sum(1 for r in agent_results.values() if "error" not in r)
        total = len(agent_results)
        agent_quality = successful / total if total > 0 else 0.5
    
    # Weighted average: 80% report quality, 20% agent quality
    return (report_quality * 0.8 + agent_quality * 0.2)


def _calculate_overall_effectiveness(
    agent_results: Optional[Dict[str, Dict[str, Any]]],
    synthesized_report: Optional[str]
) -> float:
    """Calculate overall analysis effectiveness (fallback for unknown skill types)."""
    if not agent_results:
        return 0.5
    
    successful = sum(1 for r in agent_results.values() if "error" not in r)
    total = len(agent_results)
    success_rate = successful / total if total > 0 else 0.5
    
    report_exists = 1.0 if synthesized_report and len(synthesized_report) > 100 else 0.5
    
    return (success_rate * 0.7 + report_exists * 0.3)
