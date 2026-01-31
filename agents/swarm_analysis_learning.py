"""
Swarm Analysis learning retrieval system.

Retrieves past experiences to inform current analysis.
Integrates Langfuse prompt effectiveness data for enhanced learning.
"""

import logging
from typing import Dict, Any, List, Optional

from utils.experience_storage import query_experiences, get_similar_experiences

logger = logging.getLogger(__name__)


def retrieve_relevant_experiences(
    architecture_type: Optional[str] = None,
    goal: Optional[str] = None,
    project_path: Optional[str] = None,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant past experiences for swarm analysis.
    
    Args:
        architecture_type: Architecture type to match
        goal: Analysis goal to match
        project_path: Project path to match
        limit: Maximum number of experiences to retrieve
        
    Returns:
        List of relevant experience dictionaries
    """
    try:
        # Build filters
        filters = {
            "agent_name": "Swarm Analysis"
        }
        
        if architecture_type:
            filters["context.architecture_type"] = architecture_type
        
        if goal:
            filters["goal"] = goal
        
        # Query experiences
        experiences = query_experiences(filters, limit=limit)
        
        # Convert to dictionaries if needed
        experience_dicts = []
        for exp in experiences:
            if isinstance(exp, dict):
                experience_dicts.append(exp)
            elif hasattr(exp, '__dict__'):
                experience_dicts.append(exp.__dict__)
            else:
                # Try to extract key fields
                experience_dicts.append({
                    "experience_id": getattr(exp, 'experience_id', 'unknown'),
                    "goal": getattr(exp, 'goal', ''),
                    "context": getattr(exp, 'context', {}),
                    "results": getattr(exp, 'results', {}),
                    "performance_metrics": getattr(exp, 'performance_metrics', {}),
                    "outcome": getattr(exp, 'outcome', {})
                })
        
        logger.info(f"Retrieved {len(experience_dicts)} relevant experiences")
        return experience_dicts
        
    except Exception as e:
        logger.error(f"Error retrieving experiences: {e}", exc_info=True)
        return []


def extract_learnings_from_experiences(
    experiences: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract learnings from past experiences.
    
    Args:
        experiences: List of experience dictionaries
        
    Returns:
        Dictionary with extracted learnings:
        - effective_roles: Roles that worked well
        - effective_prompts: Prompt patterns that worked
        - common_issues: Common issues found
        - recommendations: Recommendations from past analyses
    """
    learnings = {
        "effective_roles": [],
        "effective_prompts": [],
        "common_issues": [],
        "recommendations": []
    }
    
    if not experiences:
        return learnings
    
    try:
        # Extract role information
        role_counts = {}
        for exp in experiences:
            execution_plan = exp.get("execution_plan", {})
            roles = execution_plan.get("roles", [])
            outcome = exp.get("outcome", {})
            success = outcome.get("success", False)
            
            if success:
                for role in roles:
                    role_counts[role] = role_counts.get(role, 0) + 1
        
        # Get most effective roles
        learnings["effective_roles"] = sorted(
            role_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        # Extract prompt patterns (if available in results)
        prompt_patterns = []
        for exp in experiences:
            results = exp.get("results", {})
            validation_results = results.get("validation_results", {})
            evaluation_results = exp.get("metadata", {}).get("evaluation_results", {})
            
            for role, validation in validation_results.items():
                # Get effectiveness score from evaluation if available
                effectiveness = 0.0
                if role in evaluation_results:
                    effectiveness = evaluation_results[role].get("overall_effectiveness", 0.0)
                elif validation.get("is_valid"):
                    # Fallback to validation confidence
                    effectiveness = validation.get("confidence", 0.0)
                
                if effectiveness > 0.7:  # Threshold for "effective"
                    prompt_patterns.append({
                        "role": role,
                        "confidence": validation.get("confidence", 0),
                        "effectiveness": effectiveness,
                        "validation_score": validation.get("overall_score", 0.0)
                    })
        
        # Sort by effectiveness (highest first)
        prompt_patterns.sort(key=lambda x: x.get("effectiveness", 0), reverse=True)
        learnings["effective_prompts"] = prompt_patterns[:10]  # Top 10
        
        # Also try to get Langfuse prompt effectiveness data
        try:
            from utils.prompts.prompt_analytics import get_best_prompts_for_role
            
            # Get best prompts from Langfuse for each role we've seen
            langfuse_prompts = {}
            for role, _ in role_counts.items():
                try:
                    best_prompts = get_best_prompts_for_role(role, limit=3)
                    if best_prompts:
                        langfuse_prompts[role] = best_prompts
                except Exception as langfuse_error:
                    logger.debug(f"Could not get Langfuse prompts for {role}: {langfuse_error}")
            
            # Merge Langfuse data with experience-based learnings
            if langfuse_prompts:
                learnings["langfuse_prompt_data"] = langfuse_prompts
        except ImportError:
            logger.debug("Langfuse prompt analytics not available")
        except Exception as e:
            logger.debug(f"Error integrating Langfuse prompt data: {e}")
        
        # Extract common issues from synthesized reports
        # TODO: Use NLP to extract common issues from reports
        learnings["common_issues"] = []
        
        # Extract recommendations
        recommendations = []
        for exp in experiences:
            results = exp.get("results", {})
            report = results.get("synthesized_report", "")
            if report:
                # Extract recommendations section (simplified)
                if "Recommendations" in report or "recommendations" in report.lower():
                    recommendations.append({
                        "experience_id": exp.get("experience_id", "unknown"),
                        "has_recommendations": True
                    })
        
        learnings["recommendations"] = recommendations
        
        logger.info(f"Extracted learnings from {len(experiences)} experiences")
        return learnings
        
    except Exception as e:
        logger.error(f"Error extracting learnings: {e}", exc_info=True)
        return learnings


def get_learned_skills_for_analysis(
    architecture_type: Optional[str] = None,
    goal: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get learned skills to inform current analysis.
    
    Args:
        architecture_type: Architecture type
        goal: Analysis goal
        
    Returns:
        Dictionary with learned skills organized by type
    """
    try:
        # Retrieve relevant experiences
        experiences = retrieve_relevant_experiences(
            architecture_type=architecture_type,
            goal=goal,
            limit=10
        )
        
        # Extract learnings
        learnings = extract_learnings_from_experiences(experiences)
        
        # Organize by skill type
        learned_skills = {
            "role_selection": {
                "effective_roles": [r[0] for r in learnings.get("effective_roles", [])],
                "role_effectiveness": dict(learnings.get("effective_roles", []))
            },
            "prompt_generation": {
                "effective_patterns": learnings.get("effective_prompts", []),
                "high_confidence_roles": [
                    p["role"] for p in learnings.get("effective_prompts", [])
                    if p.get("confidence", 0) > 0.8 or p.get("effectiveness", 0) > 0.8
                ],
                "high_effectiveness_roles": [
                    p["role"] for p in learnings.get("effective_prompts", [])
                    if p.get("effectiveness", 0) > 0.8
                ],
                "langfuse_data": learnings.get("langfuse_prompt_data", {})
            },
            "common_issues": learnings.get("common_issues", []),
            "recommendations": learnings.get("recommendations", [])
        }
        
        logger.info(f"Retrieved learned skills for architecture_type={architecture_type}, goal={goal}")
        return learned_skills
        
    except Exception as e:
        logger.error(f"Error getting learned skills: {e}", exc_info=True)
        return {}


def get_previous_report_from_experiences(
    architecture_type: Optional[str] = None,
    goal: Optional[str] = None,
    project_path: Optional[str] = None,
    limit: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Get a previous report candidate from the experience learning system.

    This is intended for \"Auto-select Similar\" behaviour in the UI.

    Returns:
        Dictionary with at least:
        - 'experience_id': str
        - 'report_text': str (synthesized report)
    Or None if no suitable experience is found.
    """
    try:
        # Prefer experiences that match architecture_type and/or goal.
        experiences = retrieve_relevant_experiences(
            architecture_type=architecture_type,
            goal=goal,
            project_path=project_path,
            limit=limit or 1,
        )

        if not experiences:
            logger.info("No experiences available for previous-report auto selection")
            return None

        from utils.report_loader import extract_synthesized_report_from_result

        for exp in experiences:
            # Experience schema may wrap results under different keys; treat the
            # whole experience dict as a result payload for the extractor.
            report_text = extract_synthesized_report_from_result(exp) or extract_synthesized_report_from_result(
                exp.get("results", {}) if isinstance(exp.get("results"), dict) else {}
            )
            if report_text:
                return {
                    "experience_id": exp.get("experience_id", "unknown"),
                    "report_text": report_text,
                }

        logger.info("Could not find synthesized_report in retrieved experiences")
        return None

    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Error getting previous report from experiences: {e}", exc_info=True)
        return None

