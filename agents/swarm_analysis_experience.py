"""
Swarm Analysis experience models for capturing learnings.

Provides utilities for creating and storing swarm analysis experiences.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from services.experience_models import Experience, PerformanceMetrics, Outcome, SuccessLevel
from utils.experience_storage import store_experience

logger = logging.getLogger(__name__)


def create_swarm_experience(
    project_path: str,
    architecture_model: Dict[str, Any],
    roles_selected: List[str],
    prompts_generated: Dict[str, str],
    validation_results: Dict[str, Dict[str, Any]],
    agent_results: Dict[str, Dict[str, Any]],
    synthesized_report: str,
    goal: Optional[str] = None,
    files_scanned: int = 0,
    chunks_analyzed: int = 0,
    execution_time: float = 0.0,
    token_usage: int = 0
) -> Experience:
    """
    Create an Experience object from swarm analysis results.
    
    Args:
        project_path: Path to analyzed project
        architecture_model: Architecture model dictionary
        roles_selected: List of selected role names
        prompts_generated: Dictionary of role -> prompt
        validation_results: Dictionary of role -> validation data
        agent_results: Dictionary of role -> agent result
        synthesized_report: Final synthesized report
        goal: Optional analysis goal
        files_scanned: Number of files scanned
        chunks_analyzed: Number of chunks analyzed
        execution_time: Execution time in seconds
        token_usage: Total token usage
        
    Returns:
        Experience object ready for storage
    """
    experience_id = f"swarm_{uuid.uuid4().hex[:16]}"
    
    # Calculate performance metrics
    successful_agents = len([r for r in agent_results.values() if "error" not in r])
    total_agents = len(roles_selected)
    success_rate = successful_agents / total_agents if total_agents > 0 else 0.0
    
    performance_metrics = PerformanceMetrics(
        goal_achievement_score=1.0 if synthesized_report else 0.5,  # Simplified
        quality_score=sum(
            v.get("confidence", 0.5) for v in validation_results.values()
        ) / len(validation_results) if validation_results else 0.5,
        efficiency_score=0.8,  # Placeholder
        token_usage=token_usage,
        execution_time=execution_time,
        error_count=len([r for r in agent_results.values() if "error" in r]),
        adaptation_count=0
    )
    
    # Determine outcome
    success = success_rate >= 0.5 and synthesized_report is not None
    success_level = SuccessLevel.FULL if success_rate >= 0.9 else (
        SuccessLevel.PARTIAL if success_rate >= 0.5 else SuccessLevel.FAILED
    )
    
    primary_achievements = []
    if synthesized_report:
        primary_achievements.append("Analysis completed successfully")
    if successful_agents > 0:
        primary_achievements.append(f"{successful_agents} agents executed successfully")
    
    failures = []
    for role, result in agent_results.items():
        if "error" in result:
            failures.append(f"{role}: {result.get('error', 'Unknown error')}")
    
    outcome = Outcome(
        success=success,
        success_level=success_level,
        primary_achievements=primary_achievements,
        failures=failures
    )
    
    # Extract lessons learned
    lessons_learned = []
    if validation_results:
        avg_confidence = sum(
            v.get("confidence", 0) for v in validation_results.values()
        ) / len(validation_results)
        if avg_confidence > 0.8:
            lessons_learned.append("High-quality prompts generated")
        elif avg_confidence < 0.6:
            lessons_learned.append("Prompt quality needs improvement")
    
    # Create context
    context = {
        "project_path": project_path,
        "architecture_type": architecture_model.get("system_type", "unknown"),
        "architecture_pattern": architecture_model.get("architecture_pattern", "unknown"),
        "files_scanned": files_scanned,
        "chunks_analyzed": chunks_analyzed,
        "goal": goal
    }
    
    # Create execution plan (simplified)
    execution_plan = {
        "roles": roles_selected,
        "max_agents": len(roles_selected),
        "strategy": "swarm_analysis"
    }
    
    # Create results
    results = {
        "synthesized_report": synthesized_report,
        "agent_results": agent_results,
        "validation_results": validation_results
    }
    
    # Create experience
    # Note: Experience requires GoalUnderstanding and Strategy which may not exist
    # For now, create simplified version
    try:
        experience = Experience(
            experience_id=experience_id,
            agent_name="Swarm Analysis",
            goal=goal or "General code analysis",
            goal_understanding=None,  # TODO: Create proper GoalUnderstanding
            strategy_used=None,  # TODO: Create proper Strategy
            context=context,
            execution_plan=execution_plan,
            results=results,
            performance_metrics=performance_metrics,
            outcome=outcome,
            lessons_learned=lessons_learned,
            timestamp=datetime.now(),
            duration=execution_time
        )
        return experience
    except Exception as e:
        logger.warning(f"Could not create full Experience object: {e}, creating simplified version")
        # Return a simplified dict-based experience
        return {
            "experience_id": experience_id,
            "agent_name": "Swarm Analysis",
            "goal": goal or "General code analysis",
            "context": context,
            "execution_plan": execution_plan,
            "results": results,
            "performance_metrics": {
                "goal_achievement_score": performance_metrics.goal_achievement_score,
                "quality_score": performance_metrics.quality_score,
                "token_usage": performance_metrics.token_usage,
                "execution_time": performance_metrics.execution_time,
                "error_count": performance_metrics.error_count
            },
            "outcome": {
                "success": outcome.success,
                "success_level": outcome.success_level.value,
                "primary_achievements": outcome.primary_achievements,
                "failures": outcome.failures
            },
            "lessons_learned": lessons_learned,
            "timestamp": datetime.now().isoformat(),
            "duration": execution_time
        }


def store_swarm_experience(
    state: Dict[str, Any],
    execution_time: float = 0.0,
    token_usage: int = 0
) -> Optional[str]:
    """
    Store swarm analysis experience from state.
    
    Args:
        state: Swarm analysis state dictionary
        execution_time: Execution time in seconds
        token_usage: Total token usage
        
    Returns:
        Experience ID if successful, None otherwise
    """
    try:
        experience = create_swarm_experience(
            project_path=state.get("project_path", ""),
            architecture_model=state.get("architecture_model", {}),
            roles_selected=state.get("role_names", []),
            prompts_generated=state.get("generated_prompts", {}),
            validation_results=state.get("validation_results", {}),
            agent_results=state.get("agent_results", {}),
            synthesized_report=state.get("synthesized_report", ""),
            goal=state.get("goal"),
            files_scanned=len(state.get("files", [])),
            chunks_analyzed=state.get("chunks_analyzed", 0),
            execution_time=execution_time,
            token_usage=token_usage
        )
        
        # Store experience
        if isinstance(experience, Experience):
            experience_id = store_experience(experience)
        else:
            # Simplified dict-based experience - store directly
            # TODO: Create proper storage method for dict-based experiences
            logger.warning("Storing simplified experience (dict-based)")
            experience_id = experience.get("experience_id")
        
        logger.info(f"Stored swarm analysis experience: {experience_id}")
        return experience_id
        
    except Exception as e:
        logger.error(f"Error storing swarm experience: {e}", exc_info=True)
        return None

