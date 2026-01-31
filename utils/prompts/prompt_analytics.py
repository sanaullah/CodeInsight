"""
Prompt Analytics Utilities.

Provides functions to query and analyze prompt effectiveness from Langfuse.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def get_prompt_effectiveness(
    role_name: Optional[str] = None,
    architecture_type: Optional[str] = None,
    goal_category: Optional[str] = None,
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get prompt effectiveness metrics from Langfuse.
    
    Args:
        role_name: Optional role name to filter by
        architecture_type: Optional architecture type to filter by
        goal_category: Optional goal category to filter by
        days_back: Number of days to look back (default: 30)
        
    Returns:
        Dictionary with effectiveness metrics:
        - average_scores: Average scores by type
        - prompt_count: Number of prompts analyzed
        - best_prompts: List of best performing prompts
        - trends: Score trends over time
    """
    client = None
    try:
        from utils.langfuse_integration import get_langfuse_client
        client = get_langfuse_client()
    except Exception as e:
        logger.debug(f"Cannot get Langfuse client: {e}")
        return {}
    
    if client is None:
        logger.debug("Langfuse client not available for prompt analytics")
        return {}
    
    try:
        # Build query filters
        filters = {}
        if role_name:
            filters["metadata.role"] = role_name
        if architecture_type:
            filters["metadata.architecture_type"] = architecture_type
        if goal_category:
            filters["metadata.goal_category"] = goal_category
        
        # Query traces with prompt metadata
        # Note: This depends on Langfuse API availability
        # For now, return empty structure - implementation depends on Langfuse SDK version
        
        result = {
            "average_scores": {},
            "prompt_count": 0,
            "best_prompts": [],
            "trends": []
        }
        
        logger.debug(f"Prompt effectiveness query not yet fully implemented (Langfuse API dependency)")
        return result
        
    except Exception as e:
        logger.error(f"Error querying prompt effectiveness: {e}", exc_info=True)
        return {}


def compare_prompt_versions(
    prompt_name: str,
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Compare different versions of the same prompt pattern.
    
    Args:
        prompt_name: Prompt name pattern (e.g., "Security_Analyst_web_app_security")
        days_back: Number of days to look back
        
    Returns:
        Dictionary with comparison data:
        - versions: List of versions with scores
        - best_version: Best performing version
        - improvement_trend: Whether scores are improving over time
    """
    client = None
    try:
        from utils.langfuse_integration import get_langfuse_client
        client = get_langfuse_client()
    except Exception as e:
        logger.debug(f"Cannot get Langfuse client: {e}")
        return {}
    
    if client is None:
        logger.debug("Langfuse client not available for prompt comparison")
        return {}
    
    try:
        # Query prompts by name pattern
        # This would query Langfuse for all prompts matching the name pattern
        # and compare their effectiveness scores
        
        result = {
            "versions": [],
            "best_version": None,
            "improvement_trend": "stable"
        }
        
        logger.debug(f"Prompt version comparison not yet fully implemented (Langfuse API dependency)")
        return result
        
    except Exception as e:
        logger.error(f"Error comparing prompt versions: {e}", exc_info=True)
        return {}


def get_best_prompts_for_role(
    role_name: str,
    limit: int = 10,
    days_back: int = 30
) -> List[Dict[str, Any]]:
    """
    Get the best performing prompts for a specific role.
    
    Args:
        role_name: Role name
        limit: Maximum number of prompts to return
        days_back: Number of days to look back
        
    Returns:
        List of prompt dictionaries with effectiveness scores
    """
    client = None
    try:
        from utils.langfuse_integration import get_langfuse_client
        client = get_langfuse_client()
    except Exception as e:
        logger.debug(f"Cannot get Langfuse client: {e}")
        return []
    
    if client is None:
        logger.debug("Langfuse client not available for prompt ranking")
        return []
    
    try:
        # Query traces filtered by role and order by effectiveness scores
        # Return top N prompts
        
        result = []
        logger.debug(f"Best prompts query not yet fully implemented (Langfuse API dependency)")
        return result
        
    except Exception as e:
        logger.error(f"Error getting best prompts: {e}", exc_info=True)
        return []


def get_prompt_evolution_timeline(
    role_name: str,
    architecture_type: Optional[str] = None,
    days_back: int = 90
) -> Dict[str, Any]:
    """
    Track how prompts evolved over time for a role.
    
    Args:
        role_name: Role name
        architecture_type: Optional architecture type filter
        days_back: Number of days to look back
        
    Returns:
        Dictionary with timeline data:
        - timeline: List of prompt versions with timestamps and scores
        - trends: Score trends over time
        - improvements: List of improvements made
    """
    client = None
    try:
        from utils.langfuse_integration import get_langfuse_client
        client = get_langfuse_client()
    except Exception as e:
        logger.debug(f"Cannot get Langfuse client: {e}")
        return {}
    
    if client is None:
        logger.debug("Langfuse client not available for prompt evolution tracking")
        return {}
    
    try:
        # Query prompts over time, group by time periods, track score changes
        
        result = {
            "timeline": [],
            "trends": {},
            "improvements": []
        }
        
        logger.debug(f"Prompt evolution timeline not yet fully implemented (Langfuse API dependency)")
        return result
        
    except Exception as e:
        logger.error(f"Error getting prompt evolution timeline: {e}", exc_info=True)
        return {}


def get_prompt_statistics(
    role_name: Optional[str] = None,
    days_back: int = 30
) -> Dict[str, Any]:
    """
    Get overall statistics about prompts.
    
    Args:
        role_name: Optional role name to filter by
        days_back: Number of days to look back
        
    Returns:
        Dictionary with statistics:
        - total_prompts: Total number of prompts
        - average_effectiveness: Average effectiveness score
        - score_distribution: Distribution of scores
        - top_roles: Best performing roles
    """
    client = None
    try:
        from utils.langfuse_integration import get_langfuse_client
        client = get_langfuse_client()
    except Exception as e:
        logger.debug(f"Cannot get Langfuse client: {e}")
        return {}
    
    if client is None:
        logger.debug("Langfuse client not available for prompt statistics")
        return {}
    
    try:
        # Aggregate statistics from all prompts/traces
        
        result = {
            "total_prompts": 0,
            "average_effectiveness": 0.0,
            "score_distribution": {},
            "top_roles": []
        }
        
        logger.debug(f"Prompt statistics not yet fully implemented (Langfuse API dependency)")
        return result
        
    except Exception as e:
        logger.error(f"Error getting prompt statistics: {e}", exc_info=True)
        return {}

