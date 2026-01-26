"""
Swarm skillbook utility module.

This module provides a PostgreSQL-based API for managing swarm skillbook.
All data is stored in PostgreSQL with Redis caching.

Backward-compatible functions:
- save_skill()
- get_skills()
- get_relevant_skills()
- update_skill_usage()
- save_reflection()
- get_similar_reflections()
- deduplicate_skills()
- get_skill_statistics()
- get_improvement_metrics()
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from services.storage.base_storage import (
    StorageError,
    StorageConnectionError,
    DatabaseConnectionError,
    ConfigurationError,
    UserFacingError
)

logger = logging.getLogger(__name__)

# Import storage
try:
    from services.storage.cached_swarm_skillbook_storage import CachedSwarmSkillbookStorage
    from agents.swarm_skillbook_models import SwarmSkill, SwarmReflection
    _storage = None
    
    def _get_storage():
        """Get or create storage instance."""
        global _storage
        if _storage is None:
            _storage = CachedSwarmSkillbookStorage()
        return _storage
except ImportError as e:
    logger.error(f"Failed to import CachedSwarmSkillbookStorage: {e}")
    _storage = None
    
    def _get_storage():
        """Get storage instance."""
        return None


def save_skill(skill: SwarmSkill) -> str:
    """
    Save skill.
    
    Args:
        skill: SwarmSkill object
    
    Returns:
        Skill ID
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Storage not available")
    
    try:
        skill_id = storage.save_skill(skill)
        if not skill_id:
            raise RuntimeError("Failed to store skill: storage returned None")
        return skill_id
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error storing skill: {e}", exc_info=True)
        raise UserFacingError("Unable to connect to analysis database. Please try again later.") from e
    except StorageError as e:
        logger.error(f"Storage error storing skill: {e}", exc_info=True)
        raise
    except ConfigurationError as e:
        logger.error(f"Configuration error storing skill: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error storing skill: {e}", exc_info=True)
        raise


def get_skills(
    skill_type: Optional[str] = None,
    skill_category: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> List[SwarmSkill]:
    """
    Get skills with optional filters.
    
    Args:
        skill_type: Optional skill type filter
        skill_category: Optional category filter
        context: Optional context filter
    
    Returns:
        List of SwarmSkill objects
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return []
    
    try:
        return storage.get_skills(
            skill_type=skill_type,
            skill_category=skill_category,
            context=context
        )
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting skills: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting skills: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error getting skills: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting skills: {e}", exc_info=True)
        return []  # Fallback


def get_relevant_skills(
    architecture_type: str,
    goal: Optional[str] = None
) -> Dict[str, List[SwarmSkill]]:
    """
    Get relevant skills organized by skill type.
    
    Args:
        architecture_type: Architecture type to match
        goal: Optional goal for additional filtering
    
    Returns:
        Dictionary mapping skill_type -> List[SwarmSkill]
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return {}
    
    try:
        return storage.get_relevant_skills(architecture_type, goal)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting relevant skills: {e}", exc_info=True)
        return {}  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting relevant skills: {e}", exc_info=True)
        return {}
    except ConfigurationError as e:
        logger.error(f"Configuration error getting relevant skills: {e}", exc_info=True)
        return {}  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting relevant skills: {e}", exc_info=True)
        return {}  # Fallback


def update_skill_usage(skill_id: str, effectiveness: float, analysis_id: str) -> None:
    """
    Update skill usage statistics.
    
    Args:
        skill_id: Skill ID
        effectiveness: Effectiveness score
        analysis_id: Analysis ID
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return
    
    try:
        storage.update_skill_usage(skill_id, effectiveness, analysis_id)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error updating skill usage: {e}", exc_info=True)
        raise UserFacingError("Unable to connect to analysis database. Please try again later.") from e
    except StorageError as e:
        logger.error(f"Storage error updating skill usage: {e}", exc_info=True)
        raise
    except ConfigurationError as e:
        logger.error(f"Configuration error updating skill usage: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error updating skill usage: {e}", exc_info=True)
        raise


def save_reflection(reflection: SwarmReflection) -> str:
    """
    Save reflection.
    
    Args:
        reflection: SwarmReflection object
    
    Returns:
        Reflection ID
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Storage not available")
    
    try:
        reflection_id = storage.save_reflection(reflection)
        if not reflection_id:
            raise RuntimeError("Failed to store reflection: storage returned None")
        return reflection_id
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error storing reflection: {e}", exc_info=True)
        raise UserFacingError("Unable to connect to analysis database. Please try again later.") from e
    except StorageError as e:
        logger.error(f"Storage error storing reflection: {e}", exc_info=True)
        raise
    except ConfigurationError as e:
        logger.error(f"Configuration error storing reflection: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error storing reflection: {e}", exc_info=True)
        raise


def get_similar_reflections(
    architecture_type: str,
    limit: int = 10
) -> List[SwarmReflection]:
    """
    Get similar reflections for a given architecture type.
    
    Args:
        architecture_type: Architecture type to match
        limit: Maximum number of results
    
    Returns:
        List of SwarmReflection objects
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return []
    
    try:
        return storage.get_similar_reflections(architecture_type, limit)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting reflections: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting reflections: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error getting reflections: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting reflections: {e}", exc_info=True)
        return []  # Fallback


def deduplicate_skills() -> Dict[str, int]:
    """
    Deduplicate similar skills by merging them.
    
    Returns:
        Dictionary with deduplication statistics
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return {"merged": 0, "deleted": 0}
    
    try:
        return storage.deduplicate_skills()
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error deduplicating skills: {e}", exc_info=True)
        return {"merged": 0, "deleted": 0}  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error deduplicating skills: {e}", exc_info=True)
        return {"merged": 0, "deleted": 0}
    except ConfigurationError as e:
        logger.error(f"Configuration error deduplicating skills: {e}", exc_info=True)
        return {"merged": 0, "deleted": 0}  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error deduplicating skills: {e}", exc_info=True)
        return {"merged": 0, "deleted": 0}  # Fallback


def get_skill_statistics() -> Dict[str, Any]:
    """
    Get statistics about skill usage and effectiveness.
    
    Returns:
        Dictionary with statistics
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return {}
    
    try:
        return storage.get_skill_statistics()
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting skill statistics: {e}", exc_info=True)
        return {}  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting skill statistics: {e}", exc_info=True)
        return {}
    except ConfigurationError as e:
        logger.error(f"Configuration error getting skill statistics: {e}", exc_info=True)
        return {}  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting skill statistics: {e}", exc_info=True)
        return {}  # Fallback


def get_improvement_metrics(
    time_period: Tuple[datetime, datetime]
) -> Dict[str, Any]:
    """
    Get improvement metrics over time.
    
    Args:
        time_period: Tuple of (start_time, end_time)
    
    Returns:
        Dictionary with improvement metrics
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return {}
    
    try:
        return storage.get_improvement_metrics(time_period)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting improvement metrics: {e}", exc_info=True)
        return {}  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting improvement metrics: {e}", exc_info=True)
        return {}
    except ConfigurationError as e:
        logger.error(f"Configuration error getting improvement metrics: {e}", exc_info=True)
        return {}  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting improvement metrics: {e}", exc_info=True)
        return {}  # Fallback

