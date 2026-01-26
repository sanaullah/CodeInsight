"""
Experience storage database management for CodeLumen.

This module provides a PostgreSQL-based API for managing agent experiences.
All data is stored in PostgreSQL with Redis caching.

Backward-compatible functions:
- store_experience()
- retrieve_experience()
- query_experiences()
- get_similar_experiences()
- get_experience_statistics()
"""

import logging
from typing import Dict, List, Optional, Any

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
    from services.storage.cached_experience_storage import CachedExperienceStorage
    _storage = None
    
    def _get_storage():
        """Get or create storage instance."""
        global _storage
        if _storage is None:
            _storage = CachedExperienceStorage()
        return _storage
except ImportError as e:
    logger.error(f"Failed to import CachedExperienceStorage: {e}")
    _storage = None
    
    def _get_storage():
        """Get storage instance."""
        return None


def store_experience(experience) -> str:
    """
    Store agent experience.
    
    Args:
        experience: Experience object
    
    Returns:
        Experience ID
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Storage not available")
    
    try:
        experience_id = storage.store_experience(experience)
        if not experience_id:
            raise RuntimeError("Failed to store experience: storage returned None")
        return experience_id
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error storing experience: {e}", exc_info=True)
        raise UserFacingError("Unable to connect to analysis database. Please try again later.") from e
    except StorageError as e:
        logger.error(f"Storage error storing experience: {e}", exc_info=True)
        raise
    except ConfigurationError as e:
        logger.error(f"Configuration error storing experience: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error storing experience: {e}", exc_info=True)
        raise


def retrieve_experience(experience_id: str):
    """
    Retrieve experience by ID.
    
    Args:
        experience_id: Experience ID
    
    Returns:
        Experience object or None
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return None
    
    try:
        return storage.retrieve_experience(experience_id)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error retrieving experience {experience_id}: {e}", exc_info=True)
        return None  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error retrieving experience {experience_id}: {e}", exc_info=True)
        return None
    except ConfigurationError as e:
        logger.error(f"Configuration error retrieving experience {experience_id}: {e}", exc_info=True)
        return None  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error retrieving experience {experience_id}: {e}", exc_info=True)
        return None  # Fallback


def query_experiences(filters: Dict[str, Any], limit: int = 100) -> List:
    """
    Query experiences with filters.
    
    Args:
        filters: Filter dictionary
        limit: Maximum number of results
    
    Returns:
        List of Experience objects
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return []
    
    try:
        return storage.query_experiences(filters, limit)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error querying experiences: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error querying experiences: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error querying experiences: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error querying experiences: {e}", exc_info=True)
        return []  # Fallback


def get_similar_experiences(experience, limit: int = 10) -> List:
    """
    Get similar experiences.
    
    Args:
        experience: Experience object
        limit: Maximum number of results
    
    Returns:
        List of similar Experience objects
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return []
    
    try:
        return storage.get_similar_experiences(experience, limit)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting similar experiences: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting similar experiences: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error getting similar experiences: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting similar experiences: {e}", exc_info=True)
        return []  # Fallback


def get_experience_statistics(agent_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get experience statistics.
    
    Args:
        agent_name: Optional agent name filter
    
    Returns:
        Statistics dictionary
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return {"total_experiences": 0}
    
    try:
        return storage.get_experience_statistics(agent_name)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting experience statistics: {e}", exc_info=True)
        return {"total_experiences": 0}  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting experience statistics: {e}", exc_info=True)
        return {"total_experiences": 0}
    except ConfigurationError as e:
        logger.error(f"Configuration error getting experience statistics: {e}", exc_info=True)
        return {"total_experiences": 0}  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting experience statistics: {e}", exc_info=True)
        return {"total_experiences": 0}  # Fallback

