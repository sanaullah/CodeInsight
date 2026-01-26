"""
Knowledge base utility module.

This module provides a PostgreSQL-based API for managing knowledge base.
All data is stored in PostgreSQL with Redis caching.

Backward-compatible functions:
- store_knowledge()
- retrieve_knowledge()
- query_knowledge()
- search_knowledge()
- get_knowledge_statistics()
- cleanup_expired_knowledge()
"""

import logging
from typing import Dict, List, Optional, Any
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
    from services.storage.cached_knowledge_storage import CachedKnowledgeStorage
    _storage = None
    
    def _get_storage():
        """Get or create storage instance."""
        global _storage
        if _storage is None:
            _storage = CachedKnowledgeStorage()
        return _storage
except ImportError as e:
    logger.error(f"Failed to import CachedKnowledgeStorage: {e}")
    _storage = None
    
    def _get_storage():
        """Get storage instance."""
        return None


def store_knowledge(knowledge) -> str:
    """
    Store knowledge.
    
    Args:
        knowledge: Knowledge object from agents.collaboration_models
    
    Returns:
        Knowledge ID
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Storage not available")
    
    try:
        knowledge_id = storage.save(knowledge)
        if not knowledge_id:
            raise RuntimeError("Failed to store knowledge: storage returned None")
        return knowledge_id
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error storing knowledge: {e}", exc_info=True)
        raise UserFacingError("Unable to connect to analysis database. Please try again later.") from e
    except StorageError as e:
        logger.error(f"Storage error storing knowledge: {e}", exc_info=True)
        raise
    except ConfigurationError as e:
        logger.error(f"Configuration error storing knowledge: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error storing knowledge: {e}", exc_info=True)
        raise


def retrieve_knowledge(knowledge_id: str):
    """
    Retrieve knowledge by ID.
    
    Args:
        knowledge_id: Knowledge ID
    
    Returns:
        Knowledge object or None
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return None
    
    try:
        knowledge_dict = storage.get(knowledge_id)
        if knowledge_dict:
            # Convert dict to Knowledge object
            from agents.collaboration_models import Knowledge, KnowledgeType
            return _dict_to_knowledge(knowledge_dict)
        return None
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error retrieving knowledge {knowledge_id}: {e}", exc_info=True)
        return None  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error retrieving knowledge {knowledge_id}: {e}", exc_info=True)
        return None
    except ConfigurationError as e:
        logger.error(f"Configuration error retrieving knowledge {knowledge_id}: {e}", exc_info=True)
        return None  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error retrieving knowledge {knowledge_id}: {e}", exc_info=True)
        return None  # Fallback


def query_knowledge(query: str, filters: Optional[Dict] = None, limit: int = 10):
    """
    Query knowledge with text search.
    
    Args:
        query: Search query string
        filters: Optional filter dictionary
        limit: Maximum number of results
    
    Returns:
        List of Knowledge objects
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Knowledge storage not available")
    
    try:
        # Convert filters to storage format
        storage_filters = {}
        if filters:
            if "agent_name" in filters:
                storage_filters["agent_name"] = filters["agent_name"]
            if "knowledge_type" in filters:
                storage_filters["knowledge_type"] = filters["knowledge_type"]
            if "min_confidence" in filters:
                storage_filters["min_confidence"] = filters["min_confidence"]
            if "date_from" in filters:
                storage_filters["date_from"] = filters["date_from"]
            if "date_to" in filters:
                storage_filters["date_to"] = filters["date_to"]
        
        # Use query method if no text search, otherwise use search
        if query:
            # Use search_knowledge for text search
            from agents.collaboration_models import KnowledgeType
            knowledge_types = filters.get("knowledge_type") if filters else None
            if knowledge_types and not isinstance(knowledge_types, list):
                knowledge_types = [knowledge_types]
            
            results = storage.search_knowledge(
                query=query,
                knowledge_types=knowledge_types,
                limit=limit
            )
        else:
            # Use query method for filtered queries
            results = storage.query(filters=storage_filters, limit=limit)
        
        # Convert dicts to Knowledge objects
        return [_dict_to_knowledge(r) for r in results]
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error querying knowledge: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error querying knowledge: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error querying knowledge: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error querying knowledge: {e}", exc_info=True)
        return []  # Fallback


def search_knowledge(query: str, **kwargs):
    """
    Search knowledge base with advanced filters.
    
    Args:
        query: Search query string
        **kwargs: Additional filters:
            - knowledge_types: List of KnowledgeType
            - agents: List of agent names
            - date_range: Tuple of (start_date, end_date)
            - limit: Maximum number of results (default: 10)
    
    Returns:
        List of Knowledge objects
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Knowledge storage not available")
    
    try:
        knowledge_types = kwargs.get("knowledge_types")
        agents = kwargs.get("agents")
        date_range = kwargs.get("date_range")
        limit = kwargs.get("limit", 10)
        
        results = storage.search_knowledge(
            query=query,
            knowledge_types=knowledge_types,
            agents=agents,
            date_range=date_range,
            limit=limit
        )
        
        # Convert dicts to Knowledge objects
        return [_dict_to_knowledge(r) for r in results]
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error searching knowledge: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error searching knowledge: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error searching knowledge: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error searching knowledge: {e}", exc_info=True)
        return []  # Fallback


def get_knowledge_statistics() -> Dict[str, Any]:
    """
    Get statistics about knowledge base.
    
    Returns:
        Dictionary with statistics
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Knowledge storage not available")
    
    try:
        return storage.get_knowledge_statistics()
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting knowledge statistics: {e}", exc_info=True)
        return {}  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting knowledge statistics: {e}", exc_info=True)
        return {}
    except ConfigurationError as e:
        logger.error(f"Configuration error getting knowledge statistics: {e}", exc_info=True)
        return {}  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting knowledge statistics: {e}", exc_info=True)
        return {}  # Fallback


def cleanup_expired_knowledge() -> int:
    """
    Remove expired knowledge items.
    
    Returns:
        Number of items deleted
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Knowledge storage not available")
    
    try:
        return storage.cleanup_expired_knowledge()
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error cleaning up expired knowledge: {e}", exc_info=True)
        return 0  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error cleaning up expired knowledge: {e}", exc_info=True)
        return 0
    except ConfigurationError as e:
        logger.error(f"Configuration error cleaning up expired knowledge: {e}", exc_info=True)
        return 0  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error cleaning up expired knowledge: {e}", exc_info=True)
        return 0  # Fallback


def store_architecture_model(project_path: str, model: Any, file_hash: str, 
                             version: int = 1, agent_name: str = "ArchitectureModelBuilder") -> str:
    """
    Store architecture model in knowledge base.
    
    Args:
        project_path: Path to the project
        model: Architecture model (dict or object)
        file_hash: Hash of the file
        version: Optional version number
        agent_name: Name of agent creating the model
    
    Returns:
        Knowledge ID
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Knowledge storage not available")
    
    try:
        return storage.store_architecture_model(project_path, model, file_hash, version, agent_name)
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error storing architecture model: {e}", exc_info=True)
        raise UserFacingError("Unable to connect to analysis database. Please try again later.") from e
    except StorageError as e:
        logger.error(f"Storage error storing architecture model: {e}", exc_info=True)
        raise
    except ConfigurationError as e:
        logger.error(f"Configuration error storing architecture model: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error storing architecture model: {e}", exc_info=True)
        raise


def get_architecture_model(project_path: str, file_hash: str, version: Optional[int] = None):
    """
    Get architecture model by project_path and file_hash.
    
    Args:
        project_path: Path to the project
        file_hash: Hash of the file
        version: Optional version number
    
    Returns:
        Knowledge object (with content containing the model dict) or None
    """
    storage = _get_storage()
    if not storage:
        raise RuntimeError("Knowledge storage not available")
    
    try:
        model_dict = storage.get_architecture_model(project_path, file_hash, version)
        if model_dict:
            return _dict_to_knowledge(model_dict)
        return None
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting architecture model: {e}", exc_info=True)
        return None  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting architecture model: {e}", exc_info=True)
        return None
    except ConfigurationError as e:
        logger.error(f"Configuration error getting architecture model: {e}", exc_info=True)
        return None  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting architecture model: {e}", exc_info=True)
        return None  # Fallback


def get_architecture_model_dict(project_path: str, file_hash: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Get architecture model as dict by project_path and file_hash.
    
    Convenience function that extracts the content dict from Knowledge object.
    
    Args:
        project_path: Path to the project
        file_hash: Hash of the file
        version: Optional version number
    
    Returns:
        Architecture model dict or None
    """
    knowledge_obj = get_architecture_model(project_path, file_hash, version)
    if knowledge_obj:
        return knowledge_obj.content if hasattr(knowledge_obj, 'content') else knowledge_obj
    return None


def list_architecture_models(project_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    List all architecture models for a project path.
    
    Args:
        project_path: Path to the project
        limit: Optional limit on number of results
        
    Returns:
        List of model dictionaries with metadata, sorted by created_at DESC (newest first)
    """
    try:
        # Use architecture model storage to query by project_path
        from services.storage.architecture_model_storage import ArchitectureModelStorage
        
        arch_storage = ArchitectureModelStorage()
        
        # Query all models for this project path
        filters = {'project_path': project_path}
        results = arch_storage.query(filters=filters, limit=limit, offset=None)
        
        # Sort by created_at DESC (newest first)
        if results:
            # Extract created_at from timestamp field
            results.sort(key=lambda x: x.get('created_at') or x.get('timestamp') or datetime.min, reverse=True)
        
        return results
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error listing architecture models: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error listing architecture models: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error listing architecture models: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error listing architecture models: {e}", exc_info=True)
        return []  # Fallback


def _dict_to_knowledge(data: Dict) -> Any:
    """Convert dict to Knowledge object."""
    from agents.collaboration_models import Knowledge, KnowledgeType
    
    # Handle knowledge_type
    knowledge_type = data.get('knowledge_type')
    if isinstance(knowledge_type, str):
        try:
            knowledge_type = KnowledgeType(knowledge_type)
        except ValueError:
            # Keep as string if not a valid enum
            pass
    elif hasattr(knowledge_type, 'value'):
        knowledge_type = KnowledgeType(knowledge_type.value)
    
    return Knowledge(
        knowledge_id=data.get('knowledge_id'),
        agent_name=data.get('agent_name', ''),
        knowledge_type=knowledge_type or KnowledgeType.FINDING,
        content=data.get('content', {}),
        relevance_tags=data.get('relevance_tags', []),
        confidence=data.get('confidence', 0.5),
        timestamp=data.get('created_at') or data.get('timestamp') or datetime.now(),
        expires_at=data.get('expires_at')
    )

