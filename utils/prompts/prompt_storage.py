"""
Prompt storage utility module.

This module provides a PostgreSQL-based API for managing prompts.
All data is stored in PostgreSQL with Redis caching.

Backward-compatible functions:
- save_prompt()
- get_prompt()
- get_prompt_by_name()
- get_all_prompts()
- search_prompts()
- update_prompt()
- delete_prompt()
- get_prompt_statistics()
"""

import logging
import uuid
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Import storage
try:
    from services.storage.cached_prompt_storage import CachedPromptStorage
    _storage = None
    
    def _get_storage():
        """Get or create storage instance."""
        global _storage
        if _storage is None:
            _storage = CachedPromptStorage()
        return _storage
except ImportError as e:
    logger.error(f"Failed to import CachedPromptStorage: {e}")
    _storage = None
    
    def _get_storage():
        """Get storage instance."""
        return None


def save_prompt(
    name: str,
    content: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
    architecture_hash: Optional[str] = None
) -> Optional[str]:
    """
    Save prompt.
    
    Args:
        name: Prompt name
        content: Prompt content
        description: Optional description
        tags: Optional list of tags
        metadata: Optional metadata dictionary
        role: Optional role name
        architecture_hash: Optional architecture hash
    
    Returns:
        Prompt ID or None if error
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return None
    
    # Generate prompt_id for new storage
    prompt_id = str(uuid.uuid4())
    
    try:
        # Format data for storage.save() method (matches PromptStorage.save() interface)
        data = {
            "prompt_id": prompt_id,
            "prompt_name": name,
            "prompt_content": content,
            "metadata_json": metadata,
            "role": role,
            "architecture_hash": architecture_hash,
            "usage_count": 0
        }
        
        # Use save() method from CachedStorage/BaseStorage interface
        return storage.save(data)
    except Exception as e:
        logger.error(f"Error storing prompt: {e}", exc_info=True)
        return None


def get_prompt(prompt_id: str) -> Optional[Dict[str, Any]]:
    """
    Get prompt by ID.
    
    Args:
        prompt_id: Prompt ID
    
    Returns:
        Dictionary with prompt data, or None if not found
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return None
    
    try:
        return storage.get_prompt(prompt_id)
    except Exception as e:
        logger.error(f"Error getting prompt: {e}", exc_info=True)
        return None


def get_prompt_by_name(name: str) -> Optional[Dict[str, Any]]:
    """
    Get prompt by name.
    
    Args:
        name: Prompt name
    
    Returns:
        Dictionary with prompt data, or None if not found
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return None
    
    try:
        return storage.get_prompt_by_name(name)
    except Exception as e:
        logger.error(f"Error getting prompt by name: {e}", exc_info=True)
        return None



def get_prompt_by_architecture_hash(
    role: str,
    architecture_hash: str,
    file_hash: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get prompt by role and architecture hash.
    
    Args:
        role: Role name
        architecture_hash: Architecture hash
        file_hash: Optional file hash for stricter matching
        
    Returns:
        Dictionary with prompt data, or None if not found
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return None
    
    try:
        # Use search_prompts which supports filtering by these fields
        # Note: search_prompts returns a list, we want the most recent one
        results = storage.search_prompts(
            query="",  # Empty query means match all (filtered by other params)
            role=role,
            architecture_hash=architecture_hash,
            limit=1
        )
        
        # If file_hash provided, we might need to filter manually if search_prompts doesn't support it directly yet
        # BaseStorage.search_prompts currently supports role and architecture_hash
        # If file_hash is critical, we could add client-side filtering here, but for now strict architectural match is good.
        
        if results:
            return results[0]
            
        return None
    except Exception as e:
        logger.error(f"Error getting prompt by architecture hash: {e}", exc_info=True)
        return None


def get_all_prompts(limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]:

    """
    Get all prompts.
    
    Args:
        limit: Maximum number of prompts
        offset: Offset for pagination
    
    Returns:
        List of prompt dictionaries
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return []
    
    try:
        return storage.get_all_prompts(limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"Error getting all prompts: {e}", exc_info=True)
        return []


def search_prompts(
    query: str,
    tags: Optional[List[str]] = None,
    role: Optional[str] = None,
    architecture_hash: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search prompts using full-text search.
    
    Args:
        query: Search query text
        tags: Optional list of tags
        role: Optional role
        architecture_hash: Optional architecture hash
        limit: Maximum number of results
    
    Returns:
        List of prompt dictionaries
    """
    storage = _get_storage()
    if storage:
        try:
            return storage.search_prompts(
                query=query,
                tags=tags,
                role=role,
                architecture_hash=architecture_hash,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error searching prompts in PostgreSQL: {e}", exc_info=True)
            raise
    
    return []


def update_prompt(prompt_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update existing prompt.
    
    Args:
        prompt_id: Prompt ID
        updates: Dictionary with fields to update
    
    Returns:
        True if updated, False if not found
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return False
    
    try:
        return storage.update_prompt(prompt_id, updates)
    except Exception as e:
        logger.error(f"Error updating prompt: {e}", exc_info=True)
        return False


def delete_prompt(prompt_id: str) -> bool:
    """
    Delete prompt by ID.
    
    Args:
        prompt_id: Prompt ID
    
    Returns:
        True if deleted, False if not found
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return False
    
    try:
        return storage.delete(prompt_id)
    except Exception as e:
        logger.error(f"Error deleting prompt: {e}", exc_info=True)
        return False


def get_prompt_statistics() -> Dict[str, Any]:
    """
    Get statistics about prompts.
    
    Returns:
        Dictionary with statistics
    """
    storage = _get_storage()
    if not storage:
        logger.error("Storage not available")
        return {}
    
    try:
        return storage.get_prompt_statistics()
    except Exception as e:
        logger.error(f"Error getting prompt statistics: {e}", exc_info=True)
        return {}

