"""
Cache utility functions for key generation, parsing, and invalidation.

Provides utilities for working with cache keys following the naming convention:
{prefix}:{service}:{type}:{identifier}
"""

import logging
import hashlib
from typing import Optional, Dict, List, Tuple, Any
from services.redis_client import RedisClient

logger = logging.getLogger(__name__)


def validate_cache_key(key: str) -> str:
    """
    Validate and sanitize cache key.
    
    This is a shared utility function that can be imported by other modules
    to ensure consistent cache key validation across the codebase.
    
    Args:
        key: Cache key to validate
        
    Returns:
        Validated and sanitized cache key
        
    Raises:
        ValueError: If key is empty or exceeds maximum length
    """
    # Check empty
    if not key:
        raise ValueError("Cache key cannot be empty")
    
    # Check length (500 chars is practical limit for Redis keys)
    if len(key) > 500:
        raise ValueError(f"Cache key exceeds maximum length of 500 characters (got {len(key)})")
    
    # Remove control characters but preserve colons (:) for key structure
    # Allow printable characters plus common safe characters: -_/.
    sanitized = ''.join(c for c in key if c.isprintable() or c in '-_/.:')
    
    return sanitized


def make_cache_key(service: str, type: str, identifier: str, prefix: Optional[str] = None) -> str:
    """
    Generate cache key following naming convention.
    
    Args:
        service: Service name (e.g., 'scan_history', 'architecture_model')
        type: Object type (e.g., 'id', 'project', 'hash')
        identifier: Unique identifier (e.g., ID, hash, path)
        prefix: Optional prefix override (defaults to RedisClient.get_key_prefix())
        
    Returns:
        Full cache key with prefix
    """
    if prefix is None:
        prefix = RedisClient.get_key_prefix()
    
    # Remove prefix from identifier if it already has it
    if identifier.startswith(prefix):
        identifier = identifier[len(prefix):]
    
    # Build key: {prefix}{service}:{type}:{identifier}
    key = f"{service}:{type}:{identifier}"
    
    # Add prefix if not already present
    if not key.startswith(prefix):
        key = f"{prefix}{key}"
    
    # Validate the final constructed key
    return validate_cache_key(key)


def parse_cache_key(key: str, prefix: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Parse cache key into components.
    
    Args:
        key: Full cache key
        prefix: Optional prefix override (defaults to RedisClient.get_key_prefix())
        
    Returns:
        Dictionary with 'prefix', 'service', 'type', 'identifier' keys, or None if invalid
    """
    if prefix is None:
        prefix = RedisClient.get_key_prefix()
    
    # Remove prefix
    if key.startswith(prefix):
        key_without_prefix = key[len(prefix):]
    else:
        key_without_prefix = key
        prefix = ""
    
    # Parse: {service}:{type}:{identifier}
    parts = key_without_prefix.split(':', 2)
    if len(parts) < 3:
        logger.warning(f"Invalid cache key format: {key}")
        return None
    
    return {
        'prefix': prefix,
        'service': parts[0],
        'type': parts[1],
        'identifier': parts[2]
    }


def get_cache_key_for_record(service: str, record_id: str, prefix: Optional[str] = None) -> str:
    """
    Get cache key for a record.
    
    Args:
        service: Service name
        record_id: Record identifier
        prefix: Optional prefix override
        
    Returns:
        Cache key for the record
    """
    return make_cache_key(service, 'id', record_id, prefix)


def get_cache_key_for_query(service: str, query_hash: Optional[str] = None, 
                           filters: Optional[Dict[str, Any]] = None, 
                           limit: Optional[int] = None, 
                           offset: Optional[int] = None,
                           prefix: Optional[str] = None) -> str:
    """
    Get cache key for query results.
    
    Args:
        service: Service name
        query_hash: Optional pre-computed query hash
        filters: Optional filter dictionary
        limit: Optional limit
        offset: Optional offset
        prefix: Optional prefix override
        
    Returns:
        Cache key for the query
    """
    if query_hash:
        return make_cache_key(service, 'query', query_hash, prefix)
    
    # Generate hash from query parameters
    query_data = {
        'filters': filters or {},
        'limit': limit,
        'offset': offset
    }
    query_str = str(sorted(query_data.items()))
    query_hash = hashlib.md5(query_str.encode('utf-8')).hexdigest()
    
    return make_cache_key(service, 'query', query_hash, prefix)


def invalidate_related_keys(cache: Any, service: str, identifier: str, 
                           patterns: Optional[List[str]] = None) -> int:
    """
    Invalidate related cache entries.
    
    Args:
        cache: Cache instance (RedisCache or RedisClient)
        service: Service name
        identifier: Identifier to invalidate
        patterns: Optional list of additional patterns to invalidate
        
    Returns:
        Number of keys invalidated
    """
    deleted = 0
    
    # Invalidate record key
    record_key = get_cache_key_for_record(service, identifier)
    if hasattr(cache, 'delete'):
        if cache.delete(record_key):
            deleted += 1
    elif hasattr(cache, 'cache_delete'):
        if cache.cache_delete(record_key):
            deleted += 1
    
    # Invalidate query keys for this service
    query_pattern = f"{service}:query:*"
    if hasattr(cache, 'delete_pattern'):
        deleted += cache.delete_pattern(query_pattern)
    elif hasattr(cache, 'cache_delete_by_pattern'):
        deleted += cache.cache_delete_by_pattern(query_pattern)
    
    # Invalidate additional patterns
    if patterns:
        for pattern in patterns:
            full_pattern = f"{service}:{pattern}"
            if hasattr(cache, 'delete_pattern'):
                deleted += cache.delete_pattern(full_pattern)
            elif hasattr(cache, 'cache_delete_by_pattern'):
                deleted += cache.cache_delete_by_pattern(full_pattern)
    
    return deleted










