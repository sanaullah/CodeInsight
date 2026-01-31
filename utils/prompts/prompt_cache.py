"""
Prompt caching utility for generated prompts.

Caches generated prompts using Redis (if available) with in-memory fallback
to improve performance and reduce LLM API calls.
"""

import logging
from typing import Optional, Dict
from datetime import timedelta

logger = logging.getLogger(__name__)

# In-memory cache (fallback if Redis unavailable)
_in_memory_cache: Dict[str, str] = {}

# Cache TTL (1 hour)
CACHE_TTL_SECONDS = 3600


def get_cached_prompt(role_name: str, architecture_hash: str, file_hash: str) -> Optional[str]:
    """
    Get cached prompt for role + architecture hash + file hash.
    
    Checks Redis cache first, then falls back to in-memory cache.
    Also checks old cache key format (without file_hash) for backward compatibility.
    
    Args:
        role_name: Name of the analysis role
        architecture_hash: SHA256 hash of architecture model
        file_hash: SHA256 hash of file structure
        
    Returns:
        Cached prompt text or None if not found
    """
    # New cache key format with file_hash
    cache_key = f"prompt:{role_name}:{file_hash}:{architecture_hash}"
    
    # Try Redis first
    try:
        from services.redis_client import RedisClient
        redis_client = RedisClient()
        if redis_client and redis_client.is_available():
            cached = redis_client.get(cache_key)
            if cached:
                if isinstance(cached, bytes):
                    cached = cached.decode('utf-8')
                logger.debug(f"Retrieved prompt from Redis cache for {role_name}")
                return cached
    except Exception as e:
        logger.debug(f"Redis cache unavailable: {e}, using in-memory cache")
    
    # Fallback to in-memory cache
    if cache_key in _in_memory_cache:
        logger.debug(f"Retrieved prompt from in-memory cache for {role_name}")
        return _in_memory_cache[cache_key]
    
    # Backward compatibility: check old cache key format (without file_hash)
    old_cache_key = f"prompt:{role_name}:{architecture_hash}"
    try:
        from services.redis_client import RedisClient
        redis_client = RedisClient()
        if redis_client and redis_client.is_available():
            cached = redis_client.get(old_cache_key)
            if cached:
                if isinstance(cached, bytes):
                    cached = cached.decode('utf-8')
                logger.debug(f"Retrieved prompt from Redis cache (old format) for {role_name}")
                # Migrate to new format
                cache_prompt(role_name, architecture_hash, file_hash, cached)
                return cached
    except Exception as e:
        logger.debug(f"Redis cache unavailable: {e}")
    
    if old_cache_key in _in_memory_cache:
        logger.debug(f"Retrieved prompt from in-memory cache (old format) for {role_name}")
        cached = _in_memory_cache[old_cache_key]
        # Migrate to new format
        cache_prompt(role_name, architecture_hash, file_hash, cached)
        return cached
    
    return None


def cache_prompt(role_name: str, architecture_hash: str, file_hash: str, prompt: str) -> None:
    """
    Cache generated prompt.
    
    Stores in Redis (if available) and in-memory cache.
    
    Args:
        role_name: Name of the analysis role
        architecture_hash: SHA256 hash of architecture model
        file_hash: SHA256 hash of file structure
        prompt: Generated prompt text to cache
    """
    if not prompt:
        return
    
    cache_key = f"prompt:{role_name}:{file_hash}:{architecture_hash}"
    
    # Try Redis first
    try:
        from services.redis_client import RedisClient
        redis_client = RedisClient()
        if redis_client and redis_client.is_available():
            redis_client.setex(cache_key, CACHE_TTL_SECONDS, prompt)
            logger.debug(f"Cached prompt in Redis for {role_name}")
    except Exception as e:
        logger.debug(f"Redis cache unavailable: {e}, using in-memory cache only")
    
    # Always store in in-memory cache (fallback)
    _in_memory_cache[cache_key] = prompt
    logger.debug(f"Cached prompt in memory for {role_name}")


def clear_cache() -> None:
    """Clear all cached prompts (both Redis and in-memory)."""
    global _in_memory_cache
    
    # Clear Redis cache
    try:
        from services.redis_client import RedisClient
        redis_client = RedisClient()
        if redis_client and redis_client.is_available():
            # Clear all prompt keys (pattern matching)
            keys = redis_client.keys("prompt:*")
            if keys:
                redis_client.delete(*keys)
                logger.info(f"Cleared {len(keys)} prompts from Redis cache")
    except Exception as e:
        logger.debug(f"Could not clear Redis cache: {e}")
    
    # Clear in-memory cache
    count = len(_in_memory_cache)
    _in_memory_cache.clear()
    logger.info(f"Cleared {count} prompts from in-memory cache")

