"""
Enhanced Redis Caching Layer.

Provides cache decorator, TTL management, cache invalidation patterns,
cache statistics, and cache stampede prevention.
"""

import logging
import time
import json
import hashlib
import threading
from typing import Any, Optional, Callable, Dict, List
from functools import wraps
from datetime import datetime, date
from enum import Enum
from services.db_config import get_db_config, RedisConfig
from services.redis_client import RedisClient

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Enhanced Redis caching layer with advanced features.
    
    Features:
    - TTL management
    - Cache invalidation patterns
    - Cache statistics
    - Cache stampede prevention
    - Automatic serialization/deserialization
    """
    
    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis cache.
        
        Args:
            config: Optional Redis configuration (defaults to from_env)
        """
        self.config = config or get_db_config().redis
        self.key_prefix = self.config.key_prefix
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0,
        }
    
    def _validate_cache_key(self, key: str) -> str:
        """
        Validate and sanitize cache key.
        
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
    
    def _get_client(self):
        """Get Redis client."""
        return RedisClient.get_client()
    
    def _make_key(self, key: str) -> str:
        """
        Make full cache key with prefix.
        
        Args:
            key: Cache key
            
        Returns:
            Full cache key with prefix
            
        Raises:
            ValueError: If key is invalid after validation
        """
        if key.startswith(self.key_prefix):
            full_key = key
        else:
            full_key = f"{self.key_prefix}{key}"
        
        # Validate the final constructed key
        return self._validate_cache_key(full_key)
    
    def _serialize_value(self, value: Any) -> Any:
        """
        Recursively serialize a value, handling datetime, date, Enum, and nested structures.
        
        Args:
            value: Value to serialize
            
        Returns:
            JSON-serializable value
        """
        if value is None:
            return None
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, date):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            # For other types, try to serialize as-is
            # If it fails, json.dumps will raise TypeError which we'll catch
            return value
    
    def _deserialize_value(self, value: Any) -> Any:
        """
        Recursively deserialize a value, converting ISO datetime strings back to datetime objects.
        
        Args:
            value: Value to deserialize
            
        Returns:
            Deserialized value with datetime objects restored
        """
        if value is None:
            return None
        elif isinstance(value, str):
            # Try to parse as ISO datetime string
            # Check if it looks like an ISO datetime string (contains 'T' or is in ISO date format)
            if 'T' in value or (len(value) >= 10 and value[4] == '-' and value[7] == '-'):
                try:
                    # Try datetime first (handles both naive and timezone-aware)
                    if 'T' in value:
                        return datetime.fromisoformat(value.replace('Z', '+00:00'))
                    else:
                        # Just a date string, convert to datetime
                        return datetime.fromisoformat(value)
                except (ValueError, AttributeError):
                    # Not a valid datetime string, return as-is
                    pass
            return value
        elif isinstance(value, (list, tuple)):
            return [self._deserialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._deserialize_value(v) for k, v in value.items()}
        else:
            return value
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.
        
        Handles datetime, date, Enum, and nested structures by converting them
        to JSON-serializable formats before encoding.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized bytes
        """
        try:
            # Pre-process value to handle datetime and other non-serializable types
            serialized_value = self._serialize_value(value)
            return json.dumps(serialized_value).encode('utf-8')
        except (TypeError, ValueError) as e:
            logger.error(f"Serialization failed: {e}, value type: {type(value)}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from storage.
        
        Converts ISO datetime strings back to datetime objects and handles
        nested structures recursively.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized value with datetime objects restored
        """
        try:
            decoded = json.loads(data.decode('utf-8'))
            return self._deserialize_value(decoded)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Deserialization failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        client = self._get_client()
        if client is None:
            logger.debug("Redis not available, returning default")
            return default
        
        full_key = self._make_key(key)
        
        # Track cache operation
        try:
            from utils.langfuse.db_tracking import track_cache_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_cache_operation(
                operation="get",
                cache_key=key,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ) as obs:
                try:
                    data = client.get(full_key)
                    hit = data is not None
                    result_size = len(data) if data else None
                    
                    if data is None:
                        with self._lock:
                            self._stats["misses"] += 1
                        if obs:
                            try:
                                obs.update(output={"hit": False}, metadata={"hit": False})
                            except Exception:
                                pass
                        return default
                    
                    value = self._deserialize(data)
                    with self._lock:
                        self._stats["hits"] += 1
                    
                    if obs:
                        try:
                            obs.update(
                                output={"hit": True, "result_size": result_size},
                                metadata={"hit": True, "result_size": result_size}
                            )
                        except Exception:
                            pass
                    
                    return value
                except Exception as e:
                    logger.warning(f"Cache get failed for {key}: {e}")
                    with self._lock:
                        self._stats["errors"] += 1
                    return default
        except ImportError:
            # Fallback if tracking not available
            try:
                data = client.get(full_key)
                if data is None:
                    with self._lock:
                        self._stats["misses"] += 1
                    return default
                
                value = self._deserialize(data)
                with self._lock:
                    self._stats["hits"] += 1
                return value
            except Exception as e:
                logger.warning(f"Cache get failed for {key}: {e}")
                with self._lock:
                    self._stats["errors"] += 1
                return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds
            
        Returns:
            True if successful, False otherwise
        """
        client = self._get_client()
        if client is None:
            logger.debug("Redis not available, skipping cache set")
            return False
        
        full_key = self._make_key(key)
        
        # Track cache operation
        try:
            from utils.langfuse.db_tracking import track_cache_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            result_size = len(str(value)) if value else 0
            
            with track_cache_operation(
                operation="set",
                cache_key=key,
                ttl=ttl,
                result_size=result_size,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ) as obs:
                try:
                    data = self._serialize(value)
                    if ttl:
                        result = client.setex(full_key, ttl, data)
                    else:
                        result = client.set(full_key, data)
                    
                    if result:
                        with self._lock:
                            self._stats["sets"] += 1
                    
                    if obs:
                        try:
                            obs.update(
                                output={"success": bool(result), "result_size": result_size},
                                metadata={"success": bool(result), "result_size": result_size}
                            )
                        except Exception:
                            pass
                    
                    return bool(result)
                except Exception as e:
                    logger.warning(f"Cache set failed for {key}: {e}")
                    with self._lock:
                        self._stats["errors"] += 1
                    return False
        except ImportError:
            # Fallback if tracking not available
            try:
                data = self._serialize(value)
                if ttl:
                    result = client.setex(full_key, ttl, data)
                else:
                    result = client.set(full_key, data)
                
                if result:
                    with self._lock:
                        self._stats["sets"] += 1
                return bool(result)
            except Exception as e:
                logger.warning(f"Cache set failed for {key}: {e}")
                with self._lock:
                    self._stats["errors"] += 1
                return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False otherwise
        """
        client = self._get_client()
        if client is None:
            logger.debug("Redis not available, skipping cache delete")
            return False
        
        full_key = self._make_key(key)
        
        # Track cache operation
        try:
            from utils.langfuse.db_tracking import track_cache_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_cache_operation(
                operation="delete",
                cache_key=key,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ) as obs:
                try:
                    result = client.delete(full_key)
                    if result:
                        with self._lock:
                            self._stats["deletes"] += 1
                    
                    if obs:
                        try:
                            obs.update(
                                output={"success": bool(result)},
                                metadata={"success": bool(result)}
                            )
                        except Exception:
                            pass
                    
                    return bool(result)
                except Exception as e:
                    logger.warning(f"Cache delete failed for {key}: {e}")
                    with self._lock:
                        self._stats["errors"] += 1
                    return False
        except ImportError:
            # Fallback if tracking not available
            try:
                result = client.delete(full_key)
                if result:
                    with self._lock:
                        self._stats["deletes"] += 1
                return bool(result)
            except Exception as e:
                logger.warning(f"Cache delete failed for {key}: {e}")
                with self._lock:
                    self._stats["errors"] += 1
                return False
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Key pattern (supports * wildcard)
            
        Returns:
            Number of keys deleted
        """
        client = self._get_client()
        if client is None:
            logger.debug("Redis not available, skipping cache delete pattern")
            return 0
        
        full_pattern = self._make_key(pattern)
        
        try:
            # Use SCAN to find matching keys
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=full_pattern, count=100)
                if keys:
                    deleted += client.delete(*keys)
                if cursor == 0:
                    break
            
            if deleted > 0:
                with self._lock:
                    self._stats["deletes"] += deleted
            return deleted
        except Exception as e:
            logger.warning(f"Cache delete pattern failed for {pattern}: {e}")
            with self._lock:
                self._stats["errors"] += 1
            return 0
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        client = self._get_client()
        if client is None:
            return False
        
        full_key = self._make_key(key)
        
        try:
            return bool(client.exists(full_key))
        except Exception as e:
            logger.warning(f"Cache exists check failed for {key}: {e}")
            return False
    
    def ttl(self, key: str) -> Optional[int]:
        """
        Get TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, None if key doesn't exist
        """
        client = self._get_client()
        if client is None:
            return None
        
        full_key = self._make_key(key)
        
        try:
            ttl = client.ttl(full_key)
            return ttl if ttl >= 0 else None
        except Exception as e:
            logger.warning(f"Cache TTL check failed for {key}: {e}")
            return None
    
    def get_with_lock(self, key: str, compute_func: Callable[[], Any], ttl: int = 3600, lock_ttl: int = 10) -> Any:
        """
        Get value with cache stampede prevention using lock.
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: TTL for cached value
            lock_ttl: TTL for lock (should be short)
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached = self.get(key)
        if cached is not None:
            return cached
        
        # Try to acquire lock
        lock_key = f"{key}:lock"
        lock_acquired = self.set(lock_key, "locked", ttl=lock_ttl)
        
        if lock_acquired:
            try:
                # Compute value
                value = compute_func()
                self.set(key, value, ttl=ttl)
                return value
            finally:
                # Release lock
                self.delete(lock_key)
        else:
            # Wait for lock holder to finish
            time.sleep(0.1)
            # Retry getting from cache
            return self.get(key, default=None) or self.get_with_lock(key, compute_func, ttl, lock_ttl)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0.0
            miss_rate = self._stats["misses"] / total if total > 0 else 0.0
            
            return {
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "errors": self._stats["errors"],
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "total_operations": total,
            }
    
    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._lock:
            self._stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "deletes": 0,
                "errors": 0,
            }


# Global cache instance
_cache: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """
    Get or create global cache instance.
    
    Returns:
        RedisCache instance
    """
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache


def cache_result(key_func: Optional[Callable] = None, ttl: int = 3600, key_prefix: str = "",
                 condition: Optional[Callable] = None, invalidate_on: Optional[List[str]] = None):
    """
    Decorator to cache function results with advanced features.
    
    Args:
        key_func: Optional function to generate cache key from args/kwargs
        ttl: TTL in seconds
        key_prefix: Optional prefix for cache key
        condition: Optional function to determine if caching should be used (args, kwargs) -> bool
        invalidate_on: Optional list of function names whose calls should invalidate this cache
        
    Usage:
        @cache_result(ttl=3600)
        def expensive_function(arg1, arg2):
            return compute_result()
        
        @cache_result(key_func=lambda args, kwargs: f"key_{args[0]}", ttl=1800)
        def another_function(arg1):
            return compute_result()
        
        @cache_result(condition=lambda args, kwargs: kwargs.get('use_cache', True))
        def conditional_function(arg1, use_cache=True):
            return compute_result()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Check condition
            if condition and not condition(args, kwargs):
                # Skip caching, call function directly
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(args, kwargs)
            else:
                # Default: use function name and args hash
                key_data = f"{func.__name__}:{args}:{kwargs}"
                key_hash = hashlib.md5(key_data.encode()).hexdigest()
                cache_key = f"{key_prefix}{func.__name__}:{key_hash}"
            
            # Validate the cache key
            try:
                cache_key = cache._validate_cache_key(cache_key)
            except ValueError as e:
                logger.warning(f"Invalid cache key for {func.__name__}: {e}, using sanitized key")
                # Re-raise to fail fast - caller should handle invalid keys
                raise
            
            # Try to get from cache
            try:
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Cache get failed for {func.__name__}: {e}, computing result")
            
            # Compute result
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")
                raise
            
            # Cache result
            try:
                cache.set(cache_key, result, ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache set failed for {func.__name__}: {e}")
            
            return result
        
        # Store metadata for invalidation
        wrapper._cache_key_prefix = key_prefix
        wrapper._cache_ttl = ttl
        wrapper._cache_key_func = key_func
        wrapper._cache_invalidate_on = invalidate_on or []
        
        return wrapper
    return decorator


# TTL configuration by cache type
CACHE_TTL = {
    "architecture_model": 86400,  # 24 hours
    "prompt": 43200,  # 12 hours
    "knowledge_base": 21600,  # 6 hours
    "experience": 21600,  # 6 hours
    "scan_history": 3600,  # 1 hour
    "query_result": 900,  # 15 minutes
    "metadata": 1800,  # 30 minutes
}

