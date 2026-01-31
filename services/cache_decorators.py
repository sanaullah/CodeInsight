"""
Advanced cache decorators for functions and methods.

Provides decorators for caching methods, properties, and handling
cache invalidation and stale-while-revalidate patterns.
"""

import logging
import time
import hashlib
from typing import Any, Optional, Callable, Dict, List
from functools import wraps
from services.redis_cache import get_cache

logger = logging.getLogger(__name__)


def cached_method(ttl: int = 3600, key_func: Optional[Callable] = None, 
                  invalidate_on: Optional[List[str]] = None):
    """
    Decorator to cache method results.
    
    Args:
        ttl: TTL in seconds
        key_func: Optional function to generate cache key from (self, args, kwargs)
        invalidate_on: Optional list of method names whose calls should invalidate this cache
    
    Usage:
        class MyClass:
            @cached_method(ttl=1800)
            def expensive_method(self, arg1, arg2):
                return compute_result()
            
            @cached_method(key_func=lambda self, args, kwargs: f"key_{args[0]}")
            def another_method(self, arg1):
                return compute_result()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(self, args, kwargs)
            else:
                # Default: use class name, method name, and args hash
                class_name = self.__class__.__name__
                key_data = f"{class_name}:{func.__name__}:{args}:{kwargs}"
                key_hash = hashlib.md5(key_data.encode()).hexdigest()
                cache_key = f"{class_name}:{func.__name__}:{key_hash}"
            
            # Try to get from cache
            try:
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Cache get failed for {func.__name__}: {e}")
            
            # Compute result
            try:
                result = func(self, *args, **kwargs)
            except Exception as e:
                logger.error(f"Method {func.__name__} failed: {e}")
                raise
            
            # Cache result
            try:
                cache.set(cache_key, result, ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache set failed for {func.__name__}: {e}")
            
            return result
        
        # Store metadata
        wrapper._cache_ttl = ttl
        wrapper._cache_key_func = key_func
        wrapper._cache_invalidate_on = invalidate_on or []
        
        return wrapper
    return decorator


def cached_property(ttl: int = 3600):
    """
    Decorator to cache property values.
    
    Args:
        ttl: TTL in seconds
    
    Usage:
        class MyClass:
            @cached_property(ttl=3600)
            def expensive_property(self):
                return compute_value()
    """
    def decorator(func: Callable) -> Callable:
        cache_key_attr = f"_cache_key_{func.__name__}"
        
        @property
        @wraps(func)
        def wrapper(self):
            cache = get_cache()
            
            # Generate cache key
            if not hasattr(self, cache_key_attr):
                class_name = self.__class__.__name__
                instance_id = id(self)
                setattr(self, cache_key_attr, f"{class_name}:{func.__name__}:{instance_id}")
            
            cache_key = getattr(self, cache_key_attr)
            
            # Try to get from cache
            try:
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
            except Exception as e:
                logger.warning(f"Cache get failed for property {func.__name__}: {e}")
            
            # Compute result
            try:
                result = func(self)
            except Exception as e:
                logger.error(f"Property {func.__name__} failed: {e}")
                raise
            
            # Cache result
            try:
                cache.set(cache_key, result, ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache set failed for property {func.__name__}: {e}")
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(pattern_func: Optional[Callable] = None):
    """
    Decorator to invalidate cache on method call.
    
    Args:
        pattern_func: Optional function to generate cache key pattern from (self, args, kwargs)
    
    Usage:
        class MyClass:
            @cache_invalidate(pattern_func=lambda self, args, kwargs: f"key_{args[0]}:*")
            def update_method(self, id, data):
                # This will invalidate cache keys matching the pattern
                return update_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            cache = get_cache()
            
            # Generate pattern
            if pattern_func:
                pattern = pattern_func(self, args, kwargs)
            else:
                # Default: invalidate all keys for this class and method
                class_name = self.__class__.__name__
                pattern = f"{class_name}:{func.__name__}:*"
            
            # Invalidate cache
            try:
                deleted = cache.delete_pattern(pattern)
                logger.debug(f"Invalidated {deleted} cache keys matching pattern: {pattern}")
            except Exception as e:
                logger.warning(f"Cache invalidation failed for {func.__name__}: {e}")
            
            # Call original method
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator


def stale_while_revalidate(ttl: int = 3600, stale_ttl: int = 7200):
    """
    Decorator implementing stale-while-revalidate pattern.
    
    Serves stale data immediately while refreshing in background.
    
    Args:
        ttl: TTL for fresh data
        stale_ttl: TTL for stale data (should be > ttl)
    
    Usage:
        @stale_while_revalidate(ttl=3600, stale_ttl=7200)
        def get_data():
            return fetch_from_database()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            key_data = f"{func.__name__}:{args}:{kwargs}"
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            cache_key = f"{func.__name__}:{key_hash}"
            
            # Try to get from cache
            try:
                cached = cache.get(cache_key)
                if cached is not None:
                    # Check if stale
                    remaining_ttl = cache.ttl(cache_key)
                    if remaining_ttl is not None and remaining_ttl < (stale_ttl - ttl):
                        # Stale but still valid, refresh in background
                        logger.debug(f"Cache entry for {func.__name__} is stale, refreshing in background")
                        # In a real implementation, you might want to use a background task
                        # For now, we'll just refresh synchronously but return stale data
                        try:
                            result = func(*args, **kwargs)
                            cache.set(cache_key, result, ttl=stale_ttl)
                        except Exception as e:
                            logger.warning(f"Background refresh failed for {func.__name__}: {e}")
                    return cached
            except Exception as e:
                logger.warning(f"Cache get failed for {func.__name__}: {e}")
            
            # Compute result
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Function {func.__name__} failed: {e}")
                raise
            
            # Cache result with stale TTL
            try:
                cache.set(cache_key, result, ttl=stale_ttl)
            except Exception as e:
                logger.warning(f"Cache set failed for {func.__name__}: {e}")
            
            return result
        
        return wrapper
    return decorator










