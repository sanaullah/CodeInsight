"""
Cached Storage Wrapper.

Provides automatic caching for BaseStorage implementations with
cache invalidation on writes and configurable TTL per operation.
"""

import logging
import hashlib
from typing import Any, Dict, List, Optional
from contextlib import contextmanager
from services.storage.base_storage import BaseStorage, StorageError
from services.redis_cache import get_cache
from services.cache_utils import (
    get_cache_key_for_record,
    get_cache_key_for_query,
    invalidate_related_keys
)
from services.cache_config import get_ttl_for_cache_type

logger = logging.getLogger(__name__)


class CachedStorage:
    """
    Generic cache wrapper for storage services.
    
    Wraps any BaseStorage implementation with automatic caching:
    - Caches get() operations
    - Caches query() results
    - Invalidates cache on save(), delete(), bulk operations
    - Supports pattern-based invalidation
    - Configurable TTL per operation type
    """
    
    def __init__(self, storage: BaseStorage, service_name: str, 
                 get_ttl: Optional[int] = None,
                 query_ttl: Optional[int] = None,
                 enable_query_cache: bool = True):
        """
        Initialize cached storage wrapper.
        
        Args:
            storage: BaseStorage implementation to wrap
            service_name: Service name for cache key generation
            get_ttl: TTL for get() operations (defaults to cache type TTL)
            query_ttl: TTL for query() results (defaults to query_result TTL)
            enable_query_cache: Whether to cache query results
        """
        self.storage = storage
        self.service_name = service_name
        self.cache = get_cache()
        
        # Get TTLs
        if get_ttl is None:
            get_ttl = get_ttl_for_cache_type(service_name)
        if query_ttl is None:
            query_ttl = get_ttl_for_cache_type("query_result")
        
        self.get_ttl = get_ttl
        self.query_ttl = query_ttl
        self.enable_query_cache = enable_query_cache
    
    def _get_cache_key(self, record_id: str) -> str:
        """Get cache key for a record."""
        return get_cache_key_for_record(self.service_name, record_id)
    
    def _get_query_cache_key(self, filters: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None,
                             offset: Optional[int] = None) -> str:
        """Get cache key for a query."""
        return get_cache_key_for_query(
            self.service_name,
            query_hash=None,
            filters=filters,
            limit=limit,
            offset=offset
        )
    
    def _invalidate_record(self, record_id: str) -> None:
        """Invalidate cache for a record and related queries."""
        try:
            # Invalidate record
            cache_key = self._get_cache_key(record_id)
            self.cache.delete(cache_key)
            
            # Invalidate query cache
            if self.enable_query_cache:
                query_pattern = f"{self.service_name}:query:*"
                self.cache.delete_pattern(query_pattern)
        except Exception as e:
            logger.warning(f"Cache invalidation failed for {record_id}: {e}")
    
    def _invalidate_all_queries(self) -> None:
        """Invalidate all query cache for this service."""
        try:
            query_pattern = f"{self.service_name}:query:*"
            self.cache.delete_pattern(query_pattern)
        except Exception as e:
            logger.warning(f"Query cache invalidation failed: {e}")
    
    def _get_with_fallback(self, record_id: str, obs: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        Core cache retrieval logic with optional observation tracking.
        
        Args:
            record_id: Record identifier
            obs: Optional observation object for metadata tracking
            
        Returns:
            Record dictionary or None if not found
        """
        cache_key = self._get_cache_key(record_id)
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {record_id}")
                if obs:
                    try:
                        obs.update(metadata={"cache_hit": True})
                    except Exception:
                        pass
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for {record_id}: {e}")
        
        # Cache miss, get from storage
        logger.debug(f"Cache miss for {record_id}")
        record = self.storage.get(record_id)
        
        if obs:
            try:
                obs.update(metadata={"cache_hit": False})
            except Exception:
                pass
        
        # Cache result if found
        if record is not None:
            try:
                self.cache.set(cache_key, record, ttl=self.get_ttl)
            except Exception as e:
                logger.warning(f"Cache set failed for {record_id}: {e}")
        
        return record
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save data to storage and invalidate cache.
        
        Args:
            data: Data dictionary to save
            
        Returns:
            ID of saved record
        """
        # Track cached storage operation
        try:
            from utils.langfuse.db_tracking import track_storage_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_storage_operation(
                operation="cached_save",
                service_name=f"Cached{self.service_name}",
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ):
                pass  # Tracking context manager handles the observation
        except Exception as track_error:
            logger.debug(f"Failed to track cached storage save operation: {track_error}")
        
        record_id = self.storage.save(data)
        
        # Invalidate cache for this record and queries
        self._invalidate_record(record_id)
        
        return record_id
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get record by ID with caching.
        
        Args:
            record_id: Record identifier
            
        Returns:
            Record dictionary or None if not found
        """
        # Track cached storage operation
        try:
            from utils.langfuse.db_tracking import track_storage_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_storage_operation(
                operation="cached_get",
                service_name=f"Cached{self.service_name}",
                record_id=record_id,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ) as obs:
                return self._get_with_fallback(record_id, obs)
        except Exception as track_error:
            logger.debug(f"Failed to track cached storage get operation: {track_error}")
            # Fallback to original logic
            return self._get_with_fallback(record_id, obs=None)
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query records with optional result caching.
        
        Args:
            filters: Optional filter dictionary
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of record dictionaries
        """
        # Track cached storage operation
        try:
            from utils.langfuse.db_tracking import track_storage_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_storage_operation(
                operation="cached_query",
                service_name=f"Cached{self.service_name}",
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ) as obs:
                if not self.enable_query_cache:
                    results = self.storage.query(filters=filters, limit=limit, offset=offset)
                    if obs:
                        try:
                            obs.update(metadata={"cache_enabled": False, "result_count": len(results)})
                        except Exception:
                            pass
                    return results
                
                cache_key = self._get_query_cache_key(filters, limit, offset)
                
                # Try to get from cache
                try:
                    cached = self.cache.get(cache_key)
                    if cached is not None:
                        logger.debug(f"Query cache hit")
                        if obs:
                            try:
                                obs.update(metadata={"cache_hit": True, "result_count": len(cached)})
                            except Exception:
                                pass
                        return cached
                except Exception as e:
                    logger.warning(f"Query cache get failed: {e}")
                
                # Cache miss, query storage
                logger.debug(f"Query cache miss")
                results = self.storage.query(filters=filters, limit=limit, offset=offset)
                
                if obs:
                    try:
                        obs.update(metadata={"cache_hit": False, "result_count": len(results)})
                    except Exception:
                        pass
                
                # Cache results
                try:
                    self.cache.set(cache_key, results, ttl=self.query_ttl)
                except Exception as e:
                    logger.warning(f"Query cache set failed: {e}")
                
                return results
        except Exception as track_error:
            logger.debug(f"Failed to track cached storage query operation: {track_error}")
            # Fallback to original logic
            if not self.enable_query_cache:
                return self.storage.query(filters=filters, limit=limit, offset=offset)
            
            cache_key = self._get_query_cache_key(filters, limit, offset)
            
            try:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    logger.debug(f"Query cache hit")
                    return cached
            except Exception as e:
                logger.warning(f"Query cache get failed: {e}")
            
            logger.debug(f"Query cache miss")
            results = self.storage.query(filters=filters, limit=limit, offset=offset)
            
            try:
                self.cache.set(cache_key, results, ttl=self.query_ttl)
            except Exception as e:
                logger.warning(f"Query cache set failed: {e}")
            
            return results
    
    def delete(self, record_id: str) -> bool:
        """
        Delete record and invalidate cache.
        
        Args:
            record_id: Record identifier
            
        Returns:
            True if deleted, False if not found
        """
        result = self.storage.delete(record_id)
        
        if result:
            # Invalidate cache
            self._invalidate_record(record_id)
        
        return result
    
    def bulk_save(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Save multiple records and invalidate cache.
        
        Args:
            records: List of data dictionaries to save
            
        Returns:
            List of IDs of saved records
        """
        ids = self.storage.bulk_save(records)
        
        # Invalidate cache for all records and queries
        for record_id in ids:
            self._invalidate_record(record_id)
        self._invalidate_all_queries()
        
        return ids
    
    def bulk_delete(self, record_ids: List[str]) -> int:
        """
        Delete multiple records and invalidate cache.
        
        Args:
            record_ids: List of record identifiers to delete
            
        Returns:
            Number of records deleted
        """
        deleted = self.storage.bulk_delete(record_ids)
        
        if deleted > 0:
            # Invalidate cache for deleted records and queries
            for record_id in record_ids:
                self._invalidate_record(record_id)
            self._invalidate_all_queries()
        
        return deleted
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records matching filters."""
        return self.storage.count(filters=filters)
    
    def exists(self, record_id: str) -> bool:
        """Check if record exists."""
        return self.storage.exists(record_id)
    
    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        with self.storage.transaction():
            yield
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate data before saving."""
        return self.storage.validate_data(data)
    
    def log_operation(self, operation: str, record_id: Optional[str] = None, 
                     success: bool = True, error: Optional[Exception] = None) -> None:
        """Log storage operation."""
        self.storage.log_operation(operation, record_id, success, error)
    
    def _init_database(self) -> None:
        """Initialize database schema (delegates to wrapped storage)."""
        self.storage._init_database()




