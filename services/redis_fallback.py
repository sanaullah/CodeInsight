"""
Redis Fallback Implementation.

Provides in-memory fallback for Redis operations when Redis is unavailable.
Thread-safe implementation using locks.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InMemoryFallback:
    """
    In-memory fallback for Redis operations.
    
    Provides thread-safe implementations of common Redis operations:
    - SET, GET, HSET, HGET, EXPIRE, SETNX
    - Automatic expiration handling
    """
    
    def __init__(self):
        """Initialize in-memory fallback storage."""
        self._storage: Dict[str, Any] = {}
        self._hash_storage: Dict[str, Dict[str, Any]] = {}
        self._expirations: Dict[str, float] = {}  # key -> expiration timestamp
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        logger.info("InMemoryFallback initialized")
    
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """
        Set a key-value pair.
        
        Args:
            key: Key name
            value: Value to store
            ex: Optional expiration time in seconds
        
        Returns:
            True if successful
        """
        with self._lock:
            self._storage[key] = value
            if ex:
                self._expirations[key] = time.time() + ex
            else:
                # Remove expiration if key exists without expiration
                self._expirations.pop(key, None)
            logger.debug(f"InMemoryFallback: SET {key}")
            return True
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value by key.
        
        Args:
            key: Key name
        
        Returns:
            Value or None if not found or expired
        """
        with self._lock:
            # Check expiration
            if key in self._expirations:
                if time.time() > self._expirations[key]:
                    # Expired, remove it
                    self._storage.pop(key, None)
                    self._expirations.pop(key, None)
                    logger.debug(f"InMemoryFallback: GET {key} (expired)")
                    return None
            
            value = self._storage.get(key)
            logger.debug(f"InMemoryFallback: GET {key} = {value is not None}")
            return value
    
    def delete(self, key: str) -> int:
        """
        Delete a key.
        
        Args:
            key: Key name
        
        Returns:
            Number of keys deleted (0 or 1)
        """
        with self._lock:
            deleted = 0
            if key in self._storage:
                del self._storage[key]
                deleted = 1
            self._expirations.pop(key, None)
            self._hash_storage.pop(key, None)
            logger.debug(f"InMemoryFallback: DELETE {key}")
            return deleted
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists.
        
        Args:
            key: Key name
        
        Returns:
            True if key exists and is not expired
        """
        with self._lock:
            # Check expiration
            if key in self._expirations:
                if time.time() > self._expirations[key]:
                    # Expired, remove it
                    self._storage.pop(key, None)
                    self._expirations.pop(key, None)
                    return False
            
            return key in self._storage
    
    def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Key name
            seconds: Expiration time in seconds
        
        Returns:
            True if key exists and expiration was set
        """
        with self._lock:
            if key in self._storage:
                self._expirations[key] = time.time() + seconds
                logger.debug(f"InMemoryFallback: EXPIRE {key} {seconds}s")
                return True
            return False
    
    def setnx(self, key: str, value: Any) -> bool:
        """
        Set a key only if it doesn't exist (SET if Not eXists).
        
        Args:
            key: Key name
            value: Value to store
        
        Returns:
            True if key was set, False if key already exists
        """
        with self._lock:
            if key not in self._storage:
                self._storage[key] = value
                logger.debug(f"InMemoryFallback: SETNX {key} (set)")
                return True
            logger.debug(f"InMemoryFallback: SETNX {key} (already exists)")
            return False
    
    def hset(self, name: str, key: Optional[str] = None, value: Optional[Any] = None, mapping: Optional[Dict[str, Any]] = None) -> int:
        """
        Set hash field(s).
        
        Args:
            name: Hash name
            key: Optional single field name
            value: Optional single field value
            mapping: Optional dictionary of field-value pairs
        
        Returns:
            Number of fields set
        """
        with self._lock:
            if name not in self._hash_storage:
                self._hash_storage[name] = {}
            
            count = 0
            if key is not None and value is not None:
                self._hash_storage[name][key] = value
                count = 1
            elif mapping:
                for k, v in mapping.items():
                    self._hash_storage[name][k] = v
                    count += 1
            
            logger.debug(f"InMemoryFallback: HSET {name} ({count} fields)")
            return count
    
    def hget(self, name: str, key: str) -> Optional[Any]:
        """
        Get hash field value.
        
        Args:
            name: Hash name
            key: Field name
        
        Returns:
            Field value or None
        """
        with self._lock:
            if name not in self._hash_storage:
                return None
            value = self._hash_storage[name].get(key)
            logger.debug(f"InMemoryFallback: HGET {name}.{key} = {value is not None}")
            return value
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """
        Get all hash fields and values.
        
        Args:
            name: Hash name
        
        Returns:
            Dictionary of field-value pairs
        """
        with self._lock:
            if name not in self._hash_storage:
                return {}
            result = self._hash_storage[name].copy()
            logger.debug(f"InMemoryFallback: HGETALL {name} ({len(result)} fields)")
            return result
    
    def hdel(self, name: str, *keys: str) -> int:
        """
        Delete hash field(s).
        
        Args:
            name: Hash name
            *keys: Field names to delete
        
        Returns:
            Number of fields deleted
        """
        with self._lock:
            if name not in self._hash_storage:
                return 0
            
            count = 0
            for key in keys:
                if key in self._hash_storage[name]:
                    del self._hash_storage[name][key]
                    count += 1
            
            logger.debug(f"InMemoryFallback: HDEL {name} ({count} fields)")
            return count
    
    def clear(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._storage.clear()
            self._hash_storage.clear()
            self._expirations.clear()
            logger.info("InMemoryFallback: Cleared all data")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get fallback storage statistics.
        
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            # Clean up expired keys
            now = time.time()
            expired_keys = [k for k, exp_time in self._expirations.items() if now > exp_time]
            for key in expired_keys:
                self._storage.pop(key, None)
                self._expirations.pop(key, None)
            
            return {
                "keys": len(self._storage),
                "hashes": len(self._hash_storage),
                "expirations": len(self._expirations),
                "total_hash_fields": sum(len(h) for h in self._hash_storage.values())
            }

