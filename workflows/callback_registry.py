"""
Callback registry for stream callbacks in Swarm Analysis workflow.

Provides thread-safe storage and retrieval of callback functions using unique IDs.
This enables checkpoint serialization and distributed execution by storing callback
IDs in state instead of callable objects.
"""

import uuid
import threading
import logging
from typing import Callable, Optional, Dict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class StreamCallbackRegistry:
    """
    Thread-safe registry for stream callbacks.
    
    Stores callbacks with unique IDs and provides lookup functionality.
    Callbacks automatically expire after TTL (default 1 hour) to prevent memory leaks.
    """
    
    def __init__(self, default_ttl: timedelta = timedelta(hours=1)):
        """
        Initialize callback registry.
        
        Args:
            default_ttl: Default time-to-live for callbacks (default: 1 hour)
        """
        self._registry: Dict[str, Dict[str, any]] = {}
        self._lock = threading.Lock()
        self._default_ttl = default_ttl
    
    def register(self, callback: Callable) -> str:
        """
        Register a callback and return its unique ID.
        
        Args:
            callback: Callback function to register
            
        Returns:
            Unique callback ID (UUID4 string)
        """
        if not callable(callback):
            raise ValueError("Callback must be callable")
        
        callback_id = str(uuid.uuid4())
        expires_at = datetime.now() + self._default_ttl
        
        with self._lock:
            self._registry[callback_id] = {
                "callback": callback,
                "registered_at": datetime.now(),
                "expires_at": expires_at
            }
        
        logger.debug(f"Registered callback with ID: {callback_id}")
        return callback_id
    
    def get(self, callback_id: str) -> Optional[Callable]:
        """
        Get callback by ID.
        
        Args:
            callback_id: Callback ID to lookup
            
        Returns:
            Callback function if found and not expired, None otherwise
        """
        if not callback_id:
            return None
        
        with self._lock:
            entry = self._registry.get(callback_id)
            
            if entry is None:
                logger.debug(f"Callback not found: {callback_id}")
                return None
            
            # Check expiration
            if datetime.now() > entry["expires_at"]:
                logger.debug(f"Callback expired: {callback_id}")
                del self._registry[callback_id]
                return None
            
            return entry["callback"]
    
    def unregister(self, callback_id: str) -> bool:
        """
        Unregister a callback by ID.
        
        Args:
            callback_id: Callback ID to remove
            
        Returns:
            True if callback was removed, False if not found
        """
        if not callback_id:
            return False
        
        with self._lock:
            if callback_id in self._registry:
                del self._registry[callback_id]
                logger.debug(f"Unregistered callback: {callback_id}")
                return True
            else:
                logger.debug(f"Callback not found for unregister: {callback_id}")
                return False
    
    def cleanup(self) -> int:
        """
        Remove expired callbacks.
        
        Returns:
            Number of callbacks removed
        """
        now = datetime.now()
        removed = 0
        
        with self._lock:
            expired_ids = [
                callback_id
                for callback_id, entry in self._registry.items()
                if now > entry["expires_at"]
            ]
            
            for callback_id in expired_ids:
                del self._registry[callback_id]
                removed += 1
        
        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired callbacks")
        
        return removed
    
    def clear(self) -> int:
        """
        Clear all callbacks (primarily for testing).
        
        Returns:
            Number of callbacks removed
        """
        with self._lock:
            count = len(self._registry)
            self._registry.clear()
        
        logger.debug(f"Cleared {count} callbacks")
        return count
    
    def size(self) -> int:
        """
        Get current number of registered callbacks.
        
        Returns:
            Number of callbacks in registry
        """
        with self._lock:
            return len(self._registry)


# Singleton instance
_registry_instance: Optional[StreamCallbackRegistry] = None
_registry_lock = threading.Lock()


def get_callback_registry() -> StreamCallbackRegistry:
    """
    Get singleton instance of callback registry.
    
    Returns:
        StreamCallbackRegistry instance
    """
    global _registry_instance
    
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = StreamCallbackRegistry()
    
    return _registry_instance
