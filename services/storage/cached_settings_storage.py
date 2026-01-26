"""
Cached Settings Storage.

Redis-cached wrapper for SettingsStorage providing automatic caching
with invalidation on writes and configurable TTL.
"""

import logging
from services.storage.settings_storage import SettingsStorage
from services.storage.cached_storage import CachedStorage
from services.cache_config import get_ttl_for_cache_type

logger = logging.getLogger(__name__)


class CachedSettingsStorage(CachedStorage):
    """
    Cached settings storage with Redis.
    
    Wraps SettingsStorage with automatic caching:
    - Caches get() operations (15 minutes TTL)
    - Caches query() results (15 minutes TTL)
    - Caches get_setting() and get_all_settings() results
    - Invalidates cache on save(), delete(), bulk operations
    """
    
    def __init__(self):
        """
        Initialize cached settings storage.
        
        Creates underlying SettingsStorage and wraps it with caching.
        """
        storage = SettingsStorage()
        super().__init__(
            storage=storage,
            service_name="settings",
            get_ttl=get_ttl_for_cache_type("settings"),  # 15 minutes default
            query_ttl=get_ttl_for_cache_type("query_result"),  # 15 min default
            enable_query_cache=True
        )
        logger.debug("CachedSettingsStorage initialized")
    
    # Expose settings-specific convenience methods
    
    def get_setting(self, user_id: str, key: str, default=None):
        """Get a setting value for a user (with caching)."""
        # Use composite key for caching
        record_id = f"{user_id}:{key}"
        
        # Try cache first
        cache_key = self._get_cache_key(record_id)
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for setting {key} for user {user_id}")
            return cached.get("setting_value") if isinstance(cached, dict) else cached
        
        # Cache miss - get from storage
        setting = self.storage.get_setting(user_id, key, default)
        
        # Cache the result
        if setting is not None:
            try:
                self.cache.set(cache_key, {"setting_value": setting}, ttl=self.get_ttl)
            except Exception as e:
                logger.warning(f"Failed to cache setting: {e}")
        
        return setting
    
    def set_setting(self, user_id: str, key: str, value) -> bool:
        """Set or update a setting for a user (with cache invalidation)."""
        success = self.storage.set_setting(user_id, key, value)
        if success:
            # Invalidate cache
            record_id = f"{user_id}:{key}"
            self._invalidate_record(record_id)
        return success
    
    def get_all_settings(self, user_id: str) -> dict:
        """Get all settings for a user (with caching)."""
        # Use query cache
        filters = {"user_id": user_id}
        cache_key = self._get_query_cache_key(filters=filters)
        
        # Try cache first
        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for all settings for user {user_id}")
            return cached
        
        # Cache miss - get from storage
        settings = self.storage.get_all_settings(user_id)
        
        # Cache the result
        try:
            self.cache.set(cache_key, settings, ttl=self.query_ttl)
        except Exception as e:
            logger.warning(f"Failed to cache all settings: {e}")
        
        return settings
    
    def delete_user_settings(self, user_id: str) -> bool:
        """Delete all settings for a user (with cache invalidation)."""
        success = self.storage.delete_user_settings(user_id)
        if success:
            # Invalidate all queries for this user
            self._invalidate_all_queries()
        return success
    
    def delete_setting(self, user_id: str, key: str) -> bool:
        """Delete a specific setting (with cache invalidation)."""
        success = self.storage.delete_setting(user_id, key)
        if success:
            record_id = f"{user_id}:{key}"
            self._invalidate_record(record_id)
        return success

