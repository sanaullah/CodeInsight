"""
Cached Prompt Storage.

Redis-cached wrapper for PromptStorage providing automatic caching
with invalidation on writes and configurable TTL based on usage.
"""

import logging
from typing import Dict, List, Optional, Any
from services.storage.prompt_storage import PromptStorage
from services.storage.cached_storage import CachedStorage
from services.cache_config import get_ttl_for_cache_type
from services.redis_cache import get_cache
from services.cache_utils import validate_cache_key

logger = logging.getLogger(__name__)


class CachedPromptStorage(CachedStorage):
    """
    Cached prompt storage with Redis.
    
    Wraps PromptStorage with automatic caching:
    - Individual prompts: TTL 12 hours (default), 24 hours for popular prompts (usage_count > 100)
    - Search results: TTL 15 minutes
    - Invalidates cache on prompt updates
    """
    
    def __init__(self):
        """
        Initialize cached prompt storage.
        
        Creates underlying PromptStorage and wraps it with caching.
        """
        storage = PromptStorage()
        super().__init__(
            storage=storage,
            service_name="prompt",
            get_ttl=get_ttl_for_cache_type("prompt"),  # 12 hours default
            query_ttl=get_ttl_for_cache_type("query_result"),  # 15 min default
            enable_query_cache=True
        )
        self.cache = get_cache()
        logger.debug("CachedPromptStorage initialized")
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get prompt with caching and dynamic TTL based on usage.
        
        Args:
            record_id: Prompt ID
        
        Returns:
            Prompt dictionary or None if not found
        """
        cache_key = self._get_cache_key(record_id)
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for prompt: {record_id}")
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for {record_id}: {e}")
        
        # Cache miss, get from storage
        logger.debug(f"Cache miss for prompt: {record_id}")
        prompt = self.storage.get(record_id)
        
        # Cache result if found with appropriate TTL
        if prompt is not None:
            try:
                ttl = self._get_ttl_for_prompt(prompt)
                self.cache.set(cache_key, prompt, ttl=ttl)
                logger.debug(f"Cached prompt: {record_id} with TTL {ttl}")
            except Exception as e:
                logger.warning(f"Cache set failed for {record_id}: {e}")
        
        return prompt
    
    def get_prompt_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get prompt by name with caching.
        
        Args:
            name: Prompt name
            
        Returns:
            Prompt dictionary or None if not found
        """
        # Use name as cache key
        cache_key = f"{self.service_name}:name:{name}"
        # Validate the cache key
        try:
            cache_key = validate_cache_key(cache_key)
        except ValueError as e:
            logger.warning(f"Invalid cache key for prompt name: {e}, using sanitized key")
            # Re-raise to fail fast - caller should handle invalid keys
            raise
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for prompt by name: {name}")
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for prompt name {name}: {e}")
        
        # Cache miss, get from storage
        prompt = self.storage.get_prompt_by_name(name)
        
        # Cache result if found
        if prompt is not None:
            try:
                ttl = self._get_ttl_for_prompt(prompt)
                self.cache.set(cache_key, prompt, ttl=ttl)
                
                # Also cache by ID
                prompt_id = prompt.get('prompt_id')
                if prompt_id:
                    id_cache_key = self._get_cache_key(prompt_id)
                    self.cache.set(id_cache_key, prompt, ttl=ttl)
            except Exception as e:
                logger.warning(f"Cache set failed for prompt name {name}: {e}")
        
        return prompt
    
    def search_prompts(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        role: Optional[str] = None,
        architecture_hash: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search prompts with caching.
        
        Args:
            query: Search query text
            tags: Optional list of tags
            role: Optional role
            architecture_hash: Optional architecture hash
            limit: Maximum number of results
            
        Returns:
            List of prompt dictionaries
        """
        # Build cache key from search parameters
        import hashlib
        search_params = f"{query}:{tags}:{role}:{architecture_hash}:{limit}"
        search_hash = hashlib.md5(search_params.encode()).hexdigest()
        cache_key = f"{self.service_name}:search:{search_hash}"
        
        # Try to get from cache (shorter TTL for search results)
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Search cache hit for query: {query[:50]}")
                return cached
        except Exception as e:
            logger.warning(f"Search cache get failed: {e}")
        
        # Cache miss, search storage
        logger.debug(f"Search cache miss for query: {query[:50]}")
        results = self.storage.search_prompts(
            query=query,
            tags=tags,
            role=role,
            architecture_hash=architecture_hash,
            limit=limit
        )
        
        # Cache results with shorter TTL (15 minutes)
        try:
            search_ttl = get_ttl_for_cache_type("query_result")  # 15 minutes
            self.cache.set(cache_key, results, ttl=search_ttl)
            logger.debug(f"Cached search results with TTL {search_ttl}")
        except Exception as e:
            logger.warning(f"Search cache set failed: {e}")
        
        return results
    
    def update_prompt(self, prompt_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update prompt and invalidate cache.
        
        Args:
            prompt_id: Prompt ID
            updates: Dictionary with fields to update
            
        Returns:
            True if updated, False if not found
        """
        updated = self.storage.update_prompt(prompt_id, updates)
        
        if updated:
            # Invalidate cache
            self._invalidate_record(prompt_id)
            
            # Also invalidate search caches
            try:
                search_pattern = f"{self.service_name}:search:*"
                self.cache.delete_pattern(search_pattern)
            except Exception as e:
                logger.warning(f"Search cache invalidation failed: {e}")
        
        return updated
    
    def get_all_prompts(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all prompts with caching.
        
        Args:
            limit: Maximum number of prompts
            offset: Offset for pagination
            
        Returns:
            List of prompt dictionaries
        """
        # Build cache key from query parameters
        import hashlib
        query_params = f"all:{limit}:{offset}"
        query_hash = hashlib.md5(query_params.encode()).hexdigest()
        cache_key = f"{self.service_name}:query:{query_hash}"
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for get_all_prompts (limit={limit}, offset={offset})")
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for get_all_prompts: {e}")
        
        # Cache miss, get from storage
        logger.debug(f"Cache miss for get_all_prompts (limit={limit}, offset={offset})")
        prompts = self.storage.get_all_prompts(limit=limit, offset=offset)
        
        # Cache results with query TTL (15 minutes)
        try:
            query_ttl = get_ttl_for_cache_type("query_result")  # 15 minutes
            self.cache.set(cache_key, prompts, ttl=query_ttl)
            logger.debug(f"Cached get_all_prompts results with TTL {query_ttl}")
        except Exception as e:
            logger.warning(f"Cache set failed for get_all_prompts: {e}")
        
        return prompts
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get prompt statistics (delegates to storage, not cached)."""
        return self.storage.get_prompt_statistics()
    
    def _get_ttl_for_prompt(self, prompt: Dict[str, Any]) -> int:
        """
        Get TTL for prompt based on usage count.
        
        Args:
            prompt: Prompt dictionary
            
        Returns:
            TTL in seconds
        """
        usage_count = prompt.get('usage_count', 0)
        
        # Popular prompts (usage_count > 100): 24 hours
        if usage_count > 100:
            return 86400  # 24 hours
        
        # Regular prompts: 12 hours
        return get_ttl_for_cache_type("prompt")  # 12 hours

