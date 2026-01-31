"""
Cached Knowledge Storage.

Redis-cached wrapper for KnowledgeStorage providing automatic caching
with invalidation on writes and configurable TTL based on knowledge type.
"""

import logging
import json
import hashlib
from typing import Dict, List, Optional, Any
from services.storage.knowledge_storage import KnowledgeStorage
from services.storage.cached_storage import CachedStorage
from services.cache_config import get_ttl_for_cache_type
from services.redis_cache import get_cache
from services.cache_utils import validate_cache_key
from agents.collaboration_models import KnowledgeType

logger = logging.getLogger(__name__)


class CachedKnowledgeStorage(CachedStorage):
    """
    Cached knowledge storage with Redis.
    
    Wraps KnowledgeStorage with automatic caching:
    - Caches get() operations with TTL based on knowledge type:
      - Architecture models: 24 hours
      - High confidence (>=0.8): 6 hours
      - Regular knowledge: 1 hour (default from cache_config)
    - Caches search_knowledge() results: 15 minutes
    - Caches query() results: 15 minutes
    - Invalidates cache on save(), delete(), bulk operations
    """
    
    def __init__(self):
        """
        Initialize cached knowledge storage.
        
        Creates underlying KnowledgeStorage and wraps it with caching.
        """
        storage = KnowledgeStorage()
        super().__init__(
            storage=storage,
            service_name="knowledge_base",
            get_ttl=get_ttl_for_cache_type("knowledge_base"),  # 6 hours default
            query_ttl=get_ttl_for_cache_type("query_result"),  # 15 min default
            enable_query_cache=True
        )
        self.cache = get_cache()
        logger.debug("CachedKnowledgeStorage initialized")
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge with caching and dynamic TTL based on knowledge type.
        
        Args:
            record_id: Knowledge ID
        
        Returns:
            Knowledge dictionary or None if not found
        """
        cache_key = self._get_cache_key(record_id)
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for knowledge: {record_id}")
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for {record_id}: {e}")
        
        # Cache miss, get from storage
        logger.debug(f"Cache miss for knowledge: {record_id}")
        knowledge = self.storage.get(record_id)
        
        # Cache result if found with appropriate TTL
        if knowledge is not None:
            try:
                ttl = self._get_ttl_for_knowledge(knowledge)
                self.cache.set(cache_key, knowledge, ttl=ttl)
                logger.debug(f"Cached knowledge: {record_id} with TTL {ttl}")
            except Exception as e:
                logger.warning(f"Cache set failed for {record_id}: {e}")
        
        return knowledge
    
    def search_knowledge(self, query: str, 
                        knowledge_types: Optional[List[KnowledgeType]] = None,
                        agents: Optional[List[str]] = None,
                        date_range: Optional[Any] = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search knowledge with caching.
        
        Args:
            query: Search query string
            knowledge_types: Optional list of knowledge types
            agents: Optional list of agent names
            date_range: Optional date range tuple
            limit: Maximum number of results
        
        Returns:
            List of knowledge dictionaries
        """
        # Build cache key from search parameters
        cache_key = self._get_search_cache_key(query, knowledge_types, agents, date_range, limit)
        
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
        results = self.storage.search_knowledge(
            query=query,
            knowledge_types=knowledge_types,
            agents=agents,
            date_range=date_range,
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
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save knowledge and invalidate cache.
        
        Args:
            data: Knowledge data dictionary
        
        Returns:
            Knowledge ID
        """
        knowledge_id = super().save(data)
        
        # Invalidate search caches (pattern delete)
        try:
            search_pattern = f"{self.service_name}:query:*"
            self.cache.delete_pattern(search_pattern)
            logger.debug("Invalidated search caches after save")
        except Exception as e:
            logger.warning(f"Failed to invalidate search caches: {e}")
        
        return knowledge_id
    
    def cleanup_expired_knowledge(self) -> int:
        """
        Cleanup expired knowledge and invalidate all caches.
        
        Returns:
            Number of items deleted
        """
        count = self.storage.cleanup_expired_knowledge()
        
        # Invalidate all knowledge caches after cleanup
        if count > 0:
            try:
                # Invalidate all knowledge cache keys
                pattern = f"{self.service_name}:*"
                self.cache.delete_pattern(pattern)
                logger.debug(f"Invalidated all knowledge caches after cleanup ({count} items)")
            except Exception as e:
                logger.warning(f"Failed to invalidate caches after cleanup: {e}")
        
        return count
    
    def _get_ttl_for_knowledge(self, knowledge: Dict[str, Any]) -> int:
        """
        Get TTL for knowledge based on type and confidence.
        
        Args:
            knowledge: Knowledge dictionary
        
        Returns:
            TTL in seconds
        """
        knowledge_type = knowledge.get('knowledge_type')
        confidence = knowledge.get('confidence', 0.5)
        
        # Convert KnowledgeType enum to string if needed
        if hasattr(knowledge_type, 'value'):
            knowledge_type = knowledge_type.value
        elif isinstance(knowledge_type, KnowledgeType):
            knowledge_type = knowledge_type.value
        
        # Architecture models get longest TTL
        if knowledge_type == KnowledgeType.ARCHITECTURE_MODEL.value:
            return 86400  # 24 hours
        
        # High confidence knowledge gets longer TTL
        if confidence >= 0.8:
            return 21600  # 6 hours
        
        # Default TTL from cache config
        return get_ttl_for_cache_type("knowledge_base")  # 6 hours default
    
    def _get_search_cache_key(self, query: str, knowledge_types: Optional[List] = None,
                              agents: Optional[List[str]] = None, date_range: Optional[Any] = None,
                              limit: int = 10) -> str:
        """
        Get cache key for search query.
        
        Args:
            query: Search query string
            knowledge_types: Optional list of knowledge types
            agents: Optional list of agent names
            date_range: Optional date range
            limit: Result limit
        
        Returns:
            Cache key string
        """
        # Build key data
        key_data = {
            "query": query,
            "types": [kt.value if hasattr(kt, 'value') else str(kt) for kt in (knowledge_types or [])],
            "agents": sorted(agents or []),
            "limit": limit
        }
        
        # Add date range if provided
        if date_range:
            key_data["date_range"] = [str(d) for d in date_range]
        
        # Create hash of key data
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{self.service_name}:search:{key_hash}"
    
    # Delegate additional methods to storage
    def store_architecture_model(self, project_path: str, model: Any, 
                                  file_hash: str, version: int = 1,
                                  agent_name: str = "ArchitectureModelBuilder") -> str:
        """Store architecture model (delegates to storage, then invalidates cache)."""
        knowledge_id = self.storage.store_architecture_model(
            project_path, model, file_hash, version, agent_name
        )
        # Invalidate cache for this knowledge
        self._invalidate_record(knowledge_id)
        return knowledge_id
    
    def get_architecture_model(self, project_path: str, file_hash: str,
                               version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get architecture model (uses get() which handles caching)."""
        # Use a special cache key for architecture models
        cache_key = f"{self.service_name}:arch_model:{project_path}:{file_hash}:{version or 'latest'}"
        # Validate the cache key
        try:
            cache_key = validate_cache_key(cache_key)
        except ValueError as e:
            logger.warning(f"Invalid cache key for architecture model: {e}, using sanitized key")
            # Re-raise to fail fast - caller should handle invalid keys
            raise
        
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for architecture model: {project_path}:{file_hash}")
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for architecture model: {e}")
        
        # Cache miss, get from storage
        model = self.storage.get_architecture_model(project_path, file_hash, version)
        
        if model:
            try:
                # Architecture models get 24 hour TTL
                self.cache.set(cache_key, model, ttl=86400)
                logger.debug(f"Cached architecture model with TTL 86400")
            except Exception as e:
                logger.warning(f"Cache set failed for architecture model: {e}")
        
        return model
    
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """Update knowledge (delegates to storage, then invalidates cache)."""
        updated = self.storage.update_knowledge(knowledge_id, updates)
        if updated:
            self._invalidate_record(knowledge_id)
        return updated
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get knowledge statistics (not cached, always fresh)."""
        return self.storage.get_knowledge_statistics()










