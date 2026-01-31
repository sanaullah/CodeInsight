"""
Cached Architecture Model Storage.

Redis-cached wrapper for ArchitectureModelStorage providing automatic caching
with invalidation on writes. Caches metadata always, optionally caches small models.
"""

import logging
from typing import Dict, List, Optional, Any
from services.storage.architecture_model_storage import ArchitectureModelStorage
from services.storage.cached_storage import CachedStorage
from services.cache_config import get_ttl_for_cache_type
from services.redis_cache import get_cache

logger = logging.getLogger(__name__)


class CachedArchitectureModelStorage(CachedStorage):
    """
    Cached architecture model storage with Redis.
    
    Wraps ArchitectureModelStorage with automatic caching:
    - Model metadata: TTL 24 hours (always cached)
    - Small models (<1MB): TTL 24 hours (cached)
    - Large models: Metadata only, not the model data
    - Invalidates cache on model updates
    """
    
    def __init__(self, use_minio: bool = True):
        """
        Initialize cached architecture model storage.
        
        Args:
            use_minio: Whether to use MinIO for large models
        """
        storage = ArchitectureModelStorage(use_minio=use_minio)
        super().__init__(
            storage=storage,
            service_name="architecture_model",
            get_ttl=get_ttl_for_cache_type("architecture_model"),  # 24 hours default
            query_ttl=get_ttl_for_cache_type("query_result"),  # 15 min default
            enable_query_cache=True
        )
        self.cache = get_cache()
        logger.debug("CachedArchitectureModelStorage initialized")
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get architecture model with caching.
        
        For large models stored in MinIO, only metadata is cached.
        For small models, the full model is cached.
        
        Args:
            record_id: Knowledge ID
        
        Returns:
            Model dictionary or None if not found
        """
        cache_key = self._get_cache_key(record_id)
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for architecture model: {record_id}")
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for {record_id}: {e}")
        
        # Cache miss, get from storage
        logger.debug(f"Cache miss for architecture model: {record_id}")
        model = self.storage.get(record_id)
        
        # Cache result if found
        if model is not None:
            try:
                # Check if this is a large model (stored in MinIO)
                metadata = model.get('metadata', {})
                if metadata.get('stored_in') == 'minio':
                    # For large models, cache metadata only (not the full model)
                    # The model data will be loaded from MinIO when needed
                    cache_data = {
                        'knowledge_id': model.get('knowledge_id'),
                        'metadata': metadata,
                        'file_hash': model.get('file_hash'),
                        'project_path': model.get('project_path'),
                        'minio_reference': True  # Flag to indicate MinIO storage
                    }
                    ttl = get_ttl_for_cache_type("architecture_model")  # 24 hours
                    self.cache.set(cache_key, cache_data, ttl=ttl)
                    logger.debug(f"Cached architecture model metadata: {record_id}")
                else:
                    # For small models, cache the full model
                    ttl = get_ttl_for_cache_type("architecture_model")  # 24 hours
                    self.cache.set(cache_key, model, ttl=ttl)
                    logger.debug(f"Cached architecture model: {record_id} with TTL {ttl}")
            except Exception as e:
                logger.warning(f"Cache set failed for {record_id}: {e}")
        
        return model
    
    def store_architecture_model(
        self,
        project_path: str,
        model: Any,
        file_hash: str,
        version: int = 1,
        agent_name: str = "ArchitectureModelBuilder"
    ) -> str:
        """
        Store architecture model and invalidate cache.
        
        Args:
            project_path: Path to the project
            model: Architecture model
            file_hash: Hash of the file
            version: Optional version number
            agent_name: Name of agent creating the model
            
        Returns:
            Knowledge ID
        """
        knowledge_id = self.storage.store_architecture_model(
            project_path=project_path,
            model=model,
            file_hash=file_hash,
            version=version,
            agent_name=agent_name
        )
        
        # Invalidate cache
        self._invalidate_record(knowledge_id)
        
        return knowledge_id
    
    def get_architecture_model(
        self,
        project_path: str,
        file_hash: str,
        version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get architecture model (uses get() which handles caching).
        
        Args:
            project_path: Path to the project
            file_hash: Hash of the file
            version: Optional version number
            
        Returns:
            Model dictionary or None if not found
        """
        # Build knowledge_id
        if version:
            knowledge_id = f"arch_model:{project_path}:{file_hash}:v{version}"
        else:
            # Get latest version
            knowledge_id = f"arch_model:{project_path}:{file_hash}:v1"
        
        return self.get(knowledge_id)
    
    def store_large_model(
        self,
        project_path: str,
        model_data: bytes,
        file_hash: str
    ) -> str:
        """
        Store large model in MinIO and invalidate cache.
        
        Args:
            project_path: Path to the project
            model_data: Model data as bytes
            file_hash: Hash of the file
            
        Returns:
            Knowledge ID
        """
        knowledge_id = self.storage.store_large_model(
            project_path=project_path,
            model_data=model_data,
            file_hash=file_hash
        )
        
        # Invalidate cache
        self._invalidate_record(knowledge_id)
        
        return knowledge_id
    
    def get_large_model(self, model_id: str) -> Optional[bytes]:
        """
        Get large model from MinIO (delegates to storage).
        
        Args:
            model_id: Knowledge ID
            
        Returns:
            Model data as bytes, or None if not found
        """
        return self.storage.get_large_model(model_id)










