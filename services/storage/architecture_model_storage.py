"""
Architecture Model Storage Service.

PostgreSQL-only storage implementation for architecture models with MinIO support for large models.
Provides efficient storage and retrieval of architecture models.
"""

import json
import logging
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
from io import BytesIO

from services.storage.base_storage import BaseStorage, StorageError, StorageNotFoundError
from services.storage.knowledge_storage import KnowledgeStorage
from services.db_config import get_db_config, CODELUMEN_DATABASE
from services.postgresql_connection_pool import get_db_connection, transaction
from agents.collaboration_models import Knowledge, KnowledgeType

# Try to use orjson for faster JSON parsing if available
try:
    import orjson
    _json_loads = orjson.loads
    _json_dumps = lambda obj: orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode('utf-8')
    _USE_ORJSON = True
except ImportError:
    _json_loads = json.loads
    _json_dumps = lambda obj: json.dumps(obj, ensure_ascii=False, default=str)
    _USE_ORJSON = False

# MinIO support
try:
    from services.minio_client import MinIOClient
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Size threshold for using MinIO (1MB)
MINIO_SIZE_THRESHOLD = 1024 * 1024  # 1MB


class ArchitectureModelStorage(BaseStorage):
    """
    PostgreSQL storage for architecture models with MinIO support for large models.
    
    Implements BaseStorage interface and provides additional methods
    for architecture model operations with automatic MinIO integration for large models.
    """
    
    def __init__(self, use_minio: bool = True):
        """
        Initialize architecture model storage.
        
        Args:
            use_minio: Whether to use MinIO for large models (default True)
        """
        self.config = get_db_config()
        self.schema = self.config.postgresql.schema
        self._initialized = False
        self.use_minio = use_minio and MINIO_AVAILABLE
        
        # Initialize MinIO client if available
        if self.use_minio:
            try:
                self.minio_client = MinIOClient()
                self.minio_client.ensure_bucket()
                logger.info("MinIO client initialized for architecture model storage")
            except Exception as e:
                logger.warning(f"MinIO not available, will use PostgreSQL only: {e}")
                self.use_minio = False
        
        # Use knowledge storage for small models
        self.knowledge_storage = KnowledgeStorage()
        super().__init__(db_identifier=CODELUMEN_DATABASE)
    
    def _init_database(self) -> None:
        """
        Initialize database schema.
        
        Creates schema, tables, and indexes if they don't exist.
        This is called automatically by BaseStorage.__init__.
        """
        if self._initialized:
            return
        
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Ensure schema exists
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                    
                    # Tables should already exist from migration
                    logger.debug(f"Database schema initialized for {self.schema}")
                
                conn.commit()
                self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise StorageError(f"Failed to initialize database: {e}") from e
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save architecture model to storage (implements BaseStorage interface).
        
        Args:
            data: Dictionary containing model data
        
        Returns:
            Model ID as string
        """
        project_path = data.get('project_path')
        model = data.get('model') or data.get('content')
        file_hash = data.get('file_hash')
        version = data.get('version', 1)
        agent_name = data.get('agent_name', 'ArchitectureModelBuilder')
        
        if not project_path or not model or not file_hash:
            raise StorageError("project_path, model, and file_hash are required")
        
        return self.store_architecture_model(
            project_path=project_path,
            model=model,
            file_hash=file_hash,
            version=version,
            agent_name=agent_name
        )
    
    def store_architecture_model(
        self,
        project_path: str,
        model: Any,
        file_hash: str,
        version: int = 1,
        agent_name: str = "ArchitectureModelBuilder"
    ) -> str:
        """
        Store architecture model in storage.
        
        Models < 1MB: Store as JSONB in PostgreSQL
        Models >= 1MB: Store in MinIO, reference in PostgreSQL
        
        Args:
            project_path: Path to the project
            model: Architecture model (dict or object that can be serialized)
            file_hash: Hash of the file that generated this model
            version: Optional version number (default: 1)
            agent_name: Name of agent creating the model (default: "ArchitectureModelBuilder")
        
        Returns:
            Knowledge ID
        """
        # Convert model to dict if needed
        if not isinstance(model, dict):
            try:
                model_dict = model.__dict__ if hasattr(model, '__dict__') else dict(model)
            except:
                model_dict = {"model": str(model)}
        else:
            model_dict = model
        
        # Check model size
        model_size = self.check_model_size(model_dict)
        use_minio = self.use_minio and self.should_use_minio(model_size)
        
        knowledge_id = f"arch_model:{project_path}:{file_hash}:v{version}"
        
        if use_minio:
            # Store large model in MinIO
            try:
                # Serialize model to JSON bytes
                model_json = _json_dumps(model_dict)
                model_bytes = model_json.encode('utf-8')
                
                # Upload to MinIO
                object_name = f"architecture_models/{project_path}/{file_hash}/v{version}.json"
                self.minio_client.upload_fileobj(
                    BytesIO(model_bytes),
                    object_name
                )
                
                # Store metadata in PostgreSQL with MinIO reference
                metadata = {
                    "minio_object": object_name,
                    "size": model_size,
                    "stored_in": "minio"
                }
                
                knowledge = Knowledge(
                    knowledge_id=knowledge_id,
                    agent_name=agent_name,
                    knowledge_type=KnowledgeType.ARCHITECTURE_MODEL,
                    content={"minio_reference": object_name, "size": model_size},
                    confidence=1.0,
                    timestamp=datetime.now()
                )
                
                data = {
                    "knowledge_id": knowledge.knowledge_id,
                    "agent_name": knowledge.agent_name,
                    "knowledge_type": knowledge.knowledge_type,
                    "content": knowledge.content,
                    "confidence": knowledge.confidence,
                    "timestamp": knowledge.timestamp,
                    "file_hash": file_hash,
                    "project_path": project_path,
                    "metadata": metadata
                }
                
                return self.knowledge_storage.save(data)
                
            except Exception as e:
                logger.warning(f"Failed to store in MinIO, falling back to PostgreSQL: {e}")
                # Fall back to PostgreSQL
                use_minio = False
        
        if not use_minio:
            # Store small model in PostgreSQL
            knowledge = Knowledge(
                knowledge_id=knowledge_id,
                agent_name=agent_name,
                knowledge_type=KnowledgeType.ARCHITECTURE_MODEL,
                content=model_dict,
                confidence=1.0,
                timestamp=datetime.now()
            )
            
            data = {
                "knowledge_id": knowledge.knowledge_id,
                "agent_name": knowledge.agent_name,
                "knowledge_type": knowledge.knowledge_type,
                "content": knowledge.content,
                "confidence": knowledge.confidence,
                "timestamp": knowledge.timestamp,
                "file_hash": file_hash,
                "project_path": project_path,
                "metadata": {"size": model_size, "stored_in": "postgresql"}
            }
            
            return self.knowledge_storage.save(data)
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get architecture model by ID (implements BaseStorage interface).
        
        Args:
            record_id: Knowledge ID as string
        
        Returns:
            Dictionary with model data, or None if not found
        """
        # Parse knowledge_id to extract project_path and file_hash
        # Format: arch_model:{project_path}:{file_hash}:v{version}
        parts = record_id.split(':')
        if len(parts) < 4 or parts[0] != 'arch_model':
            return None
        
        project_path = parts[1]
        file_hash = parts[2]
        version_str = parts[3]
        version = int(version_str.replace('v', '')) if version_str.startswith('v') else 1
        
        return self.get_architecture_model(project_path, file_hash, version)
    
    def get_architecture_model(
        self,
        project_path: str,
        file_hash: str,
        version: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get architecture model by project_path and file_hash.
        
        Args:
            project_path: Path to the project
            file_hash: Hash of the file
            version: Optional version number (if None, gets latest)
        
        Returns:
            Model dictionary or None if not found
        """
        # Get from knowledge storage
        model_data = self.knowledge_storage.get_architecture_model(
            project_path, file_hash, version
        )
        
        if not model_data:
            return None
        
        # Check if stored in MinIO
        metadata = model_data.get('metadata', {})
        if metadata.get('stored_in') == 'minio' and 'minio_object' in metadata:
            # Retrieve from MinIO
            try:
                object_name = metadata['minio_object']
                model_bytes = self.minio_client.download_fileobj(object_name)
                model_json = model_bytes.read().decode('utf-8')
                model_dict = _json_loads(model_json)
                
                # Replace content with actual model
                model_data['content'] = model_dict
                return model_data
            except Exception as e:
                logger.error(f"Failed to retrieve model from MinIO: {e}", exc_info=True)
                return None
        
        # Already in PostgreSQL, return as-is
        return model_data
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query architecture models with filters (implements BaseStorage interface).
        
        Args:
            filters: Optional filter dictionary
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of model dictionaries
        """
        # Query knowledge storage with architecture model type filter
        query_filters = filters or {}
        query_filters['knowledge_type'] = KnowledgeType.ARCHITECTURE_MODEL.value
        
        results = self.knowledge_storage.query(
            filters=query_filters,
            limit=limit,
            offset=offset
        )
        
        # Load models from MinIO if needed
        for result in results:
            metadata = result.get('metadata', {})
            if metadata.get('stored_in') == 'minio' and 'minio_object' in metadata:
                try:
                    object_name = metadata['minio_object']
                    model_bytes = self.minio_client.download_fileobj(object_name)
                    model_json = model_bytes.read().decode('utf-8')
                    model_dict = _json_loads(model_json)
                    result['content'] = model_dict
                except Exception as e:
                    logger.warning(f"Failed to load model from MinIO: {e}")
        
        return results
    
    def delete(self, record_id: str) -> bool:
        """
        Delete architecture model by ID (implements BaseStorage interface).
        
        Args:
            record_id: Knowledge ID
            
        Returns:
            True if deleted, False if not found
        """
        # Get model to check if it's in MinIO
        model_data = self.get(record_id)
        if model_data:
            metadata = model_data.get('metadata', {})
            if metadata.get('stored_in') == 'minio' and 'minio_object' in metadata:
                # Delete from MinIO
                try:
                    object_name = metadata['minio_object']
                    self.minio_client.delete_file(object_name)
                except Exception as e:
                    logger.warning(f"Failed to delete from MinIO: {e}")
        
        # Delete from knowledge storage
        return self.knowledge_storage.delete(record_id)
    
    def store_large_model(
        self,
        project_path: str,
        model_data: bytes,
        file_hash: str
    ) -> str:
        """
        Store large model directly in MinIO.
        
        Args:
            project_path: Path to the project
            model_data: Model data as bytes
            file_hash: Hash of the file
        
        Returns:
            Knowledge ID
        """
        if not self.use_minio:
            raise StorageError("MinIO is not available")
        
        try:
            # Upload to MinIO
            object_name = f"architecture_models/{project_path}/{file_hash}/model.json"
            self.minio_client.upload_fileobj(
                BytesIO(model_data),
                object_name
            )
            
            # Store metadata in PostgreSQL
            knowledge_id = f"arch_model:{project_path}:{file_hash}:v1"
            metadata = {
                "minio_object": object_name,
                "size": len(model_data),
                "stored_in": "minio"
            }
            
            knowledge = Knowledge(
                knowledge_id=knowledge_id,
                agent_name="ArchitectureModelBuilder",
                knowledge_type=KnowledgeType.ARCHITECTURE_MODEL,
                content={"minio_reference": object_name, "size": len(model_data)},
                confidence=1.0,
                timestamp=datetime.now()
            )
            
            data = {
                "knowledge_id": knowledge.knowledge_id,
                "agent_name": knowledge.agent_name,
                "knowledge_type": knowledge.knowledge_type,
                "content": knowledge.content,
                "confidence": knowledge.confidence,
                "timestamp": knowledge.timestamp,
                "file_hash": file_hash,
                "project_path": project_path,
                "metadata": metadata
            }
            
            return self.knowledge_storage.save(data)
            
        except Exception as e:
            logger.error(f"Failed to store large model: {e}", exc_info=True)
            raise StorageError(f"Failed to store large model: {e}") from e
    
    def get_large_model(self, model_id: str) -> Optional[bytes]:
        """
        Get large model from MinIO.
        
        Args:
            model_id: Knowledge ID
            
        Returns:
            Model data as bytes, or None if not found
        """
        model_data = self.get(model_id)
        if not model_data:
            return None
        
        metadata = model_data.get('metadata', {})
        if metadata.get('stored_in') == 'minio' and 'minio_object' in metadata:
            try:
                object_name = metadata['minio_object']
                model_bytes = self.minio_client.download_fileobj(object_name)
                return model_bytes.read()
            except Exception as e:
                logger.error(f"Failed to get large model: {e}", exc_info=True)
                return None
        
        return None
    
    def check_model_size(self, model: Any) -> int:
        """
        Check size of model in bytes.
        
        Args:
            model: Model object or dict
        
        Returns:
            Size in bytes
        """
        try:
            if isinstance(model, dict):
                model_json = _json_dumps(model)
            else:
                model_json = _json_dumps(model.__dict__ if hasattr(model, '__dict__') else str(model))
            return len(model_json.encode('utf-8'))
        except Exception:
            # Fallback: estimate size
            return sys.getsizeof(str(model))
    
    def should_use_minio(self, size: int) -> bool:
        """
        Determine if model should be stored in MinIO.
        
        Args:
            size: Model size in bytes
        
        Returns:
            True if should use MinIO, False otherwise
        """
        return size >= MINIO_SIZE_THRESHOLD
    
    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        with get_db_connection(self.db_identifier, read_only=False) as conn:
            with transaction(conn):
                yield










