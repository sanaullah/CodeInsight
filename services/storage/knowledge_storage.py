"""
Knowledge Storage Service.

PostgreSQL-only storage implementation for knowledge base.
Provides CRUD operations for knowledge, search functionality, and architecture model support.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager

from services.storage.base_storage import BaseStorage, StorageError, StorageNotFoundError
from services.db_config import get_db_config, CODELUMEN_DATABASE
from services.postgresql_connection_pool import get_db_connection, transaction
from agents.collaboration_models import Knowledge, KnowledgeType
from psycopg2 import sql

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

logger = logging.getLogger(__name__)


class KnowledgeStorage(BaseStorage):
    """
    PostgreSQL storage for knowledge base.
    
    Implements BaseStorage interface and provides additional methods
    for knowledge-specific operations including search and architecture model storage.
    """
    
    def __init__(self):
        """Initialize knowledge storage."""
        self.config = get_db_config()
        self.schema = self.config.postgresql.schema
        self._initialized = False
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
                    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema)
                    ))
                    
                    # Tables should already exist from migration, but we check anyway
                    # The migration framework handles table creation
                    logger.debug(f"Database schema initialized for {self.schema}")
                
                conn.commit()
                self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise StorageError(f"Failed to initialize database: {e}") from e
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save knowledge to PostgreSQL.
        
        Args:
            data: Dictionary containing Knowledge object or knowledge data:
                - knowledge_id: Optional, will be generated if not provided
                - agent_name: Name of agent that created knowledge
                - knowledge_type: KnowledgeType enum or string
                - content: Dictionary containing knowledge content
                - relevance_tags: Optional list of tags
                - confidence: Optional confidence score (0.0-1.0, default 0.5)
                - timestamp: Optional datetime (defaults to now)
                - expires_at: Optional expiration datetime
                - metadata: Optional metadata dictionary
                - file_hash: Optional file hash for architecture models
                - project_path: Optional project path for architecture models
        
        Returns:
            Knowledge ID as string
            
        Raises:
            StorageError: If save fails
            StorageValidationError: If data validation fails
        """
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise StorageError(f"Data validation failed: {error_msg}")
        
        try:
            # Handle both Knowledge object and dict
            if isinstance(data, Knowledge):
                knowledge = data
            else:
                # Convert dict to Knowledge if needed
                knowledge = self._dict_to_knowledge(data)
            
            # Generate ID if not provided
            if not knowledge.knowledge_id:
                knowledge.knowledge_id = str(uuid.uuid4())
            
            # Serialize content and metadata
            content_json = _json_dumps(knowledge.content)
            metadata = getattr(knowledge, 'metadata', {}) or {}
            metadata_json = _json_dumps(metadata) if metadata else None
            
            # Extract optional fields
            file_hash = data.get('file_hash') if isinstance(data, dict) else getattr(knowledge, 'file_hash', None)
            project_path = data.get('project_path') if isinstance(data, dict) else getattr(knowledge, 'project_path', None)
            
            # Use timestamp from knowledge or default to now
            timestamp = knowledge.timestamp if knowledge.timestamp else datetime.now()
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Insert or update knowledge
                    query_template = sql.SQL("""
                        INSERT INTO {schema}.knowledge_base (
                            knowledge_id, agent_name, knowledge_type, content_json,
                            relevance_tags, confidence, created_at, expires_at,
                            metadata_json, access_count, last_accessed, file_hash, project_path
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (knowledge_id) DO UPDATE SET
                            agent_name = EXCLUDED.agent_name,
                            knowledge_type = EXCLUDED.knowledge_type,
                            content_json = EXCLUDED.content_json,
                            relevance_tags = EXCLUDED.relevance_tags,
                            confidence = EXCLUDED.confidence,
                            expires_at = EXCLUDED.expires_at,
                            metadata_json = EXCLUDED.metadata_json,
                            file_hash = EXCLUDED.file_hash,
                            project_path = EXCLUDED.project_path,
                            updated_at = CURRENT_TIMESTAMP
                    """).format(schema=sql.Identifier(self.schema))
                    
                    cursor.execute(query_template, (
                        knowledge.knowledge_id,
                        knowledge.agent_name,
                        knowledge.knowledge_type.value if isinstance(knowledge.knowledge_type, KnowledgeType) else str(knowledge.knowledge_type),
                        content_json,
                        knowledge.relevance_tags or [],
                        knowledge.confidence,
                        timestamp,
                        knowledge.expires_at,
                        metadata_json,
                        0,  # Initial access count
                        None,  # No access yet
                        file_hash,
                        project_path
                    ))
                
                conn.commit()
            
            logger.info(f"Saved knowledge: {knowledge.knowledge_id}")
            return knowledge.knowledge_id
            
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}", exc_info=True)
            raise StorageError(f"Failed to save knowledge: {e}") from e
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get knowledge by ID.
        
        Args:
            record_id: Knowledge ID as string
        
        Returns:
            Dictionary with knowledge data including parsed JSON fields, or None if not found
            
        Raises:
            StorageError: If get fails
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    query_template = sql.SQL("""
                        SELECT * FROM {schema}.knowledge_base 
                        WHERE knowledge_id = %s
                        AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    """).format(schema=sql.Identifier(self.schema))
                    
                    cursor.execute(query_template, (record_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    knowledge = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    knowledge = self._parse_json_fields(knowledge)
                    
                    # Increment access count
                    self.increment_access_count(record_id)
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Error getting knowledge {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to get knowledge: {e}") from e
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query knowledge with filters.
        
        Args:
            filters: Optional filter dictionary with keys:
                - agent_name: Filter by agent name (exact match)
                - knowledge_type: Filter by knowledge type (exact match)
                - min_confidence: Filter by minimum confidence
                - date_from: Filter by created_at >= date_from
                - date_to: Filter by created_at <= date_to
                - has_tags: Filter by relevance_tags containing any of these tags
            limit: Optional limit on number of results
            offset: Optional offset for pagination
        
        Returns:
            List of knowledge dictionaries
            
        Raises:
            StorageError: If query fails
        """
        try:
            filters = filters or {}
            
            # Build WHERE clause
            where_clauses = ["(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"]  # Exclude expired
            params = []
            
            if "agent_name" in filters and filters["agent_name"]:
                where_clauses.append("agent_name = %s")
                params.append(filters["agent_name"])
            
            if "knowledge_type" in filters and filters["knowledge_type"]:
                knowledge_type = filters["knowledge_type"]
                if isinstance(knowledge_type, KnowledgeType):
                    knowledge_type = knowledge_type.value
                where_clauses.append("knowledge_type = %s")
                params.append(knowledge_type)
            
            if "min_confidence" in filters and filters["min_confidence"] is not None:
                where_clauses.append("confidence >= %s")
                params.append(filters["min_confidence"])
            
            if "date_from" in filters and filters["date_from"]:
                where_clauses.append("created_at >= %s")
                params.append(filters["date_from"])
            
            if "date_to" in filters and filters["date_to"]:
                where_clauses.append("created_at <= %s")
                params.append(filters["date_to"])
            
            if "has_tags" in filters and filters["has_tags"]:
                tags = filters["has_tags"]
                if isinstance(tags, str):
                    tags = [tags]
                where_clauses.append("relevance_tags && %s")  # Array overlap operator
                params.append(tags)
            
            where_clause = sql.SQL(" AND ").join(map(sql.SQL, where_clauses)) if where_clauses else sql.SQL("1=1")
            
            # Build query
            query_template = sql.SQL("""
                SELECT * FROM {schema}.knowledge_base 
                WHERE {where}
                ORDER BY confidence DESC, created_at DESC
            """).format(
                schema=sql.Identifier(self.schema),
                where=where_clause
            )
            
            if limit:
                query_template += sql.SQL(" LIMIT {}").format(sql.Literal(limit))
                if offset:
                    query_template += sql.SQL(" OFFSET {}").format(sql.Literal(offset))
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query_template, params)
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    knowledge_list = []
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in rows:
                        knowledge = dict(zip(columns, row))
                        knowledge = self._parse_json_fields(knowledge)
                        knowledge_list.append(knowledge)
            
            return knowledge_list
            
        except Exception as e:
            logger.error(f"Error querying knowledge: {e}", exc_info=True)
            raise StorageError(f"Failed to query knowledge: {e}") from e
    
    def delete(self, record_id: str) -> bool:
        """
        Delete knowledge by ID.
        
        Args:
            record_id: Knowledge ID as string
        
        Returns:
            True if deleted, False if not found
            
        Raises:
            StorageError: If delete fails
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    query_template = sql.SQL("""
                        DELETE FROM {schema}.knowledge_base 
                        WHERE knowledge_id = %s
                    """).format(schema=sql.Identifier(self.schema))
                    
                    cursor.execute(query_template, (record_id,))
                    deleted = cursor.rowcount > 0
                
                conn.commit()
            
            if deleted:
                logger.info(f"Deleted knowledge {record_id}")
            else:
                logger.warning(f"Knowledge {record_id} not found for deletion")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting knowledge {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete knowledge: {e}") from e
    
    @contextmanager
    def transaction(self):
        """
        Transaction context manager.
        
        Usage:
            with storage.transaction():
                storage.save(data1)
                storage.save(data2)
        """
        with get_db_connection(self.db_identifier, read_only=False) as conn:
            with transaction(conn):
                yield
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate data before saving.
        
        Args:
            data: Data dictionary or Knowledge object to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle Knowledge object
        if isinstance(data, Knowledge):
            if not data.agent_name:
                return False, "Agent name is required"
            if not data.knowledge_type:
                return False, "Knowledge type is required"
            if not data.content:
                return False, "Content is required"
            if data.confidence is not None and not (0.0 <= data.confidence <= 1.0):
                return False, "Confidence must be between 0.0 and 1.0"
            return True, None
        
        # Handle dict
        if not isinstance(data, dict):
            return False, "Data must be a dictionary or Knowledge object"
        
        required_fields = ["agent_name", "knowledge_type", "content"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        if "confidence" in data and data["confidence"] is not None:
            if not (0.0 <= data["confidence"] <= 1.0):
                return False, "Confidence must be between 0.0 and 1.0"
        
        return True, None
    
    # Additional methods for knowledge-specific operations
    
    def search_knowledge(self, query: str, 
                        knowledge_types: Optional[List[KnowledgeType]] = None,
                        agents: Optional[List[str]] = None,
                        date_range: Optional[Tuple[datetime, datetime]] = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """
        Advanced search with full-text search and multiple filters.
        
        Args:
            query: Search query string (for full-text search)
            knowledge_types: Optional list of knowledge types to filter
            agents: Optional list of agent names to filter
            date_range: Optional tuple of (start_date, end_date)
            limit: Maximum number of results
        
        Returns:
            List of knowledge dictionaries
        """
        try:
            where_clauses = ["(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"]
            params = []
            
            # Full-text search
            if query:
                # Use PostgreSQL full-text search
                where_clauses.append("to_tsvector('english', content_json::text) @@ plainto_tsquery('english', %s)")
                params.append(query)
            
            # Knowledge type filter
            if knowledge_types:
                type_values = [kt.value if isinstance(kt, KnowledgeType) else str(kt) for kt in knowledge_types]
                where_clauses.append("knowledge_type = ANY(%s)")
                params.append(type_values)
            
            # Agent filter
            if agents:
                where_clauses.append("agent_name = ANY(%s)")
                params.append(agents)
            
            # Date range filter
            if date_range:
                start_date, end_date = date_range
                if start_date:
                    where_clauses.append("created_at >= %s")
                    params.append(start_date)
                if end_date:
                    where_clauses.append("created_at <= %s")
                    params.append(end_date)
            
            where_clause = sql.SQL(" AND ").join(map(sql.SQL, where_clauses)) if where_clauses else sql.SQL("1=1")
            
            # Build query with relevance scoring
            search_query = sql.SQL("""
                SELECT *, 
                       ts_rank(to_tsvector('english', content_json::text), plainto_tsquery('english', %s)) as rank
                FROM {schema}.knowledge_base 
                WHERE {where}
                ORDER BY rank DESC, confidence DESC, created_at DESC
                LIMIT %s
            """).format(
                schema=sql.Identifier(self.schema),
                where=where_clause
            )
            
            # Add query to params for ranking (even if empty, for consistency)
            search_params = [query if query else '', *params, limit]
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(search_query, search_params)
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    knowledge_list = []
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in rows:
                        knowledge = dict(zip(columns, row))
                        knowledge = self._parse_json_fields(knowledge)
                        knowledge_list.append(knowledge)
            
            return knowledge_list
            
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}", exc_info=True)
            raise StorageError(f"Failed to search knowledge: {e}") from e
    
    def store_architecture_model(self, project_path: str, model: Any, 
                                  file_hash: str, version: int = 1,
                                  agent_name: str = "ArchitectureModelBuilder") -> str:
        """
        Store architecture model in knowledge base.
        
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
        
        # Create Knowledge object
        knowledge = Knowledge(
            knowledge_id=f"arch_model:{project_path}:{file_hash}:v{version}",
            agent_name=agent_name,
            knowledge_type=KnowledgeType.ARCHITECTURE_MODEL,
            content=model_dict,
            confidence=1.0,  # Architecture models have high confidence
            timestamp=datetime.now()
        )
        
        # Add file_hash and project_path to data
        data = {
            "knowledge_id": knowledge.knowledge_id,
            "agent_name": knowledge.agent_name,
            "knowledge_type": knowledge.knowledge_type,
            "content": knowledge.content,
            "confidence": knowledge.confidence,
            "timestamp": knowledge.timestamp,
            "file_hash": file_hash,
            "project_path": project_path
        }
        
        return self.save(data)
    
    def get_architecture_model(self, project_path: str, file_hash: str,
                               version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get architecture model by project_path and file_hash.
        
        Args:
            project_path: Path to the project
            file_hash: Hash of the file
            version: Optional version number (if None, gets latest)
        
        Returns:
            Knowledge dictionary or None if not found
        """
        try:
            where_clauses = [
                "knowledge_type = %s",
                "project_path = %s",
                "file_hash = %s",
                "(expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
            ]
            params = [KnowledgeType.ARCHITECTURE_MODEL.value, project_path, file_hash]
            
            if version:
                knowledge_id = f"arch_model:{project_path}:{file_hash}:v{version}"
                where_clauses.append("knowledge_id = %s")
                params.append(knowledge_id)
            
            where_clause = sql.SQL(" AND ").join(map(sql.SQL, where_clauses)) if where_clauses else sql.SQL("1=1")
            order_by = sql.SQL("ORDER BY created_at DESC LIMIT 1") if not version else sql.SQL("")
            
            query = sql.SQL("""
                SELECT * FROM {schema}.knowledge_base 
                WHERE {where}
                {order}
            """).format(
                schema=sql.Identifier(self.schema),
                where=where_clause,
                order=order_by
            )
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    knowledge = dict(zip(columns, row))
                    knowledge = self._parse_json_fields(knowledge)
                    
                    # Increment access count
                    self.increment_access_count(knowledge['knowledge_id'])
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Error getting architecture model: {e}", exc_info=True)
            raise StorageError(f"Failed to get architecture model: {e}") from e
    
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update existing knowledge.
        
        Args:
            knowledge_id: Knowledge ID
            updates: Dictionary of fields to update
        
        Returns:
            True if updated, False if not found
        """
        try:
            if not updates:
                return False
            
            set_clauses = []
            params = []
            
            if "content" in updates:
                set_clauses.append("content_json = %s")
                params.append(_json_dumps(updates["content"]))
            
            if "confidence" in updates:
                set_clauses.append("confidence = %s")
                params.append(updates["confidence"])
            
            if "relevance_tags" in updates:
                set_clauses.append("relevance_tags = %s")
                params.append(updates["relevance_tags"])
            
            if "expires_at" in updates:
                set_clauses.append("expires_at = %s")
                params.append(updates["expires_at"])
            
            if "metadata" in updates:
                set_clauses.append("metadata_json = %s")
                params.append(_json_dumps(updates["metadata"]))
            
            if not set_clauses:
                return False
            
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(knowledge_id)
            
            # Convert set clauses to sql.SQL objects
            set_clauses_sql = [sql.SQL(clause) for clause in set_clauses]
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    query_template = sql.SQL("""
                        UPDATE {schema}.knowledge_base 
                        SET {updates}
                        WHERE knowledge_id = %s
                    """).format(
                        schema=sql.Identifier(self.schema),
                        updates=sql.SQL(", ").join(set_clauses_sql)
                    )
                    cursor.execute(query_template, params)
                    updated = cursor.rowcount > 0
                
                conn.commit()
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating knowledge {knowledge_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to update knowledge: {e}") from e
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    # Total count
                    query_total = sql.SQL("""
                        SELECT COUNT(*) FROM {schema}.knowledge_base
                        WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP
                    """).format(schema=sql.Identifier(self.schema))
                    cursor.execute(query_total)
                    total = cursor.fetchone()[0]
                    
                    # Count by type
                    query_by_type = sql.SQL("""
                        SELECT knowledge_type, COUNT(*) 
                        FROM {schema}.knowledge_base
                        WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP
                        GROUP BY knowledge_type
                    """).format(schema=sql.Identifier(self.schema))
                    cursor.execute(query_by_type)
                    by_type = dict(cursor.fetchall())
                    
                    # Count by agent
                    query_by_agent = sql.SQL("""
                        SELECT agent_name, COUNT(*) 
                        FROM {schema}.knowledge_base
                        WHERE expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP
                        GROUP BY agent_name
                        ORDER BY COUNT(*) DESC
                        LIMIT 10
                    """).format(schema=sql.Identifier(self.schema))
                    cursor.execute(query_by_agent)
                    by_agent = dict(cursor.fetchall())
                    
                    # Expired count
                    query_expired = sql.SQL("""
                        SELECT COUNT(*) FROM {schema}.knowledge_base
                        WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
                    """).format(schema=sql.Identifier(self.schema))
                    cursor.execute(query_expired)
                    expired = cursor.fetchone()[0]
            
            return {
                "total": total,
                "by_type": by_type,
                "by_agent": by_agent,
                "expired": expired
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge statistics: {e}", exc_info=True)
            raise StorageError(f"Failed to get statistics: {e}") from e
    
    def cleanup_expired_knowledge(self) -> int:
        """
        Remove expired knowledge items.
        
        Returns:
            Number of items deleted
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    query_delete = sql.SQL("""
                        DELETE FROM {schema}.knowledge_base
                        WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
                    """).format(schema=sql.Identifier(self.schema))
                    cursor.execute(query_delete)
                    deleted = cursor.rowcount
                
                conn.commit()
            
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired knowledge items")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up expired knowledge: {e}", exc_info=True)
            raise StorageError(f"Failed to cleanup expired knowledge: {e}") from e
    
    def increment_access_count(self, knowledge_id: str) -> None:
        """
        Increment access count for knowledge.
        
        Args:
            knowledge_id: Knowledge ID
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    query_update = sql.SQL("""
                        UPDATE {schema}.knowledge_base 
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE knowledge_id = %s
                    """).format(schema=sql.Identifier(self.schema))
                    cursor.execute(query_update, (knowledge_id,))
                
                conn.commit()
        except Exception as e:
            logger.warning(f"Error incrementing access count for {knowledge_id}: {e}")
            # Don't raise - this is a non-critical operation
    
    # Helper methods
    
    def _parse_json_fields(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON fields from database row."""
        if "content_json" in knowledge and knowledge["content_json"]:
            if isinstance(knowledge["content_json"], str):
                knowledge["content"] = _json_loads(knowledge["content_json"])
            else:
                knowledge["content"] = knowledge["content_json"]
            # Keep content_json for backward compatibility
        else:
            knowledge["content"] = {}
        
        if "metadata_json" in knowledge and knowledge["metadata_json"]:
            if isinstance(knowledge["metadata_json"], str):
                knowledge["metadata"] = _json_loads(knowledge["metadata_json"])
            else:
                knowledge["metadata"] = knowledge["metadata_json"]
        else:
            knowledge["metadata"] = {}
        
        # Convert knowledge_type string to enum if needed
        if "knowledge_type" in knowledge and isinstance(knowledge["knowledge_type"], str):
            try:
                knowledge["knowledge_type"] = KnowledgeType(knowledge["knowledge_type"])
            except ValueError:
                pass  # Keep as string if not a valid enum value
        
        return knowledge
    
    def _dict_to_knowledge(self, data: Dict[str, Any]) -> Knowledge:
        """Convert dict to Knowledge object."""
        # Extract knowledge_type
        knowledge_type = data.get("knowledge_type")
        if isinstance(knowledge_type, str):
            try:
                knowledge_type = KnowledgeType(knowledge_type)
            except ValueError:
                raise ValueError(f"Invalid knowledge_type: {knowledge_type}")
        elif not isinstance(knowledge_type, KnowledgeType):
            raise ValueError("knowledge_type must be KnowledgeType enum or string")
        
        return Knowledge(
            knowledge_id=data.get("knowledge_id"),
            agent_name=data["agent_name"],
            knowledge_type=knowledge_type,
            content=data["content"],
            relevance_tags=data.get("relevance_tags", []),
            confidence=data.get("confidence", 0.5),
            timestamp=data.get("timestamp", datetime.now()),
            expires_at=data.get("expires_at")
        )










