"""
Prompt Storage Service.

PostgreSQL-only storage implementation for prompts.
Provides CRUD operations for prompts with full-text search support.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager

from services.storage.base_storage import BaseStorage, StorageError, StorageNotFoundError
from services.db_config import get_db_config, CODELUMEN_DATABASE
from services.postgresql_connection_pool import get_db_connection, transaction

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


class PromptStorage(BaseStorage):
    """
    PostgreSQL storage for prompts.
    
    Implements BaseStorage interface and provides additional methods
    for prompt-specific operations including full-text search.
    """
    
    def __init__(self):
        """Initialize prompt storage."""
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
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                    
                    # Tables should already exist from migration, but we check anyway
                    logger.debug(f"Database schema initialized for {self.schema}")
                
                conn.commit()
                self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise StorageError(f"Failed to initialize database: {e}") from e
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save prompt to PostgreSQL (implements BaseStorage interface).
        
        Args:
            data: Dictionary containing prompt data:
                - prompt_id: Optional, will be generated if not provided
                - prompt_name: Prompt name (required)
                - prompt_content: Prompt content (required)
                - metadata_json: Optional metadata dictionary
                - role: Optional role name
                - architecture_hash: Optional architecture hash
                - usage_count: Optional usage count (default 0)
        
        Returns:
            Prompt ID as string
        """
        # Validate required fields
        if 'prompt_name' not in data or 'prompt_content' not in data:
            raise StorageError("prompt_name and prompt_content are required")
        
        # Generate prompt_id if not provided
        if 'prompt_id' not in data or not data['prompt_id']:
            data['prompt_id'] = str(uuid.uuid4())
        
        return self.save_prompt(
            prompt_id=data['prompt_id'],
            name=data['prompt_name'],
            content=data['prompt_content'],
            metadata=data.get('metadata_json'),
            role=data.get('role'),
            architecture_hash=data.get('architecture_hash'),
            usage_count=data.get('usage_count', 0)
        )
    
    def save_prompt(
        self,
        prompt_id: str,
        name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        architecture_hash: Optional[str] = None,
        usage_count: int = 0
    ) -> str:
        """
        Save or update a prompt.
        
        Args:
            prompt_id: Prompt ID
            name: Prompt name
            content: Prompt content
            metadata: Optional metadata dictionary
            role: Optional role name
            architecture_hash: Optional architecture hash
            usage_count: Usage count (default 0)
            
        Returns:
            Prompt ID
        """
        try:
            # Serialize metadata
            metadata_json = _json_dumps(metadata) if metadata else None
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Check if prompt exists
                    cursor.execute(f"""
                        SELECT prompt_id FROM {self.schema}.prompts 
                        WHERE prompt_id = %s
                    """, (prompt_id,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing prompt
                        cursor.execute(f"""
                            UPDATE {self.schema}.prompts SET
                                prompt_name = %s,
                                prompt_content = %s,
                                metadata_json = %s,
                                role = %s,
                                architecture_hash = %s,
                                usage_count = %s,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE prompt_id = %s
                        """, (
                            name,
                            content,
                            metadata_json,
                            role,
                            architecture_hash,
                            usage_count,
                            prompt_id
                        ))
                        logger.debug(f"Updated prompt: {prompt_id}")
                    else:
                        # Insert new prompt
                        cursor.execute(f"""
                            INSERT INTO {self.schema}.prompts (
                                prompt_id, prompt_name, prompt_content, metadata_json,
                                role, architecture_hash, usage_count
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            prompt_id,
                            name,
                            content,
                            metadata_json,
                            role,
                            architecture_hash,
                            usage_count
                        ))
                        logger.debug(f"Created new prompt: {prompt_id}")
                
                conn.commit()
            
            return prompt_id
            
        except Exception as e:
            logger.error(f"Error saving prompt: {e}", exc_info=True)
            raise StorageError(f"Failed to save prompt: {e}") from e
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get prompt by ID (implements BaseStorage interface).
        
        Args:
            record_id: Prompt ID as string
        
        Returns:
            Dictionary with prompt data, or None if not found
        """
        return self.get_prompt(record_id)
    
    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get prompt by ID.
        
        Args:
            prompt_id: Prompt ID
            
        Returns:
            Dictionary with prompt data, or None if not found
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT * FROM {self.schema}.prompts 
                        WHERE prompt_id = %s
                    """, (prompt_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    prompt_dict = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    prompt_dict = self._parse_json_fields(prompt_dict)
            
            return prompt_dict
            
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to get prompt: {e}") from e
    
    def get_prompt_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get prompt by name.
        
        Args:
            name: Prompt name
            
        Returns:
            Dictionary with prompt data, or None if not found
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT * FROM {self.schema}.prompts 
                        WHERE prompt_name = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (name,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    prompt_dict = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    prompt_dict = self._parse_json_fields(prompt_dict)
            
            return prompt_dict
            
        except Exception as e:
            logger.error(f"Error getting prompt by name {name}: {e}", exc_info=True)
            raise StorageError(f"Failed to get prompt: {e}") from e
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query prompts with filters (implements BaseStorage interface).
        
        Args:
            filters: Optional filter dictionary
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of prompt dictionaries
        """
        return self.get_all_prompts(limit=limit, offset=offset or 0)
    
    def get_all_prompts(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all prompts.
        
        Args:
            limit: Maximum number of prompts
            offset: Offset for pagination
            
        Returns:
            List of prompt dictionaries
        """
        try:
            prompts = []
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    query = f"""
                        SELECT * FROM {self.schema}.prompts 
                        ORDER BY created_at DESC
                    """
                    
                    params = []
                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)
                    if offset:
                        query += " OFFSET %s"
                        params.append(offset)
                    
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Get column names while cursor is still open
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    
                    # Convert rows to dictionaries
                    for row in rows:
                        prompt_dict = dict(zip(columns, row))
                        
                        # Parse JSON fields
                        prompt_dict = self._parse_json_fields(prompt_dict)
                        prompts.append(prompt_dict)
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error getting all prompts: {e}", exc_info=True)
            return []
    
    def search_prompts(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        role: Optional[str] = None,
        architecture_hash: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search prompts using full-text search.
        
        Args:
            query: Search query text
            tags: Optional list of tags to filter by
            role: Optional role to filter by
            architecture_hash: Optional architecture hash to filter by
            limit: Maximum number of results
            
        Returns:
            List of prompt dictionaries
        """
        try:
            conditions = []
            params = []
            
            # Full-text search
            if query:
                conditions.append(f"to_tsvector('english', prompt_content) @@ plainto_tsquery('english', %s)")
                params.append(query)
            
            # Filter by role
            if role:
                conditions.append("role = %s")
                params.append(role)
            
            # Filter by architecture_hash
            if architecture_hash:
                conditions.append("architecture_hash = %s")
                params.append(architecture_hash)
            
            # Filter by tags (if tags are stored in metadata)
            # Note: This is a simple implementation. For better tag search,
            # consider storing tags in a separate table or using JSONB operators
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            search_query = f"""
                SELECT *, ts_rank(to_tsvector('english', prompt_content), plainto_tsquery('english', %s)) as rank
                FROM {self.schema}.prompts
                WHERE {where_clause}
                ORDER BY rank DESC, created_at DESC
                LIMIT %s
            """
            
            # Add query to params for ranking
            if query:
                params.insert(0, query)
            params.append(limit)
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(search_query, params)
                    rows = cursor.fetchall()
            
            prompts = []
            for row in rows:
                # Convert row to dict
                columns = [desc[0] for desc in cursor.description]
                prompt_dict = dict(zip(columns, row))
                
                # Remove rank from result
                if 'rank' in prompt_dict:
                    del prompt_dict['rank']
                
                # Parse JSON fields
                prompt_dict = self._parse_json_fields(prompt_dict)
                prompts.append(prompt_dict)
            
            return prompts
            
        except Exception as e:
            logger.error(f"Error searching prompts: {e}", exc_info=True)
            return []
    
    def update_prompt(self, prompt_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update existing prompt.
        
        Args:
            prompt_id: Prompt ID
            updates: Dictionary with fields to update
            
        Returns:
            True if updated, False if not found
        """
        try:
            # Build update query dynamically
            set_clauses = []
            params = []
            
            if 'prompt_name' in updates:
                set_clauses.append("prompt_name = %s")
                params.append(updates['prompt_name'])
            
            if 'prompt_content' in updates:
                set_clauses.append("prompt_content = %s")
                params.append(updates['prompt_content'])
            
            if 'metadata_json' in updates:
                metadata_json = _json_dumps(updates['metadata_json']) if updates['metadata_json'] else None
                set_clauses.append("metadata_json = %s")
                params.append(metadata_json)
            
            if 'role' in updates:
                set_clauses.append("role = %s")
                params.append(updates['role'])
            
            if 'architecture_hash' in updates:
                set_clauses.append("architecture_hash = %s")
                params.append(updates['architecture_hash'])
            
            if 'usage_count' in updates:
                set_clauses.append("usage_count = %s")
                params.append(updates['usage_count'])
            
            if not set_clauses:
                return False
            
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(prompt_id)
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        UPDATE {self.schema}.prompts 
                        SET {', '.join(set_clauses)}
                        WHERE prompt_id = %s
                    """, params)
                    
                    updated = cursor.rowcount > 0
                
                conn.commit()
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to update prompt: {e}") from e
    
    def delete(self, record_id: str) -> bool:
        """
        Delete prompt by ID (implements BaseStorage interface).
        
        Args:
            record_id: Prompt ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        DELETE FROM {self.schema}.prompts 
                        WHERE prompt_id = %s
                    """, (record_id,))
                    
                    deleted = cursor.rowcount > 0
                
                conn.commit()
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting prompt {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete prompt: {e}") from e
    
    def get_prompt_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prompts.
        
        Returns:
            Dictionary with statistics
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    # Total prompts
                    cursor.execute(f"SELECT COUNT(*) FROM {self.schema}.prompts")
                    total_prompts = cursor.fetchone()[0] or 0
                    
                    # Prompts by role
                    cursor.execute(f"""
                        SELECT role, COUNT(*) as count
                        FROM {self.schema}.prompts
                        WHERE role IS NOT NULL
                        GROUP BY role
                    """)
                    prompts_by_role = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Total usage count
                    cursor.execute(f"""
                        SELECT SUM(usage_count) FROM {self.schema}.prompts
                    """)
                    total_usage = cursor.fetchone()[0] or 0
                    
                    # Most used prompts
                    cursor.execute(f"""
                        SELECT prompt_name, usage_count
                        FROM {self.schema}.prompts
                        ORDER BY usage_count DESC
                        LIMIT 10
                    """)
                    top_prompts = [{"name": row[0], "usage_count": row[1]} for row in cursor.fetchall()]
            
            return {
                "total_prompts": total_prompts,
                "prompts_by_role": prompts_by_role,
                "total_usage": total_usage,
                "top_prompts": top_prompts
            }
            
        except Exception as e:
            logger.error(f"Error getting prompt statistics: {e}", exc_info=True)
            return {}
    
    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        with get_db_connection(self.db_identifier, read_only=False) as conn:
            with transaction(conn):
                yield
    
    # Helper methods
    
    def _parse_json_fields(self, prompt_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON fields in prompt dictionary."""
        if 'metadata_json' in prompt_dict and prompt_dict['metadata_json']:
            if isinstance(prompt_dict['metadata_json'], str):
                prompt_dict['metadata'] = _json_loads(prompt_dict['metadata_json'])
            else:
                prompt_dict['metadata'] = prompt_dict['metadata_json']
        else:
            prompt_dict['metadata'] = {}
        
        return prompt_dict

