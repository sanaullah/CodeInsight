"""
Settings Storage Service.

PostgreSQL storage implementation for user settings.
Provides CRUD operations for user settings with JSON value encoding/decoding.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager

from services.storage.base_storage import BaseStorage, StorageError
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


class SettingsStorage(BaseStorage):
    """
    PostgreSQL storage for user settings.
    
    Implements BaseStorage interface with settings-specific methods.
    Settings use composite key (user_id, setting_key).
    """
    
    def __init__(self):
        """Initialize settings storage."""
        self.config = get_db_config()
        self.schema = self.config.postgresql.schema
        self._initialized = False
        super().__init__(db_identifier=CODELUMEN_DATABASE)
    
    def _init_database(self) -> None:
        """
        Initialize database schema.
        
        Creates schema and ensures table exists.
        This is called automatically by BaseStorage.__init__.
        """
        if self._initialized:
            return
        
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Ensure schema exists
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                    
                    # Table should already exist from migration, but we check anyway
                    logger.debug(f"Database schema initialized for {self.schema}")
                
                conn.commit()
                self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise StorageError(f"Failed to initialize database: {e}") from e
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save setting to PostgreSQL.
        
        Args:
            data: Dictionary with keys: user_id, setting_key, setting_value
        
        Returns:
            Composite key string: "{user_id}:{setting_key}"
            
        Raises:
            StorageError: If save fails
            StorageValidationError: If data validation fails
        """
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise StorageError(f"Data validation failed: {error_msg}")
        
        try:
            user_id = data["user_id"]
            setting_key = data["setting_key"]
            setting_value = data["setting_value"]
            
            # Encode value to JSON string
            value_str = _json_dumps(setting_value)
            updated_at = datetime.now()
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # UPSERT: Insert or update
                    cursor.execute(f"""
                        INSERT INTO {self.schema}.user_settings (
                            user_id, setting_key, setting_value, updated_at
                        ) VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id, setting_key) DO UPDATE SET
                            setting_value = EXCLUDED.setting_value,
                            updated_at = EXCLUDED.updated_at
                    """, (user_id, setting_key, value_str, updated_at))
                
                conn.commit()
            
            composite_key = f"{user_id}:{setting_key}"
            logger.debug(f"Saved setting: {composite_key}")
            return composite_key
            
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error saving setting: {e}", exc_info=True)
            raise StorageError(f"Failed to save setting: {e}") from e
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get setting by composite key.
        
        Args:
            record_id: Composite key in format "{user_id}:{setting_key}"
        
        Returns:
            Dictionary with setting data or None if not found
            
        Raises:
            StorageError: If get fails
        """
        try:
            # Parse composite key
            parts = record_id.split(":", 1)
            if len(parts) != 2:
                raise StorageError(f"Invalid record_id format: {record_id}. Expected 'user_id:setting_key'")
            
            user_id, setting_key = parts
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT user_id, setting_key, setting_value, updated_at
                        FROM {self.schema}.user_settings 
                        WHERE user_id = %s AND setting_key = %s
                    """, (user_id, setting_key))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    setting = dict(zip(columns, row))
                    
                    # Decode JSON value
                    if setting.get("setting_value"):
                        try:
                            setting["setting_value"] = _json_loads(setting["setting_value"])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to decode setting value: {e}")
                            setting["setting_value"] = setting["setting_value"]
            
            return setting
            
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error getting setting {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to get setting: {e}") from e
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query settings with filters.
        
        Args:
            filters: Optional filter dictionary with keys:
                - user_id: Filter by user ID (exact match)
                - setting_key: Filter by setting key (exact match)
            limit: Optional limit on number of results
            offset: Optional offset for pagination
        
        Returns:
            List of setting dictionaries
            
        Raises:
            StorageError: If query fails
        """
        try:
            filters = filters or {}
            
            # Build WHERE clause
            where_clauses = []
            params = []
            
            if "user_id" in filters and filters["user_id"]:
                where_clauses.append("user_id = %s")
                params.append(filters["user_id"])
            
            if "setting_key" in filters and filters["setting_key"]:
                where_clauses.append("setting_key = %s")
                params.append(filters["setting_key"])
            
            # Build query
            query = f"SELECT user_id, setting_key, setting_value, updated_at FROM {self.schema}.user_settings"
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY updated_at DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    settings = []
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in rows:
                        setting = dict(zip(columns, row))
                        # Decode JSON value
                        if setting.get("setting_value"):
                            try:
                                setting["setting_value"] = _json_loads(setting["setting_value"])
                            except (json.JSONDecodeError, TypeError) as e:
                                logger.warning(f"Failed to decode setting value: {e}")
                                # Keep as string if decode fails
                        
                        settings.append(setting)
            
            return settings
            
        except Exception as e:
            logger.error(f"Error querying settings: {e}", exc_info=True)
            raise StorageError(f"Failed to query settings: {e}") from e
    
    def delete(self, record_id: str) -> bool:
        """
        Delete setting by composite key.
        
        Args:
            record_id: Composite key in format "{user_id}:{setting_key}"
        
        Returns:
            True if deleted, False if not found
            
        Raises:
            StorageError: If delete fails
        """
        try:
            # Parse composite key
            parts = record_id.split(":", 1)
            if len(parts) != 2:
                raise StorageError(f"Invalid record_id format: {record_id}. Expected 'user_id:setting_key'")
            
            user_id, setting_key = parts
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        DELETE FROM {self.schema}.user_settings 
                        WHERE user_id = %s AND setting_key = %s
                    """, (user_id, setting_key))
                    deleted = cursor.rowcount > 0
                
                conn.commit()
            
            if deleted:
                logger.debug(f"Deleted setting {record_id}")
            else:
                logger.warning(f"Setting {record_id} not found for deletion")
            
            return deleted
            
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error deleting setting {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete setting: {e}") from e
    
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
            data: Data dictionary to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        required_fields = ["user_id", "setting_key", "setting_value"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate user_id and setting_key are not empty
        if not data["user_id"] or not isinstance(data["user_id"], str):
            return False, "user_id must be a non-empty string"
        
        if not data["setting_key"] or not isinstance(data["setting_key"], str):
            return False, "setting_key must be a non-empty string"
        
        return True, None
    
    # Settings-specific convenience methods
    
    def get_setting(self, user_id: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get a setting value for a user.
        
        Args:
            user_id: User identifier
            key: Setting key
            default: Default value if setting doesn't exist
        
        Returns:
            Setting value or default
        """
        try:
            record_id = f"{user_id}:{key}"
            setting = self.get(record_id)
            if setting:
                return setting.get("setting_value")
            return default
        except Exception as e:
            logger.error(f"Error getting setting {key} for user {user_id}: {e}", exc_info=True)
            return default
    
    def set_setting(self, user_id: str, key: str, value: Any) -> bool:
        """
        Set or update a setting for a user (UPSERT).
        
        Args:
            user_id: User identifier
            key: Setting key
            value: Setting value (will be JSON-encoded)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "user_id": user_id,
                "setting_key": key,
                "setting_value": value
            }
            self.save(data)
            logger.debug(f"Set setting {key} for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error setting {key} for user {user_id}: {e}", exc_info=True)
            return False
    
    def get_all_settings(self, user_id: str) -> Dict[str, Any]:
        """
        Get all settings for a user as a dictionary.
        
        Args:
            user_id: User identifier
        
        Returns:
            Dictionary of all settings for the user
        """
        try:
            filters = {"user_id": user_id}
            settings = self.query(filters=filters)
            
            # Convert to dictionary: key -> value
            result = {}
            for setting in settings:
                result[setting["setting_key"]] = setting["setting_value"]
            
            logger.debug(f"Retrieved {len(result)} settings for user {user_id}")
            return result
        except Exception as e:
            logger.error(f"Error getting all settings for user {user_id}: {e}", exc_info=True)
            return {}
    
    def delete_user_settings(self, user_id: str) -> bool:
        """
        Delete all settings for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        DELETE FROM {self.schema}.user_settings 
                        WHERE user_id = %s
                    """, (user_id,))
                    deleted_count = cursor.rowcount
                
                conn.commit()
            
            logger.info(f"Deleted {deleted_count} settings for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting settings for user {user_id}: {e}", exc_info=True)
            return False
    
    def delete_setting(self, user_id: str, key: str) -> bool:
        """
        Delete a specific setting for a user.
        
        Args:
            user_id: User identifier
            key: Setting key
        
        Returns:
            True if deleted, False if not found
        """
        try:
            record_id = f"{user_id}:{key}"
            return self.delete(record_id)
        except Exception as e:
            logger.error(f"Error deleting setting {key} for user {user_id}: {e}", exc_info=True)
            return False

