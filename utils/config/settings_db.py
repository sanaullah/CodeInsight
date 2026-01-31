"""
Settings database management for CodeInsight.

Manages PostgreSQL database for storing and retrieving user settings.
Uses CachedSettingsStorage for automatic Redis caching.
"""

import logging
import hashlib
from typing import Dict, Optional, Any

from services.storage.base_storage import (
    StorageError,
    StorageConnectionError,
    DatabaseConnectionError,
    ConfigurationError,
    UserFacingError
)

logger = logging.getLogger(__name__)

# Singleton storage instance
_storage = None


def _get_storage():
    """
    Get or create the settings storage instance (singleton).
    
    Returns:
        CachedSettingsStorage instance
    """
    global _storage
    if _storage is None:
        try:
            from services.storage.cached_settings_storage import CachedSettingsStorage
            _storage = CachedSettingsStorage()
            logger.debug("Settings storage initialized (PostgreSQL with Redis cache)")
        except Exception as e:
            logger.error(f"Failed to initialize settings storage: {e}", exc_info=True)
            # Fallback to non-cached storage if cache fails
            try:
                from services.storage.settings_storage import SettingsStorage
                _storage = SettingsStorage()
                logger.warning("Using non-cached settings storage (cache unavailable)")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback settings storage: {e2}", exc_info=True)
                raise
    return _storage


def init_settings_db() -> None:
    """
    Initialize PostgreSQL database and ensure schema exists.
    
    This is a no-op for PostgreSQL as schema is managed by migrations,
    but we ensure the storage is initialized.
    """
    try:
        _get_storage()
        logger.debug("Settings database initialized (PostgreSQL)")
    except Exception as e:
        logger.error(f"Error initializing settings database: {e}", exc_info=True)
        raise


def get_user_id() -> str:
    """
    Get current user ID.
    
    Uses Streamlit session state to generate a persistent user ID.
    If not present, generates one based on session state hash.
    
    Returns:
        User ID string
    """
    import streamlit as st
    
    if "user_id" not in st.session_state:
        # Generate a persistent user ID based on session state
        # This ensures the same user gets the same ID across sessions
        session_id = st.session_state.get("_session_id", str(hash(str(st.session_state))))
        user_id = f"user_{hashlib.md5(session_id.encode()).hexdigest()[:16]}"
        st.session_state["user_id"] = user_id
        logger.debug(f"Generated new user ID: {user_id}")
    
    return st.session_state["user_id"]


def get_setting(user_id: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
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
        storage = _get_storage()
        value = storage.get_setting(user_id, key, default)
        # Ensure we return default if value is None (setting doesn't exist)
        if value is None:
            return default
        logger.debug(f"Retrieved setting {key} for user {user_id}: {value}")
        return value
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting setting {key} for user {user_id}: {e}", exc_info=True)
        return default  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting setting {key} for user {user_id}: {e}", exc_info=True)
        return default
    except ConfigurationError as e:
        logger.error(f"Configuration error getting setting {key} for user {user_id}: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error getting setting {key} for user {user_id}: {e}", exc_info=True)
        return default  # Fallback


def set_setting(user_id: str, key: str, value: Any) -> bool:
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
        storage = _get_storage()
        success = storage.set_setting(user_id, key, value)
        if success:
            logger.debug(f"Set setting {key} for user {user_id}")
        return success
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error setting {key} for user {user_id}: {e}", exc_info=True)
        return False  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error setting {key} for user {user_id}: {e}", exc_info=True)
        return False
    except ConfigurationError as e:
        logger.error(f"Configuration error setting {key} for user {user_id}: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error setting {key} for user {user_id}: {e}", exc_info=True)
        return False  # Fallback


def get_all_settings(user_id: str) -> Dict[str, Any]:
    """
    Get all settings for a user as a dictionary.
    
    Args:
        user_id: User identifier
    
    Returns:
        Dictionary of all settings for the user
    """
    try:
        storage = _get_storage()
        settings = storage.get_all_settings(user_id)
        logger.debug(f"Retrieved {len(settings)} settings for user {user_id}")
        return settings
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting all settings for user {user_id}: {e}", exc_info=True)
        return {}  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting all settings for user {user_id}: {e}", exc_info=True)
        return {}
    except ConfigurationError as e:
        logger.error(f"Configuration error getting all settings for user {user_id}: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error getting all settings for user {user_id}: {e}", exc_info=True)
        return {}  # Fallback


def reset_user_settings(user_id: str) -> bool:
    """
    Reset all settings for a user to defaults.
    
    Args:
        user_id: User identifier
    
    Returns:
        True if successful, False otherwise
    """
    try:
        storage = _get_storage()
        success = storage.delete_user_settings(user_id)
        if success:
            logger.info(f"Reset all settings for user {user_id}")
        return success
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error resetting settings for user {user_id}: {e}", exc_info=True)
        return False  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error resetting settings for user {user_id}: {e}", exc_info=True)
        return False
    except ConfigurationError as e:
        logger.error(f"Configuration error resetting settings for user {user_id}: {e}", exc_info=True)
        raise UserFacingError("System configuration error - please contact support") from e
    except Exception as e:
        logger.error(f"Unexpected error resetting settings for user {user_id}: {e}", exc_info=True)
        return False  # Fallback


def verify_setting_sync(user_id: str, key: str, expected_value: Any) -> bool:
    """
    Verify that a setting is correctly stored in the database.
    
    Args:
        user_id: User identifier
        key: Setting key
        expected_value: Expected value
    
    Returns:
        True if setting matches expected value
    """
    try:
        actual_value = get_setting(user_id, key)
        return actual_value == expected_value
    except Exception as e:
        logger.error(f"Error verifying setting sync: {e}", exc_info=True)
        return False
