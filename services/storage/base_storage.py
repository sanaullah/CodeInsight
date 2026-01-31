"""
Base Storage Interface.

Abstract base class for all storage services providing common methods,
transaction handling, error handling, and logging.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class BaseStorage(ABC):
    """
    Abstract base class for all storage services.
    
    Provides common interface for:
    - CRUD operations
    - Transaction handling
    - Error handling
    - Logging
    """
    
    def __init__(self, db_identifier: str):
        """
        Initialize storage service.
        
        Args:
            db_identifier: Database identifier (database name for PostgreSQL)
        """
        self.db_identifier = db_identifier
        self._initialized = False
        self._init_database()
    
    @abstractmethod
    def _init_database(self) -> None:
        """
        Initialize database schema.
        
        Should create tables, indexes, etc. if they don't exist.
        """
        pass
    
    @abstractmethod
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save data to storage.
        
        Args:
            data: Data dictionary to save
            
        Returns:
            ID of saved record
            
        Raises:
            StorageError: If save fails
        """
        pass
    
    @abstractmethod
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get record by ID.
        
        Args:
            record_id: Record identifier
            
        Returns:
            Record dictionary or None if not found
            
        Raises:
            StorageError: If get fails
        """
        pass
    
    @abstractmethod
    def query(self, filters: Optional[Dict[str, Any]] = None, limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query records with filters.
        
        Args:
            filters: Optional filter dictionary
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of record dictionaries
            
        Raises:
            StorageError: If query fails
        """
        pass
    
    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """
        Delete record by ID.
        
        Args:
            record_id: Record identifier
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            StorageError: If delete fails
        """
        pass
    
    @abstractmethod
    @contextmanager
    def transaction(self):
        """
        Transaction context manager.
        
        Usage:
            with storage.transaction():
                storage.save(data1)
                storage.save(data2)
        """
        pass
    
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count records matching filters.
        
        Args:
            filters: Optional filter dictionary
            
        Returns:
            Count of matching records
            
        Raises:
            StorageError: If count fails
        """
        # Track storage operation
        try:
            from utils.langfuse.db_tracking import track_storage_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_storage_operation(
                operation="count",
                service_name=self.__class__.__name__,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ):
                pass  # Tracking context manager handles the observation
        except Exception as track_error:
            logger.debug(f"Failed to track storage count operation: {track_error}")
        
        # Default implementation: query and count
        # Subclasses can override for better performance
        results = self.query(filters=filters)
        return len(results)
    
    def exists(self, record_id: str) -> bool:
        """
        Check if record exists.
        
        Args:
            record_id: Record identifier
            
        Returns:
            True if exists, False otherwise
            
        Raises:
            StorageError: If check fails
        """
        # Track storage operation
        try:
            from utils.langfuse.db_tracking import track_storage_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_storage_operation(
                operation="exists",
                service_name=self.__class__.__name__,
                record_id=record_id,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ):
                pass  # Tracking context manager handles the observation
        except Exception as track_error:
            logger.debug(f"Failed to track storage exists operation: {track_error}")
        
        record = self.get(record_id)
        return record is not None
    
    def bulk_save(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Save multiple records in a single transaction.
        
        Args:
            records: List of data dictionaries to save
            
        Returns:
            List of IDs of saved records
            
        Raises:
            StorageError: If bulk save fails
        """
        # Track storage operation
        try:
            from utils.langfuse.db_tracking import track_storage_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_storage_operation(
                operation="bulk_save",
                service_name=self.__class__.__name__,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ) as obs:
                if obs:
                    try:
                        obs.update(metadata={"record_count": len(records)})
                    except Exception:
                        pass
        except Exception as track_error:
            logger.debug(f"Failed to track storage bulk_save operation: {track_error}")
        
        ids = []
        with self.transaction():
            for record in records:
                record_id = self.save(record)
                ids.append(record_id)
        return ids
    
    def bulk_delete(self, record_ids: List[str]) -> int:
        """
        Delete multiple records in a single transaction.
        
        Args:
            record_ids: List of record identifiers to delete
            
        Returns:
            Number of records deleted
            
        Raises:
            StorageError: If bulk delete fails
        """
        # Track storage operation
        try:
            from utils.langfuse.db_tracking import track_storage_operation
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_storage_operation(
                operation="bulk_delete",
                service_name=self.__class__.__name__,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ) as obs:
                if obs:
                    try:
                        obs.update(metadata={"record_count": len(record_ids)})
                    except Exception:
                        pass
        except Exception as track_error:
            logger.debug(f"Failed to track storage bulk_delete operation: {track_error}")
        
        deleted = 0
        with self.transaction():
            for record_id in record_ids:
                if self.delete(record_id):
                    deleted += 1
        return deleted
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate data before saving.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Default implementation: always valid
        Subclasses should override for specific validation
        """
        return True, None
    
    def log_operation(self, operation: str, record_id: Optional[str] = None, success: bool = True, error: Optional[Exception] = None) -> None:
        """
        Log storage operation.
        
        Args:
            operation: Operation name (e.g., 'save', 'get', 'delete')
            record_id: Optional record identifier
            success: Whether operation succeeded
            error: Optional error exception
        """
        if success:
            if record_id:
                logger.debug(f"{operation} operation succeeded for {record_id}")
            else:
                logger.debug(f"{operation} operation succeeded")
        else:
            if record_id:
                logger.error(f"{operation} operation failed for {record_id}: {error}", exc_info=error)
            else:
                logger.error(f"{operation} operation failed: {error}", exc_info=error)


class StorageError(Exception):
    """Base exception for storage operations."""
    pass


class StorageNotFoundError(StorageError):
    """Exception raised when record is not found."""
    pass


class StorageValidationError(StorageError):
    """Exception raised when data validation fails."""
    pass


class StorageConnectionError(StorageError):
    """Exception raised when database connection fails."""
    pass


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid or missing."""
    pass


class UserFacingError(Exception):
    """Base exception for errors that can be safely shown to users."""
    def __init__(self, message: str, user_message: Optional[str] = None):
        super().__init__(message)
        self.user_message = user_message or message


class DatabaseConnectionError(StorageConnectionError):
    """Exception raised when database connection fails (alias for compatibility)."""
    pass


class AuthenticationError(UserFacingError):
    """Exception raised when authentication fails."""
    pass
