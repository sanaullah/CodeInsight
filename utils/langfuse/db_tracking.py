"""
Database operation tracking utilities for Langfuse.

Provides context managers for tracking PostgreSQL queries, storage operations,
cache operations, and connection pool usage.
"""

import logging
import hashlib
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime
from .tracing import create_observation, get_current_trace_context
from .client import get_langfuse_client
from llm.config import ConfigManager

logger = logging.getLogger(__name__)


def _is_db_tracking_enabled() -> bool:
    """Check if database tracking is enabled in config."""
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        # Default to True if not specified
        return getattr(config.langfuse, 'tracking', {}).get('db_enabled', True) if hasattr(config.langfuse, 'tracking') else True
    except Exception:
        # Default to True if config check fails
        return True


def _is_cache_tracking_enabled() -> bool:
    """Check if cache tracking is enabled in config."""
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        # Default to True if not specified
        return getattr(config.langfuse, 'tracking', {}).get('cache_enabled', True) if hasattr(config.langfuse, 'tracking') else True
    except Exception:
        # Default to True if config check fails
        return True


def _sanitize_query(query: str) -> str:
    """
    Sanitize SQL query to remove sensitive data.
    
    Args:
        query: Original SQL query
        
    Returns:
        Sanitized SQL query
    """
    if not query:
        return query
    
    # Remove password, token, secret values from query
    import re
    # Pattern to match common sensitive patterns in SQL
    patterns = [
        (r"password\s*=\s*'[^']*'", "password='***REDACTED***'"),
        (r"password\s*=\s*\"[^\"]*\"", 'password="***REDACTED***"'),
        (r"token\s*=\s*'[^']*'", "token='***REDACTED***'"),
        (r"token\s*=\s*\"[^\"]*\"", 'token="***REDACTED***"'),
        (r"secret\s*=\s*'[^']*'", "secret='***REDACTED***'"),
        (r"secret\s*=\s*\"[^\"]*\"", 'secret="***REDACTED***"'),
    ]
    
    sanitized = query
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    # Truncate very long queries
    if len(sanitized) > 5000:
        sanitized = sanitized[:5000] + "... (truncated)"
    
    return sanitized


def _hash_query(query: str) -> str:
    """Generate hash of query for deduplication."""
    return hashlib.md5(query.encode('utf-8')).hexdigest()[:16]


@contextmanager
def track_db_query(
    query: str,
    db_name: Optional[str] = None,
    row_count: Optional[int] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracking SQL query execution.
    
    Args:
        query: SQL query text
        db_name: Optional database name
        row_count: Optional number of rows affected/returned
        trace_id: Optional trace ID for linking
        user_id: Optional user ID for linking
        session_id: Optional session ID for linking
        state: Optional state dictionary (for extracting trace context)
        config: Optional config dictionary (for extracting trace context)
        
    Yields:
        Observation object that can be updated with query results
    """
    if not _is_db_tracking_enabled():
        yield None
        return
    
    # Extract trace context if not provided
    if not trace_id and not user_id and not session_id:
        trace_context = get_current_trace_context(state=state, config=config)
        trace_id = trace_id or trace_context.get("trace_id")
        user_id = user_id or trace_context.get("user_id")
        session_id = session_id or trace_context.get("session_id")
    
    start_time = datetime.now()
    observation_cm = None
    observation_obj = None
    error_occurred = False
    error_message = None
    
    try:
        # Sanitize query
        sanitized_query = _sanitize_query(query)
        query_hash = _hash_query(query)
        
        # Create observation context manager
        observation_cm = create_observation(
            name="db_query",
            trace_id=trace_id,
            input_data={
                "query": sanitized_query,
                "db_name": db_name
            },
            metadata={
                "operation_type": "db_query",
                "query_hash": query_hash,
                "db_name": db_name,
                "query_length": len(query)
            }
        )
        
        if observation_cm:
            # Enter the context manager to get actual observation
            with observation_cm as observation_obj:
                try:
                    yield observation_obj
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    raise
        else:
            yield None
            
    except Exception as e:
        if not error_occurred:  # Don't double-set if already set
            error_occurred = True
            error_message = str(e)
        logger.error(f"Database query error: {e}", exc_info=True)
        raise
    finally:
        # Update observation before context exits (if we entered it)
        if observation_obj:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            try:
                update_metadata = {
                    "execution_time_ms": duration_ms,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
                
                if row_count is not None:
                    update_metadata["row_count"] = row_count
                
                if error_occurred:
                    update_metadata["error"] = error_message
                    observation_obj.update(
                        output=None,
                        error=error_message,
                        level="ERROR",
                        metadata=update_metadata
                    )
                else:
                    observation_obj.update(
                        output={"row_count": row_count},
                        metadata=update_metadata
                    )
            except Exception as update_error:
                logger.warning(f"Failed to update database query observation: {update_error}")


@contextmanager
def track_db_connection(
    db_name: str,
    read_only: bool = False,
    pool_size: Optional[int] = None,
    active_connections: Optional[int] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracking database connection acquisition.
    
    Args:
        db_name: Database name
        read_only: Whether connection is read-only
        pool_size: Optional connection pool size
        active_connections: Optional number of active connections
        trace_id: Optional trace ID for linking
        user_id: Optional user ID for linking
        session_id: Optional session ID for linking
        state: Optional state dictionary (for extracting trace context)
        config: Optional config dictionary (for extracting trace context)
        
    Yields:
        Observation object that can be updated with connection info
    """
    if not _is_db_tracking_enabled():
        yield None
        return
    
    # Extract trace context if not provided
    if not trace_id and not user_id and not session_id:
        trace_context = get_current_trace_context(state=state, config=config)
        trace_id = trace_id or trace_context.get("trace_id")
        user_id = user_id or trace_context.get("user_id")
        session_id = session_id or trace_context.get("session_id")
    
    start_time = datetime.now()
    observation_cm = None
    observation_obj = None
    error_occurred = False
    error_message = None
    
    try:
        # Create observation context manager
        observation_cm = create_observation(
            name="db_connection",
            trace_id=trace_id,
            input_data={
                "db_name": db_name,
                "read_only": read_only
            },
            metadata={
                "operation_type": "db_connection",
                "db_name": db_name,
                "read_only": read_only,
                "pool_size": pool_size,
                "active_connections": active_connections
            }
        )
        
        if observation_cm:
            # Enter the context manager to get actual observation
            with observation_cm as observation_obj:
                try:
                    yield observation_obj
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    raise
        else:
            yield None
            
    except Exception as e:
        if not error_occurred:  # Don't double-set if already set
            error_occurred = True
            error_message = str(e)
        logger.error(f"Database connection error: {e}", exc_info=True)
        raise
    finally:
        # Update observation before context exits (if we entered it)
        if observation_obj:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            try:
                update_metadata = {
                    "execution_time_ms": duration_ms,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
                
                if error_occurred:
                    update_metadata["error"] = error_message
                    observation_obj.update(
                        output=None,
                        error=error_message,
                        level="ERROR",
                        metadata=update_metadata
                    )
                else:
                    observation_obj.update(
                        output={"status": "success"},
                        metadata=update_metadata
                    )
            except Exception as update_error:
                logger.warning(f"Failed to update database connection observation: {update_error}")


@contextmanager
def track_storage_operation(
    operation: str,
    service_name: str,
    record_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracking storage service operations.
    
    Args:
        operation: Operation type (save, get, query, delete, etc.)
        service_name: Storage service name (e.g., "ExperienceStorage")
        record_id: Optional record identifier
        trace_id: Optional trace ID for linking
        user_id: Optional user ID for linking
        session_id: Optional session ID for linking
        state: Optional state dictionary (for extracting trace context)
        config: Optional config dictionary (for extracting trace context)
        
    Yields:
        Observation object that can be updated with operation results
    """
    if not _is_db_tracking_enabled():
        yield None
        return
    
    # Extract trace context if not provided
    if not trace_id and not user_id and not session_id:
        trace_context = get_current_trace_context(state=state, config=config)
        trace_id = trace_id or trace_context.get("trace_id")
        user_id = user_id or trace_context.get("user_id")
        session_id = session_id or trace_context.get("session_id")
    
    start_time = datetime.now()
    observation_cm = None
    observation_obj = None
    error_occurred = False
    error_message = None
    
    try:
        # Create observation context manager
        observation_cm = create_observation(
            name=f"storage_{operation}",
            trace_id=trace_id,
            input_data={
                "operation": operation,
                "service_name": service_name,
                "record_id": record_id
            },
            metadata={
                "operation_type": "storage_operation",
                "operation": operation,
                "service_name": service_name,
                "record_id": record_id
            }
        )
        
        if observation_cm:
            # Enter the context manager to get actual observation
            with observation_cm as observation_obj:
                try:
                    yield observation_obj
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    raise
        else:
            yield None
            
    except Exception as e:
        if not error_occurred:  # Don't double-set if already set
            error_occurred = True
            error_message = str(e)
        logger.error(f"Storage operation error: {e}", exc_info=True)
        raise
    finally:
        # Update observation before context exits (if we entered it)
        if observation_obj:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            try:
                update_metadata = {
                    "execution_time_ms": duration_ms,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }
                
                if error_occurred:
                    update_metadata["error"] = error_message
                    observation_obj.update(
                        output=None,
                        error=error_message,
                        level="ERROR",
                        metadata=update_metadata
                    )
                else:
                    observation_obj.update(
                        metadata=update_metadata
                    )
            except Exception as update_error:
                logger.warning(f"Failed to update storage operation observation: {update_error}")


@contextmanager
def track_cache_operation(
    operation: str,
    cache_key: str,
    hit: Optional[bool] = None,
    ttl: Optional[int] = None,
    result_size: Optional[int] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracking cache operations.
    
    Args:
        operation: Operation type (get, set, delete, exists, etc.)
        cache_key: Cache key (may be truncated for long keys)
        hit: Optional boolean indicating cache hit/miss (for get operations)
        ttl: Optional time to live in seconds
        result_size: Optional size of cached value in bytes
        trace_id: Optional trace ID for linking
        user_id: Optional user ID for linking
        session_id: Optional session ID for linking
        state: Optional state dictionary (for extracting trace context)
        config: Optional config dictionary (for extracting trace context)
        
    Yields:
        Observation object that can be updated with cache operation results
    """
    if not _is_cache_tracking_enabled():
        yield None
        return
    
    # Extract trace context if not provided
    if not trace_id and not user_id and not session_id:
        trace_context = get_current_trace_context(state=state, config=config)
        trace_id = trace_id or trace_context.get("trace_id")
        user_id = user_id or trace_context.get("user_id")
        session_id = session_id or trace_context.get("session_id")
    
    start_time = datetime.now()
    observation_cm = None
    observation_obj = None
    error_occurred = False
    error_message = None
    
    try:
        # Truncate long cache keys
        display_key = cache_key if len(cache_key) <= 200 else cache_key[:200] + "..."
        
        # Create observation context manager
        observation_cm = create_observation(
            name=f"cache_{operation}",
            trace_id=trace_id,
            input_data={
                "operation": operation,
                "cache_key": display_key
            },
            metadata={
                "operation_type": "cache_operation",
                "operation": operation,
                "cache_key": display_key,
                "cache_key_length": len(cache_key),
                "hit": hit,
                "ttl": ttl,
                "result_size": result_size
            }
        )
        
        if observation_cm:
            # Enter the context manager to get actual observation
            with observation_cm as observation_obj:
                try:
                    yield observation_obj
                except Exception as e:
                    error_occurred = True
                    error_message = str(e)
                    raise
        else:
            yield None
            
    except Exception as e:
        if not error_occurred:  # Don't double-set if already set
            error_occurred = True
            error_message = str(e)
        logger.error(f"Cache operation error: {e}", exc_info=True)
        raise
    finally:
        # Update observation before context exits (if we entered it)
        if observation_obj:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            try:
                update_metadata = {
                    "execution_time_ms": duration_ms,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "hit": hit,
                    "ttl": ttl,
                    "result_size": result_size
                }
                
                if error_occurred:
                    update_metadata["error"] = error_message
                    observation_obj.update(
                        output=None,
                        error=error_message,
                        level="ERROR",
                        metadata=update_metadata
                    )
                else:
                    output_data = {}
                    if hit is not None:
                        output_data["hit"] = hit
                    if result_size is not None:
                        output_data["result_size"] = result_size
                    
                    observation_obj.update(
                        output=output_data if output_data else {"status": "success"},
                        metadata=update_metadata
                    )
            except Exception as update_error:
                logger.warning(f"Failed to update cache operation observation: {update_error}")

