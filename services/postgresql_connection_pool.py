"""
PostgreSQL Connection Pool.

Provides thread-safe connection pooling for PostgreSQL databases.
"""

import logging
import threading
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor
from typing import Optional, Dict, Tuple, Any
from contextlib import contextmanager
import os

logger = logging.getLogger(__name__)


class PostgreSQLConnectionPool:
    """
    Thread-safe connection pool for PostgreSQL databases.
    
    Manages connections using psycopg2 connection pooling with thread-safe access.
    Supports both connection string and individual parameters.
    """
    
    _instance: Optional['PostgreSQLConnectionPool'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize connection pool."""
        # Dictionary mapping connection_key -> SimpleConnectionPool
        self._pools: Dict[str, pool.SimpleConnectionPool] = {}
        # RLock for thread-safe access to pools
        self._pool_lock = threading.RLock()
        logger.info("PostgreSQLConnectionPool initialized")
    
    @classmethod
    def get_instance(cls) -> 'PostgreSQLConnectionPool':
        """
        Get singleton instance of connection pool.
        
        Returns:
            PostgreSQLConnectionPool instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _get_connection_key(self, db_name: str, read_only: bool = False) -> str:
        """
        Generate a connection key for pool lookup.
        
        Args:
            db_name: Database name
            read_only: If True, use read-only connection
            
        Returns:
            Connection key string
        """
        return f"{db_name}:{'ro' if read_only else 'rw'}"
    
    def _get_connection_params(self, db_name: str, read_only: bool = False) -> Dict[str, Any]:
        """
        Get PostgreSQL connection parameters from environment or defaults.
        
        Args:
            db_name: Database name
            read_only: If True, use read-only connection
            
        Returns:
            Dictionary of connection parameters
        """
        # Get connection parameters from environment or use defaults
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD")
        
        if not password:
            raise ValueError("POSTGRES_PASSWORD environment variable is required")
        
        params = {
            "host": host,
            "port": port,
            "database": db_name,
            "user": user,
            "password": password,
        }
        
        # For read-only connections, we can set default_transaction_isolation
        if read_only:
            params["options"] = "-c default_transaction_read_only=on"
        
        return params
    
    def _create_pool(self, db_name: str, read_only: bool = False) -> pool.SimpleConnectionPool:
        """
        Create a new connection pool for a database.
        
        Args:
            db_name: Database name
            read_only: If True, create read-only pool
            
        Returns:
            SimpleConnectionPool instance
        """
        connection_key = self._get_connection_key(db_name, read_only)
        params = self._get_connection_params(db_name, read_only)
        
        # Get pool configuration from environment variables with validation
        try:
            minconn = int(os.getenv("POSTGRES_POOL_MINCONN", "1"))
            maxconn = int(os.getenv("POSTGRES_POOL_MAXCONN", "20"))
            
            # Validate minconn
            if minconn < 1:
                logger.warning(f"POSTGRES_POOL_MINCONN must be >= 1, using default 1")
                minconn = 1
            
            # Validate maxconn
            if maxconn <= minconn:
                logger.warning(f"POSTGRES_POOL_MAXCONN ({maxconn}) must be > POSTGRES_POOL_MINCONN ({minconn}), using default 20")
                maxconn = 20
            
            # Cap maxconn at reasonable limit
            if maxconn > 100:
                logger.warning(f"POSTGRES_POOL_MAXCONN ({maxconn}) exceeds recommended limit of 100, capping at 100")
                maxconn = 100
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid pool configuration (minconn/maxconn), using defaults: {e}")
            minconn = 1
            maxconn = 20
        
        # Create connection pool
        # minconn: minimum number of connections
        # maxconn: maximum number of connections
        pool_instance = pool.SimpleConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            **params
        )
        
        if pool_instance:
            logger.info(f"Created PostgreSQL connection pool for {connection_key} (minconn={minconn}, maxconn={maxconn})")
        else:
            raise RuntimeError(f"Failed to create connection pool for {connection_key}")
        
        return pool_instance
    
    @contextmanager
    def get_connection(self, db_name: str, read_only: bool = False):
        """
        Get a connection from the pool (thread-safe).
        
        Args:
            db_name: Database name
            read_only: If True, get read-only connection
            
        Yields:
            PostgreSQL connection object
        """
        connection_key = self._get_connection_key(db_name, read_only)
        
        with self._pool_lock:
            # Get or create pool for this database and mode
            if connection_key not in self._pools:
                self._pools[connection_key] = self._create_pool(db_name, read_only)
            
            pool_instance = self._pools[connection_key]
            pool_size = pool_instance.maxconn if hasattr(pool_instance, 'maxconn') else None
        
        # Track connection acquisition
        try:
            from utils.langfuse.db_tracking import track_db_connection
            from utils.langfuse.tracing import get_current_trace_context
            
            trace_context = get_current_trace_context()
            
            with track_db_connection(
                db_name=db_name,
                read_only=read_only,
                pool_size=pool_size,
                active_connections=len([p for p in self._pools.values() if hasattr(p, '_used')]) if hasattr(self, '_pools') else None,
                trace_id=trace_context.get("trace_id"),
                user_id=trace_context.get("user_id"),
                session_id=trace_context.get("session_id")
            ):
                pass  # Tracking context manager handles the observation
        except Exception as track_error:
            logger.debug(f"Failed to track database connection: {track_error}")
        
        # Get connection from pool (outside lock to avoid blocking)
        conn = None
        try:
            conn = pool_instance.getconn()
            if conn is None:
                raise RuntimeError(f"Failed to get connection from pool: {connection_key}")
            
            # Test connection
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
            except Exception as e:
                logger.warning(f"Connection test failed for {connection_key}: {e}, returning connection to pool")
                pool_instance.putconn(conn, close=True)
                raise
            
            yield conn
            
        except Exception as e:
            logger.error(f"Error using connection for {connection_key}: {e}", exc_info=True)
            # Return connection to pool (will be closed if corrupted)
            if conn:
                try:
                    pool_instance.putconn(conn, close=True)
                except Exception:
                    pass
            raise
        else:
            # Return connection to pool
            if conn:
                try:
                    pool_instance.putconn(conn)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")
    
    def close_pool(self, db_name: str, read_only: Optional[bool] = None) -> None:
        """
        Close and remove a specific connection pool.
        
        Args:
            db_name: Database name
            read_only: If specified, only close pool with this mode.
                      If None, close both read-only and read-write pools for this database.
        """
        with self._pool_lock:
            if read_only is not None:
                # Close specific pool
                connection_key = self._get_connection_key(db_name, read_only)
                if connection_key in self._pools:
                    try:
                        self._pools[connection_key].closeall()
                    except Exception as e:
                        logger.warning(f"Error closing pool for {connection_key}: {e}")
                    del self._pools[connection_key]
                    logger.debug(f"Closed and removed pool for {connection_key}")
            else:
                # Close all pools for this database (both read-only and read-write)
                keys_to_remove = [
                    key for key in self._pools.keys() 
                    if key.startswith(f"{db_name}:")
                ]
                for connection_key in keys_to_remove:
                    try:
                        self._pools[connection_key].closeall()
                    except Exception as e:
                        logger.warning(f"Error closing pool for {connection_key}: {e}")
                    del self._pools[connection_key]
                    logger.debug(f"Closed and removed pool for {connection_key}")
    
    def close_all(self) -> None:
        """Close all connection pools."""
        with self._pool_lock:
            for connection_key, pool_instance in list(self._pools.items()):
                try:
                    pool_instance.closeall()
                except Exception as e:
                    logger.warning(f"Error closing pool for {connection_key}: {e}")
            self._pools.clear()
            logger.info("Closed all connection pools")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes all pools."""
        self.close_all()
        return False


# Convenience function for getting connection
@contextmanager
def get_db_connection(db_name: str, read_only: bool = False):
    """
    Get a PostgreSQL connection from the pool (convenience function).
    
    Args:
        db_name: Database name
        read_only: If True, open connection in read-only mode
        
    Yields:
        PostgreSQL connection object
    """
    pool_instance = PostgreSQLConnectionPool.get_instance()
    with pool_instance.get_connection(db_name, read_only=read_only) as conn:
        yield conn


@contextmanager
def transaction(conn: psycopg2.extensions.connection):
    """
    PostgreSQL transaction context manager.
    
    Handles nested transaction attempts gracefully using savepoints.
    
    Args:
        conn: PostgreSQL connection object
    
    Yields:
        The connection object (for convenience)
    """
    # Check transaction status first
    transaction_level = conn.get_transaction_status()
    
    # 0 = IDLE (no transaction)
    # 1 = INTRANS (in transaction)
    # 2 = INERROR (in failed transaction)
    # 3 = UNKNOWN (connection lost)
    
    # Handle error state first
    if transaction_level == 2:  # INERROR
        try:
            conn.rollback()
            transaction_level = conn.get_transaction_status()  # Recheck after rollback
            logger.debug("Rolled back failed transaction before starting new one")
        except Exception as e:
            logger.warning(f"Error rolling back failed transaction: {e}")
    
    savepoint_name = None
    
    if transaction_level == 1:  # Already in transaction
        # Use savepoint for nested transaction support
        savepoint_name = f"sp_{id(conn)}_{threading.current_thread().ident}"
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql.SQL("SAVEPOINT {}").format(sql.Identifier(savepoint_name)))
            logger.debug(f"Created savepoint: {savepoint_name}")
        except Exception as e:
            logger.warning(f"Error creating savepoint: {e}")
            savepoint_name = None
    
    try:
        if savepoint_name is None:
            # Start new transaction only if not already in one
            if transaction_level == 0:  # IDLE - safe to set autocommit
                if conn.autocommit:
                    conn.autocommit = False
            # If transaction_level is 1, we're using savepoint (already handled above)
            # If transaction_level is still 2 after rollback, connection may be bad
        
        yield conn
        
        # Commit or release savepoint
        if savepoint_name:
            with conn.cursor() as cursor:
                cursor.execute(sql.SQL("RELEASE SAVEPOINT {}").format(sql.Identifier(savepoint_name)))
            logger.debug(f"Released savepoint: {savepoint_name}")
        else:
            # Only commit if we're in a transaction we started
            current_status = conn.get_transaction_status()
            if current_status == 1:  # INTRANS
                conn.commit()
                logger.debug("Transaction committed")
            
    except Exception as e:
        # Rollback or rollback to savepoint
        if savepoint_name:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(sql.SQL("ROLLBACK TO SAVEPOINT {}").format(sql.Identifier(savepoint_name)))
                logger.debug(f"Rolled back to savepoint: {savepoint_name}")
            except Exception as rollback_error:
                logger.warning(f"Error during savepoint rollback: {rollback_error}")
        else:
            try:
                current_status = conn.get_transaction_status()
                if current_status in (1, 2):  # INTRANS or INERROR
                    conn.rollback()
                    logger.debug("Transaction rolled back")
            except Exception as rollback_error:
                logger.warning(f"Error during transaction rollback: {rollback_error}")
        raise


# Health check implementation (kept for backward compatibility)
# New implementations should use services.health_checks.postgresql_health.PostgreSQLHealthCheck
class PostgreSQLHealthCheck:
    """Health check for PostgreSQL connection pool (legacy, kept for backward compatibility)."""
    
    def __init__(self, db_name: str):
        """
        Initialize PostgreSQL health check.
        
        Args:
            db_name: Database name to check
        """
        self.db_name = db_name
    
    def check_health(self):
        """
        Check PostgreSQL connection pool health.
        
        Returns:
            ServiceHealth object or dict
        """
        try:
            from services.health_check import ServiceHealth, HealthStatus
        except ImportError:
            # Health check module may not exist, return simple status
            return {"status": "unknown", "message": "Health check module not available"}
        
        try:
            pool_instance = PostgreSQLConnectionPool.get_instance()
            # Try to get a connection
            with pool_instance.get_connection(self.db_name, read_only=True) as conn:
                # Test connection with a simple query
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    if result and result[0] == 1:
                        with pool_instance._pool_lock:
                            pool_count = len(pool_instance._pools)
                        return ServiceHealth(
                            status=HealthStatus.HEALTHY,
                            details={
                                "db_name": self.db_name,
                                "pool_count": pool_count
                            },
                            message="PostgreSQL connection pool is healthy"
                        )
                    else:
                        return ServiceHealth(
                            status=HealthStatus.DEGRADED,
                            details={"db_name": self.db_name},
                            message="PostgreSQL connection test returned unexpected result"
                        )
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                details={"db_name": self.db_name, "error": str(e)},
                message=f"PostgreSQL connection pool health check failed: {e}"
            )
    
    def get_service_name(self) -> str:
        """Get service name."""
        return f"PostgreSQLConnectionPool({self.db_name})"

