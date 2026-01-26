"""
ClickHouse Client Wrapper.

Provides connection management, query execution, error handling, and health checks.
"""

import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from services.db_config import get_db_config, ClickHouseConfig

logger = logging.getLogger(__name__)

# Try to import clickhouse-driver, but don't fail if not available
try:
    from clickhouse_driver import Client as ClickHouseDriverClient
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False
    ClickHouseDriverClient = None  # type: ignore
    logger.warning("clickhouse-driver not installed. Install with: pip install clickhouse-driver")


class ClickHouseClient:
    """
    ClickHouse client wrapper with connection management.
    
    Provides:
    - Connection management
    - Query execution
    - Error handling
    - Health checks
    """
    
    def __init__(self, config: Optional[ClickHouseConfig] = None):
        """
        Initialize ClickHouse client.
        
        Args:
            config: Optional ClickHouse configuration (defaults to from_env)
        """
        if not CLICKHOUSE_AVAILABLE:
            raise ImportError("clickhouse-driver is not installed. Install with: pip install clickhouse-driver")
        
        self.config = config or get_db_config().clickhouse
        self._client: Optional[Any] = None
        self._connected = False
    
    def _get_client(self) -> Any:
        """
        Get or create ClickHouse client.
        
        Returns:
            ClickHouse client instance
        """
        if self._client is None or not self._connected:
            try:
                self._client = ClickHouseDriverClient(
                    host=self.config.host,
                    port=self.config.native_port,
                    user=self.config.user,
                    password=self.config.password,
                    database=self.config.database,
                    secure=self.config.secure,
                    verify=self.config.verify,
                    connect_timeout=self.config.timeout,
                    send_receive_timeout=self.config.timeout,
                )
                # Test connection
                self._client.execute("SELECT 1")
                self._connected = True
                logger.info(f"ClickHouse client connected to {self.config.host}:{self.config.native_port}")
            except Exception as e:
                logger.error(f"Failed to connect to ClickHouse: {e}")
                self._connected = False
                raise
        
        return self._client
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a query.
        
        Args:
            query: SQL query
            params: Optional query parameters
            
        Returns:
            Query result
        """
        client = self._get_client()
        try:
            if params:
                return client.execute(query, params)
            else:
                return client.execute(query)
        except Exception as e:
            logger.error(f"ClickHouse query failed: {e}")
            self._connected = False
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query
            params: Optional query parameters
            
        Returns:
            List of result tuples
        """
        return self.execute(query, params)
    
    def insert(self, table: str, data: List[Dict[str, Any]]) -> None:
        """
        Insert data into a table.
        
        Args:
            table: Table name
            data: List of dictionaries with data
        """
        if not data:
            return
        
        client = self._get_client()
        try:
            # Get column names from first row
            columns = list(data[0].keys())
            
            # Prepare data as list of tuples
            values = [[row[col] for col in columns] for row in data]
            
            # Insert
            client.execute(f"INSERT INTO {table} ({', '.join(columns)}) VALUES", values)
            logger.debug(f"Inserted {len(data)} rows into {table}")
        except Exception as e:
            logger.error(f"ClickHouse insert failed: {e}")
            self._connected = False
            raise
    
    def create_database(self, database: Optional[str] = None) -> None:
        """
        Create database if it doesn't exist.
        
        Args:
            database: Database name (defaults to config database)
        """
        db_name = database or self.config.database
        query = f"CREATE DATABASE IF NOT EXISTS {db_name}"
        self.execute(query)
        logger.info(f"Created database {db_name}")
    
    def create_table(self, table: str, schema: str, database: Optional[str] = None) -> None:
        """
        Create table if it doesn't exist.
        
        Args:
            table: Table name
            schema: Table schema (CREATE TABLE statement)
            database: Database name (defaults to config database)
        """
        db_name = database or self.config.database
        full_table = f"{db_name}.{table}" if db_name else table
        query = f"CREATE TABLE IF NOT EXISTS {full_table} {schema}"
        self.execute(query)
        logger.info(f"Created table {full_table}")
    
    def drop_table(self, table: str, database: Optional[str] = None) -> None:
        """
        Drop table if it exists.
        
        Args:
            table: Table name
            database: Database name (defaults to config database)
        """
        db_name = database or self.config.database
        full_table = f"{db_name}.{table}" if db_name else table
        query = f"DROP TABLE IF EXISTS {full_table}"
        self.execute(query)
        logger.info(f"Dropped table {full_table}")
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check ClickHouse connection health.
        
        Returns:
            Dictionary with health status
        """
        try:
            result = self.execute("SELECT 1")
            return {
                "status": "healthy",
                "connected": True,
                "host": self.config.host,
                "port": self.config.native_port,
                "database": self.config.database,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "host": self.config.host,
                "port": self.config.native_port,
            }
    
    def close(self) -> None:
        """Close ClickHouse connection."""
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
            self._connected = False
            logger.debug("ClickHouse connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Global client instance
_client: Optional[ClickHouseClient] = None


def get_clickhouse_client() -> ClickHouseClient:
    """
    Get or create global ClickHouse client instance.
    
    Returns:
        ClickHouseClient instance
    """
    global _client
    if _client is None:
        _client = ClickHouseClient()
    return _client

