"""
PostgreSQL health check implementation.
"""

import time
import logging
from typing import Optional
from services.health_check import HealthCheck, ServiceHealth, HealthStatus
from services.postgresql_connection_pool import PostgreSQLConnectionPool
from services.db_config import get_db_config

logger = logging.getLogger(__name__)


class PostgreSQLHealthCheck(HealthCheck):
    """Health check for PostgreSQL connection pool."""
    
    def __init__(self, db_name: Optional[str] = None):
        """
        Initialize PostgreSQL health check.
        
        Args:
            db_name: Database name to check (defaults to config database)
        """
        if db_name is None:
            db_name = get_db_config().postgresql.database
        self.db_name = db_name
    
    def check_health(self) -> ServiceHealth:
        """Check PostgreSQL connection health."""
        start_time = time.time()
        
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
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        return ServiceHealth(
                            status=HealthStatus.HEALTHY,
                            details={
                                "db_name": self.db_name,
                                "pool_count": pool_count
                            },
                            message="PostgreSQL connection pool is healthy",
                            response_time_ms=response_time
                        )
                    else:
                        return ServiceHealth(
                            status=HealthStatus.DEGRADED,
                            details={"db_name": self.db_name},
                            message="PostgreSQL connection test returned unexpected result",
                            response_time_ms=(time.time() - start_time) * 1000
                        )
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                details={"db_name": self.db_name, "error": str(e)},
                message=f"PostgreSQL connection pool health check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_service_name(self) -> str:
        """Get service name."""
        return f"PostgreSQL({self.db_name})"

