"""
ClickHouse health check implementation.
"""

import time
import logging
from services.health_check import HealthCheck, ServiceHealth, HealthStatus
from services.clickhouse_client import get_clickhouse_client, ClickHouseClient
from services.db_config import get_db_config

logger = logging.getLogger(__name__)


class ClickHouseHealthCheck(HealthCheck):
    """Health check for ClickHouse client."""
    
    def check_health(self) -> ServiceHealth:
        """Check ClickHouse connection health."""
        start_time = time.time()
        
        try:
            # Try to get client - this may fail if database doesn't exist
            try:
                client = get_clickhouse_client()
            except Exception as conn_error:
                error_str = str(conn_error)
                # Check if it's a database not found error
                if "does not exist" in error_str or "Code: 81" in error_str:
                    # Database doesn't exist - try to create it or connect to system database
                    try:
                        config = get_db_config().clickhouse
                        # Try connecting to system database for health check
                        from clickhouse_driver import Client as ClickHouseDriverClient
                        system_client = ClickHouseDriverClient(
                            host=config.host,
                            port=config.native_port,
                            user=config.user,
                            password=config.password,
                            database="system",  # Use system database for health check
                            secure=config.secure,
                            verify=config.verify,
                            connect_timeout=config.timeout,
                            send_receive_timeout=config.timeout,
                        )
                        result = system_client.execute("SELECT 1")
                        system_client.disconnect()
                        
                        response_time = (time.time() - start_time) * 1000
                        return ServiceHealth(
                            status=HealthStatus.DEGRADED,
                            details={
                                "host": config.host,
                                "port": config.native_port,
                                "database": config.database,
                                "issue": "Database does not exist, but ClickHouse server is reachable"
                            },
                            message=f"ClickHouse server is reachable but database '{config.database}' does not exist",
                            response_time_ms=response_time
                        )
                    except Exception as system_error:
                        # Can't even connect to system database
                        return ServiceHealth(
                            status=HealthStatus.UNHEALTHY,
                            details={"error": str(conn_error)},
                            message=f"ClickHouse connection failed: {conn_error}",
                            response_time_ms=(time.time() - start_time) * 1000
                        )
                else:
                    # Other connection error
                    raise conn_error
            
            # If we got here, client is connected
            result = client.execute("SELECT 1")
            
            if result:
                response_time = (time.time() - start_time) * 1000
                
                return ServiceHealth(
                    status=HealthStatus.HEALTHY,
                    details={
                        "host": client.config.host,
                        "port": client.config.native_port,
                        "database": client.config.database
                    },
                    message="ClickHouse is healthy",
                    response_time_ms=response_time
                )
            else:
                return ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    details={},
                    message="ClickHouse query returned no result",
                    response_time_ms=(time.time() - start_time) * 1000
                )
        except ImportError:
            return ServiceHealth(
                status=HealthStatus.UNKNOWN,
                details={"error": "clickhouse-driver not installed"},
                message="ClickHouse driver not available",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            error_str = str(e)
            # Check if it's a database not found error
            if "does not exist" in error_str or "Code: 81" in error_str:
                return ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    details={"error": error_str},
                    message=f"ClickHouse database does not exist: {error_str}",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            else:
                return ServiceHealth(
                    status=HealthStatus.UNHEALTHY,
                    details={"error": error_str},
                    message=f"ClickHouse health check failed: {e}",
                    response_time_ms=(time.time() - start_time) * 1000
                )
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "ClickHouse"

