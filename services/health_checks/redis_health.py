"""
Redis health check implementation.
"""

import time
import logging
from services.health_check import HealthCheck, ServiceHealth, HealthStatus
from services.redis_client import RedisClient
from services.redis_fallback import InMemoryFallback
from services.db_config import get_db_config
import services.redis_client as redis_client_module

logger = logging.getLogger(__name__)


class RedisHealthCheck(HealthCheck):
    """Health check for Redis client."""
    
    def check_health(self) -> ServiceHealth:
        """Check Redis connection health."""
        start_time = time.time()
        
        try:
            client = RedisClient.get_client()
            if client is None:
                # Check if using fallback
                fallback_client = RedisClient.get_client_with_fallback()
                if isinstance(fallback_client, InMemoryFallback):
                    stats = fallback_client.get_stats()
                    response_time = (time.time() - start_time) * 1000
                    
                    # Get circuit breaker state
                    try:
                        with redis_client_module._circuit_lock:
                            circuit_state = redis_client_module._circuit_state.value
                    except Exception:
                        circuit_state = "unknown"
                    
                    return ServiceHealth(
                        status=HealthStatus.DEGRADED,
                        details={
                            "using_fallback": True,
                            "fallback_stats": stats,
                            "circuit_state": circuit_state
                        },
                        message="Redis unavailable, using in-memory fallback",
                        response_time_ms=response_time
                    )
                else:
                    # Get circuit breaker state
                    try:
                        with redis_client_module._circuit_lock:
                            circuit_state = redis_client_module._circuit_state.value
                    except Exception:
                        circuit_state = "unknown"
                    
                    return ServiceHealth(
                        status=HealthStatus.UNHEALTHY,
                        details={"circuit_state": circuit_state},
                        message="Redis client unavailable",
                        response_time_ms=(time.time() - start_time) * 1000
                    )
            
            # Test connection
            client.ping()
            
            # Get circuit breaker state
            try:
                with redis_client_module._circuit_lock:
                    circuit_state = redis_client_module._circuit_state.value
                    circuit_failures = redis_client_module._circuit_failures
            except Exception:
                circuit_state = "unknown"
                circuit_failures = 0
            
            # Get config for host/port
            try:
                config = get_db_config().redis
                host = config.host
                port = config.port
            except Exception:
                host = "unknown"
                port = "unknown"
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                status=HealthStatus.HEALTHY,
                details={
                    "host": host,
                    "port": port,
                    "circuit_state": circuit_state,
                    "circuit_failures": circuit_failures
                },
                message="Redis client is healthy",
                response_time_ms=response_time
            )
        except Exception as e:
            circuit_state = "unknown"
            try:
                with redis_client_module._circuit_lock:
                    circuit_state = redis_client_module._circuit_state.value
            except Exception:
                pass
            
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e), "circuit_state": circuit_state},
                message=f"Redis health check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "Redis"

