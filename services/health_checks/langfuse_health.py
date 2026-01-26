"""
Langfuse health check implementation.
"""

import time
import logging
from services.health_check import HealthCheck, ServiceHealth, HealthStatus
from utils.langfuse.client import get_langfuse_client
from llm.config import ConfigManager

logger = logging.getLogger(__name__)


class LangfuseHealthCheck(HealthCheck):
    """Health check for Langfuse client."""
    
    def check_health(self) -> ServiceHealth:
        """Check Langfuse connection health."""
        start_time = time.time()
        
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            langfuse_config = config.langfuse
            
            # Check if Langfuse is enabled
            if not langfuse_config.enabled:
                return ServiceHealth(
                    status=HealthStatus.UNKNOWN,
                    details={"enabled": False},
                    message="Langfuse is disabled in configuration",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check if client can be created
            client = get_langfuse_client()
            if client is None:
                return ServiceHealth(
                    status=HealthStatus.UNHEALTHY,
                    details={"error": "Client initialization failed"},
                    message="Langfuse client could not be initialized",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Try to flush (tests connection)
            try:
                client.flush()
                response_time = (time.time() - start_time) * 1000
                
                return ServiceHealth(
                    status=HealthStatus.HEALTHY,
                    details={
                        "host": langfuse_config.host,
                        "enabled": True
                    },
                    message="Langfuse is healthy",
                    response_time_ms=response_time
                )
            except Exception as e:
                return ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    details={"error": str(e)},
                    message=f"Langfuse connection test failed: {e}",
                    response_time_ms=(time.time() - start_time) * 1000
                )
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNKNOWN,
                details={"error": str(e)},
                message=f"Langfuse health check error: {e}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "Langfuse"

