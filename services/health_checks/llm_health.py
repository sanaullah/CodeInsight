"""
LLM Service health check implementation.
"""

import time
import logging
from services.health_check import HealthCheck, ServiceHealth, HealthStatus
from llm.config import ConfigManager

logger = logging.getLogger(__name__)


class LLMHealthCheck(HealthCheck):
    """Health check for LLM service."""
    
    def check_health(self) -> ServiceHealth:
        """Check LLM service configuration health."""
        start_time = time.time()
        
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Check if API base is configured
            if not config.api_base:
                return ServiceHealth(
                    status=HealthStatus.UNHEALTHY,
                    details={"error": "API base not configured"},
                    message="LLM API base URL is not configured",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check if API key is configured
            if not config.api_key:
                return ServiceHealth(
                    status=HealthStatus.UNHEALTHY,
                    details={"error": "API key not configured"},
                    message="LLM API key is not configured",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Check if default model is configured
            if not config.default_model:
                return ServiceHealth(
                    status=HealthStatus.DEGRADED,
                    details={"error": "Default model not configured"},
                    message="LLM default model is not configured",
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Optional: Test actual API connection (lightweight)
            # This could be a simple ping or model list request
            # For now, we just check configuration
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                status=HealthStatus.HEALTHY,
                details={
                    "api_base": config.api_base,
                    "default_model": config.default_model,
                    "timeout": config.timeout
                },
                message="LLM service configuration is healthy",
                response_time_ms=response_time
            )
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNKNOWN,
                details={"error": str(e)},
                message=f"LLM health check error: {e}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "LLM Service"

