"""
Centralized Health Monitor for CodeInsight.

Manages health checks for all services and provides unified health status reporting.
"""

import logging
from typing import Dict, Optional
from services.health_check import HealthCheck, ServiceHealth, HealthStatus

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Centralized health monitor for all services.
    
    Manages health checks for all services and provides
    unified health status reporting.
    """
    
    def __init__(self):
        """Initialize health monitor with all service health checks."""
        self._health_checks: Dict[str, HealthCheck] = {}
        self._register_default_checks()
        logger.info("HealthMonitor initialized")
    
    def _register_default_checks(self) -> None:
        """Register all default health checks."""
        try:
            from services.health_checks.postgresql_health import PostgreSQLHealthCheck
            self.register_health_check(PostgreSQLHealthCheck())
        except Exception as e:
            logger.warning(f"Could not register PostgreSQL health check: {e}")
        
        try:
            from services.health_checks.redis_health import RedisHealthCheck
            self.register_health_check(RedisHealthCheck())
        except Exception as e:
            logger.warning(f"Could not register Redis health check: {e}")
        
        # ClickHouse is only used by Langfuse, not directly by CodeInsight
        # Skip ClickHouse health check - Langfuse health check will cover it indirectly
        # try:
        #     from services.health_checks.clickhouse_health import ClickHouseHealthCheck
        #     self.register_health_check(ClickHouseHealthCheck())
        # except Exception as e:
        #     logger.warning(f"Could not register ClickHouse health check: {e}")
        
        try:
            from services.health_checks.minio_health import MinIOHealthCheck
            self.register_health_check(MinIOHealthCheck())
        except Exception as e:
            logger.warning(f"Could not register MinIO health check: {e}")
        
        try:
            from services.health_checks.langfuse_health import LangfuseHealthCheck
            self.register_health_check(LangfuseHealthCheck())
        except Exception as e:
            logger.warning(f"Could not register Langfuse health check: {e}")
        
        try:
            from services.health_checks.llm_health import LLMHealthCheck
            self.register_health_check(LLMHealthCheck())
        except Exception as e:
            logger.warning(f"Could not register LLM health check: {e}")
    
    def register_health_check(self, health_check: HealthCheck) -> None:
        """
        Register a health check.
        
        Args:
            health_check: HealthCheck instance
        """
        service_name = health_check.get_service_name()
        self._health_checks[service_name] = health_check
        logger.info(f"Registered health check for service: {service_name}")
    
    def unregister_health_check(self, service_name: str) -> None:
        """
        Unregister a health check.
        
        Args:
            service_name: Service name
        """
        if service_name in self._health_checks:
            del self._health_checks[service_name]
            logger.info(f"Unregistered health check for service: {service_name}")
    
    def get_service_health(self, service_name: str) -> Optional[ServiceHealth]:
        """
        Get health status for a specific service.
        
        Args:
            service_name: Service name
            
        Returns:
            ServiceHealth object or None if service not registered
        """
        health_check = self._health_checks.get(service_name)
        if not health_check:
            return None
        
        try:
            return health_check.check_health()
        except Exception as e:
            logger.error(f"Error checking health for {service_name}: {e}", exc_info=True)
            return ServiceHealth(
                status=HealthStatus.UNKNOWN,
                message=f"Error checking health: {str(e)}"
            )
    
    def check_all_services(self) -> Dict[str, ServiceHealth]:
        """
        Check health of all registered services.
        
        Returns:
            Dictionary mapping service names to ServiceHealth objects
        """
        results = {}
        
        for service_name, health_check in self._health_checks.items():
            try:
                results[service_name] = health_check.check_health()
            except Exception as e:
                logger.error(f"Error checking health for {service_name}: {e}", exc_info=True)
                results[service_name] = ServiceHealth(
                    status=HealthStatus.UNKNOWN,
                    message=f"Error checking health: {str(e)}"
                )
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """
        Get overall health status (worst status among all services).
        
        Returns:
            HealthStatus enum value
        """
        all_health = self.check_all_services()
        
        if not all_health:
            return HealthStatus.UNKNOWN
        
        # Priority: UNHEALTHY > DEGRADED > UNKNOWN > HEALTHY
        statuses = [health.status for health in all_health.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, any]:
        """
        Get a summary of all service health.
        
        Returns:
            Dictionary with overall status and service details
        """
        all_health = self.check_all_services()
        overall = self.get_overall_health()
        
        return {
            "overall_status": overall.value,
            "services": {
                name: health.to_dict() for name, health in all_health.items()
            },
            "service_count": len(all_health),
            "healthy_count": sum(1 for h in all_health.values() if h.status == HealthStatus.HEALTHY),
            "degraded_count": sum(1 for h in all_health.values() if h.status == HealthStatus.DEGRADED),
            "unhealthy_count": sum(1 for h in all_health.values() if h.status == HealthStatus.UNHEALTHY),
            "unknown_count": sum(1 for h in all_health.values() if h.status == HealthStatus.UNKNOWN)
        }


# Global health monitor instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """
    Get or create the global health monitor instance.
    
    Returns:
        HealthMonitor instance
    """
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor

