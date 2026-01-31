"""
MinIO health check implementation.
"""

import time
import logging
from services.health_check import HealthCheck, ServiceHealth, HealthStatus
from services.minio_client import get_minio_client

logger = logging.getLogger(__name__)


class MinIOHealthCheck(HealthCheck):
    """Health check for MinIO client."""
    
    def check_health(self) -> ServiceHealth:
        """Check MinIO connection health."""
        start_time = time.time()
        
        try:
            client = get_minio_client()
            client._get_client().list_buckets()
            
            response_time = (time.time() - start_time) * 1000
            
            return ServiceHealth(
                status=HealthStatus.HEALTHY,
                details={
                    "endpoint": client.config.endpoint,
                    "bucket": client.config.bucket
                },
                message="MinIO is healthy",
                response_time_ms=response_time
            )
        except ImportError:
            return ServiceHealth(
                status=HealthStatus.UNKNOWN,
                details={"error": "boto3 not installed"},
                message="MinIO driver not available",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e)},
                message=f"MinIO health check failed: {e}",
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    def get_service_name(self) -> str:
        """Get service name."""
        return "MinIO"

