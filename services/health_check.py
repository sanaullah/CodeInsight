"""
Unified Health Check System for CodeInsight v3.

Provides base classes and enums for consistent health checking across all services.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Service health information."""
    status: HealthStatus
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    message: Optional[str] = None
    response_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "response_time_ms": self.response_time_ms
        }


class HealthCheck(ABC):
    """
    Abstract base class for health checks.
    
    All service health checks should inherit from this class.
    """
    
    @abstractmethod
    def check_health(self) -> ServiceHealth:
        """
        Perform health check.
        
        Returns:
            ServiceHealth object with status and details
        """
        pass
    
    @abstractmethod
    def get_service_name(self) -> str:
        """
        Get service name for identification.
        
        Returns:
            Service name string
        """
        pass

