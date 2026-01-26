"""
Health check implementations for CodeInsight services.
"""

# Import all health check classes
# These imports may fail if dependencies are missing, which is handled in health_monitor
try:
    from services.health_checks.postgresql_health import PostgreSQLHealthCheck
except ImportError:
    PostgreSQLHealthCheck = None

try:
    from services.health_checks.redis_health import RedisHealthCheck
except ImportError:
    RedisHealthCheck = None

try:
    from services.health_checks.clickhouse_health import ClickHouseHealthCheck
except ImportError:
    ClickHouseHealthCheck = None

try:
    from services.health_checks.minio_health import MinIOHealthCheck
except ImportError:
    MinIOHealthCheck = None

try:
    from services.health_checks.langfuse_health import LangfuseHealthCheck
except ImportError:
    LangfuseHealthCheck = None

try:
    from services.health_checks.llm_health import LLMHealthCheck
except ImportError:
    LLMHealthCheck = None

__all__ = [
    "PostgreSQLHealthCheck",
    "RedisHealthCheck",
    "ClickHouseHealthCheck",
    "MinIOHealthCheck",
    "LangfuseHealthCheck",
    "LLMHealthCheck",
]

