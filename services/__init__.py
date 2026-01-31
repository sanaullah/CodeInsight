"""
Services package for CodeLumen.

Provides database configuration, caching, and client implementations.
"""

# Database configuration
from .db_config import (
    get_db_config,
    reset_db_config,
    PostgreSQLConfig,
    RedisConfig,
    ClickHouseConfig,
    MinIOConfig,
    DatabaseConfig,
    CODELUMEN_DATABASE,
    CODELUMEN_SCHEMA,
    CODELUMEN_CACHE_SCHEMA,
)

# PostgreSQL connection pool
from .postgresql_connection_pool import (
    PostgreSQLConnectionPool,
    get_db_connection,
    transaction,
    PostgreSQLHealthCheck,
)

# Redis client and cache
from .redis_client import RedisClient
from .redis_fallback import InMemoryFallback
from .redis_cache import (
    RedisCache,
    get_cache,
    cache_result,
    CACHE_TTL,
)

# ClickHouse client
try:
    from .clickhouse_client import (
        ClickHouseClient,
        get_clickhouse_client,
        CLICKHOUSE_AVAILABLE,
    )
except ImportError:
    CLICKHOUSE_AVAILABLE = False

# MinIO client
try:
    from .minio_client import (
        MinIOClient,
        get_minio_client,
        MINIO_AVAILABLE,
    )
except ImportError:
    MINIO_AVAILABLE = False

__all__ = [
    # Database config
    "get_db_config",
    "reset_db_config",
    "PostgreSQLConfig",
    "RedisConfig",
    "ClickHouseConfig",
    "MinIOConfig",
    "DatabaseConfig",
    "CODELUMEN_DATABASE",
    "CODELUMEN_SCHEMA",
    "CODELUMEN_CACHE_SCHEMA",
    # PostgreSQL
    "PostgreSQLConnectionPool",
    "get_db_connection",
    "transaction",
    "PostgreSQLHealthCheck",
    # Redis
    "RedisClient",
    "InMemoryFallback",
    "RedisCache",
    "get_cache",
    "cache_result",
    "CACHE_TTL",
    # ClickHouse
    "ClickHouseClient",
    "get_clickhouse_client",
    "CLICKHOUSE_AVAILABLE",
    # MinIO
    "MinIOClient",
    "get_minio_client",
    "MINIO_AVAILABLE",
]

