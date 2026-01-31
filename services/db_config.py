"""
Unified Database Configuration Management.

Centralized configuration for PostgreSQL, Redis, ClickHouse, and MinIO.
Provides environment variable loading, validation, and connection string builders.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Schema name constants
CODELUMEN_SCHEMA = "codelumen"
CODELUMEN_CACHE_SCHEMA = "codelumen_cache"

# Database name constant
CODELUMEN_DATABASE = "codelumen"


@dataclass
class PostgreSQLConfig:
    """PostgreSQL database configuration."""
    
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    database: str = CODELUMEN_DATABASE
    schema: str = CODELUMEN_SCHEMA
    cache_schema: str = CODELUMEN_CACHE_SCHEMA
    
    @classmethod
    def from_env(cls) -> "PostgreSQLConfig":
        """Load PostgreSQL configuration from environment variables."""
        password = os.getenv("POSTGRES_PASSWORD") or ""
        if not password:
            raise ValueError("POSTGRES_PASSWORD environment variable is required")
        
        return cls(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=password,
            database=os.getenv("POSTGRES_DB", CODELUMEN_DATABASE),
            schema=os.getenv("POSTGRES_SCHEMA", CODELUMEN_SCHEMA),
            cache_schema=os.getenv("POSTGRES_CACHE_SCHEMA", CODELUMEN_CACHE_SCHEMA),
        )
    
    def get_connection_params(self, db_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get PostgreSQL connection parameters.
        
        Args:
            db_name: Optional database name override (defaults to self.database)
            
        Returns:
            Dictionary of connection parameters
        """
        return {
            "host": self.host,
            "port": self.port,
            "database": db_name or self.database,
            "user": self.user,
            "password": self.password,
        }
    
    def get_connection_string(self, db_name: Optional[str] = None) -> str:
        """
        Get PostgreSQL connection string.
        
        Args:
            db_name: Optional database name override
            
        Returns:
            PostgreSQL connection string
        """
        db = db_name or self.database
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{db}"
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate PostgreSQL configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.host:
            return False, "PostgreSQL host is required"
        if not (1 <= self.port <= 65535):
            return False, f"PostgreSQL port must be between 1 and 65535, got {self.port}"
        if not self.user:
            return False, "PostgreSQL user is required"
        if not self.password:
            return False, "PostgreSQL password is required"
        if not self.database:
            return False, "PostgreSQL database name is required"
        if not self.schema:
            return False, "PostgreSQL schema name is required"
        return True, None


@dataclass
class RedisConfig:
    """Redis configuration."""
    
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 1  # Use DB 1 for CodeLumen (separate from Langfuse)
    key_prefix: str = "codelumen:"
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    health_check_interval: int = 30
    
    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Load Redis configuration from environment variables."""
        password = os.getenv("REDIS_AUTH") or os.getenv("REDIS_PASSWORD")
        if password == "":
            password = None
        
        return cls(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=password,  # Optional - can be None for no authentication
            db=int(os.getenv("REDIS_DB", "1")),  # DB 1 for CodeLumen
            key_prefix=os.getenv("REDIS_KEY_PREFIX", "codelumen:"),
            socket_timeout=int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
            socket_connect_timeout=int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5")),
            health_check_interval=int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
        )
    
    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get Redis connection parameters.
        
        Returns:
            Dictionary of connection parameters
        """
        params = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "decode_responses": False,  # Keep binary for flexibility
            "socket_connect_timeout": self.socket_connect_timeout,
            "socket_timeout": self.socket_timeout,
            "retry_on_timeout": True,
            "health_check_interval": self.health_check_interval,
        }
        
        if self.password:
            params["password"] = self.password
        
        return params
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate Redis configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.host:
            return False, "Redis host is required"
        if not (1 <= self.port <= 65535):
            return False, f"Redis port must be between 1 and 65535, got {self.port}"
        if not (0 <= self.db <= 15):
            return False, f"Redis database must be between 0 and 15, got {self.db}"
        if not self.key_prefix:
            return False, "Redis key prefix is required"
        return True, None


@dataclass
class ClickHouseConfig:
    """ClickHouse configuration."""
    
    host: str = "localhost"
    http_port: int = 8123
    native_port: int = 9000
    user: str = "clickhouse"
    password: str = ""
    database: str = "codelumen_analytics"
    secure: bool = False
    verify: bool = True
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> "ClickHouseConfig":
        """Load ClickHouse configuration from environment variables."""
        password = os.getenv("CLICKHOUSE_PASSWORD") or ""
        if not password:
            raise ValueError("CLICKHOUSE_PASSWORD environment variable is required")
        
        return cls(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            http_port=int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123")),
            native_port=int(os.getenv("CLICKHOUSE_NATIVE_PORT", "9000")),
            user=os.getenv("CLICKHOUSE_USER", "clickhouse"),
            password=password,
            database=os.getenv("CLICKHOUSE_DB", "codelumen_analytics"),
            secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true",
            verify=os.getenv("CLICKHOUSE_VERIFY", "true").lower() == "true",
            timeout=int(os.getenv("CLICKHOUSE_TIMEOUT", "30")),
        )
    
    def get_http_url(self) -> str:
        """
        Get ClickHouse HTTP URL.
        
        Returns:
            ClickHouse HTTP URL
        """
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.host}:{self.http_port}"
    
    def get_native_url(self) -> str:
        """
        Get ClickHouse native URL.
        
        Returns:
            ClickHouse native URL
        """
        protocol = "clickhouse+secure" if self.secure else "clickhouse"
        return f"{protocol}://{self.user}:{self.password}@{self.host}:{self.native_port}/{self.database}"
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate ClickHouse configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.host:
            return False, "ClickHouse host is required"
        if not (1 <= self.http_port <= 65535):
            return False, f"ClickHouse HTTP port must be between 1 and 65535, got {self.http_port}"
        if not (1 <= self.native_port <= 65535):
            return False, f"ClickHouse native port must be between 1 and 65535, got {self.native_port}"
        if not self.user:
            return False, "ClickHouse user is required"
        if not self.password:
            return False, "ClickHouse password is required"
        if not self.database:
            return False, "ClickHouse database name is required"
        return True, None


@dataclass
class MinIOConfig:
    """MinIO/S3 configuration."""
    
    endpoint: str = "localhost:9090"
    access_key: str = "minio"
    secret_key: str = ""
    bucket: str = "codelumen"
    region: str = "us-east-1"
    secure: bool = False
    force_path_style: bool = True
    
    @classmethod
    def from_env(cls) -> "MinIOConfig":
        """Load MinIO configuration from environment variables."""
        secret_key = os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD") or ""
        if not secret_key:
            raise ValueError("MINIO_SECRET_KEY or MINIO_ROOT_PASSWORD environment variable is required")
        
        return cls(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9090"),
            access_key=os.getenv("MINIO_ACCESS_KEY", os.getenv("MINIO_ROOT_USER", "minio")),
            secret_key=secret_key,
            bucket=os.getenv("MINIO_BUCKET", "codelumen"),
            region=os.getenv("MINIO_REGION", "us-east-1"),
            secure=os.getenv("MINIO_SECURE", "false").lower() == "true",
            force_path_style=os.getenv("MINIO_FORCE_PATH_STYLE", "true").lower() == "true",
        )
    
    def get_s3_config(self) -> Dict[str, Any]:
        """
        Get S3/MinIO configuration dictionary.
        
        Returns:
            Dictionary of S3 configuration parameters
        """
        return {
            "endpoint_url": f"{'https' if self.secure else 'http'}://{self.endpoint}",
            "aws_access_key_id": self.access_key,
            "aws_secret_access_key": self.secret_key,
            "region_name": self.region,
            "use_ssl": self.secure,
            "verify": self.secure,
        }
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate MinIO configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.endpoint:
            return False, "MinIO endpoint is required"
        if not self.access_key:
            return False, "MinIO access key is required"
        if not self.secret_key:
            return False, "MinIO secret key is required"
        if not self.bucket:
            return False, "MinIO bucket name is required"
        return True, None


@dataclass
class DatabaseConfig:
    """Aggregated database configuration for all services."""
    
    postgresql: PostgreSQLConfig
    redis: RedisConfig
    clickhouse: ClickHouseConfig
    minio: MinIOConfig
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load all database configurations from environment variables."""
        return cls(
            postgresql=PostgreSQLConfig.from_env(),
            redis=RedisConfig.from_env(),
            clickhouse=ClickHouseConfig.from_env(),
            minio=MinIOConfig.from_env(),
        )
    
    def validate_all(self) -> tuple[bool, list[str]]:
        """
        Validate all database configurations.
        
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        
        pg_valid, pg_error = self.postgresql.validate()
        if not pg_valid:
            errors.append(f"PostgreSQL: {pg_error}")
        
        redis_valid, redis_error = self.redis.validate()
        if not redis_valid:
            errors.append(f"Redis: {redis_error}")
        
        ch_valid, ch_error = self.clickhouse.validate()
        if not ch_valid:
            errors.append(f"ClickHouse: {ch_error}")
        
        minio_valid, minio_error = self.minio.validate()
        if not minio_valid:
            errors.append(f"MinIO: {minio_error}")
        
        return len(errors) == 0, errors


# Global configuration instance
_config: Optional[DatabaseConfig] = None


def get_db_config() -> DatabaseConfig:
    """
    Get or create the global database configuration instance.
    
    Returns:
        DatabaseConfig instance
    """
    global _config
    if _config is None:
        _config = DatabaseConfig.from_env()
        logger.debug("Database configuration loaded from environment")
    return _config


def reset_db_config() -> None:
    """Reset the global database configuration (useful for testing)."""
    global _config
    _config = None
    logger.debug("Database configuration reset")

