"""
Redis client utility for persistent state storage.

Provides a singleton Redis client with connection pooling, retry logic,
circuit breaker pattern, and graceful error handling with in-memory fallback.
"""

import logging
import time
import threading
from typing import Optional, Any, List, Dict
from enum import Enum
from .redis_fallback import InMemoryFallback

logger = logging.getLogger(__name__)

# Global Redis client instance
_redis_client: Optional[Any] = None
_redis_config: Optional[Any] = None
_redis_available: bool = False
_fallback: Optional[InMemoryFallback] = None

# Circuit breaker state
class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered

_circuit_state = CircuitState.CLOSED
_circuit_failures = 0
_circuit_last_failure_time: Optional[float] = None
_circuit_lock = threading.RLock()

# Circuit breaker configuration
CIRCUIT_FAILURE_THRESHOLD = 5  # Open circuit after 5 failures
CIRCUIT_TIMEOUT = 60  # Try again after 60 seconds
CIRCUIT_SUCCESS_THRESHOLD = 2  # Close circuit after 2 successes


class RedisClient:
    """
    Singleton Redis client wrapper with connection pooling and error handling.
    
    Provides methods to get Redis client, check availability, and test connections.
    Uses configuration from services.db_config.
    """
    
    @staticmethod
    def _record_failure() -> None:
        """Record a circuit breaker failure."""
        global _circuit_state, _circuit_failures, _circuit_last_failure_time
        
        with _circuit_lock:
            _circuit_failures += 1
            _circuit_last_failure_time = time.time()
            
            if _circuit_failures >= CIRCUIT_FAILURE_THRESHOLD:
                _circuit_state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {_circuit_failures} failures")
    
    @staticmethod
    def _record_success() -> None:
        """Record a circuit breaker success."""
        global _circuit_state, _circuit_failures
        
        with _circuit_lock:
            if _circuit_state == CircuitState.HALF_OPEN:
                _circuit_failures += 1
                if _circuit_failures >= CIRCUIT_SUCCESS_THRESHOLD:
                    _circuit_state = CircuitState.CLOSED
                    _circuit_failures = 0
                    logger.info("Circuit breaker closed after successful operations")
            elif _circuit_state == CircuitState.CLOSED:
                _circuit_failures = 0
    
    @staticmethod
    def _should_attempt_connection() -> bool:
        """Check if we should attempt a connection based on circuit breaker state."""
        global _circuit_state, _circuit_last_failure_time
        
        with _circuit_lock:
            if _circuit_state == CircuitState.CLOSED:
                return True
            elif _circuit_state == CircuitState.OPEN:
                # Check if timeout has passed
                if _circuit_last_failure_time and (time.time() - _circuit_last_failure_time) >= CIRCUIT_TIMEOUT:
                    _circuit_state = CircuitState.HALF_OPEN
                    _circuit_failures = 0
                    logger.info("Circuit breaker entering half-open state")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    @staticmethod
    def is_circuit_open() -> bool:
        """
        Check if circuit breaker is open.
        
        Returns:
            True if circuit is open, False otherwise
        """
        global _circuit_state
        with _circuit_lock:
            return _circuit_state == CircuitState.OPEN
    
    @staticmethod
    def get_client() -> Optional[Any]:
        """
        Get or create Redis client instance with retry logic.
        
        Returns:
            Redis client instance or None if Redis is disabled or unavailable
        """
        global _redis_client, _redis_config, _redis_available
        
        # Return existing client if available and circuit is not open
        if _redis_client is not None and _redis_available and not RedisClient.is_circuit_open():
            try:
                # Quick health check
                _redis_client.ping()
                RedisClient._record_success()
                return _redis_client
            except Exception:
                # Connection lost, reset and try to reconnect
                _redis_client = None
                _redis_available = False
        
        # Check circuit breaker
        if not RedisClient._should_attempt_connection():
            logger.debug("Circuit breaker is open, skipping Redis connection attempt")
            return None
        
        # Load config from db_config
        if _redis_config is None:
            try:
                from services.db_config import get_db_config
                config = get_db_config()
                _redis_config = config.redis
            except Exception as e:
                logger.warning(f"Failed to load Redis config: {e}")
                _redis_available = False
                RedisClient._record_failure()
                return None
        
        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 0.1
        
        for attempt in range(max_retries):
            try:
                import redis
                
                _redis_client = redis.Redis(
                    host=_redis_config.host,
                    port=_redis_config.port,
                    password=_redis_config.password if _redis_config.password else None,
                    db=_redis_config.db,
                    decode_responses=False,  # Keep binary for flexibility
                    socket_connect_timeout=_redis_config.socket_connect_timeout,
                    socket_timeout=_redis_config.socket_timeout,
                    retry_on_timeout=True,
                    health_check_interval=_redis_config.health_check_interval
                )
                
                # Test connection
                _redis_client.ping()
                _redis_available = True
                RedisClient._record_success()
                logger.info(f"Redis client connected to {_redis_config.host}:{_redis_config.port}")
                return _redis_client
                
            except ImportError:
                logger.warning("Redis library not installed. Install with: pip install redis>=5.0.0")
                _redis_available = False
                return None
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.debug(f"Redis connection attempt {attempt + 1} failed: {e}, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.warning(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                    _redis_available = False
                    _redis_client = None
                    RedisClient._record_failure()
                    return None
        
        return None
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if Redis is available and enabled.
        
        Returns:
            True if Redis is enabled and connected, False otherwise
        """
        global _redis_available
        
        if not _redis_available:
            # Try to get client (will set _redis_available)
            RedisClient.get_client()
        
        return _redis_available
    
    @staticmethod
    def test_connection() -> bool:
        """
        Test Redis connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        client = RedisClient.get_client()
        if client is None:
            return False
        
        try:
            client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis connection test failed: {e}")
            return False
    
    @staticmethod
    def get_key_prefix() -> str:
        """
        Get Redis key prefix from config.
        
        Returns:
            Key prefix string (e.g., "codelumen:")
        """
        global _redis_config
        
        if _redis_config is None:
            try:
                from services.db_config import get_db_config
                config = get_db_config()
                _redis_config = config.redis
            except Exception:
                return "codelumen:"
        
        return _redis_config.key_prefix if _redis_config else "codelumen:"
    
    @staticmethod
    def get_stream_maxlen() -> int:
        """
        Get maximum stream length from config.
        
        Returns:
            Maximum number of events per stream
        """
        global _redis_config
        
        if _redis_config is None:
            try:
                from services.db_config import get_db_config
                config = get_db_config()
                _redis_config = config.redis
            except Exception:
                return 10000
        
        # Default stream maxlen if not in config
        return getattr(_redis_config, 'stream_maxlen', 10000) if _redis_config else 10000
    
    @staticmethod
    def get_client_with_fallback() -> Any:
        """
        Get Redis client with automatic fallback to in-memory storage.
        
        Returns:
            Redis client if available, otherwise InMemoryFallback instance
        """
        global _fallback
        
        client = RedisClient.get_client()
        if client is not None:
            return client
        
        # Return fallback
        if _fallback is None:
            _fallback = InMemoryFallback()
            logger.info("Using in-memory fallback for Redis operations")
        
        return _fallback
    
    @staticmethod
    def reset_client() -> None:
        """
        Reset Redis client (useful for testing or reconnection).
        """
        global _redis_client, _redis_available, _circuit_state, _circuit_failures
        _redis_client = None
        _redis_available = False
        with _circuit_lock:
            _circuit_state = CircuitState.CLOSED
            _circuit_failures = 0
    
    @staticmethod
    def check_health():
        """
        Check Redis client health.
        
        Returns:
            ServiceHealth object (or dict if health_check module not available)
        """
        try:
            from services.health_check import ServiceHealth, HealthStatus
        except ImportError:
            # Health check module may not exist, return simple status
            # This is a fallback for backward compatibility
            try:
                client = RedisClient.get_client()
                if client is None:
                    return {"status": "unhealthy", "message": "Redis client unavailable"}
                client.ping()
                return {"status": "healthy", "message": "Redis client is healthy"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        
        try:
            client = RedisClient.get_client()
            if client is None:
                # Check if using fallback
                fallback_client = RedisClient.get_client_with_fallback()
                if isinstance(fallback_client, InMemoryFallback):
                    stats = fallback_client.get_stats()
                    return ServiceHealth(
                        status=HealthStatus.DEGRADED,
                        details={
                            "using_fallback": True,
                            "fallback_stats": stats,
                            "circuit_state": _circuit_state.value
                        },
                        message="Redis unavailable, using in-memory fallback"
                    )
                else:
                    return ServiceHealth(
                        status=HealthStatus.UNHEALTHY,
                        details={"circuit_state": _circuit_state.value},
                        message="Redis client unavailable"
                    )
            
            # Test connection
            client.ping()
            
            with _circuit_lock:
                circuit_state = _circuit_state.value
                circuit_failures = _circuit_failures
            
            return ServiceHealth(
                status=HealthStatus.HEALTHY,
                details={
                    "host": _redis_config.host if _redis_config else "unknown",
                    "port": _redis_config.port if _redis_config else "unknown",
                    "circuit_state": circuit_state,
                    "circuit_failures": circuit_failures
                },
                message="Redis client is healthy"
            )
        except Exception as e:
            with _circuit_lock:
                circuit_state = _circuit_state.value
            
            return ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                details={"error": str(e), "circuit_state": circuit_state},
                message=f"Redis health check failed: {e}"
            )
    
    @staticmethod
    def cache_get(key: str, default: Any = None) -> Any:
        """
        Get value from cache with automatic prefix handling.
        
        Args:
            key: Cache key (without prefix)
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        import json
        
        client = RedisClient.get_client()
        if client is None:
            logger.debug(f"Redis not available, returning default for key: {key}")
            return default
        
        prefix = RedisClient.get_key_prefix()
        full_key = key if key.startswith(prefix) else f"{prefix}{key}"
        
        try:
            data = client.get(full_key)
            if data is None:
                return default
            
            # Deserialize JSON data
            if isinstance(data, bytes):
                return json.loads(data.decode('utf-8'))
            return data
        except Exception as e:
            logger.warning(f"Cache get failed for {key}: {e}")
            return default
    
    @staticmethod
    def cache_set(key: str, value: Any, ttl: Optional[int] = 3600) -> bool:
        """
        Set value in cache with TTL and automatic prefix handling.
        
        Args:
            key: Cache key (without prefix)
            value: Value to cache
            ttl: TTL in seconds (default: 3600)
            
        Returns:
            True if successful, False otherwise
        """
        import json
        
        client = RedisClient.get_client()
        if client is None:
            logger.debug(f"Redis not available, skipping cache set for key: {key}")
            return False
        
        prefix = RedisClient.get_key_prefix()
        full_key = key if key.startswith(prefix) else f"{prefix}{key}"
        
        try:
            # Serialize value to JSON
            data = json.dumps(value).encode('utf-8')
            
            if ttl and ttl > 0:
                result = client.setex(full_key, ttl, data)
            else:
                result = client.set(full_key, data)
            
            return bool(result)
        except Exception as e:
            logger.warning(f"Cache set failed for {key}: {e}")
            return False
    
    @staticmethod
    def cache_delete(key: str) -> bool:
        """
        Delete key from cache with automatic prefix handling.
        
        Args:
            key: Cache key (without prefix)
            
        Returns:
            True if deleted, False otherwise
        """
        client = RedisClient.get_client()
        if client is None:
            logger.debug(f"Redis not available, skipping cache delete for key: {key}")
            return False
        
        prefix = RedisClient.get_key_prefix()
        full_key = key if key.startswith(prefix) else f"{prefix}{key}"
        
        try:
            result = client.delete(full_key)
            return bool(result)
        except Exception as e:
            logger.warning(f"Cache delete failed for {key}: {e}")
            return False
    
    @staticmethod
    def cache_exists(key: str) -> bool:
        """
        Check if key exists in cache with automatic prefix handling.
        
        Args:
            key: Cache key (without prefix)
            
        Returns:
            True if exists, False otherwise
        """
        client = RedisClient.get_client()
        if client is None:
            return False
        
        prefix = RedisClient.get_key_prefix()
        full_key = key if key.startswith(prefix) else f"{prefix}{key}"
        
        try:
            return bool(client.exists(full_key))
        except Exception as e:
            logger.warning(f"Cache exists check failed for {key}: {e}")
            return False
    
    @staticmethod
    def cache_ttl(key: str) -> Optional[int]:
        """
        Get remaining TTL for key with automatic prefix handling.
        
        Args:
            key: Cache key (without prefix)
            
        Returns:
            TTL in seconds, None if key doesn't exist or has no expiration
        """
        client = RedisClient.get_client()
        if client is None:
            return None
        
        prefix = RedisClient.get_key_prefix()
        full_key = key if key.startswith(prefix) else f"{prefix}{key}"
        
        try:
            ttl = client.ttl(full_key)
            return ttl if ttl >= 0 else None
        except Exception as e:
            logger.warning(f"Cache TTL check failed for {key}: {e}")
            return None
    
    @staticmethod
    def cache_delete_by_prefix(prefix: str) -> int:
        """
        Delete all keys with the given prefix.
        
        Uses SCAN for production-safe pattern matching.
        
        Args:
            prefix: Key prefix to match (without global prefix)
            
        Returns:
            Number of keys deleted
        """
        client = RedisClient.get_client()
        if client is None:
            logger.debug(f"Redis not available, skipping cache delete by prefix: {prefix}")
            return 0
        
        global_prefix = RedisClient.get_key_prefix()
        full_prefix = prefix if prefix.startswith(global_prefix) else f"{global_prefix}{prefix}"
        
        # Ensure prefix ends with * for pattern matching
        pattern = f"{full_prefix}*" if not full_prefix.endswith('*') else full_prefix
        
        try:
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += client.delete(*keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as e:
            logger.warning(f"Cache delete by prefix failed for {prefix}: {e}")
            return 0
    
    @staticmethod
    def cache_delete_by_pattern(pattern: str) -> int:
        """
        Delete all keys matching pattern (supports * wildcard).
        
        Uses SCAN for production-safe pattern matching.
        
        Args:
            pattern: Key pattern to match (without global prefix)
            
        Returns:
            Number of keys deleted
        """
        client = RedisClient.get_client()
        if client is None:
            logger.debug(f"Redis not available, skipping cache delete by pattern: {pattern}")
            return 0
        
        global_prefix = RedisClient.get_key_prefix()
        full_pattern = pattern if pattern.startswith(global_prefix) else f"{global_prefix}{pattern}"
        
        try:
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=full_pattern, count=100)
                if keys:
                    deleted += client.delete(*keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as e:
            logger.warning(f"Cache delete by pattern failed for {pattern}: {e}")
            return 0
    
    @staticmethod
    def cache_batch_delete(keys: List[str]) -> int:
        """
        Batch delete multiple keys with automatic prefix handling.
        
        Args:
            keys: List of cache keys (without prefix)
            
        Returns:
            Number of keys deleted
        """
        client = RedisClient.get_client()
        if client is None:
            logger.debug(f"Redis not available, skipping cache batch delete for {len(keys)} keys")
            return 0
        
        if not keys:
            return 0
        
        prefix = RedisClient.get_key_prefix()
        full_keys = [key if key.startswith(prefix) else f"{prefix}{key}" for key in keys]
        
        try:
            result = client.delete(*full_keys)
            return result if result else 0
        except Exception as e:
            logger.warning(f"Cache batch delete failed: {e}")
            return 0
    
    @staticmethod
    def cache_get_size() -> Optional[int]:
        """
        Get approximate cache size (number of keys).
        
        Returns:
            Approximate number of keys in cache, None if unavailable
        """
        client = RedisClient.get_client()
        if client is None:
            return None
        
        prefix = RedisClient.get_key_prefix()
        pattern = f"{prefix}*"
        
        try:
            count = 0
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                count += len(keys)
                if cursor == 0:
                    break
            return count
        except Exception as e:
            logger.warning(f"Cache get size failed: {e}")
            return None
    
    @staticmethod
    def cache_get_keys(pattern: str = "*", count: int = 100) -> List[str]:
        """
        Get keys matching pattern (supports * wildcard).
        
        Uses SCAN for production-safe pattern matching.
        
        Args:
            pattern: Key pattern to match (without global prefix)
            count: Maximum number of keys to return
            
        Returns:
            List of matching keys (without global prefix)
        """
        client = RedisClient.get_client()
        if client is None:
            return []
        
        global_prefix = RedisClient.get_key_prefix()
        full_pattern = pattern if pattern.startswith(global_prefix) else f"{global_prefix}{pattern}"
        
        try:
            keys = []
            cursor = 0
            while len(keys) < count:
                cursor, found_keys = client.scan(cursor, match=full_pattern, count=count)
                for key in found_keys:
                    if len(keys) >= count:
                        break
                    # Remove global prefix from returned keys
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    if key_str.startswith(global_prefix):
                        keys.append(key_str[len(global_prefix):])
                    else:
                        keys.append(key_str)
                if cursor == 0:
                    break
            return keys[:count]
        except Exception as e:
            logger.warning(f"Cache get keys failed for pattern {pattern}: {e}")
            return []
    
    @staticmethod
    def cache_get_info() -> Optional[Dict[str, Any]]:
        """
        Get Redis INFO for cache database.
        
        Returns:
            Dictionary with Redis INFO data, None if unavailable
        """
        client = RedisClient.get_client()
        if client is None:
            return None
        
        try:
            info = client.info()
            # Convert bytes keys to strings if needed
            if isinstance(info, dict):
                return {k.decode('utf-8') if isinstance(k, bytes) else k: 
                        v.decode('utf-8') if isinstance(v, bytes) else v 
                        for k, v in info.items()}
            return info
        except Exception as e:
            logger.warning(f"Cache get info failed: {e}")
            return None

