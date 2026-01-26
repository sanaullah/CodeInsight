"""
Centralized cache configuration and TTL management.

Provides TTL values, cache key patterns, and validation functions.
"""

import re
from typing import Optional, Dict, Any
from services.redis_client import RedisClient

# TTL configuration by cache type (in seconds)
CACHE_TTL = {
    "architecture_model": 86400,  # 24 hours
    "prompt": 43200,  # 12 hours
    "knowledge_base": 21600,  # 6 hours
    "experience": 21600,  # 6 hours
    "scan_history": 3600,  # 1 hour
    "swarm_skillbook": 3600,  # 1 hour (default, adjusted by confidence)
    "query_result": 900,  # 15 minutes
    "metadata": 1800,  # 30 minutes
    "settings": 900,  # 15 minutes (settings change infrequently but need reasonable freshness)
}

# Cache key patterns for each service
CACHE_KEY_PATTERNS = {
    "scan_history": {
        "id": "scan_history:id:{id}",
        "project": "scan_history:project:{project_path}",
        "query": "scan_history:query:{query_hash}",
    },
    "architecture_model": {
        "project": "architecture_model:project:{project_hash}",
        "hash": "architecture_model:hash:{model_hash}",
    },
    "prompt": {
        "role": "prompt:role:{role}:hash:{hash}",
        "id": "prompt:id:{prompt_id}",
    },
    "knowledge_base": {
        "id": "knowledge_base:id:{knowledge_id}",
        "type": "knowledge_base:type:{type}",
    },
    "experience": {
        "id": "experience:id:{experience_id}",
        "agent": "experience:agent:{agent_name}",
    },
    "swarm_skillbook": {
        "skill": "swarm_skillbook:skill:{skill_id}",
        "type": "swarm_skillbook:type:{skill_type}",
    },
}


def get_ttl_for_cache_type(cache_type: str) -> int:
    """
    Get TTL for cache type.
    
    Args:
        cache_type: Cache type (e.g., 'architecture_model', 'scan_history')
        
    Returns:
        TTL in seconds, or default 3600 if not found
    """
    return CACHE_TTL.get(cache_type, 3600)  # Default: 1 hour


def get_cache_key_pattern(service: str, type: str) -> Optional[str]:
    """
    Get key pattern for service/type.
    
    Args:
        service: Service name
        type: Object type
        
    Returns:
        Key pattern string or None if not found
    """
    service_patterns = CACHE_KEY_PATTERNS.get(service)
    if service_patterns:
        return service_patterns.get(type)
    return None


def validate_cache_key(key: str, prefix: Optional[str] = None) -> tuple[bool, Optional[str]]:
    """
    Validate cache key format.
    
    Expected format: {prefix}{service}:{type}:{identifier}
    
    Args:
        key: Cache key to validate
        prefix: Optional prefix override (defaults to RedisClient.get_key_prefix())
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if prefix is None:
        prefix = RedisClient.get_key_prefix()
    
    # Check if key starts with prefix
    if not key.startswith(prefix):
        return False, f"Key must start with prefix '{prefix}'"
    
    # Remove prefix
    key_without_prefix = key[len(prefix):]
    
    # Check format: {service}:{type}:{identifier}
    # Must have at least 3 parts separated by colons
    parts = key_without_prefix.split(':')
    if len(parts) < 3:
        return False, f"Key must have format '{prefix}{{service}}:{{type}}:{{identifier}}'"
    
    # Validate parts are not empty
    if not all(part for part in parts):
        return False, "Key parts cannot be empty"
    
    # Validate identifier doesn't contain invalid characters
    identifier = parts[2]
    if not re.match(r'^[a-zA-Z0-9_\-./]+$', identifier):
        return False, "Identifier contains invalid characters"
    
    return True, None


def format_cache_key(service: str, type: str, identifier: str, prefix: Optional[str] = None) -> str:
    """
    Format cache key using pattern if available, otherwise use convention.
    
    Args:
        service: Service name
        type: Object type
        identifier: Unique identifier
        prefix: Optional prefix override
        
    Returns:
        Formatted cache key
    """
    if prefix is None:
        prefix = RedisClient.get_key_prefix()
    
    # Try to get pattern
    pattern = get_cache_key_pattern(service, type)
    if pattern:
        # Replace placeholders
        key = pattern.format(**{type: identifier})
        if not key.startswith(prefix):
            key = f"{prefix}{key}"
        return key
    
    # Fall back to convention
    return f"{prefix}{service}:{type}:{identifier}"

