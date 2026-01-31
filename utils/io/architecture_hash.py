"""
Architecture hash computation utility for prompt constraint enforcement.

Computes stable SHA256 hashes of architecture models for use in:
- Prompt caching (cache key generation)
- Constraint enforcement (preventing technology hallucinations)
- Prompt validation (ensuring architecture consistency)
"""

import json
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def compute_architecture_hash(architecture_model: Dict[str, Any]) -> str:
    """
    Compute SHA256 hash of architecture model for constraint enforcement.
    
    Creates a stable hash based on architecture model content, excluding
    timestamps and version numbers for consistency. This hash uniquely
    identifies the detected technologies, frameworks, and patterns.
    
    Args:
        architecture_model: Architecture model dictionary (from build_architecture_node)
        
    Returns:
        SHA256 hash as hexadecimal string
    """
    if not architecture_model:
        logger.warning("Empty architecture model provided for hash computation")
        return hashlib.sha256(b"").hexdigest()
    
    try:
        # Create stable representation (exclude created_at, version, and other non-deterministic fields)
        arch_dict = architecture_model.copy()
        
        # Remove non-deterministic fields
        arch_dict.pop("created_at", None)
        arch_dict.pop("version", None)
        
        # Ensure consistent serialization
        # Sort keys for consistency regardless of insertion order
        arch_str = json.dumps(arch_dict, sort_keys=True, default=str)
        
        # Compute hash
        hash_value = hashlib.sha256(arch_str.encode()).hexdigest()
        
        logger.debug(f"Computed architecture hash: {hash_value[:8]}...")
        return hash_value
        
    except Exception as e:
        logger.error(f"Error computing architecture hash: {e}", exc_info=True)
        # Return empty hash on error
        return hashlib.sha256(b"").hexdigest()

