"""
Model capability detection utilities.

Detects which models support tool calling and other capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# In-memory cache for model capabilities
_capability_cache: Dict[str, bool] = {}


def check_tool_calling_support(model_name: str, test_results_path: Optional[str] = None) -> bool:
    """
    Check if a model supports tool calling.
    
    Args:
        model_name: Name of the model to check
        test_results_path: Optional path to tool_calling_test_results.json
    
    Returns:
        True if model supports tool calling, False otherwise
    """
    # Check cache first
    if model_name in _capability_cache:
        return _capability_cache[model_name]
    
    # Default path - check archived location first, then root as fallback
    if test_results_path is None:
        project_root = Path(__file__).parent.parent
        # Check archived location first (where historical results are stored)
        archived_path = project_root / "scripts" / "archived" / "verification" / "tool_calling_test_results.json"
        root_path = project_root / "tool_calling_test_results.json"
        # Prefer archived location if it exists, otherwise use root (for newly generated files)
        test_results_path = archived_path if archived_path.exists() else root_path
    
    # Load test results
    try:
        if not Path(test_results_path).exists():
            logger.warning(f"Tool calling test results not found at {test_results_path}")
            # Default to False if we can't determine
            _capability_cache[model_name] = False
            return False
        
        with open(test_results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Find model in results
        for result in results:
            if result.get("model") == model_name:
                supports = result.get("supports_tool_calling", False)
                _capability_cache[model_name] = supports
                logger.debug(f"Model {model_name} tool calling support: {supports}")
                return supports
        
        # Model not found in test results
        logger.debug(f"Model {model_name} not found in test results, assuming no tool calling support")
        _capability_cache[model_name] = False
        return False
        
    except Exception as e:
        logger.warning(f"Error checking tool calling support for {model_name}: {e}")
        # Default to False on error
        _capability_cache[model_name] = False
        return False


def clear_capability_cache():
    """Clear the capability cache."""
    global _capability_cache
    _capability_cache.clear()
    logger.debug("Capability cache cleared")


def get_capability_cache() -> Dict[str, bool]:
    """Get the current capability cache."""
    return _capability_cache.copy()

