"""
Settings validation for CodeInsight.
"""

import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


def validate_setting(key: str, value: Any) -> Tuple[bool, Optional[str]]:
    """
    Validate a setting value.
    
    Args:
        key: Setting key
        value: Setting value to validate
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Define valid values for each setting
    validations = {
        "ui_layout_mode": lambda v: v in ["wide", "centered"],
        "ui_display_density": lambda v: v in ["compact", "spacious"],
        "ui_sidebar_default": lambda v: v in ["expanded", "collapsed"],
        "ui_animations_enabled": lambda v: isinstance(v, bool),
        "ui_auto_expand_sections": lambda v: isinstance(v, bool),
        "ui_show_help_text": lambda v: isinstance(v, bool),
        "ui_default_results_tab": lambda v: isinstance(v, str),
        "ui_progress_style": lambda v: v in ["minimal", "detailed"],
        "ui_items_per_page": lambda v: isinstance(v, int) and v > 0,
        "selected_model": lambda v: isinstance(v, str),
    }
    
    if key in validations:
        if not validations[key](value):
            return False, f"Invalid value for {key}: {value}"
    
    return True, None


def get_all_defaults() -> Dict[str, Any]:
    """
    Get all default settings.
    
    Returns:
        Dictionary of default settings
    """
    try:
        from llm.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        defaults = {
            "ui_layout_mode": "wide",
            "ui_display_density": "spacious",
            "ui_sidebar_default": "expanded",
            "ui_animations_enabled": True,
            "ui_auto_expand_sections": False,
            "ui_show_help_text": True,
            "ui_default_results_tab": "Summary",
            "ui_progress_style": "detailed",
            "ui_items_per_page": 10,
            "selected_model": getattr(config, 'default_model', 'qwen/qwen3-coder'),
        }
        
        return defaults
    except Exception as e:
        logger.warning(f"Error loading defaults from config: {e}")
        # Return hardcoded defaults
        return {
            "ui_layout_mode": "wide",
            "ui_display_density": "spacious",
            "ui_sidebar_default": "expanded",
            "ui_animations_enabled": True,
            "ui_auto_expand_sections": False,
            "ui_show_help_text": True,
            "ui_default_results_tab": "Summary",
            "ui_progress_style": "detailed",
            "ui_items_per_page": 10,
            "selected_model": "qwen/qwen3-coder",
        }

