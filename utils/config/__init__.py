"""
Configuration and Environment configuration utilities.
"""

from .env_config import get_env_var
from .env_loader import load_env, ensure_env_loaded
from .logging_config import setup_logging, setup_logging_from_config
from .project_settings import (
    get_project_key,
    get_project_selected_directories,
    save_project_selected_directories
)
from .settings_db import (
    init_settings_db,
    get_setting,
    set_setting,
    get_all_settings,
    reset_user_settings,
    verify_setting_sync,
    get_user_id
)
from .settings_validator import validate_setting, get_all_defaults
