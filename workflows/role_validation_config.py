"""
Role validation configuration loader.

Loads role validation rules from config.yaml with fallback to hardcoded defaults.
"""

import re
import logging
import threading
from typing import List, Dict, Optional, Pattern

logger = logging.getLogger(__name__)

# Hardcoded fallback defaults (used if config.yaml is missing or invalid)
FALLBACK_SAFE_PHRASES = [
    "command pattern",
    "command pattern validator",
    "execution pattern",
    "execution context",
    "execution flow",
    "execution path",
    "execution model",
    "execution environment",
    "system architecture",
    "system design",
    "system analyst",
    "system reviewer",
    "architecture pattern",
    "design pattern",
    "pattern validator",
    "pattern analyzer",
    "tool execution",
    "code execution",
    "shell script",
    "shell integration",
    "root cause",
    "root analysis",
    "admin panel",
    "admin interface",
    "admin dashboard",
]

FALLBACK_DANGEROUS_PATTERNS = [
    r"(?i)\b(delete|drop|truncate|remove)\s+(all|everything|data|database|table|file|system)\b",
    r"(?i)\b(execute|exec|eval)\s+(arbitrary|system|shell|command|code)\b",
    r"(?i)\b(sudo|root)\s+(access|privilege|permission|execute)\b",
    r"(?i)\b(grant|revoke)\s+(all|admin|root|sudo)\b",
    r"(?i)\b(alter|update|insert)\s+(system|database|config|security)\b",
    r"(?i)^(admin|root|sudo|system|shell|command|execute|delete|drop|truncate|remove|alter|update|insert|grant|revoke|eval|exec)\s*$",
]

FALLBACK_CHAR_NORMALIZATION = {
    "&": "and",
    "@": "at",
    "#": "hash",
    "$": "dollar",
    "%": "percent",
    "+": "plus",
    "=": "equals",
    "|": "pipe",
    "\\": "backslash",
    "/": "slash",
    "<": "less",
    ">": "greater",
    "?": "question",
    "!": "exclamation",
    "*": "star",
    "~": "tilde",
    "^": "caret",
    "[": "left_bracket",
    "]": "right_bracket",
    "{": "left_brace",
    "}": "right_brace",
    "(": "left_paren",
    ")": "right_paren",
}

FALLBACK_ALLOWED_CHAR_PATTERN = r'^[a-zA-Z0-9\s\-_&()]+$'
FALLBACK_MAX_LENGTH = 100


class RoleValidationConfig:
    """Role validation configuration loaded from config.yaml."""
    
    def __init__(self):
        self.safe_phrases: List[str] = []
        self.dangerous_patterns: List[str] = []
        self.compiled_dangerous_patterns: List[Pattern] = []
        self.char_normalization: Dict[str, str] = {}
        self.allowed_char_pattern: str = ""
        self.compiled_allowed_char_pattern: Optional[Pattern] = None
        self.max_role_name_length: int = 100
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from config.yaml with fallback to defaults."""
        try:
            from llm.config import ConfigManager
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Try to get role_validation from config
            if hasattr(config, 'role_validation') and config.role_validation:
                role_config = config.role_validation
                
                # role_validation is stored as a dict
                if isinstance(role_config, dict):
                    self.safe_phrases = role_config.get("safe_phrases", FALLBACK_SAFE_PHRASES)
                    self.dangerous_patterns = role_config.get("dangerous_patterns", FALLBACK_DANGEROUS_PATTERNS)
                    self.char_normalization = role_config.get("char_normalization", FALLBACK_CHAR_NORMALIZATION)
                    self.allowed_char_pattern = role_config.get("allowed_char_pattern", FALLBACK_ALLOWED_CHAR_PATTERN)
                    self.max_role_name_length = role_config.get("max_role_name_length", FALLBACK_MAX_LENGTH)
                else:
                    # If it's not a dict, use fallbacks
                    self._load_fallbacks()
                    logger.warning("Role validation config is not a dict, using fallback defaults")
                    return
                
                # Compile patterns
                self.compiled_dangerous_patterns = [
                    re.compile(pattern) for pattern in self.dangerous_patterns
                ]
                self.compiled_allowed_char_pattern = re.compile(self.allowed_char_pattern)
                
                logger.info(f"Loaded role validation config: {len(self.safe_phrases)} safe phrases, "
                          f"{len(self.dangerous_patterns)} dangerous patterns")
            else:
                # No config found, use fallbacks
                self._load_fallbacks()
                logger.warning("Role validation config not found in config.yaml, using fallback defaults")
        
        except Exception as e:
            logger.error(f"Error loading role validation config: {e}", exc_info=True)
            self._load_fallbacks()
            logger.warning("Using fallback role validation defaults due to error")
    
    def _load_fallbacks(self) -> None:
        """Load fallback defaults."""
        self.safe_phrases = FALLBACK_SAFE_PHRASES.copy()
        self.dangerous_patterns = FALLBACK_DANGEROUS_PATTERNS.copy()
        self.char_normalization = FALLBACK_CHAR_NORMALIZATION.copy()
        self.allowed_char_pattern = FALLBACK_ALLOWED_CHAR_PATTERN
        self.max_role_name_length = FALLBACK_MAX_LENGTH
        
        # Compile patterns
        self.compiled_dangerous_patterns = [
            re.compile(pattern) for pattern in self.dangerous_patterns
        ]
        self.compiled_allowed_char_pattern = re.compile(self.allowed_char_pattern)


# Singleton instance
_config_instance: Optional[RoleValidationConfig] = None
_config_lock = threading.Lock()


def get_role_validation_config() -> RoleValidationConfig:
    """
    Get role validation configuration (singleton).
    
    Returns:
        RoleValidationConfig instance
    """
    global _config_instance
    
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = RoleValidationConfig()
    
    return _config_instance

