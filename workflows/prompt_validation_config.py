"""
Prompt validation configuration loader.

Loads prompt validation thresholds from config.yaml with fallback to hardcoded defaults.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# Hardcoded fallback defaults (used if config.yaml is missing or invalid)
FALLBACK_MIN_VALIDITY_SCORE = 0.7
FALLBACK_HIGH_CONFIDENCE_THRESHOLD = 0.9


class PromptValidationConfig:
    """Prompt validation configuration loaded from config.yaml."""
    
    def __init__(self):
        self.min_validity_score: float = FALLBACK_MIN_VALIDITY_SCORE
        self.high_confidence_threshold: float = FALLBACK_HIGH_CONFIDENCE_THRESHOLD
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from config.yaml with fallback to defaults."""
        try:
            from llm.config import ConfigManager
            
            config_manager = ConfigManager()
            config = config_manager.load_config()
            
            # Try to get prompt_validation from config
            if hasattr(config, 'prompt_validation') and config.prompt_validation:
                prompt_config = config.prompt_validation
                
                # prompt_validation is stored as a dict
                if isinstance(prompt_config, dict):
                    # Load min_validity_score with validation
                    min_score = prompt_config.get("min_validity_score", FALLBACK_MIN_VALIDITY_SCORE)
                    if self._validate_score(min_score, "min_validity_score"):
                        self.min_validity_score = float(min_score)
                    else:
                        self.min_validity_score = FALLBACK_MIN_VALIDITY_SCORE
                    
                    # Load high_confidence_threshold with validation
                    high_threshold = prompt_config.get("high_confidence_threshold", FALLBACK_HIGH_CONFIDENCE_THRESHOLD)
                    if self._validate_score(high_threshold, "high_confidence_threshold"):
                        self.high_confidence_threshold = float(high_threshold)
                    else:
                        self.high_confidence_threshold = FALLBACK_HIGH_CONFIDENCE_THRESHOLD
                    
                    # Validate that high_confidence_threshold >= min_validity_score
                    if self.high_confidence_threshold < self.min_validity_score:
                        logger.warning(
                            f"high_confidence_threshold ({self.high_confidence_threshold}) is less than "
                            f"min_validity_score ({self.min_validity_score}). This may cause unexpected behavior. "
                            f"Using defaults."
                        )
                        self._load_fallbacks()
                        return
                    
                    logger.info(
                        f"Loaded prompt validation config: min_validity_score={self.min_validity_score}, "
                        f"high_confidence_threshold={self.high_confidence_threshold}"
                    )
                else:
                    # If it's not a dict, use fallbacks
                    self._load_fallbacks()
                    logger.warning("Prompt validation config is not a dict, using fallback defaults")
                    return
            else:
                # No config found, use fallbacks
                self._load_fallbacks()
                logger.debug("Prompt validation config not found in config.yaml, using fallback defaults")
        
        except Exception as e:
            logger.error(f"Error loading prompt validation config: {e}", exc_info=True)
            self._load_fallbacks()
            logger.warning("Using fallback prompt validation defaults due to error")
    
    def _validate_score(self, value: any, name: str) -> bool:
        """
        Validate that a score value is numeric and in range [0.0, 1.0].
        
        Args:
            value: Value to validate
            name: Name of the config field (for logging)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            float_value = float(value)
            if 0.0 <= float_value <= 1.0:
                return True
            else:
                logger.warning(
                    f"Invalid {name} value: {value}. Must be in range [0.0, 1.0]. Using fallback default."
                )
                return False
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid {name} value: {value}. Must be a numeric value. Using fallback default."
            )
            return False
    
    def _load_fallbacks(self) -> None:
        """Load fallback defaults."""
        self.min_validity_score = FALLBACK_MIN_VALIDITY_SCORE
        self.high_confidence_threshold = FALLBACK_HIGH_CONFIDENCE_THRESHOLD


# Singleton instance
_config_instance: Optional[PromptValidationConfig] = None
_config_lock = threading.Lock()


def get_prompt_validation_config() -> PromptValidationConfig:
    """
    Get prompt validation configuration (singleton).
    
    Returns:
        PromptValidationConfig instance
    """
    global _config_instance
    
    if _config_instance is None:
        with _config_lock:
            if _config_instance is None:
                _config_instance = PromptValidationConfig()
    
    return _config_instance
