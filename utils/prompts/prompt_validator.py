"""
Shared prompt validation logic for prompt generation and validation nodes.

Provides consistent validation rules (technology filtering, token limits) that
are applied in both generation and validation nodes to prevent inconsistencies.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PromptValidationResult:
    """Result of prompt validation."""
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptValidator:
    """
    Shared prompt validator for consistent validation across generation and validation nodes.
    
    Provides lenient validation (warns but doesn't fail) to match current LLM-based
    validation behavior.
    """
    
    def __init__(self):
        """Initialize the prompt validator."""
        pass
    
    def validate_technology_filtering(
        self,
        prompt: str,
        architecture_model: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that prompt doesn't contain forbidden technologies.
        
        Checks if prompt contains technologies not present in architecture model.
        Uses the same logic as remove_hardcoded_technologies() but returns violations
        instead of removing them.
        
        Args:
            prompt: Prompt text to validate
            architecture_model: Architecture model dictionary
            
        Returns:
            Tuple of (is_valid, violations_list)
        """
        if not prompt or not architecture_model:
            return True, []
        
        try:
            from .prompt_technology_filter import check_forbidden_technologies
            
            # Use helper function from prompt_technology_filter
            violations = check_forbidden_technologies(prompt, architecture_model)
            is_valid = len(violations) == 0
            
            return is_valid, violations
            
        except ImportError:
            # If helper doesn't exist yet, fall back to basic check
            logger.warning("check_forbidden_technologies not available, using basic validation")
            return True, []
        except Exception as e:
            logger.error(f"Error validating technology filtering: {e}", exc_info=True)
            # Return valid on error (lenient validation)
            return True, []
    
    def validate_token_limit(
        self,
        prompt: str,
        model_name: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that prompt doesn't exceed model's token limit.
        
        Args:
            prompt: Prompt text to validate
            model_name: Optional model name to get token limit from config
            
        Returns:
            Tuple of (is_valid, warning_message)
        """
        if not prompt:
            return True, None
        
        if not model_name:
            # No model specified, can't validate
            return True, None
        
        try:
            from llm.config import ConfigManager
            
            # Get model config
            config_manager = ConfigManager()
            model_config = config_manager.get_model_config(model_name)
            
            if not model_config:
                # Model not in registry, can't validate
                logger.debug(f"Model '{model_name}' not in config, skipping token limit validation")
                return True, None
            
            max_tokens = model_config.max_input_tokens
            if not max_tokens:
                # No limit specified
                return True, None
            
            # Count tokens
            token_count = self._count_tokens(prompt, model_name)
            
            if token_count > max_tokens:
                warning = (
                    f"Prompt exceeds token limit: {token_count} tokens "
                    f"(limit: {max_tokens} for model '{model_name}')"
                )
                return False, warning
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating token limit: {e}", exc_info=True)
            # Return valid on error (lenient validation)
            return True, None
    
    def _count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """
        Count tokens in text.
        
        Uses TokenCounter if available, otherwise falls back to estimate.
        
        Args:
            text: Text to count tokens for
            model_name: Optional model name for accurate counting
            
        Returns:
            Token count
        """
        try:
            from chunking.token_counter import TokenCounter
            
            counter = TokenCounter(model_name=model_name)
            return counter.count_tokens(text)
        except Exception as e:
            logger.debug(f"TokenCounter not available: {e}, using estimate")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def validate_all(
        self,
        prompt: str,
        architecture_model: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> PromptValidationResult:
        """
        Run all validation rules and return consolidated result.
        
        Args:
            prompt: Prompt text to validate
            architecture_model: Optional architecture model for technology filtering
            model_name: Optional model name for token limit validation
            **kwargs: Additional validation parameters (unused for now)
            
        Returns:
            PromptValidationResult with all validation results
        """
        warnings = []
        violations = []
        metadata = {}
        
        # Validate technology filtering
        if architecture_model:
            tech_valid, tech_violations = self.validate_technology_filtering(
                prompt, architecture_model
            )
            if not tech_valid:
                violations.extend(tech_violations)
                warnings.append(
                    f"Found {len(tech_violations)} forbidden technology references: "
                    f"{', '.join(tech_violations[:5])}"
                    + (f" and {len(tech_violations) - 5} more" if len(tech_violations) > 5 else "")
                )
                metadata["technology_violations"] = tech_violations
        
        # Validate token limit
        if model_name:
            token_valid, token_warning = self.validate_token_limit(prompt, model_name)
            if not token_valid and token_warning:
                warnings.append(token_warning)
                metadata["token_limit_exceeded"] = True
                # Get actual token count for metadata
                token_count = self._count_tokens(prompt, model_name)
                metadata["token_count"] = token_count
                try:
                    from llm.config import ConfigManager
                    config_manager = ConfigManager()
                    model_config = config_manager.get_model_config(model_name)
                    if model_config:
                        metadata["token_limit"] = model_config.max_input_tokens
                except Exception:
                    pass
        
        # Overall validation result (lenient: warnings don't make it invalid)
        is_valid = True  # Always valid (lenient validation)
        
        return PromptValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            violations=violations,
            metadata=metadata
        )


def format_shared_validation_results(result: PromptValidationResult) -> str:
    """
    Format validation results for inclusion in LLM validation prompt.
    
    Args:
        result: PromptValidationResult to format
        
    Returns:
        Formatted string for LLM prompt
    """
    if not result.warnings and not result.violations:
        return "**Pre-validation Checks:** All checks passed (no warnings or violations)."
    
    lines = ["**Pre-validation Checks:**"]
    
    if result.warnings:
        lines.append("\n**Warnings:**")
        for warning in result.warnings:
            lines.append(f"- {warning}")
    
    if result.violations:
        lines.append("\n**Violations:**")
        for violation in result.violations[:10]:  # Limit to first 10
            lines.append(f"- {violation}")
        if len(result.violations) > 10:
            lines.append(f"- ... and {len(result.violations) - 10} more violations")
    
    if result.metadata:
        lines.append("\n**Metadata:**")
        if "token_count" in result.metadata:
            lines.append(f"- Token count: {result.metadata['token_count']}")
        if "token_limit" in result.metadata:
            lines.append(f"- Token limit: {result.metadata['token_limit']}")
    
    return "\n".join(lines)




