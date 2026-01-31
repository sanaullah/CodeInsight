"""
Role definitions and validation for Swarm Analysis workflow.

Provides type-safe role definitions and validation to prevent security vulnerabilities.
"""

from enum import StrEnum
from typing import List, Optional, Pattern
import re
import logging
from pydantic import BaseModel, Field, field_validator
from workflows.role_validation_config import get_role_validation_config

logger = logging.getLogger(__name__)


class RoleEnum(StrEnum):
    """
    Enumeration of standard analysis roles.
    
    These are the predefined roles that are always safe to use.
    Custom roles are allowed but must pass validation.
    """
    CODE_QUALITY_REVIEWER = "Code Quality Reviewer"
    SECURITY_ANALYST = "Security Analyst"
    PERFORMANCE_OPTIMIZER = "Performance Optimizer"
    ARCHITECTURE_REVIEWER = "Architecture Reviewer"
    BEST_PRACTICES_REVIEWER = "Best Practices Reviewer"
    DOCUMENTATION_REVIEWER = "Documentation Reviewer"
    TESTING_REVIEWER = "Testing Reviewer"
    DEPENDENCY_ANALYZER = "Dependency Analyzer"


def is_safe_phrase(role_name: str) -> bool:
    """
    Check if role name contains a known safe phrase.
    
    Args:
        role_name: Role name to check
        
    Returns:
        True if role name contains a safe phrase
    """
    config = get_role_validation_config()
    role_lower = role_name.lower()
    
    for safe_phrase in config.safe_phrases:
        if safe_phrase in role_lower:
            return True
    return False


def normalize_role_name(role_name: str) -> str:
    """
    Normalize role name by replacing special characters.
    
    Uses char_normalization map from config.
    
    Args:
        role_name: Raw role name
        
    Returns:
        Normalized role name
    """
    config = get_role_validation_config()
    normalized = role_name.strip()
    
    for char, replacement in config.char_normalization.items():
        normalized = normalized.replace(char, replacement)
    
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


class RoleDefinition(BaseModel):
    """
    Pydantic model for role definition with validation.
    
    Ensures role names are safe and follow expected patterns.
    """
    name: str = Field(..., description="Role name (validated against RoleEnum or custom pattern)")
    description: str = Field(default="", description="Role description")
    focus_areas: List[str] = Field(default_factory=list, description="Focus areas for this role")
    is_custom: bool = Field(default=False, description="Whether this is a custom role")
    
    @field_validator('name')
    @classmethod
    def validate_role_name(cls, v: str) -> str:
        """
        Validate role name against security patterns.
        
        Raises:
            ValueError: If role name is suspicious or invalid
        """
        if not v or not v.strip():
            raise ValueError("Role name cannot be empty")
        
        role_name = v.strip()
        config = get_role_validation_config()
        
        # Check if it's a safe phrase first (allows legitimate uses of suspicious words)
        if is_safe_phrase(role_name):
            # Safe phrase, but still need to check length and characters
            if len(role_name) > config.max_role_name_length:
                raise ValueError(f"Role name too long (max {config.max_role_name_length} characters): {len(role_name)}")
            
            # For safe phrases, allow normalized characters
            normalized = normalize_role_name(role_name)
            if config.compiled_allowed_char_pattern and not config.compiled_allowed_char_pattern.match(normalized):
                raise ValueError(f"Role name contains invalid characters: {role_name}")
            
            return role_name
        
        # Check against dangerous patterns (context-independent security risks)
        for pattern in config.compiled_dangerous_patterns:
            if pattern.search(role_name):
                logger.warning(f"Rejected suspicious role name: {role_name} (matched pattern: {pattern.pattern})")
                raise ValueError(f"Role name '{role_name}' contains suspicious pattern and is rejected for security reasons")
        
        # Check length
        if len(role_name) > config.max_role_name_length:
            raise ValueError(f"Role name too long (max {config.max_role_name_length} characters): {len(role_name)}")
        
        # Check for valid characters
        if config.compiled_allowed_char_pattern and not config.compiled_allowed_char_pattern.match(role_name):
            raise ValueError(f"Role name contains invalid characters: {role_name}")
        
        return role_name
    
    def model_post_init(self, __context) -> None:
        """Set is_custom flag after validation."""
        # Check if role matches any enum value
        try:
            RoleEnum(self.name)
            self.is_custom = False
        except ValueError:
            # Not in enum, so it's a custom role
            self.is_custom = True
            logger.debug(f"Custom role detected: {self.name}")


def validate_role_name(role_name: str) -> bool:
    """
    Validate role name against security patterns and enum.
    
    Args:
        role_name: Role name to validate
        
    Returns:
        True if role is valid, False otherwise
    """
    try:
        RoleDefinition(name=role_name)
        return True
    except ValueError:
        return False


def sanitize_role_name(role_name: str) -> Optional[str]:
    """
    Sanitize role name from LLM output.
    
    Removes suspicious patterns and normalizes the name.
    
    Args:
        role_name: Raw role name from LLM
        
    Returns:
        Sanitized role name or None if too suspicious
    """
    if not role_name:
        return None
    
    config = get_role_validation_config()
    
    # Strip whitespace
    sanitized = role_name.strip()
    
    # Remove common LLM artifacts
    sanitized = re.sub(r'^[:\-\s]+', '', sanitized)  # Remove leading punctuation
    sanitized = re.sub(r'[:\-\s]+$', '', sanitized)  # Remove trailing punctuation
    
    # Normalize characters (e.g., & -> and)
    sanitized = normalize_role_name(sanitized)
    
    # Check if it's a safe phrase (allows legitimate uses)
    if is_safe_phrase(sanitized):
        return sanitized
    
    # Check against dangerous patterns (context-independent security risks)
    for pattern in config.compiled_dangerous_patterns:
        if pattern.search(sanitized):
            logger.warning(f"Role name too suspicious to sanitize: {sanitized}")
            return None
    
    return sanitized if sanitized else None


def validate_role_definition(role_dict: dict) -> Optional[RoleDefinition]:
    """
    Validate and convert role dictionary to RoleDefinition.
    
    Args:
        role_dict: Role dictionary from LLM output
        
    Returns:
        RoleDefinition if valid, None otherwise
    """
    try:
        # Extract name
        name = role_dict.get("name") if isinstance(role_dict, dict) else str(role_dict)
        if not name:
            return None
        
        # Sanitize name
        sanitized_name = sanitize_role_name(name)
        if not sanitized_name:
            return None
        
        # Create RoleDefinition
        role_def = RoleDefinition(
            name=sanitized_name,
            description=str(role_dict.get("description", "")) if isinstance(role_dict, dict) else "",
            focus_areas=role_dict.get("focus_areas", []) if isinstance(role_dict, dict) else [],
        )
        
        return role_def
    except ValueError as e:
        logger.warning(f"Invalid role definition: {e}")
        return None
    except Exception as e:
        logger.error(f"Error validating role definition: {e}", exc_info=True)
        return None


def reject_invalid_roles(roles: List[dict]) -> List[RoleDefinition]:
    """
    Filter and validate roles, rejecting invalid ones.
    
    Args:
        roles: List of role dictionaries from LLM output
        
    Returns:
        List of validated RoleDefinition objects
    """
    valid_roles = []
    rejected_roles = []
    
    for role in roles:
        role_def = validate_role_definition(role)
        if role_def:
            valid_roles.append(role_def)
        else:
            role_name = role.get("name") if isinstance(role, dict) else str(role)
            rejected_roles.append(role_name)
            logger.warning(f"Rejected invalid role: {role_name}")
    
    if rejected_roles:
        logger.info(f"Rejected {len(rejected_roles)} invalid roles: {rejected_roles}")
    
    return valid_roles

