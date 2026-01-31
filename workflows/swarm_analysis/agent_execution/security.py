"""
Security-related prompt utilities for agent execution.
"""
from typing import Dict, Any, Optional

def build_agent_system_prompt(
    role_name: str, 
    architecture_hash: Optional[str] = None,
    architecture_model: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build system prompt for agent execution with architecture hash constraint.

    Args:
        role_name: Name of the analysis role
        architecture_hash: Optional architecture hash for constraint enforcement
        architecture_model: Optional architecture model for strict technology whitelisting

    Returns:
        System prompt string with hash constraint and allowed technologies
    """
    base_prompt = f"You are a {role_name}. Analyze code according to your role."

    if architecture_hash:
        # Strict Structural Enforcement
        allowed_tech = "None detected"
        if architecture_model:
            # Extract technologies from various fields in the model
            technologies = set()
            
            # 1. Frameworks and Libraries
            technologies.update(architecture_model.get("frameworks", []))
            technologies.update(architecture_model.get("libraries", []))
            
            # 2. Tech Stack dictionary
            tech_stack = architecture_model.get("tech_stack", {})
            if isinstance(tech_stack, dict):
                for category, techs in tech_stack.items():
                    if isinstance(techs, list):
                        technologies.update(techs)
            
            # 3. System Technologies property (if available in dict)
            if "system_technologies" in architecture_model:
                technologies.update(architecture_model.get("system_technologies", []))
                
            if technologies:
                allowed_tech = ", ".join(sorted(list(technologies)))
        
        prompt = f"""{base_prompt}

YOUR ANALYSIS IS CRYPTOGRAPHICALLY CONSTRAINED.

ALLOWED TECHNOLOGIES (SHA256: {architecture_hash[:8]}...):
{allowed_tech}

VERIFICATION RULE: 
Before mentioning ANY technology, framework, or library, verify it appears in the ALLOWED TECHNOLOGIES list above. 
- If it is in the list -> You may reference it.
- If it is NOT in the list -> You MUST NOT mention it. Omit it entirely or use generic terms (e.g., "the database", "the frontend framework").
- Violation of this rule will cause the analysis to be rejected check."""
        return prompt

    return base_prompt
