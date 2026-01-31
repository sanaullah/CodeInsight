"""
Technology filtering utility for prompt generation.

Removes hardcoded technologies from prompts that are not present in the
architecture model, preventing technology hallucinations in generated prompts.
"""

import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def remove_hardcoded_technologies(
    prompt: str, 
    architecture_model: Dict[str, Any]
) -> str:
    """
    Remove technologies from prompt that are not in architecture model.
    
    Prevents technology hallucinations by removing references to common
    technologies that are often incorrectly assumed to be present. Only
    technologies explicitly detected in the architecture model are preserved.
    
    Args:
        prompt: Generated prompt text
        architecture_model: Architecture model dictionary with system_technologies
        
    Returns:
        Cleaned prompt with hardcoded technologies removed
    """
    if not prompt or not architecture_model:
        return prompt
    
    try:
        # Get allowed technologies from architecture model
        # Handle both dict format (v3) and ArchitectureModel object (v2)
        allowed_techs = set()
        
        # Try to get system_technologies property (if ArchitectureModel object)
        if hasattr(architecture_model, 'system_technologies'):
            allowed_techs = {tech.lower() for tech in architecture_model.system_technologies}
        # Try to get from dict directly
        elif isinstance(architecture_model, dict):
            # Check if system_technologies is in the dict
            if 'system_technologies' in architecture_model:
                allowed_techs = {tech.lower() for tech in architecture_model['system_technologies']}
            else:
                # Build from frameworks, libraries, and tech_stack
                frameworks = architecture_model.get('frameworks', [])
                libraries = architecture_model.get('libraries', [])
                tech_stack = architecture_model.get('tech_stack', {})
                
                allowed_techs.update(tech.lower() for tech in frameworks)
                allowed_techs.update(tech.lower() for tech in libraries)
                
                # Extract from tech_stack dict
                if isinstance(tech_stack, dict):
                    for category, techs in tech_stack.items():
                        if isinstance(techs, list):
                            allowed_techs.update(tech.lower() for tech in techs)
        
        # Common hardcoded technologies that should be removed if not in architecture
        forbidden_techs = [
            "docker", "kubernetes", "k8s", "containerization", "containers",
            "spring boot", "spring framework", "spring",
            "express.js", "expressjs", "express",
            "selenium", "beautifulsoup", "beautiful soup",
            "react", "vue", "angular", "next.js", "nextjs",
            "node.js", "nodejs", "node",
            "nginx", "apache",
            "postgresql", "postgres", "mysql", "mongodb", "mongo"
        ]
        
        # Build regex patterns to match forbidden technologies (word boundaries)
        patterns_to_remove = []
        for tech in forbidden_techs:
            tech_lower = tech.lower()
            # Only remove if not in allowed technologies
            if tech_lower not in allowed_techs:
                # Create pattern that matches the technology as a whole word
                # Handle variations like "Express.js" vs "express.js"
                pattern = r'\b' + re.escape(tech) + r'\b'
                patterns_to_remove.append((pattern, re.IGNORECASE))
        
        # Remove forbidden technologies from prompt
        cleaned_prompt = prompt
        for pattern, flags in patterns_to_remove:
            # Remove the technology (case-insensitive)
            cleaned_prompt = re.sub(
                pattern,
                '',  # Remove the technology
                cleaned_prompt,
                flags=flags
            )
        
        # Clean up extra whitespace and empty lines
        lines = cleaned_prompt.split('\n')
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                # Keep one empty line max
                cleaned_lines.append('')
                prev_empty = True
        
        cleaned_prompt = '\n'.join(cleaned_lines).strip()
        
        # Log if we removed anything
        if cleaned_prompt != prompt:
            removed_count = sum(
                1 for pattern, flags in patterns_to_remove 
                if re.search(pattern, prompt, flags=flags)
            )
            if removed_count > 0:
                logger.debug(f"Removed {removed_count} hardcoded technology references from generated prompt")
        
        return cleaned_prompt
        
    except Exception as e:
        logger.error(f"Error filtering technologies from prompt: {e}", exc_info=True)
        # Return original prompt on error
        return prompt


def check_forbidden_technologies(
    prompt: str,
    architecture_model: Dict[str, Any]
) -> List[str]:
    """
    Check for forbidden technologies in prompt without removing them.
    
    Returns a list of forbidden technology references found in the prompt.
    This is used by the validator to report violations without modifying the prompt.
    
    Args:
        prompt: Prompt text to check
        architecture_model: Architecture model dictionary with system_technologies
        
    Returns:
        List of forbidden technology names found in the prompt
    """
    if not prompt or not architecture_model:
        return []
    
    try:
        # Get allowed technologies from architecture model
        # Handle both dict format (v3) and ArchitectureModel object (v2)
        allowed_techs = set()
        
        # Try to get system_technologies property (if ArchitectureModel object)
        if hasattr(architecture_model, 'system_technologies'):
            allowed_techs = {tech.lower() for tech in architecture_model.system_technologies}
        # Try to get from dict directly
        elif isinstance(architecture_model, dict):
            # Check if system_technologies is in the dict
            if 'system_technologies' in architecture_model:
                allowed_techs = {tech.lower() for tech in architecture_model['system_technologies']}
            else:
                # Build from frameworks, libraries, and tech_stack
                frameworks = architecture_model.get('frameworks', [])
                libraries = architecture_model.get('libraries', [])
                tech_stack = architecture_model.get('tech_stack', {})
                
                allowed_techs.update(tech.lower() for tech in frameworks)
                allowed_techs.update(tech.lower() for tech in libraries)
                
                # Extract from tech_stack dict
                if isinstance(tech_stack, dict):
                    for category, techs in tech_stack.items():
                        if isinstance(techs, list):
                            allowed_techs.update(tech.lower() for tech in techs)
        
        # Common hardcoded technologies that should be checked
        forbidden_techs = [
            "docker", "kubernetes", "k8s", "containerization", "containers",
            "spring boot", "spring framework", "spring",
            "express.js", "expressjs", "express",
            "selenium", "beautifulsoup", "beautiful soup",
            "react", "vue", "angular", "next.js", "nextjs",
            "node.js", "nodejs", "node",
            "nginx", "apache",
            "postgresql", "postgres", "mysql", "mongodb", "mongo"
        ]
        
        # Find forbidden technologies in prompt
        violations = []
        prompt_lower = prompt.lower()
        
        for tech in forbidden_techs:
            tech_lower = tech.lower()
            # Only check if not in allowed technologies
            if tech_lower not in allowed_techs:
                # Check if technology appears in prompt (word boundary)
                pattern = r'\b' + re.escape(tech) + r'\b'
                if re.search(pattern, prompt, re.IGNORECASE):
                    violations.append(tech)
        
        return violations
        
    except Exception as e:
        logger.error(f"Error checking forbidden technologies: {e}", exc_info=True)
        # Return empty list on error
        return []
