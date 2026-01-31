"""
Skill integration utilities for formatting skills for LLM prompts.

Provides functions to format SwarmSkill objects for inclusion in LLM prompts,
enabling the ACE (Agent Capability Enhancement) learning loop.
"""

import logging
from typing import List
from agents.swarm_skillbook_models import SwarmSkill

logger = logging.getLogger(__name__)


def format_skills_for_prompt(
    skills: List[SwarmSkill],
    max_skills: int = 5,
    include_metadata: bool = True,
    category_label: str = "Best Practices"
) -> str:
    """
    Format skills for inclusion in LLM prompts.
    
    Args:
        skills: List of SwarmSkill objects
        max_skills: Maximum number of skills to include
        include_metadata: Whether to include success_rate and usage_count
        category_label: Label for the skills section
        
    Returns:
        Formatted string for prompt inclusion (empty string if no skills)
    """
    if not skills:
        return ""
    
    # Sort by effectiveness (success_rate * usage_count as tiebreaker)
    sorted_skills = sorted(
        skills,
        key=lambda s: (s.success_rate, s.usage_count),
        reverse=True
    )[:max_skills]
    
    formatted = f"\n## {category_label} (from past analyses):\n\n"
    for skill in sorted_skills:
        formatted += f"- {skill.content}\n"
        if include_metadata:
            formatted += f"  (Success rate: {skill.success_rate:.0%}, Used {skill.usage_count} times)\n"
    
    return formatted


def format_harmful_patterns(
    skills: List[SwarmSkill],
    max_skills: int = 3
) -> str:
    """
    Format harmful skills as warnings to avoid.
    
    Args:
        skills: List of SwarmSkill objects with skill_category="harmful"
        max_skills: Maximum number of patterns to include
        
    Returns:
        Formatted warning string (empty string if no harmful skills)
    """
    if not skills:
        return ""
    
    sorted_skills = sorted(
        skills,
        key=lambda s: (s.usage_count, s.success_rate),
        reverse=True
    )[:max_skills]
    
    formatted = "\n## Patterns to Avoid:\n\n"
    for skill in sorted_skills:
        formatted += f"- ⚠️ {skill.content}\n"
        formatted += f"  (Occurred {skill.usage_count} times)\n"
    
    return formatted

