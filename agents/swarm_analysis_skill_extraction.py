"""
Swarm Analysis skill extraction from experiences.

Extracts reusable skills from swarm analysis experiences using the ACE framework.
Ports logic from CodeInsight v2 SwarmSkillManager, adapted for v3 architecture.
"""

import logging
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.experience_models import Experience
from agents.swarm_skillbook_models import SwarmSkill
from utils.swarm_skillbook import save_skill, get_skills
from workflows.nodes import llm_node

logger = logging.getLogger(__name__)


async def experience_to_skills(
    experience: Experience,
    architecture_model: Dict[str, Any],
    model_name: Optional[str] = None
) -> List[SwarmSkill]:
    """
    Convert experience to SwarmSkill objects.
    
    This is the main entry point for the ACE learning loop.
    Extracts skills from an experience, checks for duplicates,
    and creates/updates SwarmSkill objects in the skillbook.
    
    Args:
        experience: Experience object from swarm analysis
        architecture_model: Architecture model dictionary
        model_name: Optional LLM model name for skill extraction
        
    Returns:
        List of SwarmSkill objects that were created or updated
    """
    try:
        # Extract skills from experience
        extracted_skills = await extract_skills_from_experience(
            experience=experience,
            architecture_model=architecture_model,
            model_name=model_name
        )
        
        if not extracted_skills:
            logger.info("No skills extracted from experience")
            return []
        
        # Process and save skills
        saved_skills = []
        added_count = 0
        updated_count = 0
        
        for skill_data in extracted_skills:
            # Check for existing similar skills
            existing_skill = _find_similar_skill(skill_data)
            
            if existing_skill:
                # Update existing skill
                existing_skill.usage_count += 1
                existing_skill.last_used = datetime.now()
                
                # Merge confidence (weighted average)
                total_usage = existing_skill.usage_count
                existing_skill.confidence = (
                    (existing_skill.confidence * (total_usage - 1) + skill_data["confidence"]) / total_usage
                )
                
                # Update metadata
                if "extracted_from" not in existing_skill.metadata:
                    existing_skill.metadata["extracted_from"] = []
                if experience.experience_id not in existing_skill.metadata["extracted_from"]:
                    existing_skill.metadata["extracted_from"].append(experience.experience_id)
                
                save_skill(existing_skill)
                saved_skills.append(existing_skill)
                updated_count += 1
            else:
                # Create new skill
                skill = SwarmSkill(
                    skill_id=str(uuid.uuid4()),
                    skill_type=skill_data["skill_type"],
                    skill_category=skill_data["skill_category"],
                    content=skill_data["content"],
                    context=skill_data.get("context", {}),
                    confidence=skill_data["confidence"],
                    usage_count=1,
                    success_rate=0.8 if skill_data["skill_category"] == "helpful" else 0.2,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    metadata={
                        "extracted_from": [experience.experience_id],
                        "architecture_type": experience.context.get("architecture_type", "unknown")
                    }
                )
                
                # Add architecture type to context if not present
                if "architecture_type" not in skill.context:
                    skill.context["architecture_type"] = experience.context.get("architecture_type", "unknown")
                
                save_skill(skill)
                saved_skills.append(skill)
                added_count += 1
        
        logger.info(f"Extracted {len(extracted_skills)} skills: {added_count} added, {updated_count} updated")
        return saved_skills
        
    except Exception as e:
        logger.error(f"Error converting experience to skills: {e}", exc_info=True)
        return []


async def extract_skills_from_experience(
    experience: Experience,
    architecture_model: Dict[str, Any],
    model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract skills from experience using LLM.
    
    Args:
        experience: Experience object
        architecture_model: Architecture model dictionary
        model_name: Optional model name for LLM calls
        
    Returns:
        List of skill dictionaries
    """
    try:
        # Build extraction prompt
        extraction_prompt = _build_extraction_prompt(
            experience=experience,
            architecture_model=architecture_model
        )
        
        # Create system prompt
        system_prompt = """You are a Skill Extractor, responsible for extracting structured skills and patterns from swarm analysis experiences.

Your task is to identify actionable patterns that can be reused in future analyses.

Extract skills in JSON format:
{
  "skills": [
    {
      "skill_type": "role_selection|prompt_generation|synthesis|execution|general",
      "skill_category": "helpful|harmful|neutral",
      "content": "The actual skill/pattern description",
      "context": {"architecture_type": "...", ...},
      "confidence": 0.0-1.0
    }
  ]
}"""
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extraction_prompt}
        ]
        
        # Get model name
        if not model_name:
            from llm.config import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.load_config()
            model_name = config.default_model
        
        # Use llm_node for LLM call (v3 pattern)
        llm_state = {
            "messages": messages,
            "model": model_name,
            "temperature": 0.3
        }
        
        # Call llm_node
        llm_result = await llm_node(llm_state)
        
        # Extract response
        response_text = _extract_response(llm_result)
        
        # Parse skills from response
        skills = _parse_skills_response(response_text)
        
        return skills
        
    except Exception as e:
        logger.error(f"Error extracting skills with LLM: {e}", exc_info=True)
        # Fallback: extract skills directly from experience
        return _extract_skills_directly_from_experience(experience)


def _build_extraction_prompt(
    experience: Experience,
    architecture_model: Dict[str, Any]
) -> str:
    """Build skill extraction prompt from experience."""
    context = experience.context
    architecture_type = context.get("architecture_type", "unknown")
    
    # Extract roles from execution plan
    roles_selected = experience.execution_plan.get("roles", [])
    if isinstance(roles_selected, list):
        roles_str = ", ".join(roles_selected)
    else:
        roles_str = str(roles_selected)
    
    # Build prompt sections
    prompt_parts = [
        f"Extract skills and patterns from this swarm analysis experience.",
        "",
        f"Architecture Type: {architecture_type}",
        f"Architecture Pattern: {architecture_model.get('architecture_pattern', 'unknown')}",
        f"Goal: {experience.goal}",
        f"Roles Selected: {roles_str}",
        "",
        "Lessons Learned:",
        "\n".join(f"- {lesson}" for lesson in experience.lessons_learned) if experience.lessons_learned else "- None",
        "",
        "Primary Achievements:",
        "\n".join(f"- {achievement}" for achievement in experience.outcome.primary_achievements) if experience.outcome.primary_achievements else "- None",
        "",
        "Failures:",
        "\n".join(f"- {failure}" for failure in experience.outcome.failures) if experience.outcome.failures else "- None",
        "",
        f"Quality Score: {experience.performance_metrics.quality_score}",
        f"Success: {experience.outcome.success}",
        "",
        "Focus on extracting:",
        "1. Role selection patterns (what roles worked well for this architecture type)",
        "2. Prompt generation patterns (effective prompt structures)",
        "3. Synthesis approaches (how to combine agent results)",
        "4. Execution strategies (parallelization, chunking, etc.)",
        "",
        "Provide skills in JSON format with skill_type, skill_category, content, context, and confidence."
    ]
    
    return "\n".join(prompt_parts)


def _parse_skills_response(response_text: str) -> List[Dict[str, Any]]:
    """Parse skills from LLM response."""
    try:
        # Try to extract JSON from response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            data = json.loads(json_text)
            skills = data.get("skills", [])
            
            # Validate and normalize skills
            normalized_skills = []
            for skill in skills:
                if _validate_skill(skill):
                    normalized_skills.append(_normalize_skill(skill))
            
            return normalized_skills
        else:
            # Fallback parsing
            return _parse_skills_from_text(response_text)
            
    except Exception as e:
        logger.warning(f"Error parsing skills response: {e}")
        return []


def _validate_skill(skill: Dict[str, Any]) -> bool:
    """Validate skill structure."""
    required_fields = ["skill_type", "skill_category", "content"]
    return all(field in skill for field in required_fields)


def _normalize_skill(skill: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize skill data structure."""
    return {
        "skill_type": skill.get("skill_type", "general"),
        "skill_category": skill.get("skill_category", "neutral"),
        "content": skill.get("content", ""),
        "context": skill.get("context", {}),
        "confidence": float(skill.get("confidence", 0.7))
    }


def _parse_skills_from_text(text: str) -> List[Dict[str, Any]]:
    """Parse skills from unstructured text (fallback)."""
    skills = []
    lines = text.split("\n")
    
    current_skill = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for skill indicators
        if "skill_type" in line.lower() or "pattern" in line.lower():
            if current_skill:
                skills.append(current_skill)
            current_skill = {"skill_type": "general", "skill_category": "neutral", "content": "", "context": {}, "confidence": 0.7}
        
        if current_skill:
            if "helpful" in line.lower():
                current_skill["skill_category"] = "helpful"
            elif "harmful" in line.lower() or "avoid" in line.lower():
                current_skill["skill_category"] = "harmful"
            elif "content" in line.lower() or ":" in line:
                content = line.split(":", 1)[-1].strip()
                if content:
                    current_skill["content"] = content
    
    if current_skill:
        skills.append(current_skill)
    
    return skills


def _extract_skills_directly_from_experience(experience: Experience) -> List[Dict[str, Any]]:
    """Extract skills directly from experience patterns (fallback)."""
    skills = []
    context = experience.context
    architecture_type = context.get("architecture_type", "unknown")
    
    # Extract from lessons learned (helpful patterns)
    for lesson in experience.lessons_learned:
        skill_type = _infer_skill_type(lesson)
        skills.append({
            "skill_type": skill_type,
            "skill_category": "helpful",
            "content": lesson,
            "context": {"architecture_type": architecture_type},
            "confidence": experience.performance_metrics.quality_score
        })
    
    # Extract from primary achievements (helpful patterns)
    for achievement in experience.outcome.primary_achievements:
        skill_type = _infer_skill_type(achievement)
        skills.append({
            "skill_type": skill_type,
            "skill_category": "helpful",
            "content": achievement,
            "context": {"architecture_type": architecture_type},
            "confidence": experience.performance_metrics.quality_score
        })
    
    # Extract from failures (harmful patterns)
    for failure in experience.outcome.failures:
        skill_type = _infer_skill_type(failure)
        skills.append({
            "skill_type": skill_type,
            "skill_category": "harmful",
            "content": failure,
            "context": {"architecture_type": architecture_type},
            "confidence": 0.7  # Lower confidence for failure-based skills
        })
    
    return skills


def _infer_skill_type(pattern: str) -> str:
    """Infer skill type from pattern text."""
    pattern_lower = pattern.lower()
    
    if "role" in pattern_lower or "select" in pattern_lower:
        return "role_selection"
    elif "prompt" in pattern_lower or "generate" in pattern_lower:
        return "prompt_generation"
    elif "synthesis" in pattern_lower or "combine" in pattern_lower:
        return "synthesis"
    elif "execution" in pattern_lower or "parallel" in pattern_lower:
        return "execution"
    else:
        return "general"


def _find_similar_skill(skill_data: Dict[str, Any]) -> Optional[SwarmSkill]:
    """Find similar existing skill to avoid duplicates."""
    try:
        # Get skills of same type and category
        existing_skills = get_skills(
            skill_type=skill_data["skill_type"],
            skill_category=skill_data["skill_category"]
        )
        
        # Simple similarity check: content similarity
        content = skill_data["content"].lower()
        for skill in existing_skills:
            existing_content = skill.content.lower()
            # Check if content is similar (simple substring match for now)
            # Can be enhanced with semantic similarity
            if content in existing_content or existing_content in content:
                # Check if architecture types match
                skill_arch = skill.context.get("architecture_type", "")
                data_arch = skill_data.get("context", {}).get("architecture_type", "")
                if skill_arch == data_arch or not skill_arch or not data_arch:
                    return skill
        
        return None
        
    except Exception as e:
        logger.warning(f"Error finding similar skill: {e}")
        return None


def _extract_response(result_state: Dict[str, Any]) -> str:
    """Extract LLM response from result state."""
    # Try to get response from last_response first
    response = result_state.get("last_response", "")
    
    # If empty, try to extract from messages array
    if not response or not response.strip():
        messages = result_state.get("messages", [])
        if messages:
            # Get the last message (should be the AI response)
            last_message = messages[-1]
            
            # Handle different message formats
            if hasattr(last_message, 'content'):
                response = last_message.content
            elif isinstance(last_message, dict):
                response = last_message.get("content", "")
            elif isinstance(last_message, str):
                response = last_message
    
    return response if response else ""

