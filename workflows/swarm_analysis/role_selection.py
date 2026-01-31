"""
Role selection node for Swarm Analysis workflow.

Selects analysis roles using LLM to create dynamic, project-specific roles.
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from workflows.nodes import llm_node
from workflows.swarm_analysis.architecture import summarize_architecture_for_roles
from workflows.swarm_analysis_roles import (
    RoleDefinition,
    validate_role_definition,
    reject_invalid_roles,
    RoleEnum
)
from utils.error_context import create_error_context

logger = logging.getLogger(__name__)


@validate_state_fields(["architecture_model"], "select_roles")
async def select_roles_node(state: SwarmAnalysisState, config: Optional[Dict[str, Any]] = None) -> SwarmAnalysisState:
    """
    Select analysis roles using LLM to create dynamic, project-specific roles.
    
    Uses the v2-style prompt approach to generate custom roles tailored to the
    project's architecture instead of selecting from a static list.
    
    Args:
        state: Current swarm analysis state
        config: Optional LangGraph config
        
    Returns:
        Updated state with selected_roles
    """
    architecture_model: Optional[Dict[str, Any]] = state.get(StateKeys.ARCHITECTURE_MODEL)
    goal: Optional[str] = state.get(StateKeys.GOAL)
    max_agents: int = state.get(StateKeys.MAX_AGENTS, 10)
    model_name: Optional[str] = state.get(StateKeys.MODEL_NAME)
    
    if not architecture_model:
        error_msg = (
            "architecture_model is required for role selection. "
            "Please ensure the architecture model was built successfully before selecting roles."
        )
        state[StateKeys.ERROR] = error_msg
        state[StateKeys.ERROR_STAGE] = "select_roles"
        logger.error(error_msg)
        return state
    
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback("role_selection", {
            "message": "Determining analysis roles..."
        })
    
    try:
        from utils.langfuse_integration import create_llm_observation
        
        # Create LLM observation for role selection
        role_selection_obs = create_llm_observation(
            operation_name="role_selection",
            model=model_name,
            input_data={
                "architecture_type": architecture_model.get("system_type", "Unknown"),
                "goal": goal,
                "max_agents": max_agents
            },
            metadata={
                "operation": "select_roles"
            }
        )
        
        try:
            # Build comprehensive architecture summary
            arch_summary = summarize_architecture_for_roles(architecture_model)
            
            # Goal information
            goal_section = ""
            if goal:
                goal_section = f"""
Analysis Goal:
{goal}
"""
            
            # Retrieve relevant skills for role selection (ACE Phase 2: use pre-fetched if available)
            skills_section = ""
            harmful_section = ""
            try:
                from utils.swarm_skillbook import get_relevant_skills
                from utils.skill_integration import format_skills_for_prompt, format_harmful_patterns
                from agents.swarm_skillbook_models import SwarmSkill
                
                # Check for pre-fetched skills first (Phase 2 optimization)
                learned_skills = state.get(StateKeys.LEARNED_SKILLS, {})
                role_selection_skills_data = learned_skills.get("role_selection", [])
                
                if role_selection_skills_data:
                    # Use pre-fetched skills (convert dicts back to SwarmSkill objects)
                    role_selection_skills = [
                        SwarmSkill(
                            skill_id=s["skill_id"],
                            skill_type=s["skill_type"],
                            skill_category=s["skill_category"],
                            content=s["content"],
                            context=s.get("context", {}),
                            confidence=s.get("confidence", 0.5),
                            success_rate=s.get("success_rate", 0.0),
                            usage_count=s.get("usage_count", 0)
                        )
                        for s in role_selection_skills_data
                    ]
                    logger.debug("Using pre-fetched role selection skills")
                else:
                    # Fallback to direct retrieval (backward compatibility)
                    architecture_type = architecture_model.get("system_type", "unknown")
                    relevant_skills = get_relevant_skills(
                        architecture_type=architecture_type,
                        goal=goal
                    )
                    role_selection_skills = relevant_skills.get("role_selection", [])
                    logger.debug("Retrieved role selection skills directly")
                
                # Extract helpful vs harmful
                helpful_skills = [s for s in role_selection_skills if s.skill_category == "helpful"]
                harmful_skills = [s for s in role_selection_skills if s.skill_category == "harmful"]
                
                # Format skills for prompt
                skills_section = format_skills_for_prompt(
                    helpful_skills,
                    max_skills=5,
                    category_label="Learned Best Practices"
                )
                
                harmful_section = format_harmful_patterns(harmful_skills, max_skills=3)
                
                if skills_section or harmful_section:
                    logger.info(f"Retrieved {len(helpful_skills)} helpful and {len(harmful_skills)} harmful role selection skills")
                
            except Exception as skill_error:
                # Non-blocking: log but continue without skills
                logger.warning(f"Failed to retrieve skills for role selection: {skill_error}")
                skills_section = ""
                harmful_section = ""
            
            # Build the v2-style dynamic role creation prompt
            role_selection_prompt = f"""Analyze the following software project architecture and create custom, project-specific analysis roles.

{arch_summary}
{goal_section}
{skills_section}
{harmful_section}

Based on the architecture, project context, and goal (if provided), create specialized analysis roles that are tailored to this specific project. Do NOT limit yourself to generic categories like "security" or "performance". Instead, create specific roles that address the unique aspects of this codebase.

Consider:
1. What specific technologies, frameworks, or patterns are used that need specialized analysis?
2. What unique architectural components or modules exist that require focused review?
3. What project-specific concerns or challenges are evident from the architecture?
4. If a goal is provided, create roles that directly help achieve that goal.

Examples of good custom roles:
- "Database Schema Reviewer" (for projects with complex database schemas)
- "Django ORM Query Optimizer" (for Django projects with ORM usage)
- "REST API Design Specialist" (for projects with REST APIs)
- "Microservices Communication Reviewer" (for microservices architectures)
- "Frontend Performance Specialist" (for web applications)
- "Dependency Security Scanner" (for projects with many dependencies)

Return a JSON array of role objects, each with:
- "name": A descriptive role name (e.g., "Database Schema Reviewer")
- "description": A clear description of what this role analyzes
- "focus_areas": An array of specific focus areas for this role (e.g., ["schema normalization", "index optimization", "query performance"])

Order roles by relevance (most relevant first). Only create roles that are genuinely relevant to this specific project. Create up to {max_agents} roles.

Example response:
[
  {{
    "name": "Database Schema Reviewer",
    "description": "Analyzes database schema design, relationships, indexing strategies, and query optimization opportunities",
    "focus_areas": ["schema normalization", "index optimization", "foreign key relationships", "query performance", "data integrity constraints"]
  }},
  {{
    "name": "API Design Specialist",
    "description": "Reviews REST API endpoints for design consistency, error handling, and best practices",
    "focus_areas": ["endpoint design", "HTTP methods", "error responses", "API versioning", "documentation"]
  }}
]"""
        
            # Use llm_node for role selection (will be traced via llm_node's internal tracing)
            llm_state = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert software architect who analyzes codebases to determine which specialized analysis roles are needed. Create custom, project-specific roles based on the architecture, not limited to generic categories."
                    },
                    {
                        "role": "user",
                        "content": role_selection_prompt
                    }
                ],
                "model": model_name,
                "temperature": 0.7  # Higher temperature for creativity in role creation
            }
            
            # Call llm_node (tracing happens inside llm_node)
            llm_result = await llm_node(llm_state, config=config)
            
            # Extract response
            roles_text = llm_result.get("last_response", "")
            
            # Parse roles from JSON response
            selected_roles: List[Dict[str, Any]] = []
            
            # Try to extract JSON array from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])', roles_text, re.DOTALL)
            if json_match:
                try:
                    selected_roles = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON array from code block")
            
            # If no code block, try to find JSON array directly in text
            if not selected_roles:
                json_array_match = re.search(r'\[[^\]]*\{[^}]*"name"[^}]*\}[^\]]*\]', roles_text, re.DOTALL)
                if json_array_match:
                    try:
                        selected_roles = json.loads(json_array_match.group(0))
                    except json.JSONDecodeError:
                        pass
            
            # If still no roles, try parsing the entire response as JSON
            if not selected_roles:
                try:
                    # Remove markdown code blocks if present
                    cleaned_text = roles_text.strip()
                    if cleaned_text.startswith("```"):
                        lines = cleaned_text.split("\n")
                        cleaned_text = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned_text
                    
                    parsed = json.loads(cleaned_text)
                    if isinstance(parsed, list):
                        selected_roles = parsed
                    elif isinstance(parsed, dict):
                        selected_roles = [parsed]
                except (json.JSONDecodeError, ValueError):
                    logger.warning("Could not parse roles from LLM response as JSON")
            
            # Validate and normalize roles using RoleDefinition
            # This ensures security by rejecting suspicious role names
            validated_role_definitions = reject_invalid_roles(selected_roles)
            
            # Convert RoleDefinition objects back to dicts for state storage
            valid_roles: List[Dict[str, Any]] = []
            for role_def in validated_role_definitions:
                valid_roles.append({
                    "name": role_def.name,
                    "description": role_def.description,
                    "focus_areas": role_def.focus_areas,
                    "is_custom": role_def.is_custom
                })
            
            # Log security audit information
            if len(validated_role_definitions) < len(selected_roles):
                rejected_count = len(selected_roles) - len(validated_role_definitions)
                logger.warning(
                    f"Security: Rejected {rejected_count} invalid/suspicious roles. "
                    f"Accepted {len(validated_role_definitions)} valid roles."
                )
            
            # Fallback: Use default roles if parsing fails
            if not valid_roles:
                logger.warning("Could not parse roles from LLM response, using fallback")
                valid_roles = [
                    {
                        "name": "Code Quality Reviewer",
                        "description": "Assesses code organization, maintainability, readability, and best practices",
                        "focus_areas": ["code organization", "maintainability", "readability", "code smells", "refactoring opportunities"]
                    },
                    {
                        "name": "Security Analyst",
                        "description": "Analyzes code for security vulnerabilities, authentication issues, and data protection concerns",
                        "focus_areas": ["authentication", "authorization", "vulnerabilities", "data protection", "secure communication"]
                    },
                    {
                        "name": "Performance Optimizer",
                        "description": "Identifies performance bottlenecks, optimization opportunities, and scalability concerns",
                        "focus_areas": ["bottlenecks", "optimization", "caching", "algorithms", "scalability"]
                    }
                ]
            
            # Limit to max_agents
            valid_roles = valid_roles[:max_agents]
            
            role_names: List[str] = [r["name"] if isinstance(r, dict) else str(r) for r in valid_roles]
            
            # Update observation with results
            if role_selection_obs:
                try:
                    role_selection_obs.update(
                        output={
                            "roles_selected": role_names,
                            "role_count": len(role_names)
                        },
                        metadata={"status": "success", "role_count": len(role_names)}
                    )
                except AttributeError:
                    # Observation is a context manager, update not supported directly
                    pass
            
            # Update state
            updated_state = state.copy()
            updated_state[StateKeys.SELECTED_ROLES] = valid_roles
            updated_state[StateKeys.ROLE_NAMES] = role_names
        finally:
            if role_selection_obs:
                try:
                    role_selection_obs.end()
                except Exception:
                    pass
        
        if stream_callback:
            stream_callback("roles_selected", {
                "roles": role_names,
                "count": len(valid_roles)
            })
        
        logger.info(f"Selected {len(valid_roles)} dynamic roles: {role_names}")
        return updated_state
        
    except Exception as e:
        error_context = create_error_context("select_roles", state, {
            "architecture_type": architecture_model.get("system_type", "Unknown") if architecture_model else "Unknown",
            "goal": goal,
            "max_agents": max_agents,
            "model_name": model_name
        })
        error_msg = (
            f"Failed to select roles for architecture '{architecture_model.get('system_name', 'Unknown') if architecture_model else 'Unknown'}': {str(e)}. "
            f"Check that the architecture model is valid and the LLM model is accessible."
        )
        logger.error(error_msg, extra=error_context, exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = error_msg
        updated_state[StateKeys.ERROR_STAGE] = "select_roles"
        updated_state[StateKeys.ERROR_CONTEXT] = error_context
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": error_msg})
        return updated_state


