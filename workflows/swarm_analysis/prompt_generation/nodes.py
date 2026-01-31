"""
Prompt generation nodes for Swarm Analysis workflow.

Handles parallel prompt generation for multiple roles.
"""

import logging
from typing import Dict, Any, Optional

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from workflows.nodes import llm_node
from .builders import build_enhanced_system_prompt, build_enhanced_user_prompt
from .fallback import generate_fallback_prompt

logger = logging.getLogger(__name__)


@validate_state_fields(["selected_roles"], "dispatch_prompt_generation")
def dispatch_prompt_generation_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Dispatch prompt generation node - prepares state for parallel execution.
    
    This node validates roles and prepares state. The actual Send objects
    are created in the conditional edge function (route_after_dispatch_prompts).
    
    Args:
        state: Current swarm analysis state
        
    Returns:
        Updated state (roles should already be set by select_roles_node)
    """
    selected_roles = state.get(StateKeys.SELECTED_ROLES, [])
    
    if not selected_roles:
        state[StateKeys.ERROR] = "selected_roles are required"
        state[StateKeys.ERROR_STAGE] = "dispatch_prompt_generation"
        return state
    
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback("prompt_generation_start", {
            "message": "Dispatching prompt generation for parallel execution...",
            "role_count": len(selected_roles)
        })
    
    # Just validate and return state - Send objects are created in conditional edge
    logger.info(f"Prepared {len(selected_roles)} roles for parallel prompt generation")
    
    # Return state unchanged (roles are already in state from select_roles_node)
    return state


@validate_state_fields(["role_info"], "generate_single_prompt")
async def generate_single_prompt_node(state: SwarmAnalysisState, config: Optional[Dict[str, Any]] = None) -> SwarmAnalysisState:
    """
    Generate prompt for a single role.
    
    This node processes one role's prompt generation including LLM call.
    Designed to be called in parallel via Send from dispatch_prompt_generation_node.
    
    Args:
        state: State containing role_info and all necessary context
        config: Optional LangGraph config (contains callbacks from graph invocation)
        
    Returns:
        Updated state with prompt result in generated_prompts_list
    """
    # Extract role info from state (set by Send)
    role_info = state.get(StateKeys.ROLE_INFO)
    role_name = state.get(StateKeys.ROLE_NAME)
    
    if not role_info and not role_name:
        # Fallback: try to get from selected_roles (for backward compatibility)
        selected_roles = state.get(StateKeys.SELECTED_ROLES, [])
        if selected_roles:
            role_info = selected_roles[0]
            role_name = role_info["name"] if isinstance(role_info, dict) else str(role_info)
        else:
            state[StateKeys.ERROR] = "role_info or role_name is required"
            state[StateKeys.ERROR_STAGE] = "generate_single_prompt"
            return state
    
    if not role_name:
        role_name = role_info["name"] if isinstance(role_info, dict) else str(role_info)
    
    # Get configuration from state
    architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL, {})
    goal = state.get(StateKeys.GOAL)
    model_name = state.get(StateKeys.MODEL_NAME)
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    if stream_callback:
        stream_callback("prompt_processing", {
            "role": role_name,
            "message": "Generating prompt..."
        })
    
    try:
        from utils.langfuse_integration import create_llm_observation, create_langfuse_prompt
        
        # Create LLM observation for prompt generation
        prompt_gen_obs = create_llm_observation(
            operation_name="prompt_generation",
            model=model_name,
            input_data={
                "role": role_name,
                "total_roles": 1
            },
            metadata={
                "operation": "generate_prompt",
                "role": role_name
            }
        )
        
        try:
            # Get hashes from state
            architecture_hash = state.get(StateKeys.ARCHITECTURE_HASH)
            file_hash = state.get(StateKeys.FILE_HASH)
            
            # Debug: Log hash availability
            logger.debug(
                f"Prompt generation for {role_name}: "
                f"architecture_hash={'present' if architecture_hash else 'MISSING'}, "
                f"file_hash={'present' if file_hash else 'MISSING'}"
            )
            
            # STEP 1: Check Langfuse for existing prompt matching all criteria
            existing_prompt = None
            if architecture_hash and file_hash:
                from utils.langfuse.prompts import get_langfuse_prompt_by_metadata
                architecture_type = architecture_model.get('system_type', 'unknown') if architecture_model else 'unknown'
                
                existing_prompt = get_langfuse_prompt_by_metadata(
                    role=role_name,
                    architecture_hash=architecture_hash,
                    file_hash=file_hash,
                    architecture_type=architecture_type,
                    label="production"
                )
                
                if existing_prompt:
                    logger.info(
                        f"Found existing Langfuse prompt for {role_name} "
                        f"(id: {existing_prompt['id']}, version: {existing_prompt['version']})"
                    )
                    
                    # Notify UI about source
                    if stream_callback:
                        stream_callback("prompt_source_detected", {
                            "role": role_name,
                            "source": "Langfuse",
                            "message": f"Retrieved {role_name} prompt from Langfuse"
                        })
                    
                    # Cache it for faster future access
                    from utils.prompts.prompt_cache import cache_prompt
                    cache_prompt(role_name, architecture_hash, file_hash, existing_prompt["prompt"])
                    
                    # Return existing prompt
                    return {
                        "generated_prompts_list": [{
                            "role_name": role_name,
                            "prompt": existing_prompt["prompt"],
                            "langfuse_prompt_id": existing_prompt["id"],
                            "langfuse_prompt_version": existing_prompt["version"]
                        }]
                    }
            
            # STEP 2: Check Redis/in-memory cache (with file_hash)
            from utils.prompts.prompt_cache import get_cached_prompt, cache_prompt
            cached_prompt = None
            if architecture_hash and file_hash:
                cached_prompt = get_cached_prompt(role_name, architecture_hash, file_hash)
                if cached_prompt:
                    logger.info(f"Using cached prompt for {role_name}")
                    
                    # Notify UI about source
                    if stream_callback:
                        stream_callback("prompt_source_detected", {
                            "role": role_name,
                            "source": "Cache",
                            "message": f"Retrieved {role_name} prompt from Redis Cache"
                        })

                    # Return cached prompt
                    return {
                        "generated_prompts_list": [{
                            "role_name": role_name,
                            "prompt": cached_prompt,
                            "langfuse_prompt_id": None  # Cached prompts don't need new Langfuse IDs
                        }]
                    }
            
            # STEP 3: Check CodeLumen DB (fallback)
            # If Langfuse (Step 1) and Redis (Step 2) missed, check local DB
            if architecture_hash:
                try:
                    from utils.prompts.prompt_storage import get_prompt_by_architecture_hash
                    logger.debug(f"Checking CodeLumen DB for prompt: role={role_name}, arch_hash={architecture_hash}")
                    
                    db_prompt = get_prompt_by_architecture_hash(role_name, architecture_hash)
                    
                    if db_prompt and db_prompt.get("prompt_content"):
                        logger.info(f"Found existing prompt in CodeLumen DB for {role_name}")
                        content = db_prompt["prompt_content"]
                        
                        # Notify UI about source
                        if stream_callback:
                            stream_callback("prompt_source_detected", {
                                "role": role_name,
                                "source": "CodeLumen DB",
                                "message": f"Retrieved {role_name} prompt from CodeLumen DB"
                            })
                        
                        # Cache it for next time
                        if file_hash:
                            cache_prompt(role_name, architecture_hash, file_hash, content)
                            
                        # Recover Langfuse ID if stored in metadata
                        langfuse_id = None
                        langfuse_version = None
                        if "metadata" in db_prompt:
                            langfuse_id = db_prompt["metadata"].get("langfuse_prompt_id")
                            langfuse_version = db_prompt["metadata"].get("langfuse_prompt_version")
                        
                        return {
                            "generated_prompts_list": [{
                                "role_name": role_name,
                                "prompt": content,
                                "langfuse_prompt_id": langfuse_id,
                                "langfuse_prompt_version": langfuse_version
                            }]
                        }
                except Exception as db_fallback_error:
                    # Non-blocking: log and continue to generation
                    logger.warning(f"Error checking CodeLumen DB fallback: {db_fallback_error}")

            # STEP 4: Generate new prompt via LLM (only if not found in previous levels)
            
            # Retrieve prompt generation skills for this role (ACE Phase 2: use pre-fetched if available)
            prompt_skills = []
            try:
                from utils.swarm_skillbook import get_skills
                from utils.skill_integration import format_skills_for_prompt
                from agents.swarm_skillbook_models import SwarmSkill
                
                architecture_type = architecture_model.get("system_type", "unknown") if architecture_model else "unknown"
                
                # Check for pre-fetched skills first (Phase 2 optimization)
                learned_skills = state.get(StateKeys.LEARNED_SKILLS, {})
                prompt_generation_skills_data = learned_skills.get("prompt_generation", [])
                
                if prompt_generation_skills_data:
                    # Use pre-fetched skills (convert dicts back to SwarmSkill objects)
                    base_prompt_skills = [
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
                        for s in prompt_generation_skills_data
                        if s.get("skill_category") == "helpful"  # Only helpful skills
                    ]
                    
                    # Filter by role if role is specified in context (prefer role-specific)
                    role_specific_skills = [
                        s for s in base_prompt_skills
                        if s.context and s.context.get("role") == role_name
                    ]
                    
                    # Use role-specific if available, otherwise use general
                    if role_specific_skills:
                        prompt_skills = role_specific_skills
                    else:
                        prompt_skills = base_prompt_skills
                    
                    logger.debug(f"Using pre-fetched prompt generation skills for {role_name}")
                else:
                    # Fallback to direct retrieval (backward compatibility, includes role-specific filtering)
                    # Get skills filtered by role and architecture
                    prompt_skills = get_skills(
                        skill_type="prompt_generation",
                        skill_category="helpful",
                        context={
                            "architecture_type": architecture_type,
                            "role": role_name
                        }
                    )
                    
                    # Also get general prompt_generation skills (not role-specific)
                    general_prompt_skills = get_skills(
                        skill_type="prompt_generation",
                        skill_category="helpful",
                        context={
                            "architecture_type": architecture_type
                        }
                    )
                    
                    # Combine and deduplicate (prefer role-specific)
                    skill_ids_seen = {s.skill_id for s in prompt_skills}
                    for skill in general_prompt_skills:
                        if skill.skill_id not in skill_ids_seen:
                            prompt_skills.append(skill)
                    
                    logger.debug(f"Retrieved prompt generation skills directly for {role_name}")
                
                # Sort by effectiveness
                prompt_skills = sorted(
                    prompt_skills,
                    key=lambda s: (s.success_rate, s.usage_count),
                    reverse=True
                )[:3]  # Top 3 most effective
                
                if prompt_skills:
                    logger.debug(f"Retrieved {len(prompt_skills)} prompt generation skills for {role_name}")
                
            except Exception as skill_error:
                # Non-blocking: log but continue without skills
                logger.warning(f"Failed to retrieve skills for prompt generation ({role_name}): {skill_error}")
                prompt_skills = []
            
            # Prepare prompt generation request using enhanced prompts
            role_obj = role_info if isinstance(role_info, dict) else {"name": role_name, "description": "", "focus_areas": []}
            
            # Build enhanced system prompt
            base_system_prompt = build_enhanced_system_prompt(architecture_hash)
            
            # Add skills section if available
            skills_section = ""
            if prompt_skills:
                skills_section = format_skills_for_prompt(
                    prompt_skills,
                    max_skills=3,
                    category_label="Effective Prompt Patterns"
                )
            
            system_prompt = base_system_prompt + skills_section
            
            # Build enhanced user prompt
            user_prompt = build_enhanced_user_prompt(
                role_name=role_name,
                role_description=role_obj.get('description', '') if isinstance(role_obj, dict) else '',
                role_focus_areas=role_obj.get('focus_areas', []) if isinstance(role_obj, dict) else [],
                architecture_model=architecture_model,
                goal=goal,
                architecture_hash=architecture_hash
            )
            
            # Use llm_node for prompt generation
            llm_state = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "model": model_name,
                "temperature": 0.7
            }
            
            # Call llm_node with fallback handling
            try:
                llm_result = await llm_node(llm_state, config=config)
                
                # Extract prompt
                prompt = llm_result.get("last_response", "")
            except Exception as llm_error:
                logger.warning(f"LLM generation failed for {role_name}: {llm_error}, using fallback")
                # Use fallback prompt
                prompt = generate_fallback_prompt(role_name, architecture_model, architecture_hash)
            
            # Clean response (remove markdown code blocks if present)
            prompt = prompt.strip()
            if prompt.startswith("```"):
                lines = prompt.split("\n")
                if len(lines) > 1:
                    lines = lines[1:]  # Remove first line (```markdown or ```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]  # Remove last line (```)
                prompt = "\n".join(lines).strip()
            
            # Add architecture hash constraint if not present
            if architecture_hash and architecture_hash not in prompt:
                hash_section = f"""

## Architecture Model Constraint

**CRITICAL**: This analysis must only reference technologies, frameworks, and patterns explicitly present in the architecture model.

Architecture Model SHA256 Hash: `{architecture_hash}`

This hash uniquely identifies the detected technologies, frameworks, and patterns. Any technology not explicitly detected in the architecture model MUST NOT be mentioned or assumed.
"""
                prompt = prompt + hash_section
                logger.debug(f"Injected architecture hash into generated prompt for {role_name}")
            
            # Apply technology filtering to remove hardcoded technologies
            from utils.prompts.prompt_technology_filter import remove_hardcoded_technologies
            prompt = remove_hardcoded_technologies(prompt, architecture_model)
            
            # Run shared validation (lenient: warns but continues)
            from utils.prompts.prompt_validator import PromptValidator
            validator = PromptValidator()
            validation_result = validator.validate_all(
                prompt=prompt,
                architecture_model=architecture_model,
                model_name=model_name
            )
            
            # Log warnings if any
            if validation_result.warnings:
                logger.warning(
                    f"Prompt validation warnings for {role_name}: {validation_result.warnings}"
                )
            
            # Cache the generated prompt (with file_hash)
            if architecture_hash and file_hash:
                cache_prompt(role_name, architecture_hash, file_hash, prompt)
            
            # Store prompt in Langfuse (non-blocking, respects config)
            langfuse_prompt_id = None
            langfuse_prompt_version = None
            try:
                architecture_type = architecture_model.get('system_type', 'unknown') if architecture_model else 'unknown'
                
                # Debug: Log hash values before passing to create_langfuse_prompt
                logger.debug(
                    f"Creating Langfuse prompt for {role_name}: "
                    f"architecture_hash={'present' if architecture_hash else 'MISSING'}, "
                    f"file_hash={'present' if file_hash else 'MISSING'}"
                )
                
                langfuse_prompt_result = create_langfuse_prompt(
                    name=f"LLM-generated Architecture-aware prompt for {role_name}",
                    prompt=prompt,
                    role=role_name,
                    architecture_type=architecture_type,
                    goal=goal,
                    architecture_hash=architecture_hash,
                    file_hash=file_hash,
                    labels=None  # Will be set automatically in create_langfuse_prompt
                )
                if langfuse_prompt_result:
                    langfuse_prompt_id = langfuse_prompt_result.get("id")
                    langfuse_prompt_version = langfuse_prompt_result.get("version")
                    if langfuse_prompt_id:
                        logger.debug(f"Stored prompt for {role_name} in Langfuse (id: {langfuse_prompt_id}, version: {langfuse_prompt_version})")
            except Exception as langfuse_error:
                # Non-blocking: log but continue
                logger.warning(f"Failed to store prompt in Langfuse for {role_name}: {langfuse_error}")
            
            # Update observation with result
            if prompt_gen_obs:
                try:
                    prompt_gen_obs.update(
                        output={"prompt_length": len(prompt), "role": role_name},
                        metadata={"status": "success"}
                    )
                except AttributeError:
                    # Observation is a context manager, update not supported directly
                    pass
            
            # Notify UI about source (LLM)
            if stream_callback:
                stream_callback("prompt_source_detected", {
                    "role": role_name,
                    "source": "LLM Generation",
                    "message": f"Generated new {role_name} prompt via LLM"
                })

            # Return ONLY the fields that should be updated (generated_prompts_list)
            # The reducer (operator.add) will aggregate generated_prompts_list from all parallel nodes
            logger.info(f"Generated prompt for role: {role_name}")
            return {
                "generated_prompts_list": [{
                    "role_name": role_name,
                    "prompt": prompt,
                    "langfuse_prompt_id": langfuse_prompt_id,
                    "langfuse_prompt_version": langfuse_prompt_version
                }]
            }
            
        finally:
            if prompt_gen_obs:
                try:
                    prompt_gen_obs.end()
                except Exception:
                    pass
        
    except Exception as e:
        logger.error(f"Error generating prompt for {role_name}: {e}", exc_info=True)
        # Return error result in generated_prompts_list
        error_result = {
            "role_name": role_name,
            "error": str(e),
            "status": "error"
        }
        return {
            "generated_prompts_list": [error_result]
        }


def collect_generated_prompts_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Collect and aggregate results from parallel prompt generation.
    
    Converts generated_prompts_list (populated by parallel nodes) into generated_prompts Dict format.
    Extracts langfuse_prompt_ids and validates all prompts generated successfully.
    
    Args:
        state: Current swarm analysis state with generated_prompts_list
        
    Returns:
        Updated state with generated_prompts Dict and langfuse_prompt_ids Dict
    """
    generated_prompts_list = state.get(StateKeys.GENERATED_PROMPTS_LIST, [])
    selected_roles = state.get(StateKeys.SELECTED_ROLES, [])
    
    if not generated_prompts_list:
        # No results collected - check if we have roles
        if selected_roles:
            state[StateKeys.ERROR] = "No prompt results collected from parallel execution"
            state[StateKeys.ERROR_STAGE] = "collect_generated_prompts"
            return state
        else:
            # No roles to process - this is fine
            updated_state = state.copy()
            updated_state[StateKeys.GENERATED_PROMPTS] = {}
            updated_state[StateKeys.LANGFUSE_PROMPT_IDS] = {}
            updated_state[StateKeys.LANGFUSE_PROMPT_VERSIONS] = {}
            return updated_state
    
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    try:
        # Convert list to Dict format
        generated_prompts = {}
        langfuse_prompt_ids = {}
        langfuse_prompt_versions = {}
        successful_prompts = 0
        failed_prompts = 0
        
        for result in generated_prompts_list:
            role_name = result.get("role_name", "Unknown")
            
            if result.get("status") == "error" or "error" in result:
                # Error case
                error_msg = result.get("error", "Unknown error")
                logger.warning(f"Failed to generate prompt for {role_name}: {error_msg}")
                failed_prompts += 1
                # Still add to dict with empty prompt or error indicator
                generated_prompts[role_name] = f"[ERROR: {error_msg}]"
            else:
                # Success case
                prompt = result.get("prompt", "")
                if prompt:
                    generated_prompts[role_name] = prompt
                    successful_prompts += 1
                    
                    # Extract Langfuse prompt ID and version if available
                    langfuse_prompt_id = result.get("langfuse_prompt_id")
                    langfuse_prompt_version = result.get("langfuse_prompt_version")
                    if langfuse_prompt_id:
                        langfuse_prompt_ids[role_name] = langfuse_prompt_id
                    if langfuse_prompt_version is not None:
                        langfuse_prompt_versions[role_name] = langfuse_prompt_version
                else:
                    logger.warning(f"Generated prompt for {role_name} is empty")
                    failed_prompts += 1
        
        # Validate all prompts generated successfully
        expected_count = len(selected_roles) if selected_roles else len(generated_prompts_list)
        actual_count = len(generated_prompts_list)
        
        if actual_count < expected_count:
            logger.warning(
                f"Only {actual_count} of {expected_count} prompts generated. "
                f"Some prompts may have failed silently."
            )
        
        # Set workflow-level error only if all prompts failed
        if failed_prompts > 0 and successful_prompts == 0:
            logger.error("All prompts failed during generation")
            # Don't set error here - let store_prompts_node handle it
            # But log the issue
        
        # Update state
        updated_state = state.copy()
        updated_state[StateKeys.GENERATED_PROMPTS] = generated_prompts
        if langfuse_prompt_ids:
            updated_state[StateKeys.LANGFUSE_PROMPT_IDS] = langfuse_prompt_ids
        if langfuse_prompt_versions:
            updated_state[StateKeys.LANGFUSE_PROMPT_VERSIONS] = langfuse_prompt_versions
        
        if stream_callback:
            stream_callback("prompt_generation_complete", {
                "message": "All prompts generated",
                "successful": successful_prompts,
                "failed": failed_prompts,
                "total": actual_count
            })
        
        logger.info(
            f"Collected results from {actual_count} prompt generations: "
            f"{successful_prompts} successful, {failed_prompts} failed"
        )
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error collecting generated prompts: {e}", exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = str(e)
        updated_state[StateKeys.ERROR_STAGE] = "collect_generated_prompts"
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": str(e)})
        return updated_state


