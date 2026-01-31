"""
Prompt storage node for Swarm Analysis workflow.

Stores validated prompts to PostgreSQL database with metadata.
"""

import logging
from typing import Dict, Any

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.state_keys import StateKeys

logger = logging.getLogger(__name__)


def store_prompts_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Store validated prompts to PostgreSQL database.
    
    Saves each validated prompt with metadata including:
    - Role name
    - Architecture hash (for reuse)
    - Validation results (confidence, clarity, etc.)
    - Project path
    - LLM-generated flag
    
    Args:
        state: Current swarm analysis state
        
    Returns:
        Updated state (unchanged, prompts stored separately)
    """
    generated_prompts = state.get(StateKeys.GENERATED_PROMPTS, {})
    validation_results = state.get(StateKeys.VALIDATION_RESULTS, {})
    architecture_hash = state.get(StateKeys.ARCHITECTURE_HASH)
    project_path = state.get(StateKeys.PROJECT_PATH)
    role_names = state.get(StateKeys.ROLE_NAMES, [])
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    
    if not generated_prompts:
        # No prompts to store - this is fine, just continue
        logger.debug("No prompts to store")
        return state
    
    if stream_callback:
        stream_callback("prompt_storage", {
            "message": "Storing prompts to database...",
            "prompt_count": len(generated_prompts)
        })
    
    try:
        from utils.prompts.prompt_storage import save_prompt
        from utils.langfuse_integration import create_langfuse_prompt
        
        stored_count = 0
        failed_count = 0
        langfuse_prompt_ids = state.get(StateKeys.LANGFUSE_PROMPT_IDS, {})
        architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL, {})
        goal = state.get(StateKeys.GOAL)
        
        # Iterate through generated prompts
        # Use role_names for order if available, otherwise use generated_prompts keys
        roles_to_process = role_names if role_names else list(generated_prompts.keys())
        
        for role_name in roles_to_process:
            if role_name not in generated_prompts:
                logger.warning(f"Role {role_name} in role_names but not in generated_prompts, skipping")
                continue
            
            prompt_content = generated_prompts[role_name]
            validation_result = validation_results.get(role_name)
            
            try:
                # Build metadata dict
                metadata = {
                    "project_path": project_path,
                    "architecture_aware": True,
                    "llm_generated": True,
                }
                
                # Add validation results if available
                validation_metadata = None
                if validation_result:
                    validation_metadata = {
                        "validated": validation_result.get("is_valid", True),
                        "confidence": validation_result.get("confidence", 0.0),
                        "clarity": validation_result.get("clarity", 0.0),
                        "completeness": validation_result.get("completeness", 0.0),
                        "relevance": validation_result.get("relevance", 0.0),
                        "accuracy": validation_result.get("accuracy", 0.0),
                        "overall_score": validation_result.get("overall_score", 0.0),
                        "feedback": validation_result.get("feedback", "")
                    }
                    metadata.update(validation_metadata)
                else:
                    metadata["validated"] = False
                
                # Generate prompt name
                prompt_name = f"LLM-generated Architecture-aware prompt for {role_name}"
                
                # Generate description
                description = f"LLM-generated Architecture-aware prompt for {role_name}"
                
                # Save prompt to PostgreSQL database
                prompt_id = save_prompt(
                    name=prompt_name,
                    content=prompt_content,
                    description=description,
                    metadata=metadata,
                    role=role_name,
                    architecture_hash=architecture_hash
                )
                
                # Also store/update in Langfuse (respects config)
                # create_langfuse_prompt() and update_langfuse_prompt_config() will check config and return None/False if disabled
                langfuse_prompt_versions = state.get(StateKeys.LANGFUSE_PROMPT_VERSIONS, {})
                existing_langfuse_prompt_id = langfuse_prompt_ids.get(role_name)
                
                if existing_langfuse_prompt_id and validation_result:
                    # Update existing prompt with validation metadata
                    try:
                        from utils.langfuse.prompts import update_langfuse_prompt_config
                        
                        # Extract validation metadata from validation_result
                        validation_metadata_for_update = {
                            "is_valid": validation_result.get("is_valid", True),
                            "confidence": validation_result.get("confidence", 0.0),
                            "clarity": validation_result.get("clarity", 0.0),
                            "completeness": validation_result.get("completeness", 0.0),
                            "relevance": validation_result.get("relevance", 0.0),
                            "accuracy": validation_result.get("accuracy", 0.0),
                            "overall_score": validation_result.get("overall_score", 0.0),
                            "feedback": validation_result.get("feedback", "")
                        }
                        
                        success = update_langfuse_prompt_config(
                            prompt_id=existing_langfuse_prompt_id,
                            validation_metadata=validation_metadata_for_update
                        )
                        
                        if success:
                            logger.info(f"Updated Langfuse prompt for {role_name} with validation scores")
                        else:
                            logger.debug(f"Could not update Langfuse prompt for {role_name} (non-blocking)")
                            
                    except Exception as update_error:
                        # Non-blocking: log but continue
                        logger.debug(f"Failed to update Langfuse prompt config for {role_name}: {update_error}")
                elif role_name not in langfuse_prompt_ids:
                    # Create new prompt if it doesn't exist
                    try:
                        architecture_type = architecture_model.get('system_type', 'unknown') if architecture_model else 'unknown'
                        langfuse_prompt_result = create_langfuse_prompt(
                            name=prompt_name,
                            prompt=prompt_content,
                            role=role_name,
                            architecture_type=architecture_type,
                            goal=goal,
                            validation_metadata=validation_metadata,
                            architecture_hash=state.get(StateKeys.ARCHITECTURE_HASH),
                            file_hash=state.get(StateKeys.FILE_HASH),
                            labels=None
                        )
                        if langfuse_prompt_result:
                            langfuse_prompt_id = langfuse_prompt_result.get("id")
                            langfuse_prompt_version = langfuse_prompt_result.get("version")
                            if langfuse_prompt_id:
                                langfuse_prompt_ids[role_name] = langfuse_prompt_id
                            if langfuse_prompt_version is not None:
                                langfuse_prompt_versions[role_name] = langfuse_prompt_version
                            # Link PostgreSQL and Langfuse IDs in metadata
                            if prompt_id:
                                try:
                                    from utils.prompts.prompt_storage import update_prompt
                                    update_prompt(prompt_id, {
                                        "metadata_json": {
                                            **metadata,
                                            "langfuse_prompt_id": langfuse_prompt_id,
                                            "langfuse_prompt_version": langfuse_prompt_version
                                        }
                                    })
                                except Exception as link_error:
                                    logger.debug(f"Could not link Langfuse prompt ID to PostgreSQL: {link_error}")
                    except Exception as langfuse_error:
                        # Non-blocking: log but continue
                        # Isolated error handling - doesn't affect workflow or observability
                        logger.debug(f"Failed to store prompt in Langfuse for {role_name}: {langfuse_error}")
                        # Don't re-raise - continue with workflow
                
                if prompt_id:
                    stored_count += 1
                    logger.info(f"Stored prompt for {role_name} (PostgreSQL id: {prompt_id})")
                    
                    if stream_callback:
                        stream_callback("prompt_stored", {
                            "role": role_name,
                            "prompt_id": prompt_id,
                            "langfuse_prompt_id": langfuse_prompt_ids.get(role_name),
                            "stored_count": stored_count
                        })
                else:
                    failed_count += 1
                    logger.warning(f"Failed to store prompt for {role_name} (save_prompt returned None)")
                    
            except Exception as e:
                failed_count += 1
                logger.warning(f"Error storing prompt for {role_name}: {e}", exc_info=True)
                # Continue processing other prompts - don't fail the workflow
        
        # Update state with Langfuse prompt IDs and versions if any were created
        if langfuse_prompt_ids:
            state[StateKeys.LANGFUSE_PROMPT_IDS] = langfuse_prompt_ids
        langfuse_prompt_versions = state.get(StateKeys.LANGFUSE_PROMPT_VERSIONS, {})
        if langfuse_prompt_versions:
            state[StateKeys.LANGFUSE_PROMPT_VERSIONS] = langfuse_prompt_versions
        
        logger.info(f"Stored {stored_count} prompts, {failed_count} failed")
        
        if stream_callback:
            stream_callback("prompt_storage", {
                "message": f"Stored {stored_count} prompts",
                "stored_count": stored_count,
                "failed_count": failed_count
            })
        
        # Return state unchanged (prompts stored separately)
        return state
        
    except Exception as e:
        # Log error but don't fail the workflow - prompt storage is non-critical
        from services.storage.base_storage import (
            StorageError,
            StorageConnectionError,
            DatabaseConnectionError,
            ConfigurationError
        )
        from utils.error_formatting import format_user_error
        
        # Log technical details
        if isinstance(e, (StorageConnectionError, DatabaseConnectionError)):
            logger.error(f"Database connection error in store_prompts_node: {e}", exc_info=True)
        elif isinstance(e, StorageError):
            logger.error(f"Storage error in store_prompts_node: {e}", exc_info=True)
        elif isinstance(e, ConfigurationError):
            logger.error(f"Configuration error in store_prompts_node: {e}", exc_info=True)
        else:
            logger.error(f"Unexpected error in store_prompts_node: {e}", exc_info=True)
        
        if stream_callback:
            stream_callback("prompt_storage_error", {
                "error": format_user_error(e),
                "message": "Failed to store prompts, but continuing workflow"
            })
        # Return state unchanged - don't set error to allow workflow to continue
        return state










