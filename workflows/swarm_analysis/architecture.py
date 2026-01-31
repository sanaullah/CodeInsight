"""
Architecture building node for Swarm Analysis workflow.

Builds or retrieves architecture models and provides helper functions
for summarizing architecture information.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from utils.error_context import create_error_context

logger = logging.getLogger(__name__)


@validate_state_fields(["project_path", "files"], "build_architecture")
async def build_architecture_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Build or retrieve architecture model.
    
    Uses ArchitectureModelBuilder for proper model construction.
    
    Args:
        state: Current swarm analysis state
        
    Returns:
        Updated state with architecture_model
    """
    project_path: Optional[str] = state.get(StateKeys.PROJECT_PATH)
    files: List[Dict[str, Any]] = state.get(StateKeys.FILES, [])
    file_hash: Optional[str] = state.get(StateKeys.FILE_HASH)
    model_name: Optional[str] = state.get(StateKeys.MODEL_NAME)
    
    if not project_path or not files:
        error_msg = (
            "project_path and files are required for architecture building. "
            "Please ensure the project was scanned successfully before building architecture."
        )
        state[StateKeys.ERROR] = error_msg
        state[StateKeys.ERROR_STAGE] = "build_architecture"
        logger.error(error_msg)
        return state
    
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback("architecture_building_start", {
            "message": "Building architecture model...",
            "file_count": len(files)
        })
    
    try:
        from agents.architecture_model_builder import ArchitectureModelBuilder
        from utils.knowledge_base import get_architecture_model
        from scanners.project_scanner import FileInfo
        from utils.langfuse_integration import create_observation, create_llm_observation
        
        logger.info(f"Building architecture model for {project_path}")

        if stream_callback:
            stream_callback("architecture_status_update", {
                "status": "checking_cache",
                "message": "🔍 Checking for cached architecture model..."
            })

        
        # Create observation for architecture building operation
        arch_obs = create_observation(
            name="architecture_building",
            input_data={
                "project_path": project_path,
                "file_count": len(files),
                "file_hash": file_hash
            },
            metadata={
                "operation": "build_architecture",
                "model": model_name
            }
        )
        
        try:
            # Convert dict files back to FileInfo objects if needed
            file_objects = files
            if files and isinstance(files[0], dict):
                file_objects = [
                    FileInfo(
                        path=Path(f["path"]),
                        relative_path=f["relative_path"],
                        content=f.get("content", ""),
                        size=f.get("size", 0),
                        line_count=f.get("line_count", 0),
                        encoding=f.get("encoding", "utf-8")
                    ) for f in files
                ]
            
            # Check cache first
            architecture_model = None
            if file_hash:
                try:
                    knowledge_obj = get_architecture_model(project_path, file_hash)
                    if knowledge_obj:
                        # Extract content from Knowledge object
                        model_dict = knowledge_obj.content if hasattr(knowledge_obj, 'content') else knowledge_obj
                        if model_dict:
                            from agents.architecture_models import ArchitectureModel
                            architecture_model = ArchitectureModel.model_validate(model_dict)
                            logger.info(f"Using cached architecture model for {project_path}")
                            if stream_callback:
                                stream_callback("architecture_status_update", {
                                    "status": "cache_hit",
                                    "message": "✅ Found cached architecture model (No API Check)"
                                })
                            if arch_obs:
                                try:
                                    arch_obs.update(
                                        output={"status": "cached", "system_name": architecture_model.system_name},
                                        metadata={"cached": True}
                                    )
                                except AttributeError:
                                    # Observation is a context manager, update not supported directly
                                    pass
                except Exception as e:
                    logger.warning(f"Error checking cache for architecture model: {e}")
                    # Continue to build new model
            
            # Build new model if not cached
            if not architecture_model:
                # Create LLM observation for architecture deep analysis
                llm_obs = create_llm_observation(
                    operation_name="architecture_deep_analysis",
                    model=model_name,
                    input_data={
                        "file_count": len(file_objects),
                        "project_path": project_path
                    },
                    metadata={
                        "operation": "deep_analysis"
                    }
                )
                
                if stream_callback:
                    stream_callback("architecture_status_update", {
                        "status": "building_llm",
                        "message": "🏗️ Constructing new architecture model (LLM API Call)..."
                    })
                
                try:
                    builder = ArchitectureModelBuilder(model_name=model_name)
                    architecture_model = await builder.build_architecture_model(
                        project_path=project_path,
                        files=file_objects,
                        model_name=model_name,
                        use_cache=True
                    )
                    
                    if not architecture_model:
                        raise ValueError("Failed to build architecture model")
                    
                    if llm_obs:
                        try:
                            llm_obs.update(
                                output={
                                    "system_name": architecture_model.system_name,
                                    "system_type": architecture_model.system_type,
                                    "architecture_pattern": architecture_model.architecture_pattern
                                },
                                metadata={"status": "success"}
                            )
                        except AttributeError:
                            # Observation is a context manager, update not supported directly
                            pass
                finally:
                    if llm_obs:
                        try:
                            llm_obs.end()
                        except Exception:
                            pass
            
            # Convert Pydantic model to dict for state storage
            architecture_model_dict = architecture_model.model_dump()
            
            # Extract architecture details for trace tags
            system_type = architecture_model_dict.get("system_type", "unknown")
            architecture_pattern = architecture_model_dict.get("architecture_pattern", "unknown")
            frameworks = architecture_model_dict.get("frameworks", [])
            primary_framework = frameworks[0] if frameworks else "unknown"
            
            # Compute architecture hash using utility
            from utils.io.architecture_hash import compute_architecture_hash
            arch_hash = compute_architecture_hash(architecture_model_dict)
            
            # Update observation with final result
            if arch_obs:
                try:
                    arch_obs.update(
                        output={
                            "system_name": architecture_model.system_name,
                            "system_type": architecture_model.system_type,
                            "architecture_pattern": architecture_model.architecture_pattern,
                            "components_count": len(architecture_model.modules) if hasattr(architecture_model, 'modules') else 0
                        },
                        metadata={"status": "success"}
                    )
                    
                    # Update root trace with architecture tags
                    try:
                        from utils.langfuse.client import get_langfuse_client
                        
                        # Extract trace_id from observation using multiple fallback methods
                        trace_id = None
                        if hasattr(arch_obs, 'trace_id'):
                            trace_id = getattr(arch_obs, 'trace_id', None)
                        elif hasattr(arch_obs, 'trace'):
                            trace_obj = getattr(arch_obs, 'trace', None)
                            if trace_obj and hasattr(trace_obj, 'id'):
                                trace_id = getattr(trace_obj, 'id', None)
                        elif hasattr(arch_obs, 'id'):
                            # Fallback: may be observation ID, but try it
                            trace_id = getattr(arch_obs, 'id', None)
                            logger.debug("Using arch_obs.id as trace ID (may be observation ID)")
                        
                        if trace_id:
                            client = get_langfuse_client()
                            if client:
                                trace_tags = [
                                    f"arch:{system_type}",
                                    f"pattern:{architecture_pattern}",
                                    f"framework:{primary_framework}"
                                ]
                                try:
                                    trace = client.trace(id=trace_id)
                                    if trace:
                                        trace.update(tags=trace_tags)
                                        logger.debug(f"Updated root trace {trace_id} with architecture tags: {trace_tags}")
                                except Exception as trace_update_error:
                                    logger.debug(f"Could not update trace tags via client.trace().update(): {trace_update_error}")
                                    # Try alternative method: update via observation's trace
                                    try:
                                        if hasattr(arch_obs, 'update_trace'):
                                            arch_obs.update_trace(tags=trace_tags)
                                            logger.debug(f"Updated root trace via observation.update_trace() with tags: {trace_tags}")
                                    except Exception as alt_error:
                                        logger.debug(f"Could not update trace tags via alternative method: {alt_error}")
                        else:
                            logger.debug("No trace_id available from observation, skipping trace tag update")
                    except Exception as e:
                        # Don't fail the node if trace update fails
                        logger.debug(f"Could not update root trace tags: {e}")
                        
                except AttributeError:
                    # Observation is a context manager, update not supported directly
                    pass
            
            # Update state
            updated_state = state.copy()
            updated_state[StateKeys.ARCHITECTURE_MODEL] = architecture_model_dict
            updated_state[StateKeys.ARCHITECTURE_HASH] = arch_hash
            
            if stream_callback:
                stream_callback("architecture_building_complete", {
                    "system_name": architecture_model.system_name,
                    "system_type": architecture_model.system_type,
                    "architecture_pattern": architecture_model.architecture_pattern,
                    "components": [m.name for m in architecture_model.modules[:10]]
                })
            
            logger.info(f"Architecture model built: {architecture_model.system_name}")
            return updated_state
        finally:
            if arch_obs:
                try:
                    arch_obs.end()
                except Exception:
                    pass
        
    except Exception as e:
        error_context = create_error_context("build_architecture", state, {
            "file_count": len(files) if files else 0,
            "file_hash": file_hash,
            "model_name": model_name
        })
        error_msg = (
            f"Failed to build architecture model for '{project_path}': {str(e)}. "
            f"Processed {len(files) if files else 0} files before error. "
            f"Check that the project structure is valid and files are readable."
        )
        logger.error(error_msg, extra=error_context, exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = error_msg
        updated_state[StateKeys.ERROR_STAGE] = "build_architecture"
        updated_state[StateKeys.ERROR_CONTEXT] = error_context
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": error_msg})
        return updated_state


def summarize_architecture_for_roles(architecture_model: Dict[str, Any]) -> str:
    """
    Create a comprehensive summary of the architecture model for role creation.
    
    Args:
        architecture_model: Architecture model dictionary
        
    Returns:
        Formatted architecture summary string
    """
    summary_parts = [
        f"System: {architecture_model.get('system_name', 'Unknown')}",
        f"Type: {architecture_model.get('system_type', 'Unknown')}",
        f"Architecture Pattern: {architecture_model.get('architecture_pattern', 'Unknown')}",
    ]
    
    # Add modules if available
    modules = architecture_model.get('modules', [])
    if modules:
        summary_parts.append(f"\nModules ({len(modules)}):")
        for module in modules[:5]:  # Limit to first 5
            if isinstance(module, dict):
                module_name = module.get('name', 'Unknown')
                module_purpose = module.get('purpose', '')
                summary_parts.append(f"  - {module_name}: {module_purpose}")
            else:
                # Handle case where module might be a string or other type
                summary_parts.append(f"  - {str(module)}")
        if len(modules) > 5:
            summary_parts.append(f"  ... and {len(modules) - 5} more")
    
    # Add frameworks
    frameworks = architecture_model.get('frameworks', [])
    if frameworks:
        if isinstance(frameworks, list):
            summary_parts.append(f"\nFrameworks: {', '.join(frameworks)}")
        else:
            summary_parts.append(f"\nFrameworks: {frameworks}")
    
    # Add libraries
    libraries = architecture_model.get('libraries', [])
    if libraries:
        if isinstance(libraries, list):
            summary_parts.append(f"\nLibraries: {', '.join(libraries[:10])}")
        else:
            summary_parts.append(f"\nLibraries: {libraries}")
    
    # Add API endpoints
    api_endpoints = architecture_model.get('api_endpoints', [])
    if api_endpoints:
        if isinstance(api_endpoints, list):
            summary_parts.append(f"\nAPI Endpoints: {len(api_endpoints)} endpoints")
        else:
            summary_parts.append(f"\nAPI Endpoints: Present")
    
    # Add security architecture
    security_architecture = architecture_model.get('security_architecture')
    if security_architecture:
        if isinstance(security_architecture, dict) and security_architecture:
            summary_parts.append(f"\nSecurity Architecture: Present")
        elif security_architecture:
            summary_parts.append(f"\nSecurity Architecture: Present")
    
    # Add database schema
    database_schema = architecture_model.get('database_schema')
    if database_schema:
        if isinstance(database_schema, dict) and database_schema:
            summary_parts.append(f"\nDatabase Schema: Present")
        elif database_schema:
            summary_parts.append(f"\nDatabase Schema: Present")
    
    # Add design patterns
    design_patterns = architecture_model.get('design_patterns', [])
    if design_patterns:
        if isinstance(design_patterns, list) and design_patterns:
            summary_parts.append(f"\nDesign Patterns: {', '.join(design_patterns[:5])}")
        elif design_patterns:
            summary_parts.append(f"\nDesign Patterns: Present")
    
    # Add anti-patterns
    anti_patterns = architecture_model.get('anti_patterns', [])
    if anti_patterns:
        if isinstance(anti_patterns, list):
            summary_parts.append(f"\nAnti-patterns Detected: {len(anti_patterns)}")
        else:
            summary_parts.append(f"\nAnti-patterns Detected: Present")
    
    return "\n".join(summary_parts)


