"""
Swarm Analysis nodes for LangGraph workflow.

This module is now a backward compatibility wrapper.
All functionality has been moved to workflows.swarm_analysis package.

For new code, import directly from workflows.swarm_analysis:
    from workflows.swarm_analysis import scan_project_node, build_architecture_node, ...

This wrapper maintains backward compatibility for existing imports:
    from workflows.swarm_analysis_nodes import scan_project_node, ...
"""

# Re-export everything from the new package structure
from workflows.swarm_analysis import (
    # Project scanning
    scan_project_node,
    # Architecture
    build_architecture_node,
    summarize_architecture_for_roles,
    # Previous reports
    load_previous_reports_node,
    # Role selection
    select_roles_node,
    # Prompt generation
    dispatch_prompt_generation_node,
    generate_single_prompt_node,
    collect_generated_prompts_node,
    build_enhanced_system_prompt,
    build_enhanced_user_prompt,
    generate_fallback_prompt,
    # Prompt validation
    dispatch_prompt_validation_node,
    validate_single_prompt_node,
    collect_validation_results_node,
    store_prompts_node,
    # Agent execution
    spawn_agents_node,
    dispatch_agents_node,
    execute_single_agent_node,
    collect_agent_results_node,
    # Content utilities
    prepare_chunk_content,
    validate_chunk_content,
    diagnose_content_issues,
    build_chunk_analysis_message,
    synthesize_chunk_results,
    # Synthesis
    synthesize_results_node,
)

# Also export for backward compatibility (old private function names)
# These are now public in the new structure
_build_enhanced_system_prompt = build_enhanced_system_prompt
_build_enhanced_user_prompt = build_enhanced_user_prompt
_generate_fallback_prompt = generate_fallback_prompt
_prepare_chunk_content = prepare_chunk_content
_validate_chunk_content = validate_chunk_content
_diagnose_content_issues = diagnose_content_issues
_build_chunk_analysis_message = build_chunk_analysis_message
_synthesize_chunk_results = synthesize_chunk_results
_summarize_architecture_for_roles = summarize_architecture_for_roles

__all__ = [
    # Project scanning
    "scan_project_node",
    # Architecture
    "build_architecture_node",
    "summarize_architecture_for_roles",
    "_summarize_architecture_for_roles",  # Backward compat
    # Role selection
    "select_roles_node",
    # Prompt generation
    "dispatch_prompt_generation_node",
    "generate_single_prompt_node",
    "collect_generated_prompts_node",
    "build_enhanced_system_prompt",
    "build_enhanced_user_prompt",
    "generate_fallback_prompt",
    "_build_enhanced_system_prompt",  # Backward compat
    "_build_enhanced_user_prompt",  # Backward compat
    "_generate_fallback_prompt",  # Backward compat
    # Prompt validation
    "dispatch_prompt_validation_node",
    "validate_single_prompt_node",
    "collect_validation_results_node",
    "store_prompts_node",
    # Agent execution
    "spawn_agents_node",
    "dispatch_agents_node",
    "execute_single_agent_node",
    "collect_agent_results_node",
    # Content utilities
    "prepare_chunk_content",
    "validate_chunk_content",
    "diagnose_content_issues",
    "build_chunk_analysis_message",
    "synthesize_chunk_results",
    "_prepare_chunk_content",  # Backward compat
    "_validate_chunk_content",  # Backward compat
    "_diagnose_content_issues",  # Backward compat
    "_build_chunk_analysis_message",  # Backward compat
    "_synthesize_chunk_results",  # Backward compat
    # Synthesis
    "synthesize_results_node",
]
