"""
Swarm Analysis workflow nodes.

This package contains all workflow nodes for the Swarm Analysis system,
organized by domain for better maintainability and navigation.

Public API: All node functions are exported here for backward compatibility.
"""

# Re-export all nodes for backward compatibility
from .project_scanning import scan_project_node
from .architecture import build_architecture_node, summarize_architecture_for_roles
from .previous_reports import load_previous_reports_node
from .role_selection import select_roles_node
from .prompt_generation.nodes import (
    dispatch_prompt_generation_node,
    generate_single_prompt_node,
    collect_generated_prompts_node,
)
from .prompt_generation.builders import (
    build_enhanced_system_prompt,
    build_enhanced_user_prompt,
)
from .prompt_generation.fallback import generate_fallback_prompt
from .prompt_validation.nodes import (
    dispatch_prompt_validation_node,
    validate_single_prompt_node,
    collect_validation_results_node,
)
from .prompt_validation.storage import store_prompts_node
from .agent_execution.nodes import (
    spawn_agents_node,
    dispatch_agents_node,
    execute_single_agent_node,
    collect_agent_results_node,
)
from .agent_execution.content_utils import (
    prepare_chunk_content,
    validate_chunk_content,
    diagnose_content_issues,
    build_chunk_analysis_message,
    synthesize_chunk_results,
)
from .synthesis import synthesize_results_node

__all__ = [
    # Project scanning
    "scan_project_node",
    # Architecture
    "build_architecture_node",
    "summarize_architecture_for_roles",
    # Previous reports
    "load_previous_reports_node",
    # Role selection
    "select_roles_node",
    # Prompt generation
    "dispatch_prompt_generation_node",
    "generate_single_prompt_node",
    "collect_generated_prompts_node",
    "build_enhanced_system_prompt",
    "build_enhanced_user_prompt",
    "generate_fallback_prompt",
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
    # Synthesis
    "synthesize_results_node",
]



