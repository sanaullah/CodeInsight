"""
Prompt building for agent-driven file selection.

Provides functions to build structured prompts for LLM file selection.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def build_file_selection_prompt(
    role_name: str,
    prompt: str,
    file_list: str,
    architecture_model: Optional[Dict[str, Any]] = None,
    goal: Optional[str] = None
) -> str:
    """
    Build structured prompt for LLM file selection.
    
    Creates a comprehensive prompt that includes:
    - Role and analysis goal context
    - Architecture model summary
    - Available files list (formatted)
    - Selection criteria and guidelines
    - JSON response format specification
    
    Args:
        role_name: Name of the agent role
        prompt: Agent's analysis prompt
        file_list: Formatted list of available files
        architecture_model: Optional architecture model for context
        goal: Optional analysis goal
        
    Returns:
        Complete prompt string for file selection
    """
    # Build role context section
    role_context = f"""## Your Role: {role_name}

You are a {role_name} analyzing a codebase. Before you begin your analysis, you need to identify which files are relevant to your role and analysis goals.

## Your Analysis Goal

{prompt}

"""
    
    # Build architecture context if available
    arch_context = ""
    if architecture_model:
        system_name = architecture_model.get("system_name", "Unknown")
        system_type = architecture_model.get("system_type", "Unknown")
        architecture_pattern = architecture_model.get("architecture_pattern", "Unknown")
        components = architecture_model.get("components", [])
        
        arch_context = f"""## Architecture Context

**System**: {system_name} ({system_type})
**Architecture Pattern**: {architecture_pattern}
"""
        if components:
            arch_context += f"**Key Components**: {', '.join(components[:10])}"
            if len(components) > 10:
                arch_context += f" (and {len(components) - 10} more)"
            arch_context += "\n\n"
    
    # Build goal context if provided
    goal_context = ""
    if goal:
        goal_context = f"""## Analysis Goal

{goal}

"""
    
    # Build selection criteria
    selection_criteria = """## File Selection Criteria

Based on your role and analysis goal, identify which files you need to analyze. Consider:

1. **Direct Dependencies**: Files that are imported, included, or directly referenced
2. **Related Modules**: Shared utilities, common libraries, helper functions
3. **Configuration Files**: Settings, environment configs, deployment configs (if relevant to your role)
4. **Test Files**: Test suites, test utilities (if reviewing test coverage or quality)
5. **Documentation**: README, docs (if reviewing documentation quality)
6. **Entry Points**: Main files, application entry points, API endpoints
7. **Role-Specific Files**: 
   - **Security Analyst**: Authentication, authorization, input validation, encryption
   - **Performance Optimizer**: Database queries, caching, async operations, bottlenecks
   - **Code Quality Reviewer**: All source files, test files, configuration
   - **Architecture Reviewer**: Main modules, interfaces, design patterns
   - **Dependency Analyzer**: Requirements files, package managers, dependency declarations

**Selection Guidelines**:
- Be selective: Only choose files directly relevant to your analysis
- Consider transitive dependencies: If you need file A, and A imports B, include B
- Prioritize by relevance: Focus on files most critical to your role
- Don't over-select: Avoid selecting entire directories unless necessary
- Don't under-select: Ensure you have enough context for thorough analysis

"""
    
    # Build response format
    response_format = """## Response Format

Respond with a JSON object containing:

```json
{
    "files": ["path/to/file1.py", "path/to/file2.py", ...],
    "reasoning": "Brief explanation of why each file is needed for your analysis. Explain the relationship between files and how they support your analysis goal.",
    "confidence": 0.85
}
```

**Important**:
- Use relative paths (e.g., `src/auth.py` not `/absolute/path/src/auth.py`)
- Include all files you need, even if they seem obvious
- Provide clear reasoning for your selection
- Be specific about why each file is relevant
- **Confidence** (optional): A number between 0.0 and 1.0 indicating how confident you are in this file selection. Use higher values (0.8-1.0) when you're very certain these are the right files, lower values (0.5-0.7) when you're less certain, and omit this field if you cannot assess confidence

"""
    
    # Combine all sections
    full_prompt = f"""{role_context}{goal_context}{arch_context}{selection_criteria}{file_list}

{response_format}

Now, select the files you need for your analysis as {role_name}:
"""
    
    return full_prompt

