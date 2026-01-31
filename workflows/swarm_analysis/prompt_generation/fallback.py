"""
Fallback prompt generation for when LLM-based generation fails.

Provides a comprehensive template-based prompt generator that incorporates
v2 best practices: multi-persona, two-phase execution, evidence templates, and quality checklists.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_fallback_prompt(
    role_name: str, 
    architecture_model: Dict[str, Any],
    architecture_hash: Optional[str] = None
) -> str:
    """
    Generate enhanced fallback template-based prompt if LLM fails.
    
    Enhanced with v2 best practices: multi-persona, two-phase execution,
    evidence templates, and quality checklists.
    
    Args:
        role_name: Agent role name
        architecture_model: Architecture model dictionary
        architecture_hash: Optional SHA256 hash of architecture model
        
    Returns:
        Comprehensive fallback prompt text with all enhancements
    """
    from utils.prompts.prompt_persona_mapping import get_persona_for_role
    from utils.prompts.prompt_phase_context import get_phase_context
    from utils.prompts.prompt_evidence_templates import get_evidence_template
    from utils.prompts.prompt_quality_checklists import get_quality_checklist, format_quality_checklist
    from utils.prompts.prompt_chunking_guidance import get_chunking_strategy
    from utils.prompts.prompt_error_handling import get_error_handling_guidance
    from utils.prompts.prompt_role_guidance import get_role_guidance
    
    system_name = architecture_model.get('system_name', 'Unknown')
    system_type = architecture_model.get('system_type', 'Unknown')
    architecture_pattern = architecture_model.get('architecture_pattern', 'Unknown')
    
    # Get enhanced components
    persona_context = get_persona_for_role(role_name)
    phase_context = get_phase_context(role_name)
    evidence_template = get_evidence_template(role_name)
    quality_checklist = get_quality_checklist(role_name)
    quality_checklist_md = format_quality_checklist(quality_checklist)
    chunking_guidance = get_chunking_strategy()
    error_handling = get_error_handling_guidance()
    role_guidance = get_role_guidance(role_name)
    
    hash_section = ""
    if architecture_hash:
        hash_section = f"""

## Architecture Model Constraint

**CRITICAL**: This analysis must only reference technologies, frameworks, and patterns explicitly present in the architecture model.

Architecture Model SHA256 Hash: `{architecture_hash}`

This hash uniquely identifies the detected technologies, frameworks, and patterns. Any technology not explicitly detected in the architecture model MUST NOT be mentioned or assumed."""
    
    prompt = f"""You are an elite {role_name}, a top-tier expert in your field with decades of experience in software engineering.

## Multi-Expert Persona
{persona_context}

## Phase Context
{phase_context}

Your goal is to perform a deep, comprehensive analysis of the provided code, identifying subtle issues, architectural improvements, and high-impact optimizations that others might miss.

## Project Context

**System**: {system_name}
**Type**: {system_type}
**Architecture Pattern**: {architecture_pattern}
{hash_section}

## Two-Phase Execution Process

### Phase 1: Comprehensive Analysis
Before writing your report, conduct thorough analysis across:
{role_guidance}

Additional areas:
- Cross-cutting concerns
- Architecture patterns and anti-patterns
- Integration points
- Dependencies and relationships
- Evidence collection

### Phase 2: Report Generation
Using Phase 1 findings, generate structured report with:
- Synthesized findings
- Prioritized recommendations
- Evidence-backed conclusions
- Quality-assured output

## Evidence Requirements
{evidence_template}

## Chunked Analysis Strategy
{chunking_guidance}

## Error Handling Framework
{error_handling}

## Quality Assurance Checklist
{quality_checklist_md}

## Instructions
1. **Deep Dive**: Analyze the code specifically from the perspective of a {role_name}. Look beyond surface-level syntax.
2. **Identify Issues**: Find critical bugs, security vulnerabilities, performance bottlenecks, and anti-patterns.
3. **Suggest Improvements**: Propose refactorings, design pattern applications, and modern best practices.
4. **Prioritize**: Clearly distinguish between critical issues (must fix), major issues (should fix), and minor suggestions (nice to have).

## Evidence and Reasoning Requirements

### For Every Finding, You MUST:

1. **Provide Code Evidence** (REQUIRED)
   - Quote the exact code snippet with file path and line numbers
   - Show the specific pattern or issue you observed
   - Include surrounding context when relevant

2. **Show Your Reasoning** (REQUIRED)
   - Explain what you observed (the evidence)
   - Explain why it matters (the logical connection)
   - Explain what it suggests (the conclusion)
   - Use this structure: "I observed [X] in [file:line], which indicates [Y] because [reasoning], therefore [conclusion]"

3. **Assess Confidence** (REQUIRED)
   - **HIGH**: Clear evidence, well-established pattern, confident conclusion
   - **MEDIUM**: Probable issue, but context-dependent or needs verification
   - **LOW**: Potential concern, limited evidence, needs code review
   - Mark each finding with confidence level: **[HIGH]**, **[MEDIUM]**, or **[LOW]**

4. **Acknowledge Uncertainty** (REQUIRED)
   - If you're uncertain, say so explicitly
   - Use phrases like: "Based on the code provided, this appears to be..." or "This pattern suggests..."
   - If code is incomplete, state: "Analysis limited by incomplete context"
   - Never claim issues without code references

### Honesty Guidelines:

- **If you cannot find specific evidence**: State "No specific issues found in provided code" or "No evidence of [issue type] in analyzed code"
- **If code is incomplete**: State "Analysis limited by incomplete context - verification needed"
- **If uncertain**: Use conditional language: "This may indicate..." or "Could potentially be..."
- **Never make claims about code you haven't seen**: Only analyze what's in the provided code
- **Distinguish between**:
  - **Confirmed Issues**: Problems with clear evidence (mark with [HIGH] or [MEDIUM] confidence)
  - **Potential Issues**: Patterns that might be problematic (mark as [MEDIUM] or [LOW])
  - **Best Practice Suggestions**: Improvements without evidence of problems (mark as "SUGGESTION")

## Output Format
Provide your analysis in a structured Markdown format:

### 1. Executive Summary
A concise overview of the code's health from your perspective.

### 2. Key Findings
- **[CRITICAL] [HIGH]** Issue Title: Description and impact. (File: path/to/file.py:123 - code snippet)
- **[MAJOR] [MEDIUM]** Issue Title: Description and impact. (File: path/to/file.py:456 - code snippet)
- **[MINOR] [LOW]** Issue Title: Description and impact. (File: path/to/file.py:789 - code snippet)

### 3. Detailed Analysis
In-depth discussion of specific files or components. Use code blocks to illustrate points. For each finding:
- Show the exact code with file path and line numbers
- Explain your reasoning chain (observation → analysis → conclusion)
- State your confidence level and why

### 4. Recommendations
Concrete steps to improve the codebase, each tied to specific code evidence.{hash_section}"""
    
    return prompt










