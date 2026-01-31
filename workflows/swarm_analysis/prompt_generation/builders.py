"""
Prompt builders for generating enhanced analysis prompts.

Provides functions for building system and user prompts for LLM-based
prompt generation with architecture awareness and quality constraints.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def build_enhanced_system_prompt(architecture_hash: Optional[str] = None) -> str:
    """
    Build comprehensive system prompt for LLM prompt generation with architecture hash constraint.
    
    Ports v2's comprehensive system prompt with 7 key principles and detailed requirements.
    Enhanced with multi-persona approach, two-phase execution, evidence templates, and quality checklists.
    
    Args:
        architecture_hash: Optional SHA256 hash of architecture model for constraint enforcement
        
    Returns:
        Comprehensive system prompt string
    """
    from utils.prompts.prompt_evidence_templates import get_confidence_level_guidance
    from utils.prompts.prompt_quality_checklists import get_universal_checklist, format_quality_checklist
    from utils.prompts.prompt_chunking_guidance import get_chunking_strategy
    
    # Get quality checklist
    universal_checklist = get_universal_checklist()
    quality_checklist_dict = {
        "universal": universal_checklist
    }
    quality_checklist_md = format_quality_checklist(quality_checklist_dict)
    
    # Get chunking guidance
    chunking_guidance = get_chunking_strategy()
    
    # Get confidence level guidance
    confidence_guidance = get_confidence_level_guidance()
    
    base_prompt = f"""You are an expert at creating analysis prompts for AI code analysis agents.

Your task is to generate comprehensive, context-specific prompts that guide AI agents to perform deep, thorough analysis of codebases.

## Multi-Expert Approach

Generated prompts should activate multiple expert perspectives for richer analysis.
Each agent prompt should define a team of complementary experts working together.
This multi-persona approach yields diverse perspectives and more comprehensive analysis.

## Two-Phase Execution Framework

Prompts MUST structure analysis in two distinct phases:

### Phase 1: Comprehensive Analysis
Before writing the report, conduct thorough review across:
- Role-specific analysis areas (security, performance, architecture, etc.)
- Cross-cutting concerns
- Architecture and context
- Evidence collection
- Pattern identification

### Phase 2: Report Generation
Using Phase 1 findings, generate structured, evidence-based report with:
- Synthesized findings
- Prioritized recommendations
- Evidence-backed conclusions
- Quality-assured output

## Evidence-Based Requirements (CRITICAL)

Every finding MUST follow a structured evidence-based format:
- **Location**: Specific file and line numbers (file.py:123-145)
- **Evidence**: Actual code snippets showing the issue
- **Reasoning**: Step-by-step logical chain from observation to conclusion
- **Impact**: Why this matters (security, performance, maintainability, etc.)
- **Recommendation**: Specific, actionable steps to address
- **Uncertainty**: What you're unsure about (if any)

{confidence_guidance}

## Quality Assurance

Generated prompts MUST include self-validation checklist to ensure quality:

{quality_checklist_md}

## Chunked Analysis Strategy

{chunking_guidance}

## Key Principles

1. **Context-Aware**: Tailor the prompt to the specific project architecture, technologies, and characteristics
2. **Role-Specific**: Focus on the specific role's expertise (security, performance, architecture, code quality, etc.)
3. **Actionable**: Provide clear instructions on what to analyze and how
4. **Comprehensive**: Cover all relevant aspects for the role and project type
5. **Structured**: Use clear sections and formatting for readability
6. **Evidence-Based**: REQUIRE agents to provide code evidence for every finding
7. **Honest**: REQUIRE agents to acknowledge uncertainty and distinguish between confirmed issues and suggestions
8. **Constructive and Balanced**: REQUIRE agents to provide honest assessments that acknowledge strengths, frame issues as learning opportunities, and offer constructive guidance rather than just criticism

The prompt MUST include:
- Evidence requirements: Every finding must reference specific code (file:line)
- Reasoning structure: Agents must explain their logical chain from observation to conclusion
- Confidence levels: Agents must mark findings as [HIGH], [MEDIUM], or [LOW]
- Uncertainty handling: Agents must acknowledge when uncertain or when code is incomplete
- Honesty guidelines: Agents must distinguish between confirmed issues, potential concerns, and suggestions
- Constructive tone: Agents must use supportive, educational language and frame findings as guidance opportunities
- Balanced reporting: Agents must acknowledge strengths and good practices alongside areas for improvement

The prompt should:
- Define the agent's role and expertise clearly (using multi-expert persona)
- Specify what aspects to analyze based on the architecture
- Structure analysis in two phases (comprehensive analysis, then report generation)
- REQUIRE evidence-based reasoning for all findings
- REQUIRE confidence assessment for all findings
- REQUIRE uncertainty acknowledgment when appropriate
- REQUIRE acknowledgment of strengths and good practices (not just issues)
- REQUIRE constructive, guidance-oriented language (frame issues as learning opportunities)
- Provide guidance on how to structure findings with evidence
- Include examples or patterns relevant to the project
- Be specific to the technologies and patterns detected in the architecture
- Include quality assurance checklist for self-validation
- Provide chunked analysis strategy for context preservation
- Emphasize supportive, educational tone that helps developers improve

Output only the prompt text, without any additional commentary or markdown formatting."""
    
    if architecture_hash:
        constraint = f"""

CRITICAL CONSTRAINT - NO HARDCODED TECHNOLOGIES:
The architecture model used for this project has SHA256 hash: {architecture_hash}
You MUST only reference technologies, frameworks, and patterns that are explicitly present in this architecture.

FORBIDDEN TECHNOLOGIES LIST (NEVER MENTION THESE UNLESS IN ARCHITECTURE):
The following technologies are COMMONLY HALLUCINATED and MUST NOT be mentioned unless explicitly in the architecture model:
- Docker, docker, containerization, containers
- Kubernetes, kubernetes, k8s, orchestration
- Spring Boot, spring boot, spring framework
- Express.js, express.js, express, Express
- Selenium, selenium, web automation
- BeautifulSoup, beautifulsoup, web scraping
- React, react, Vue, vue, Angular, angular (unless in architecture)
- Node.js, node.js, node (unless in architecture)
- Nginx, nginx, Apache, apache (unless in architecture)
- PostgreSQL, postgresql, MySQL, mysql, MongoDB, mongodb (unless in architecture)

FORBIDDEN PRACTICES (DO NOT DO THESE):
- DO NOT mention ANY technology from the forbidden list above unless it is explicitly in the architecture model
- DO NOT assume common technologies are present - only use what is detected
- DO NOT use generic technology lists - be specific to what's actually detected
- DO NOT reference technologies "if applicable" or "when available" - only reference what IS available
- DO NOT use phrases like "common technologies" or "standard tools" - be explicit

REQUIRED PRACTICES:
- ONLY reference technologies explicitly listed in the architecture model's system_technologies
- Use conditional language: "If [technology] is detected..." only when that technology is actually in the architecture
- When uncertain, omit the technology reference entirely
- Be specific: reference exact technology names from the architecture, not generic categories

If you are uncertain about a technology's presence, DO NOT include it in the prompt. Better to omit a technology than to hallucinate its presence."""
        return base_prompt + constraint
    
    return base_prompt


def build_enhanced_user_prompt(
    role_name: str,
    role_description: str,
    role_focus_areas: List[str],
    architecture_model: Dict[str, Any],
    goal: Optional[str] = None,
    architecture_hash: Optional[str] = None
) -> str:
    """
    Build comprehensive user prompt for LLM prompt generation with rich context.
    
    Ports v2's comprehensive user prompt with architecture summary, role guidance, and constraints.
    
    Args:
        role_name: Name of the analysis role
        role_description: Description of the role
        role_focus_areas: List of focus areas for the role
        architecture_model: Architecture model dictionary
        goal: Optional analysis goal
        architecture_hash: Optional SHA256 hash of architecture model
        
    Returns:
        Comprehensive user prompt string
    """
    from utils.prompts.prompt_role_guidance import get_role_guidance
    from utils.prompts.prompt_persona_mapping import get_persona_for_role
    from utils.prompts.prompt_phase_context import get_phase_context
    from utils.prompts.prompt_chunking_guidance import get_chunking_strategy
    from utils.prompts.prompt_error_handling import get_error_handling_guidance
    from utils.prompts.prompt_evidence_templates import get_evidence_template
    from utils.prompts.prompt_quality_checklists import get_quality_checklist, format_quality_checklist
    from agents.architecture_models import ArchitectureModel
    
    # Convert dict to ArchitectureModel if needed for to_natural_language()
    if isinstance(architecture_model, dict):
        try:
            arch_model_obj = ArchitectureModel.model_validate(architecture_model)
            arch_summary = arch_model_obj.to_natural_language()
        except Exception as e:
            logger.warning(f"Error converting architecture model: {e}, using dict format")
            # Fallback to basic summary
            arch_summary = f"""# Architecture Overview

**System Name**: {architecture_model.get('system_name', 'Unknown')}
**System Type**: {architecture_model.get('system_type', 'Unknown')}
**Architecture Pattern**: {architecture_model.get('architecture_pattern', 'Unknown')}
"""
    else:
        arch_summary = architecture_model.to_natural_language() if hasattr(architecture_model, 'to_natural_language') else str(architecture_model)
    
    # Get system technologies
    if isinstance(architecture_model, dict):
        # Build from dict
        frameworks = architecture_model.get('frameworks', [])
        libraries = architecture_model.get('libraries', [])
        tech_stack = architecture_model.get('tech_stack', {})
        system_technologies = set(frameworks)
        system_technologies.update(libraries)
        if isinstance(tech_stack, dict):
            for category, techs in tech_stack.items():
                if isinstance(techs, list):
                    system_technologies.update(techs)
        system_technologies = sorted(list(system_technologies))
    else:
        # Use property if available
        system_technologies = architecture_model.system_technologies if hasattr(architecture_model, 'system_technologies') else []
    
    # Build project context summary
    tech_list = ', '.join(sorted(system_technologies)) if system_technologies else 'None detected'
    context_summary = f"""
Project Context:
- Technologies: {tech_list}

CRITICAL: ONLY reference these technologies in your prompt: {tech_list}
DO NOT mention any other technologies, even common ones like Docker, Kubernetes, Spring Boot, Express.js, etc.
"""
    
    # Goal information
    goal_section = ""
    if goal:
        goal_section = f"""
Analysis Goal:
{goal}

The prompt should be tailored to help achieve this specific goal.
"""
    
    # Role-specific information
    role_section = f"""
Role Information:
- Name: {role_name}
"""
    if role_description:
        role_section += f"- Description: {role_description}\n"
    if role_focus_areas:
        role_section += f"- Focus Areas: {', '.join(role_focus_areas)}\n"
    
    # Build persona context
    persona_context = get_persona_for_role(role_name)
    
    # Build phase context
    phase_context = get_phase_context(role_name)
    
    # Build evidence template
    evidence_template = get_evidence_template(role_name)
    
    # Build quality checklist
    quality_checklist = get_quality_checklist(role_name)
    quality_checklist_md = format_quality_checklist(quality_checklist)
    
    # Build chunking guidance
    chunking_guidance = get_chunking_strategy()
    
    # Build error handling
    error_handling = get_error_handling_guidance()
    
    # Role-specific guidance
    if role_focus_areas:
        role_guidance = f"""Focus Areas for {role_name}:
{chr(10).join(f"- {area}" for area in role_focus_areas)}"""
    else:
        role_guidance = get_role_guidance(role_name)
    
    # Add hash constraint section
    hash_constraint = ""
    if architecture_hash:
        hash_constraint = f"""

ARCHITECTURE MODEL CONSTRAINT:
The architecture model for this project has SHA256 hash: {architecture_hash}
This hash uniquely identifies the detected technologies, frameworks, and patterns.
You MUST ensure the generated prompt only references technologies present in this architecture.
Any technology not explicitly detected in the architecture model MUST NOT be mentioned."""
    
    prompt = f"""Generate a comprehensive analysis prompt for a {role_name} agent.

## Multi-Expert Persona
{persona_context}

## Phase Context
{phase_context}

{role_section}
## Architecture Overview
{arch_summary}

## Project Context
{context_summary}
{goal_section}

## Two-Phase Execution Structure

The generated prompt must structure analysis in two phases:

### Phase 1: Comprehensive Analysis
Analyze across these dimensions:
{role_guidance}

Additional analysis areas:
- Cross-cutting concerns
- Architecture patterns and anti-patterns
- Integration points
- Dependencies and relationships
- Evidence collection

### Phase 2: Report Generation
Generate structured report using Phase 1 findings:
- Acknowledge strengths and good practices found in the codebase
- Synthesize findings across all analysis dimensions
- Frame issues as learning opportunities and improvement areas
- Prioritize findings by impact and severity
- Provide evidence-backed recommendations with constructive guidance
- Distinguish between confirmed issues, potential concerns, and enhancement suggestions
- Ensure quality checklist is addressed

## Evidence Requirements
{evidence_template}

## Chunked Analysis Strategy
{chunking_guidance}

## Error Handling Framework
{error_handling}

## Quality Assurance Checklist
{quality_checklist_md}
{hash_constraint}

Generate a prompt that:
1. Clearly defines the {role_name} role using the multi-expert persona above{f' based on: {role_description}' if role_description else ''}
2. Incorporates the SDLC phase context for situational awareness
3. Structures analysis in two phases (comprehensive analysis, then report generation)
4. Tailors analysis instructions to this specific project's architecture and technologies
5. Provides specific guidance on what to look for based on the detected patterns, modules, and components
{f'6. Emphasizes the following focus areas: {", ".join(role_focus_areas)}' if role_focus_areas else '6. Includes relevant examples or patterns from the architecture'}
7. REQUIRES evidence-based reasoning using the evidence template above
8. REQUIRES acknowledgment of strengths and good practices (not just issues)
9. REQUIRES constructive, guidance-oriented language (frame findings as learning opportunities)
10. Includes chunked analysis strategy for context preservation
11. Provides error handling guidance for edge cases
12. Includes quality assurance checklist for self-validation
13. Structures the expected output format with a "Strengths and Good Practices" section
14. Focuses on actionable, specific analysis rather than generic advice
15. Uses supportive, educational tone that helps developers improve
{f'16. ONLY references technologies, frameworks, and patterns that are explicitly present in the architecture model (hash: {architecture_hash})' if architecture_hash else '16. ONLY references technologies, frameworks, and patterns that are explicitly present in the architecture model'}

The prompt should be comprehensive but focused, guiding the agent to perform deep, systematic analysis for this specific codebase. The prompt MUST include all the components above: multi-expert persona, phase context, two-phase execution, evidence requirements, chunked analysis strategy, error handling, and quality assurance."""
    
    return prompt




