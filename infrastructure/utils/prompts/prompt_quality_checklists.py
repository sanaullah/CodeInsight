"""
Quality assurance checklists for agent self-validation.
Provides role-specific and universal quality criteria.
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def get_universal_checklist() -> List[str]:
    """
    Get checklist items applicable to all roles.
    
    Returns:
        List of universal quality checklist items
    """
    return [
        "All code elements analyzed",
        "Cross-cutting concerns addressed",
        "Dependencies documented",
        "Edge cases considered",
        "Strengths and good practices acknowledged",
        "Every finding has code reference (file:line)",
        "Confidence levels assigned [HIGH/MEDIUM/LOW]",
        "Reasoning chain documented",
        "Uncertainties acknowledged",
        "Constructive, guidance-oriented language used",
        "Recommendations are actionable",
        "Findings are prioritized",
        "Report follows structure",
        "Technical terms explained where needed"
    ]


def get_quality_checklist(role_name: str) -> Dict[str, List[str]]:
    """
    Get quality checklist for a role.
    
    Returns a dictionary with categories:
    - analysis_completeness: Items for ensuring thorough analysis
    - evidence_quality: Items for ensuring high-quality evidence
    - output_quality: Items for ensuring high-quality output
    - role_specific: Role-specific quality criteria
    
    Args:
        role_name: Name of the analysis role
        
    Returns:
        Dictionary with quality checklist categories and items
    """
    role_lower = role_name.lower()
    
    # Base checklist with universal items
    checklist = {
        "analysis_completeness": [
            "All code elements analyzed",
            "Cross-cutting concerns addressed",
            "Dependencies documented",
            "Edge cases considered",
            "Architecture patterns examined",
            "Integration points reviewed"
        ],
        "evidence_quality": [
            "Every finding has code reference (file:line)",
            "Confidence levels assigned [HIGH/MEDIUM/LOW]",
            "Reasoning chain documented",
            "Uncertainties acknowledged",
            "Code snippets provided for key findings",
            "Impact assessment included"
        ],
        "output_quality": [
            "Strengths and good practices section included",
            "Issues framed as learning opportunities",
            "Constructive, supportive language used throughout",
            "Distinction made between confirmed issues, potential concerns, and suggestions",
            "Recommendations are actionable",
            "Findings are prioritized",
            "Report follows structure",
            "Technical terms explained where needed",
            "Examples provided where helpful",
            "Clear next steps identified"
        ],
        "role_specific": []
    }
    
    # Add role-specific criteria
    if "security" in role_lower or "auth" in role_lower:
        checklist["role_specific"] = [
            "Every vulnerability has OWASP mapping",
            "Severity justified with impact analysis",
            "Remediation steps are specific",
            "Compliance implications noted",
            "Attack vectors identified",
            "Security controls assessed"
        ]
    
    elif "performance" in role_lower or "optimizer" in role_lower:
        checklist["role_specific"] = [
            "Performance bottlenecks quantified",
            "Optimization opportunities prioritized",
            "Scalability concerns identified",
            "Resource usage patterns analyzed",
            "Caching opportunities identified",
            "Database query efficiency assessed"
        ]
    
    elif "architecture" in role_lower or "design" in role_lower:
        checklist["role_specific"] = [
            "Design patterns identified and evaluated",
            "Architectural decisions assessed",
            "Component relationships documented",
            "Scalability implications analyzed",
            "Maintainability concerns identified",
            "Architectural anti-patterns flagged"
        ]
    
    elif "quality" in role_lower or "reviewer" in role_lower:
        checklist["role_specific"] = [
            "Code smells identified",
            "Refactoring opportunities prioritized",
            "Complexity metrics considered",
            "Code duplication assessed",
            "Test coverage evaluated",
            "Documentation quality reviewed"
        ]
    
    elif "best practice" in role_lower or "advisor" in role_lower:
        checklist["role_specific"] = [
            "Industry standards referenced",
            "Framework conventions followed",
            "Language idioms identified",
            "Error handling patterns assessed",
            "Logging practices reviewed",
            "Testing practices evaluated"
        ]
    
    elif "documentation" in role_lower or "doc" in role_lower:
        checklist["role_specific"] = [
            "Documentation completeness assessed",
            "API documentation quality reviewed",
            "Code comments evaluated",
            "README quality assessed",
            "Documentation structure reviewed",
            "Developer experience considered"
        ]
    
    elif "testing" in role_lower or "test" in role_lower:
        checklist["role_specific"] = [
            "Test coverage analyzed",
            "Test quality assessed",
            "Test strategies evaluated",
            "Test automation reviewed",
            "Test maintainability considered",
            "Test effectiveness measured"
        ]
    
    elif "dependency" in role_lower:
        checklist["role_specific"] = [
            "Dependency vulnerabilities identified",
            "Version conflicts assessed",
            "License compliance checked",
            "Dependency hygiene evaluated",
            "Update strategies recommended",
            "Security risks documented"
        ]
    
    else:
        checklist["role_specific"] = [
            "Role-specific concerns addressed",
            "Best practices identified",
            "Improvement opportunities documented",
            "Quality standards assessed"
        ]
    
    return checklist


def format_quality_checklist(checklist: Dict[str, List[str]]) -> str:
    """
    Format quality checklist as markdown for inclusion in prompts.
    
    Args:
        checklist: Dictionary with quality checklist categories and items
        
    Returns:
        Formatted markdown string
    """
    lines = ["## Quality Assurance Checklist", ""]
    
    for category, items in checklist.items():
        if not items:
            continue
        
        # Format category name
        category_name = category.replace("_", " ").title()
        lines.append(f"### {category_name}")
        lines.append("")
        
        for item in items:
            lines.append(f"- [ ] {item}")
        
        lines.append("")
    
    return "\n".join(lines)

