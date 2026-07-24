"""
SDLC phase context mapping for situational awareness.
Maps analysis roles to development lifecycle phases.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def get_phase_context(role_name: str) -> str:
    """
    Get SDLC phase context for a role.
    
    Returns formatted markdown describing:
    - Phase name
    - Phase objectives
    - Key focus areas
    - Success criteria
    
    Args:
        role_name: Name of the analysis role
        
    Returns:
        Formatted markdown string with phase context
    """
    role_lower = role_name.lower()
    
    if "security" in role_lower or "auth" in role_lower:
        return """## Phase Context

You are operating in the **Security Assessment** phase of the SDLC, where the focus is on identifying security vulnerabilities, security anti-patterns, and compliance gaps in the codebase.

**Phase Objectives**:
- Identify security vulnerabilities and weaknesses
- Assess compliance with security standards (OWASP, CWE, etc.)
- Evaluate security controls and mechanisms
- Document security risks and remediation strategies
- Ensure security best practices are followed

**Key Focus Areas**:
- Authentication and authorization mechanisms
- Data protection and encryption
- Input validation and sanitization
- Common vulnerabilities (OWASP Top 10)
- API security
- Dependency vulnerabilities
- Security configurations

**Success Criteria**:
- All security-sensitive code sections analyzed
- Vulnerabilities identified with severity assessment
- Remediation steps provided for each finding
- Compliance implications documented
- Security best practices recommendations included"""
    
    elif "performance" in role_lower or "optimizer" in role_lower:
        return """## Phase Context

You are operating in the **Performance Optimization** phase of the SDLC, where the focus is on identifying performance bottlenecks, optimization opportunities, and scalability concerns.

**Phase Objectives**:
- Identify performance bottlenecks and inefficiencies
- Assess algorithm efficiency and complexity
- Evaluate database query performance
- Identify caching opportunities
- Recommend scalability improvements
- Optimize resource usage patterns

**Key Focus Areas**:
- Algorithm efficiency and complexity
- Database query optimization
- Caching strategies
- Async/parallel processing opportunities
- Resource usage and bottlenecks
- Scalability concerns
- Response time optimization

**Success Criteria**:
- Performance bottlenecks identified and quantified
- Optimization opportunities prioritized
- Scalability concerns documented
- Resource usage patterns analyzed
- Specific optimization recommendations provided"""
    
    elif "architecture" in role_lower or "design" in role_lower:
        return """## Phase Context

You are operating in the **Architecture Review** phase of the SDLC, where the focus is on evaluating architectural decisions, design patterns, and system structure.

**Phase Objectives**:
- Evaluate architectural decisions and trade-offs
- Assess design pattern usage and appropriateness
- Review component structure and organization
- Identify architectural anti-patterns
- Recommend architectural improvements
- Ensure scalability and maintainability

**Key Focus Areas**:
- Design patterns and their appropriate use
- Architectural decisions and trade-offs
- Component structure and organization
- Dependency management
- Separation of concerns
- Modularity and reusability
- Scalability and maintainability

**Success Criteria**:
- Architectural decisions evaluated
- Design patterns identified and assessed
- Component relationships documented
- Architectural anti-patterns flagged
- Improvement recommendations provided
- Scalability implications analyzed"""
    
    elif "quality" in role_lower or "reviewer" in role_lower:
        return """## Phase Context

You are operating in the **Code Quality Assessment** phase of the SDLC, where the focus is on evaluating code quality, maintainability, and adherence to best practices.

**Phase Objectives**:
- Assess code organization and structure
- Identify code smells and refactoring opportunities
- Evaluate maintainability and readability
- Review documentation quality
- Assess testing coverage
- Recommend quality improvements

**Key Focus Areas**:
- Code organization and structure
- Maintainability and readability
- Code smells and refactoring opportunities
- Documentation quality
- Naming conventions
- Complexity metrics
- Code duplication
- Testing coverage

**Success Criteria**:
- Code smells identified and prioritized
- Refactoring opportunities documented
- Maintainability concerns assessed
- Documentation quality reviewed
- Testing coverage evaluated
- Quality improvement recommendations provided"""
    
    elif "best practice" in role_lower or "advisor" in role_lower:
        return """## Phase Context

You are operating in the **Best Practices Review** phase of the SDLC, where the focus is on ensuring adherence to industry standards, framework conventions, and best practices.

**Phase Objectives**:
- Assess adherence to industry standards
- Evaluate framework-specific best practices
- Review language idioms and patterns
- Assess error handling strategies
- Evaluate logging and monitoring practices
- Recommend best practice improvements

**Key Focus Areas**:
- Industry standards and conventions
- Framework-specific best practices
- Language idioms and patterns
- Error handling strategies
- Logging and monitoring
- Testing practices
- Documentation standards
- Deployment practices

**Success Criteria**:
- Standards compliance assessed
- Framework conventions evaluated
- Best practice gaps identified
- Improvement recommendations provided
- Industry alignment verified"""
    
    elif "documentation" in role_lower or "doc" in role_lower:
        return """## Phase Context

You are operating in the **Documentation** phase of the SDLC, where the focus is on evaluating documentation quality, completeness, and usability.

**Phase Objectives**:
- Assess documentation completeness
- Evaluate documentation quality and clarity
- Review API documentation
- Assess developer experience
- Identify documentation gaps
- Recommend documentation improvements

**Key Focus Areas**:
- Documentation completeness
- Documentation quality and clarity
- API documentation
- Code comments
- README quality
- Developer onboarding materials
- Documentation structure

**Success Criteria**:
- Documentation completeness assessed
- Documentation quality evaluated
- API documentation reviewed
- Developer experience considered
- Documentation gaps identified
- Improvement recommendations provided"""
    
    elif "testing" in role_lower or "test" in role_lower:
        return """## Phase Context

You are operating in the **Testing Assessment** phase of the SDLC, where the focus is on evaluating test quality, coverage, and testing practices.

**Phase Objectives**:
- Assess test coverage and quality
- Evaluate test strategies and patterns
- Review test automation
- Assess test maintainability
- Identify testing gaps
- Recommend testing improvements

**Key Focus Areas**:
- Test coverage
- Test quality
- Test strategies
- Test automation
- Test maintainability
- Testing best practices
- Test effectiveness

**Success Criteria**:
- Test coverage analyzed
- Test quality assessed
- Test strategies evaluated
- Testing gaps identified
- Improvement recommendations provided
- Testing best practices documented"""
    
    elif "dependency" in role_lower:
        return """## Phase Context

You are operating in the **Dependency Analysis** phase of the SDLC, where the focus is on evaluating dependencies, vulnerabilities, and dependency management.

**Phase Objectives**:
- Assess dependency vulnerabilities
- Evaluate dependency versions and conflicts
- Review license compliance
- Assess dependency hygiene
- Identify security risks
- Recommend dependency improvements

**Key Focus Areas**:
- Dependency vulnerabilities
- Version conflicts
- License compliance
- Dependency hygiene
- Security risks
- Update strategies

**Success Criteria**:
- Vulnerabilities identified
- Version conflicts assessed
- License compliance checked
- Security risks documented
- Update strategies recommended
- Dependency hygiene evaluated"""
    
    else:
        return f"""## Phase Context

You are operating in the **Code Analysis** phase of the SDLC, where the focus is on comprehensive code analysis and quality assessment.

**Phase Objectives**:
- Perform comprehensive code analysis
- Identify improvement opportunities
- Assess code quality
- Recommend best practices
- Document findings and recommendations

**Key Focus Areas**:
- Code analysis
- Quality assessment
- Best practices
- Improvement opportunities

**Success Criteria**:
- Comprehensive analysis completed
- Findings documented
- Recommendations provided
- Quality assessed"""


def get_phase_objectives(role_name: str) -> List[str]:
    """
    Get specific objectives for the phase.
    
    Args:
        role_name: Name of the analysis role
        
    Returns:
        List of phase objectives
    """
    role_lower = role_name.lower()
    
    if "security" in role_lower or "auth" in role_lower:
        return [
            "Identify security vulnerabilities and weaknesses",
            "Assess compliance with security standards",
            "Evaluate security controls and mechanisms",
            "Document security risks and remediation strategies"
        ]
    
    elif "performance" in role_lower or "optimizer" in role_lower:
        return [
            "Identify performance bottlenecks and inefficiencies",
            "Assess algorithm efficiency and complexity",
            "Evaluate database query performance",
            "Identify caching opportunities"
        ]
    
    elif "architecture" in role_lower or "design" in role_lower:
        return [
            "Evaluate architectural decisions and trade-offs",
            "Assess design pattern usage and appropriateness",
            "Review component structure and organization",
            "Identify architectural anti-patterns"
        ]
    
    elif "quality" in role_lower or "reviewer" in role_lower:
        return [
            "Assess code organization and structure",
            "Identify code smells and refactoring opportunities",
            "Evaluate maintainability and readability",
            "Review documentation quality"
        ]
    
    elif "best practice" in role_lower or "advisor" in role_lower:
        return [
            "Assess adherence to industry standards",
            "Evaluate framework-specific best practices",
            "Review language idioms and patterns",
            "Assess error handling strategies"
        ]
    
    else:
        return [
            "Perform comprehensive code analysis",
            "Identify improvement opportunities",
            "Assess code quality",
            "Recommend best practices"
        ]

