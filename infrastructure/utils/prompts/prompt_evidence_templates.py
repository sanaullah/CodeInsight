"""
Evidence-based finding templates with confidence levels.
Provides structured formats for high-quality findings.
"""

import logging

logger = logging.getLogger(__name__)


def get_evidence_template(role_name: str) -> str:
    """
    Get evidence-based finding template for a role.
    
    Returns markdown template with:
    - Confidence level format
    - Location specification
    - Evidence presentation
    - Reasoning structure
    - Impact assessment
    - Recommendation format
    
    Args:
        role_name: Name of the analysis role
        
    Returns:
        Formatted markdown string with evidence template
    """
    role_lower = role_name.lower()
    
    # Base template
    base_template = """## Evidence-Based Finding Template

Every finding MUST follow this structured format:

**[CONFIDENCE: HIGH/MEDIUM/LOW] Finding Title**
- **Location**: `file.py:123-145` (specific file and line numbers)
- **Evidence**: 
  ```python
  # Actual code snippet showing the issue
  ```
- **Reasoning**: Step-by-step logical chain from observation to conclusion
  1. What you observed in the code
  2. Why this is significant
  3. What the implications are
  4. How you reached your conclusion
- **Impact**: Why this matters (security, performance, maintainability, etc.)
- **Recommendation**: Specific, actionable steps to address the issue (frame constructively as guidance)
- **Uncertainty**: What you're unsure about (if any)

### Constructive Language Guidelines

When presenting findings, use constructive, guidance-oriented language:

**Instead of critical language:**
- ❌ "Critical security vulnerability found"
- ❌ "Poor performance"
- ❌ "Bad architecture"
- ❌ "This is wrong"

**Use constructive language:**
- ✅ "Security enhancement opportunity: Consider implementing..."
- ✅ "Performance optimization opportunity: This could be improved by..."
- ✅ "Architectural enhancement: Consider refactoring to..."
- ✅ "Consider reviewing this approach: An alternative might be..."

**Frame findings as:**
- **Learning opportunities** rather than failures
- **Improvement areas** rather than problems
- **Enhancement suggestions** when confidence is low
- **Guidance** rather than criticism

### Confidence Level Guidelines

- **[HIGH]**: Clear evidence, well-understood issue, confident in assessment
- **[MEDIUM]**: Some evidence, partially understood, some uncertainty
- **[LOW]**: Limited evidence, unclear understanding, significant uncertainty

### Example Finding

**[CONFIDENCE: HIGH] Security Enhancement Opportunity: SQL Injection Prevention in User Authentication**
- **Location**: `auth/login.py:45-67`
- **Evidence**:
  ```python
  username = request.form.get('username')
  query = f"SELECT * FROM users WHERE username='{username}'"
  result = db.execute(query)
  ```
- **Reasoning**:
  1. User input `username` comes directly from HTTP request (line 45)
  2. Input is concatenated into SQL query without sanitization (line 46)
  3. No parameterized query or ORM used
  4. Database connection has elevated privileges (config.py:12)
  5. This allows SQL injection attacks
- **Impact**: This pattern allows SQL injection attacks, which could lead to data breach, privilege escalation, or data manipulation. Addressing this will significantly improve security posture.
- **Recommendation** (constructive guidance):
  1. Consider using parameterized queries: `db.execute("SELECT * FROM users WHERE username=?", (username,))` to prevent SQL injection
  2. Implement input validation (whitelist allowed characters) to add defense in depth
  3. Apply principle of least privilege to DB user to limit potential impact
  4. Add SQL injection tests to prevent regressions
- **Uncertainty**: None - this pattern is a well-documented security concern that should be addressed"""
    
    # Add role-specific enhancements
    if "security" in role_lower or "auth" in role_lower:
        return base_template + """

### Security-Specific Enhancements

- **OWASP Category**: Map to OWASP Top 10 (e.g., A03:2021-Injection)
- **CWE ID**: Common Weakness Enumeration ID if applicable
- **Severity**: [CRITICAL/HIGH/MEDIUM/LOW] based on impact
- **Attack Vector**: How an attacker could exploit this
- **Remediation Priority**: [IMMEDIATE/HIGH/MEDIUM/LOW] based on severity"""
    
    elif "performance" in role_lower or "optimizer" in role_lower:
        return base_template + """

### Performance-Specific Enhancements

- **Performance Impact**: Quantify impact (e.g., "adds 200ms latency", "uses 50MB memory")
- **Bottleneck Type**: [CPU/Memory/I/O/Network/Database]
- **Scalability Impact**: How this affects system scalability
- **Optimization Priority**: [IMMEDIATE/HIGH/MEDIUM/LOW] based on impact
- **Expected Improvement**: Quantify expected improvement from optimization"""
    
    elif "architecture" in role_lower or "design" in role_lower:
        return base_template + """

### Architecture-Specific Enhancements

- **Pattern Type**: Design pattern or anti-pattern identified
- **Architectural Impact**: How this affects system architecture
- **Scalability Impact**: How this affects system scalability
- **Maintainability Impact**: How this affects code maintainability
- **Refactoring Priority**: [IMMEDIATE/HIGH/MEDIUM/LOW] based on impact"""
    
    elif "quality" in role_lower or "reviewer" in role_lower:
        return base_template + """

### Quality-Specific Enhancements

- **Code Smell Type**: Type of code smell (e.g., "Long Method", "God Class")
- **Complexity Impact**: How this affects code complexity
- **Maintainability Impact**: How this affects code maintainability
- **Refactoring Priority**: [IMMEDIATE/HIGH/MEDIUM/LOW] based on impact
- **Technical Debt**: Estimated effort to fix"""
    
    else:
        return base_template


def get_confidence_level_guidance() -> str:
    """
    Get guidance for assigning confidence levels.
    
    Returns:
        Formatted markdown string with confidence level guidance
    """
    return """## Confidence Level Assignment Guidelines

### [HIGH] Confidence

Use [HIGH] confidence when:
- Clear, unambiguous evidence in the code
- Well-understood issue with established patterns
- No significant uncertainty about the finding
- Can be verified through static analysis
- Impact is clear and well-documented

**Example**: SQL injection vulnerability with unsanitized user input directly in SQL query.

### [MEDIUM] Confidence

Use [MEDIUM] confidence when:
- Some evidence but not completely clear
- Partially understood issue
- Some uncertainty about implications
- May require runtime analysis to confirm
- Impact assessment has some uncertainty

**Example**: Potential performance issue that depends on runtime data volume.

### [LOW] Confidence

Use [LOW] confidence when:
- Limited evidence available
- Unclear understanding of the issue
- Significant uncertainty about implications
- Requires additional context or runtime analysis
- Based on inference or assumptions

**Example**: Suspected security issue in obfuscated code where logic is unclear.

### When to Use Each Level

- **Always mark confidence**: Every finding must have a confidence level
- **Be honest**: Don't inflate confidence - it's better to be conservative
- **Explain rationale**: Briefly explain why you chose this confidence level
- **Update as needed**: If you gain more information, update confidence level"""


def get_uncertainty_handling_guidance() -> str:
    """
    Get guidance for acknowledging uncertainty.
    
    Returns:
        Formatted markdown string with uncertainty handling guidance
    """
    return """## Uncertainty Handling Guidelines

### When to Acknowledge Uncertainty

Acknowledge uncertainty when:
- Code is incomplete or partially visible
- Logic is complex or unclear
- Dependencies on external systems or runtime behavior
- Based on assumptions rather than direct evidence
- Requires additional context to fully understand
- Analysis is limited by chunking or missing files

### How to Acknowledge Uncertainty

1. **Explicit Statements**: Use clear language like "I'm uncertain about..." or "This is unclear because..."
2. **Confidence Levels**: Use [LOW] or [MEDIUM] confidence to indicate uncertainty
3. **Assumption Documentation**: Document what you're assuming and why
4. **Gap Identification**: Clearly identify what information is missing
5. **Recommendations**: Suggest how to reduce uncertainty (e.g., "Runtime testing needed")

### Example Uncertainty Statements

- "Uncertain about the security implications without seeing the authentication middleware"
- "[LOW] confidence - logic is complex and requires runtime analysis to confirm"
- "Assuming database connection uses connection pooling based on framework patterns"
- "Unable to verify API endpoint authentication without runtime testing"
- "Analysis limited by incomplete chunk - function implementation is cut off"

### Honesty Guidelines

- **Distinguish confirmed issues from potential concerns**: Clearly separate what you know from what you suspect
- **Separate suggestions from findings**: Distinguish between confirmed issues and improvement suggestions
- **Acknowledge limitations**: Be upfront about what you couldn't analyze
- **Avoid speculation**: Don't make claims you can't support with evidence
- **Be transparent**: It's better to acknowledge uncertainty than to make unsupported claims

### Constructive Framing for Uncertainty

When uncertain, frame findings supportively:
- Instead of: "This might be a problem" → "This is worth reviewing to ensure..."
- Instead of: "Potential issue" → "Enhancement opportunity: Consider..."
- Instead of: "Unclear if this is correct" → "Consider verifying this approach: An alternative might be..."
- Use phrases like: "Worth reviewing", "Consider evaluating", "Enhancement opportunity", "Learning opportunity"

**Remember**: Frame uncertainty as an opportunity for review and improvement, not as a failure or problem."""

