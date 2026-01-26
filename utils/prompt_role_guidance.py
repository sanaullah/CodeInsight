"""
Role-specific guidance utility for prompt generation.

Provides detailed focus areas and guidance for different analysis roles,
helping generate more targeted and effective prompts.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_role_guidance(role_name: str) -> str:
    """
    Get comprehensive role-specific guidance for prompt generation.
    
    Returns detailed focus areas, analysis methodology, common patterns,
    quality criteria, and examples tailored to the specific role type.
    
    Args:
        role_name: Name of the analysis role (e.g., "Security Analyst", "Performance Optimizer")
        
    Returns:
        String containing comprehensive role-specific guidance
    """
    role_lower = role_name.lower()
    
    if "security" in role_lower or "auth" in role_lower:
        return """## Security Analysis Focus

### Core Areas
- Authentication and authorization mechanisms
- Data protection and encryption
- Input validation and sanitization
- Common vulnerabilities (OWASP Top 10)
- Secure communication protocols
- Session management
- API security
- Dependency vulnerabilities
- Security best practices for the detected frameworks and technologies

### Analysis Methodology
1. Map authentication flows and identify authentication mechanisms
2. Trace authorization checks and permission validation
3. Identify data protection points and encryption usage
4. Scan for injection vulnerabilities (SQL, XSS, command injection)
5. Review security configurations and security-related settings
6. Assess API security and endpoint protection
7. Evaluate dependency vulnerabilities and outdated packages
8. Review session management and token handling

### Common Patterns to Look For
- JWT implementations and token validation
- OAuth flows and third-party authentication
- Password hashing and storage
- RBAC (Role-Based Access Control) systems
- CSRF protection mechanisms
- SQL injection vulnerabilities
- XSS vulnerabilities
- Insecure direct object references
- Security misconfigurations

### Quality Criteria
- Every vulnerability has OWASP mapping (e.g., A03:2021-Injection)
- Severity justified with impact analysis
- Remediation steps are specific and actionable
- Compliance implications noted (GDPR, HIPAA, etc.)
- Attack vectors identified
- Security controls assessed

### Example Finding
**[CONFIDENCE: HIGH] [SEVERITY: CRITICAL] SQL Injection via Unsanitized Input**
- Location: `auth/login.py:45-67`
- Evidence: 
  ```python
  username = request.form.get('username')
  query = f"SELECT * FROM users WHERE username='{username}'"
  result = db.execute(query)
  ```
- Reasoning: 
  1. User input `username` comes directly from HTTP request (line 45)
  2. Input is concatenated into SQL query without sanitization (line 46)
  3. No parameterized query or ORM used
  4. Database connection has elevated privileges (config.py:12)
  5. This allows SQL injection attacks
- OWASP: A03:2021-Injection
- Impact: Critical security vulnerability - attackers can execute arbitrary SQL, leading to data breach, privilege escalation, or data manipulation
- Recommendation: 
  1. Use parameterized queries: `db.execute("SELECT * FROM users WHERE username=?", (username,))`
  2. Implement input validation (whitelist allowed characters)
  3. Apply principle of least privilege to DB user
  4. Add SQL injection tests"""
    
    elif "performance" in role_lower or "optimizer" in role_lower:
        return """## Performance Analysis Focus

### Core Areas
- Algorithm efficiency and complexity
- Database query optimization
- Caching strategies
- Async/parallel processing opportunities
- Resource usage and bottlenecks
- Scalability concerns
- Response time optimization
- Memory usage patterns
- Performance best practices for the detected technologies

### Analysis Methodology
1. Identify performance bottlenecks and slow operations
2. Assess algorithm efficiency and time complexity
3. Evaluate database query performance and N+1 problems
4. Analyze caching opportunities and cache hit rates
5. Review async/parallel processing usage
6. Assess resource usage patterns (CPU, memory, I/O)
7. Evaluate scalability concerns and bottlenecks
8. Review response time and latency patterns

### Common Patterns to Look For
- N+1 query problems
- Missing database indexes
- Inefficient loops and nested iterations
- Memory leaks and excessive memory usage
- Synchronous blocking operations
- Missing caching opportunities
- Inefficient data structures
- Unoptimized algorithms

### Quality Criteria
- Performance bottlenecks quantified (e.g., "adds 200ms latency")
- Optimization opportunities prioritized by impact
- Scalability concerns identified
- Resource usage patterns analyzed
- Caching opportunities identified
- Database query efficiency assessed

### Example Finding
**[CONFIDENCE: HIGH] N+1 Query Problem in User List Endpoint**
- Location: `api/users.py:123-145`
- Evidence:
  ```python
  users = User.query.all()
  for user in users:
      posts = Post.query.filter_by(user_id=user.id).all()
  ```
- Reasoning:
  1. User list query fetches all users (line 123)
  2. For each user, separate query fetches posts (line 125)
  3. Results in N+1 queries (1 for users + N for posts)
  4. With 100 users, this executes 101 database queries
- Impact: High performance impact - adds ~500ms latency for 100 users, scales linearly
- Recommendation:
  1. Use eager loading: `users = User.query.options(joinedload(User.posts)).all()`
  2. Or use single query with JOIN
  3. Consider pagination for large datasets"""
    
    elif "architecture" in role_lower or "design" in role_lower:
        return """## Architecture Analysis Focus

### Core Areas
- Design patterns and their appropriate use
- Architectural decisions and trade-offs
- Component structure and organization
- Dependency management
- Separation of concerns
- Modularity and reusability
- Scalability and maintainability
- Architectural anti-patterns
- System design improvements

### Analysis Methodology
1. Identify design patterns and assess their appropriateness
2. Evaluate architectural decisions and trade-offs
3. Review component structure and organization
4. Assess dependency management and coupling
5. Evaluate separation of concerns
6. Assess modularity and reusability
7. Identify architectural anti-patterns
8. Recommend architectural improvements

### Common Patterns to Look For
- MVC, MVP, MVVM patterns
- Repository pattern
- Factory pattern
- Singleton pattern
- Observer pattern
- Dependency injection
- Layered architecture
- Microservices vs monolith trade-offs

### Quality Criteria
- Design patterns identified and evaluated
- Architectural decisions assessed
- Component relationships documented
- Scalability implications analyzed
- Maintainability concerns identified
- Architectural anti-patterns flagged

### Example Finding
**[CONFIDENCE: HIGH] Tight Coupling Between Business Logic and Data Access**
- Location: `services/order_service.py:45-89`
- Evidence:
  ```python
  class OrderService:
      def process_order(self, order_id):
          db = DatabaseConnection()
          order = db.execute(f"SELECT * FROM orders WHERE id={order_id}")
          # Business logic mixed with database access
  ```
- Reasoning:
  1. Business logic directly accesses database (line 47)
  2. No abstraction layer between service and data access
  3. Makes testing difficult and creates tight coupling
  4. Violates separation of concerns principle
- Impact: Reduces maintainability, makes testing difficult, creates tight coupling
- Recommendation:
  1. Introduce repository pattern for data access
  2. Inject repository dependency into service
  3. Separate business logic from data access layer"""
    
    elif "quality" in role_lower or "reviewer" in role_lower:
        return """## Code Quality Analysis Focus

### Core Areas
- Code organization and structure
- Maintainability and readability
- Code smells and refactoring opportunities
- Documentation quality
- Naming conventions
- Complexity metrics
- Code duplication
- Testing coverage
- Best practices adherence

### Analysis Methodology
1. Assess code organization and structure
2. Evaluate maintainability and readability
3. Identify code smells and anti-patterns
4. Review documentation quality and completeness
5. Assess naming conventions and consistency
6. Evaluate complexity metrics (cyclomatic complexity)
7. Identify code duplication opportunities
8. Assess testing coverage and quality

### Common Patterns to Look For
- Long methods and functions
- God classes (too many responsibilities)
- Code duplication
- Magic numbers and strings
- Poor naming conventions
- Missing documentation
- High cyclomatic complexity
- Inconsistent code style

### Quality Criteria
- Code smells identified and prioritized
- Refactoring opportunities documented
- Complexity metrics considered
- Code duplication assessed
- Test coverage evaluated
- Documentation quality reviewed

### Example Finding
**[CONFIDENCE: HIGH] Long Method with High Cyclomatic Complexity**
- Location: `utils/order_processor.py:123-245`
- Evidence:
  ```python
  def process_order(self, order):
      # 120+ lines of nested if/else statements
      # Multiple responsibilities mixed together
  ```
- Reasoning:
  1. Method is 120+ lines long (exceeds recommended 20-30 lines)
  2. Cyclomatic complexity of 15 (exceeds recommended 10)
  3. Multiple responsibilities: validation, processing, notification
  4. Difficult to test and maintain
- Impact: Reduces maintainability, increases bug risk, makes testing difficult
- Recommendation:
  1. Extract validation logic to separate method
  2. Extract processing logic to separate method
  3. Extract notification logic to separate method
  4. Reduce cyclomatic complexity to < 10"""
    
    elif "best practice" in role_lower or "advisor" in role_lower:
        return """## Best Practices Analysis Focus

### Core Areas
- Industry standards and conventions
- Framework-specific best practices
- Language idioms and patterns
- Error handling strategies
- Logging and monitoring
- Testing practices
- Documentation standards
- Version control practices
- Deployment and DevOps practices

### Analysis Methodology
1. Assess adherence to industry standards
2. Evaluate framework-specific best practices
3. Review language idioms and patterns
4. Assess error handling strategies
5. Evaluate logging and monitoring practices
6. Review testing practices and coverage
7. Assess documentation standards
8. Evaluate deployment and DevOps practices

### Common Patterns to Look For
- Error handling patterns
- Logging best practices
- Testing strategies
- Documentation standards
- Code style consistency
- Framework conventions
- Language idioms

### Quality Criteria
- Standards compliance assessed
- Framework conventions evaluated
- Best practice gaps identified
- Improvement recommendations provided
- Industry alignment verified

### Example Finding
**[CONFIDENCE: MEDIUM] Inconsistent Error Handling Across Codebase**
- Location: Multiple files (auth.py:45, payment.py:123, order.py:89)
- Evidence:
  ```python
  # auth.py - uses exceptions
  if not user:
      raise AuthenticationError("User not found")
  
  # payment.py - returns None
  if not payment:
      return None
  
  # order.py - uses error codes
  if not order:
      return {"error": 404, "message": "Not found"}
  ```
- Reasoning:
  1. Three different error handling patterns used
  2. No consistent error handling strategy
  3. Makes error handling unpredictable
  4. Difficult to maintain and test
- Impact: Reduces code consistency, makes error handling unpredictable, increases maintenance burden
- Recommendation:
  1. Standardize on exception-based error handling
  2. Create custom exception hierarchy
  3. Use consistent error response format
  4. Document error handling strategy"""
    
    else:
        return f"""## {role_name} Analysis Focus

### Core Areas
- General code analysis
- Best practices
- Quality improvements
- Architecture considerations

### Analysis Methodology
1. Perform comprehensive code analysis
2. Identify improvement opportunities
3. Assess code quality
4. Recommend best practices

### Quality Criteria
- Comprehensive analysis completed
- Findings documented
- Recommendations provided
- Quality assessed"""


def get_role_analysis_type(role_name: str) -> str:
    """
    Map role name to analysis type for architecture prompt generation.
    
    Args:
        role_name: Agent role name
        
    Returns:
        Analysis type string (security, performance, code_quality, architecture)
    """
    role_lower = role_name.lower()
    
    # Security-related roles
    if any(keyword in role_lower for keyword in ['security', 'auth', 'jwt', 'oauth', 'vulnerability']):
        return "security"
    
    # Performance-related roles
    if any(keyword in role_lower for keyword in ['performance', 'optimizer', 'optimization', 'cache', 'async', 'queue']):
        return "performance"
    
    # Architecture-related roles
    if any(keyword in role_lower for keyword in ['architecture', 'design', 'api', 'rest', 'endpoint']):
        return "architecture"
    
    # Code quality (default)
    return "code_quality"

