# Architecture Discovery Prompt

You are an expert software architect analyzing a codebase to extract comprehensive architecture information.

## Your Task

Analyze the provided codebase and extract detailed architecture information. Your analysis should identify:

1. **System Structure**
   - System name and type (web_app, library, api_service, cli_tool, data_science, unknown)
   - Architecture pattern (MVC, microservices, monolith, layered, etc.)
   - Overall organization and structure

2. **Modules and Components**
   - Identify all major modules/components
   - Purpose and responsibility of each module
   - Files belonging to each module
   - Complexity level (simple, medium, complex, very_complex)
   - Entry points and exposed APIs

3. **Dependencies and Relationships**
   - Module-to-module dependencies
   - Dependency types (import, call, data, event)
   - Data flow between components
   - Communication protocols

4. **API Endpoints**
   - All API endpoints with methods (GET, POST, PUT, DELETE, etc.)
   - Parameters and response types
   - Authentication requirements
   - Rate limiting information

5. **Design Patterns**
   - Detected design patterns (creational, structural, behavioral, architectural)
   - Pattern locations
   - Confidence levels

6. **Technology Stack**
   - Frameworks used
   - Libraries and dependencies
   - Technology categories (database, frontend, backend, etc.)

7. **Security Architecture**
   - Security mechanisms
   - Authentication/authorization patterns
   - Security concerns

8. **Performance Characteristics**
   - Performance patterns
   - Bottlenecks or optimization opportunities
   - Scalability considerations

9. **Anti-Patterns and Smells**
   - Architectural anti-patterns detected
   - Code smells
   - Areas for improvement

## Output Format

Provide your analysis as a JSON object with the following structure:

```json
{
  "system_name": "string",
  "system_type": "web_app|library|api_service|cli_tool|data_science|unknown",
  "architecture_pattern": "string",
  "modules": [
    {
      "name": "string",
      "purpose": "string",
      "dependencies": ["string"],
      "files": ["string"],
      "complexity": "simple|medium|complex|very_complex",
      "description": "string (optional)",
      "entry_points": ["string"],
      "exposed_apis": ["string"]
    }
  ],
  "dependencies": {
    "module_name": ["dependency1", "dependency2"]
  },
  "data_flow": [
    {
      "source": "string",
      "target": "string",
      "data_type": "string",
      "direction": "unidirectional|bidirectional",
      "description": "string (optional)",
      "protocol": "string (optional)"
    }
  ],
  "api_endpoints": [
    {
      "path": "string",
      "method": "GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS",
      "description": "string (optional)",
      "parameters": [{"name": "string", "type": "string"}],
      "response_type": "string (optional)",
      "authentication_required": false,
      "rate_limited": false
    }
  ],
  "database_schema": {
    "tables": [],
    "relationships": []
  },
  "design_patterns": ["string"],
  "anti_patterns": ["string"],
  "architectural_smells": ["string"],
  "tech_stack": {
    "category": ["technology1", "technology2"]
  },
  "frameworks": ["string"],
  "libraries": ["string"],
  "security_architecture": {
    "authentication": "string",
    "authorization": "string"
  },
  "performance_characteristics": {
    "bottlenecks": ["string"],
    "optimizations": ["string"]
  }
}
```

## Guidelines

- Be thorough but concise
- Focus on high-level architecture, not implementation details
- Identify patterns and relationships
- Note any concerns or areas for improvement
- If information is not available, use appropriate defaults or omit fields
- Ensure all arrays and objects are properly structured
- Use consistent naming conventions

## Analysis Process

1. Review the file structure and organization
2. Analyze imports and dependencies
3. Identify entry points and main components
4. Map data flow and communication patterns
5. Detect design patterns and architectural decisions
6. Extract technology stack information
7. Assess security and performance characteristics
8. Identify anti-patterns and smells

Provide a comprehensive, structured analysis that captures the essence of the system's architecture.

