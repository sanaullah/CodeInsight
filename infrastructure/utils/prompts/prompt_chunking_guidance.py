"""
Comprehensive chunked analysis guidance.
Provides strategies for maintaining context across code chunks.
"""

import logging

logger = logging.getLogger(__name__)


def get_chunking_strategy() -> str:
    """
    Get comprehensive chunked analysis strategy.
    
    Returns markdown with:
    - Context preservation techniques
    - Progressive understanding approach
    - Gap management strategies
    - Chunk-specific templates
    
    Returns:
        Formatted markdown string with chunking strategy
    """
    return """## Chunked Analysis Strategy

You will receive code in chunks to manage token limits and processing efficiency. This requires special attention to context preservation and progressive understanding.

### Understanding the Chunked Analysis Process

The system delivers file contents in chunks to manage token limits and processing efficiency. This requires special attention to:

1. **Context Preservation**: Maintain understanding across multiple chunks of the same file
2. **Progressive Building**: Incrementally build your understanding as more chunks arrive
3. **Cross-Chunk Relationships**: Identify relationships that span multiple chunks or files
4. **Gap Management**: Recognize and document information gaps due to chunking

### Chunk Analysis Strategy

#### For Single Chunk Analysis:
- Perform the most comprehensive analysis possible with available information
- Clearly state limitations due to incomplete data
- Focus on extracting maximum value from visible code
- Highlight areas where additional information would be beneficial

#### For Multi-Chunk Analysis:

**First Chunk**:
- Establish initial understanding of the file's purpose
- Identify key components, classes, and functions
- Note imports and dependencies
- Map the overall structure
- Identify patterns and architectural elements

**Middle Chunks**:
- Build upon previous understanding
- Identify implementation details
- Connect components and relationships
- Note data flow and control flow
- Identify patterns and anti-patterns

**Final Chunk**:
- Complete understanding of the file
- Identify system connections and patterns
- Synthesize findings across chunks
- Document complete picture
- Note any remaining gaps or uncertainties

### Incremental Analysis Approach

1. **Progressive Documentation**: Begin documenting findings as you analyze each chunk
2. **Cross-Chunk Relationships**: Look for relationships between different chunks of the same file
3. **Gap Identification**: Note areas where information is missing or unclear
4. **Assumption Management**: Make reasonable assumptions but state them explicitly
5. **Context Maintenance**: Keep mental model of the system as you analyze each chunk

### Context Preservation Techniques

- **Mental Model Maintenance**: Maintain a mental model of the system as you analyze each chunk
- **Relationship Tracking**: Track relationships between different chunks of the same file
- **Pattern Recognition**: Identify patterns that emerge across chunks
- **Component Mapping**: Map components and their relationships across chunks
- **Dependency Tracking**: Track dependencies and imports across chunks

### Gap Management Strategies

- **Missing Information**: Clearly identify what information is missing from your analysis
- **Explicit Assumptions**: Make reasonable assumptions where necessary, but state them explicitly
- **Critical Gaps**: Highlight critical areas where additional information would be valuable
- **Incomplete Understanding**: Acknowledge when your understanding is incomplete due to chunking
- **Synthesis Notes**: Document items requiring synthesis with other chunks

### Chunk Prioritization

- Focus on the most significant parts of each chunk
- Identify key patterns, relationships, and implementations
- Note any unusual or critical functionality that appears in any chunk
- Prioritize security-sensitive, performance-critical, or architecturally important code
- Balance thoroughness with efficiency"""


def get_chunk_analysis_template() -> str:
    """
    Get template for analyzing individual chunks.
    
    Returns:
        Formatted markdown template for chunk analysis
    """
    return """## Chunk Analysis [X/Y]

### Files Analyzed
- List files included in this chunk with their purposes

### Functional Capabilities Identified
- Document functions, classes, and features visible in this chunk
- Note business logic and workflows

### Data Models Found
- Identify data structures, schemas, and entity definitions
- Note relationships and constraints

### Security Measures
- Document authentication, authorization, and security implementations
- Note potential vulnerabilities or security considerations

### Non-Functional Aspects
- Identify performance, scalability, and reliability considerations
- Note error handling and logging implementations

### Dependencies
- Document external libraries, services, and system dependencies
- Note integration points and APIs

### Technical Debt Identified
- Flag code quality issues, anti-patterns, and improvement opportunities
- Note outdated dependencies or deprecated approaches

### Synthesis Notes
- Note incomplete implementations, cross-file references
- Identify items requiring synthesis with other chunks
- Document assumptions made due to incomplete information"""


def get_synthesis_guidance() -> str:
    """
    Get guidance for synthesizing across chunks.
    
    Returns:
        Formatted markdown string with synthesis guidance
    """
    return """## Synthesis Across Chunks

When you have analyzed multiple chunks, synthesize your findings:

### Cross-Chunk Synthesis

1. **Component Relationships**: Identify how components from different chunks relate
2. **Data Flow**: Map data flow across chunks and files
3. **Control Flow**: Understand control flow across chunks
4. **Pattern Recognition**: Identify patterns that span multiple chunks
5. **Architectural Understanding**: Build complete architectural picture

### Final Synthesis

1. **Complete Picture**: Assemble complete understanding from all chunks
2. **Relationship Mapping**: Map all relationships and dependencies
3. **Pattern Documentation**: Document all identified patterns
4. **Gap Documentation**: Document any remaining gaps or uncertainties
5. **Comprehensive Findings**: Synthesize all findings into comprehensive report

### Quality Assurance for Synthesis

- Verify consistency across chunks
- Resolve any contradictions between chunks
- Complete any incomplete understanding
- Document synthesis process
- Ensure comprehensive coverage"""

