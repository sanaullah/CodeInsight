"""
Comprehensive error handling framework for edge cases.
Provides guidance for handling various analysis challenges.
"""

import logging

logger = logging.getLogger(__name__)


def get_error_handling_guidance() -> str:
    """
    Get comprehensive error handling guidance.
    
    Returns markdown covering:
    - Missing code handling
    - Incomplete chunks
    - Unclear logic
    - Obfuscated code
    - Legacy patterns
    - Chunk boundary issues
    
    Returns:
        Formatted markdown string with error handling guidance
    """
    return """## Handling Special Cases and Edge Cases

During analysis, you may encounter various challenges. Here's how to handle them:

### Missing File Contents

**If file contents are missing or unavailable**:
- Focus on file names, paths, and types to infer purpose
- Use file structure and naming conventions to understand intent
- Document what you can infer and what remains unknown
- Clearly state limitations in your analysis
- Mark findings as [LOW] confidence when based on inference

### Incomplete Chunks

**If chunks are incomplete or cut off**:
- Identify when a chunk cuts off in the middle of a function or class
- Note when imports or dependencies are split across chunks
- Handle cases where comments or documentation are split across chunks
- Document what appears to be missing
- Make reasonable assumptions but state them explicitly
- Mark analysis as partial and note what's missing

### Unclear Logic

**If code logic is unclear or complex**:
- Break down complex logic step by step
- Document your understanding and any uncertainties
- Mark findings as [MEDIUM] or [LOW] confidence when logic is unclear
- Request clarification or additional context if needed
- Focus on what you can understand and document uncertainties

### Obfuscated Code

**If code is obfuscated or minified**:
- Focus on interfaces and external interactions
- Analyze function signatures and API contracts
- Document what can be determined from structure
- Note that code is obfuscated and analysis is limited
- Mark all findings as [LOW] confidence
- Recommend deobfuscation if security analysis is critical

### Legacy Patterns

**If code uses legacy or outdated patterns**:
- Identify the pattern and its vintage
- Assess whether it's still appropriate or needs modernization
- Don't judge without understanding historical context
- Document the pattern and its implications
- Recommend modernization if appropriate, but acknowledge context
- Consider backward compatibility requirements

### Chunk Boundary Issues

**If chunks split code awkwardly**:
- Note where chunks might be splitting code awkwardly
- Identify when functions or classes are split across chunks
- Document how this affects your analysis
- Synthesize understanding across chunk boundaries
- Mark findings that span chunks appropriately

### Unrecognized File Types

**If file types are unrecognized or unusual**:
- Research common uses for these extensions
- Infer purpose from file location and naming
- Document assumptions about file purpose
- Mark analysis as [LOW] confidence
- Recommend investigation if file is critical

### Incomplete Implementations

**If code appears incomplete or in progress**:
- Identify what appears to be missing
- Document incomplete functionality
- Note TODOs, FIXMEs, and incomplete features
- Assess impact of incomplete implementation
- Recommend completion strategies

### Error Handling and Limitations

### If you encounter issues during analysis:

1. **Clearly state the nature of the problem**: Document what went wrong or what's unclear
2. **Proceed with available information**: Continue analysis based on what you can determine
3. **Note limitations in output**: Explicitly state what you couldn't analyze or understand
4. **Make explicit assumptions**: If you must assume, state assumptions clearly
5. **Mark confidence appropriately**: Use [LOW] confidence for uncertain findings

### For different types of errors:

- **Missing file contents**: Focus on file names, paths, and types to infer purpose
- **Unrecognized file types**: Research common uses for these extensions
- **Incomplete code**: Identify what appears to be missing and note the uncertainty
- **Obfuscated code**: Focus on interfaces and external interactions
- **Chunk boundary issues**: Note where chunks might be splitting code awkwardly

### Limitation Documentation

Always document limitations in your analysis:

- **What you couldn't analyze**: Clearly state what was unavailable
- **What you're uncertain about**: Document uncertainties explicitly
- **What you assumed**: List all assumptions made
- **Confidence levels**: Use [HIGH/MEDIUM/LOW] to indicate certainty
- **Recommendations for improvement**: Suggest how to get better information"""


def get_limitation_documentation_guidance() -> str:
    """
    Get guidance for documenting analysis limitations.
    
    Returns:
        Formatted markdown string with limitation documentation guidance
    """
    return """## Documenting Analysis Limitations

It's critical to document limitations in your analysis to maintain honesty and transparency.

### What to Document

1. **Missing Information**: What information was unavailable or incomplete
2. **Uncertainties**: What you're uncertain about and why
3. **Assumptions**: What assumptions you made and why
4. **Confidence Levels**: Use [HIGH/MEDIUM/LOW] to indicate certainty
5. **Gaps**: What gaps exist in your understanding

### How to Document

- Use explicit language: "I could not analyze X because..."
- Mark confidence levels: [HIGH/MEDIUM/LOW]
- State assumptions: "Assuming that..."
- Note uncertainties: "Uncertain about..."
- Recommend improvements: "To improve analysis, consider..."

### Example Limitation Documentation

```
## Analysis Limitations

- **Missing Code**: File `auth.py` was not available for analysis
- **Incomplete Chunk**: Function `process_payment()` was cut off mid-implementation
- **Uncertainty**: [LOW] confidence in security assessment due to obfuscated code
- **Assumption**: Assuming database connection uses connection pooling
- **Gap**: Unable to verify API endpoint authentication without runtime testing
```"""

