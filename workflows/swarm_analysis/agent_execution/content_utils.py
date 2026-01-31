"""
Content preparation and validation utilities for agent execution.

Provides functions for preparing chunk content, validating content,
diagnosing content issues, building analysis messages, and synthesizing results.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from workflows.nodes import llm_node

logger = logging.getLogger(__name__)


def prepare_chunk_content(chunk, files_dict: List[Dict[str, Any]]) -> str:
    """
    Prepare chunk content for analysis with improved lookup.
    
    Formats files in chunk with headers and full content.
    Uses multiple lookup strategies to ensure content is found.
    
    Args:
        chunk: Chunk object containing files to prepare
        files_dict: Dictionary mapping file paths to file data
        
    Returns:
        Formatted string with chunk content
    """
    content_parts = []
    
    # Create multiple lookup strategies for better path matching
    files_by_relative = {f.get("relative_path", ""): f for f in files_dict}
    files_by_path = {f.get("path", ""): f for f in files_dict}
    files_by_filename = {Path(f.get("path", "")).name: f for f in files_dict if f.get("path")}
    
    for file_chunk in chunk.files:
        file_info = file_chunk.file_info
        relative_path = file_info.relative_path if hasattr(file_info, 'relative_path') else str(file_info.path)
        
        # Try multiple lookup strategies
        file_data = (
            files_by_relative.get(relative_path) or
            files_by_relative.get(str(Path(relative_path))) or
            files_by_path.get(str(file_info.path)) or
            files_by_filename.get(Path(relative_path).name) or
            {}
        )
        
        # Add file header
        if file_chunk.start_line is not None:
            header = f"# File: {relative_path} (lines {file_chunk.start_line}-{file_chunk.end_line})\n\n"
        else:
            header = f"# File: {relative_path}\n\n"
        
        content_parts.append(header)
        
        # Get content with improved fallback chain
        content = None
        if file_chunk.content:
            content = file_chunk.content
        elif hasattr(file_info, 'content') and file_info.content:
            content = file_info.content
        elif file_data.get("content"):
            content = file_data["content"]
        
        # Validate content
        if not content:
            logger.warning(
                f"No content found for {relative_path}. "
                f"file_chunk.content={bool(file_chunk.content)}, "
                f"file_info.content={bool(getattr(file_info, 'content', None))}, "
                f"file_data.content={bool(file_data.get('content'))}"
            )
            content = ""  # Empty content, but still include file header
        else:
            logger.debug(f"Content found for {relative_path}: {len(content)} chars")
        
        content_parts.append(content)
        content_parts.append("\n\n---\n\n")
    
    return "".join(content_parts)


def validate_chunk_content(chunk_content: str, chunk_num: int, total_chunks: int, role_name: str = "Unknown") -> Dict[str, Any]:
    """
    Validate chunk content and return metrics.
    
    Args:
        chunk_content: Prepared chunk content to validate
        chunk_num: Current chunk number
        total_chunks: Total number of chunks
        role_name: Agent role name for logging
        
    Returns:
        Dictionary with validation metrics
    """
    content_length = len(chunk_content)
    token_estimate = content_length // 4  # Rough estimate (1 token ≈ 4 characters)
    
    metrics = {
        "content_length": content_length,
        "estimated_tokens": token_estimate,
        "chunk_num": chunk_num,
        "total_chunks": total_chunks,
        "role_name": role_name
    }
    
    if content_length == 0:
        logger.error(
            f"Chunk {chunk_num}/{total_chunks} for agent {role_name} has empty content! "
            "This will result in poor analysis quality."
        )
    elif content_length < 100:
        logger.warning(
            f"Chunk {chunk_num}/{total_chunks} for agent {role_name} has very small content: "
            f"{content_length} chars (~{token_estimate} tokens). This may indicate content loss."
        )
    else:
        logger.info(
            f"Chunk {chunk_num}/{total_chunks} for agent {role_name} prepared: "
            f"{content_length} chars, ~{token_estimate} tokens"
        )
    
    return metrics


def diagnose_content_issues(files: List[Dict[str, Any]], file_objects: List[Any]) -> Dict[str, Any]:
    """
    Diagnose content preservation issues.
    
    Compares files dict with FileInfo objects to identify content loss.
    
    Args:
        files: List of file dictionaries from state
        file_objects: List of FileInfo objects created from files
        
    Returns:
        Dictionary with diagnostic statistics
    """
    from scanners.project_scanner import FileInfo
    
    stats = {
        "total_files": len(files),
        "files_with_content": 0,
        "files_without_content": 0,
        "total_content_length": 0,
        "file_objects_with_content": 0,
        "file_objects_without_content": 0,
        "content_preserved": True,
        "content_loss_count": 0
    }
    
    # Analyze files dict
    for f in files:
        content = f.get('content', '')
        if content:
            stats["files_with_content"] += 1
            stats["total_content_length"] += len(content)
        else:
            stats["files_without_content"] += 1
    
    # Analyze FileInfo objects
    for file_info in file_objects:
        if isinstance(file_info, FileInfo):
            if file_info.content:
                stats["file_objects_with_content"] += 1
            else:
                stats["file_objects_without_content"] += 1
    
    # Check for content preservation issues
    if stats["files_with_content"] > stats["file_objects_with_content"]:
        stats["content_preserved"] = False
        stats["content_loss_count"] = stats["files_with_content"] - stats["file_objects_with_content"]
        logger.warning(
            f"Content preservation issue detected: {stats['files_with_content']} files with content in dict, "
            f"but only {stats['file_objects_with_content']} FileInfo objects have content. "
            f"Lost content for {stats['content_loss_count']} files."
        )
    elif stats["files_with_content"] == stats["file_objects_with_content"]:
        logger.debug(
            f"Content preservation verified: {stats['files_with_content']} files with content, "
            f"{stats['file_objects_with_content']} FileInfo objects with content"
        )
    
    return stats


def build_chunk_analysis_message(
    chunk_content: str,
    chunk_num: int,
    total_chunks: int,
    prompt: str,
    architecture_model: Optional[Dict[str, Any]] = None,
    tools_available: bool = False
) -> str:
    """
    Build user message for chunk analysis.
    
    Args:
        chunk_content: Prepared chunk content
        chunk_num: Current chunk number
        total_chunks: Total number of chunks
        prompt: Agent prompt
        architecture_model: Optional architecture model for context
        
    Returns:
        User message string
    """
    # Build architecture context if available
    arch_context = ""
    if architecture_model:
        arch_context = f"\n\n## Architecture Context\nSystem: {architecture_model.get('system_name', 'Unknown')} ({architecture_model.get('system_type', 'Unknown')})\nArchitecture Pattern: {architecture_model.get('architecture_pattern', 'Unknown')}\n"
    
    # Prepare date context for the report
    current_date = datetime.now()
    date_context = f"""
## Report Date Context

IMPORTANT: Use the following date when generating your analysis report:
- Current Date: {current_date.strftime('%B %d, %Y')}
- Current Date (ISO): {current_date.strftime('%Y-%m-%d')}

When referencing dates in your report, use: {current_date.strftime('%B %d, %Y')}
"""
    
    # Add tools section if tools are available
    tools_section = ""
    if tools_available:
        tools_section = """

## Available Tools

You have access to tools to read additional files if needed:
- **read_file(file_path)**: Read the content of any file in the project by its relative path (e.g., 'utils/experience_storage.py')
- **list_directory(directory_path)**: List files in a directory to explore the project structure
- **get_file_info(file_path)**: Get metadata about a file (size, line count, encoding)
"""
    
    # Check if prompt already contains evidence requirements
    # If not, inject v2-style evidence requirements section
    evidence_requirements_section = ""
    if "Evidence-Based Analysis" not in prompt and "Evidence and Reasoning Requirements" not in prompt:
        evidence_requirements_section = """

## Analysis Requirements

1. **Evidence-Based Analysis**: Every finding must reference specific code
   - Include file paths and line numbers
   - Quote relevant code snippets
   - Show the exact pattern you're analyzing

2. **Logical Reasoning**: For each finding, explain:
   - What you observed (the evidence)
   - Why it matters (the reasoning)
   - What it suggests (the conclusion)
   - Your confidence level: [HIGH], [MEDIUM], or [LOW]

3. **Code-Specific Analysis**: 
   - Avoid generic advice that could apply to any project
   - Tie every recommendation to specific code patterns you observed
   - If you find yourself giving generic advice, ask: "What specific code makes this relevant?"

4. **Honest Assessment**:
   - If you cannot find specific evidence, state: "No specific issues found in this chunk"
   - If uncertain, use phrases like: "This pattern suggests..." or "May indicate..."
   - Distinguish between confirmed issues and potential concerns
   - Acknowledge when code is incomplete or context is missing

"""
    
    # Build structured message with evidence requirements
    message = f"""{prompt}{tools_section}{evidence_requirements_section}## Code to Analyze (Chunk {chunk_num}/{total_chunks})

{chunk_content}{arch_context}

{date_context}

Remember: Your analysis must be rooted in the code provided. Every finding must have evidence. If you're uncertain, say so explicitly.
"""
    
    return message


async def synthesize_chunk_results(
    chunk_results: List[Dict[str, Any]],
    role_name: str,
    prompt: str,
    model_name: str,
    architecture_model: Optional[Dict[str, Any]] = None,
    langfuse_prompt_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Synthesize analysis results from multiple chunks into a final report.
    
    Args:
        chunk_results: List of chunk analysis results
        role_name: Agent role name
        prompt: Agent prompt
        model_name: Model name to use for synthesis
        architecture_model: Optional architecture model
        
    Returns:
        Synthesized report text
    """
    if not chunk_results:
        return "No analysis results to synthesize."
    
    # Prepare synthesis prompt
    results_text = "\n\n---\n\n".join([
        f"## Chunk {r.get('chunk_num', '?')} Analysis\n\n{r.get('analysis', 'No analysis available')}"
        for r in chunk_results
    ])
    
    arch_context = ""
    if architecture_model:
        arch_context = f"\n\n## Architecture Context\nSystem: {architecture_model.get('system_name', 'Unknown')} ({architecture_model.get('system_type', 'Unknown')})\nArchitecture Pattern: {architecture_model.get('architecture_pattern', 'Unknown')}\n"
    
    # Prepare date context for the report
    current_date = datetime.now()
    date_context = f"""
## Report Date Context

IMPORTANT: Use the following date when generating your analysis report:
- Current Date: {current_date.strftime('%B %d, %Y')}
- Current Date (ISO): {current_date.strftime('%Y-%m-%d')}

When referencing dates in your report, use: {current_date.strftime('%B %d, %Y')}
"""
    
    synthesis_prompt = f"""{prompt}

## Synthesis Requirements

1. **Evidence Preservation**: Maintain all code references and evidence from individual analyses
   - Preserve file paths and line numbers
   - Keep code snippets that support findings
   - Don't remove evidence when synthesizing

2. **Balanced Reporting**: Acknowledge both strengths and areas for improvement
   - Include a "Strengths and Good Practices" section highlighting what's working well
   - Frame issues as learning opportunities and improvement areas
   - Use constructive, guidance-oriented language throughout

3. **Logical Coherence**: Show how findings relate to each other
   - Identify patterns across chunks
   - Show relationships between findings (both positive patterns and improvement areas)
   - Explain how findings in one area relate to others

4. **Confidence Aggregation**: 
   - When multiple chunks mention similar issues, increase confidence
   - When only one chunk mentions an issue, maintain or lower confidence
   - Clearly mark findings with confidence levels: [HIGH], [MEDIUM], [LOW]
   - Distinguish between confirmed issues, potential concerns, and enhancement suggestions

5. **Uncertainty Handling**: 
   - Clearly mark findings with low confidence or incomplete evidence
   - Distinguish between confirmed issues and potential concerns
   - Frame uncertainty supportively: "Worth reviewing" or "Consider evaluating" rather than "Potential problem"
   - Acknowledge any gaps in analysis due to incomplete code or context

6. **Constructive Language**: 
   - Frame findings as learning opportunities: "Enhancement opportunity" rather than "Problem found"
   - Use supportive, educational tone: "Consider implementing..." rather than "You must fix..."
   - Distinguish between critical issues (frame as "Important to address") and suggestions (frame as "Consider improving")

7. **No Hallucination**: 
   - Only synthesize what was actually found - don't add new findings without evidence
   - If you notice patterns that weren't explicitly mentioned, mark them as "OBSERVATION" not "FINDING"
   - Never claim issues that weren't in the individual analyses

## Individual Chunk Analysis Results

{results_text}{arch_context}

{date_context}

Synthesize these analysis results into a comprehensive, unified report for the {role_name} role. Remember to:
- Acknowledge strengths and good practices found in the code
- Frame all findings constructively as guidance opportunities
- Use supportive, educational language
- Distinguish between confirmed issues, potential concerns, and enhancement suggestions"""
    
    # Use llm_node for synthesis
    llm_state = {
        "messages": [
            {"role": "system", "content": f"You are a {role_name} providing constructive, balanced analysis. Synthesize results from multiple code chunks into a comprehensive report that acknowledges strengths, frames issues as learning opportunities, and provides supportive guidance to help developers improve their codebase."},
            {"role": "user", "content": synthesis_prompt}
        ],
        "model": model_name,
        "temperature": 0.5,
        "metadata": {
            "role": role_name,
            "operation": "synthesis",
            "langfuse_prompt_id": langfuse_prompt_id  # Always include, even if None
        }
    }
    
    try:
        llm_result = await llm_node(llm_state, config=config)
        response = llm_result.get("last_response", "")
        
        if not response or not response.strip():
            logger.warning(f"Synthesis returned empty response for {role_name}")
            # Fallback: combine results without LLM
            return "\n\n---\n\n".join([
                f"## Chunk {r.get('chunk_num', '?')} Analysis\n\n{r.get('analysis', 'No analysis available')}"
                for r in chunk_results
            ])
        
        return response
    except Exception as e:
        logger.error(f"Error synthesizing chunk results for {role_name}: {e}", exc_info=True)
        # Fallback: combine results without LLM
        return "\n\n---\n\n".join([
            f"## Chunk {r.get('chunk_num', '?')} Analysis\n\n{r.get('analysis', 'No analysis available')}"
            for r in chunk_results
        ])




