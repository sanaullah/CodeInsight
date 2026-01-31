"""
Agent execution nodes for Swarm Analysis workflow.

Handles agent spawning, dispatching, execution, and result collection.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from workflows.swarm_analysis_state import SwarmAnalysisState
from workflows.swarm_analysis_state_decorators import validate_state_fields
from workflows.state_keys import StateKeys
from workflows.nodes import llm_node
from utils.error_context import create_error_context
from workflows.swarm_analysis.tools import get_tool_definitions, execute_tool
from workflows.swarm_analysis.tools.tool_executor import parse_tool_call
from utils.model_capabilities import check_tool_calling_support
from utils.langfuse.tracing import create_observation

# from workflows.swarm_analysis.mcp.filesystem_client import FilesystemMCPClient # Removed: MCP archived
from .content_utils import (
    prepare_chunk_content,
    validate_chunk_content,
    diagnose_content_issues,
    build_chunk_analysis_message,
)

try:
    from langfuse.decorators import observe
except ImportError:
    # Dummy decorator if langfuse not installed
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


logger = logging.getLogger(__name__)


def _normalize_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """
    Normalize tool call arguments for comparison.

    Handles both JSON strings and dicts, normalizes to dict format.

    Args:
        arguments: Tool call arguments (string or dict)

    Returns:
        Normalized dict of arguments
    """
    import json

    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return {"raw": arguments}
    return {"raw": str(arguments)}


@observe(as_type="generation")
async def _execute_with_tool_calling(
    llm_state: Dict[str, Any],
    project_path: str,
    state: SwarmAnalysisState,
    config: Optional[Dict[str, Any]] = None,
    max_tool_calls: int = 10,
    max_iterations: int = 15,
    max_duplicate_iterations: int = 3,
    llm_call_timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    Execute LLM call with tool calling support (multi-turn loop).

    Uses MCP filesystem server if available, falls back to embedded tools.

    Safety Mechanisms:
    - Iteration limit: Prevents infinite loops by limiting total loop iterations
    - Duplicate detection: Detects when LLM is stuck repeating identical tool calls
    - Timeout protection: Prevents hanging on slow/unresponsive LLM calls

    Args:
        llm_state: Initial LLM state with messages
        project_path: Project root path for tool execution
        state: Current workflow state
        config: Optional LangGraph config
        max_tool_calls: Maximum number of tool calls allowed
        max_iterations: Maximum loop iterations (default: 15). Prevents infinite loops
            even if LLM keeps making tool calls. Checked before each LLM call.
        max_duplicate_iterations: Number of identical iterations before breaking
            (default: 3). Detects when LLM is stuck in repetitive behavior.
        llm_call_timeout: Timeout in seconds for each LLM call (default: 60.0).
            Prevents hanging on slow or unresponsive LLM API calls.

    Returns:
        Final LLM result with analysis. If safety guards trigger, returns partial
        result with appropriate warnings logged.

    Behavior on Limits:
    - Iteration limit: Loop breaks, returns last LLM result, logs warning
    - Duplicate detection: Loop breaks, returns last LLM result, logs warning
    - Timeout: Loop breaks, no result returned, logs error
    """
    tool_calls_made = 0
    tool_calls_history = []
    current_messages = llm_state["messages"].copy()

    # Initialize file cache and tool call history in state if not present
    if StateKeys.FILE_CACHE not in state:
        state[StateKeys.FILE_CACHE] = {}
    if StateKeys.TOOL_CALLS_HISTORY not in state:
        state[StateKeys.TOOL_CALLS_HISTORY] = []

    # MCP filesystem server support removed (archived feature)
    use_mcp = False
    mcp_client = None
    mcp_tools = None
    use_embedded_tools = True

    if use_embedded_tools:
        # Use embedded tools
        tools = get_tool_definitions()
        if "tools" not in llm_state:
            llm_state["tools"] = tools

    # Initialize safety guards
    iteration = 0
    previous_iterations = []  # Track last N iterations for duplicate detection

    while True:
        iteration += 1

        # Check iteration limit BEFORE making expensive LLM call
        if iteration > max_iterations:
            logger.warning(
                f"Tool call loop terminated: max iterations ({max_iterations}) reached "
                f"(tool calls made: {tool_calls_made}/{max_tool_calls})"
            )
            break

        # Check tool call limit
        if tool_calls_made >= max_tool_calls:
            logger.warning(f"Tool call limit ({max_tool_calls}) reached")
            break
        # Call LLM
        current_llm_state = {
            "messages": current_messages,
            "model": llm_state["model"],
            "temperature": llm_state.get("temperature", 0.7),
            "max_tokens": llm_state.get("max_tokens"),
            "metadata": llm_state.get("metadata", {}),
        }

        # Add tools if available
        if "tools" in llm_state:
            current_llm_state["tools"] = llm_state["tools"]
            current_llm_state["tool_choice"] = llm_state.get("tool_choice", "auto")

        # Call LLM with timeout protection
        try:
            llm_result = await asyncio.wait_for(
                llm_node(current_llm_state, config=config), timeout=llm_call_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"LLM call timed out after {llm_call_timeout}s "
                f"(iteration: {iteration}, tool calls made: {tool_calls_made}/{max_tool_calls})"
            )
            # Break loop on timeout - don't continue with partial result
            break
        except Exception as e:
            logger.error(f"LLM call failed: {e} (iteration: {iteration})")
            break

        # Check for tool calls
        tool_calls = llm_result.get("tool_calls", [])

        # Duplicate detection: Create signature for this iteration
        if tool_calls:
            # Normalize tool call arguments for comparison
            iteration_signature = {
                "tool_calls": [
                    {
                        "name": tc.get("function", {}).get("name")
                        if isinstance(tc.get("function"), dict)
                        else None,
                        "arguments": _normalize_tool_arguments(
                            tc.get("function", {}).get("arguments")
                            if isinstance(tc.get("function"), dict)
                            else None
                        ),
                    }
                    for tc in tool_calls
                ]
            }

            # Check for duplicates in recent iterations
            if len(previous_iterations) >= max_duplicate_iterations:
                # Check if last N iterations are identical
                recent_iterations = previous_iterations[-max_duplicate_iterations:]
                if all(prev == iteration_signature for prev in recent_iterations):
                    logger.warning(
                        f"Tool call loop terminated: detected {max_duplicate_iterations} "
                        f"identical iterations (likely stuck in loop). "
                        f"Iteration: {iteration}, tool calls made: {tool_calls_made}/{max_tool_calls}"
                    )
                    break

            # Store this iteration's signature
            previous_iterations.append(iteration_signature)
            # Keep only last max_duplicate_iterations + 1 for efficiency
            if len(previous_iterations) > max_duplicate_iterations + 1:
                previous_iterations.pop(0)

            # No tool calls or limit reached - return final result
            if tool_calls_made >= max_tool_calls:
                logger.warning(f"Tool call limit ({max_tool_calls}) reached for agent")
            # return llm_result  # FIXED: This prevented tool execution!

        # Execute tool calls
        tool_results = []
        for tool_call in tool_calls:
            tool_calls_made += 1
            if tool_calls_made > max_tool_calls:
                logger.warning(f"Tool call limit exceeded, stopping tool execution")
                break

            tool_name, arguments = parse_tool_call(tool_call)
            if not tool_name:
                logger.warning(f"Failed to parse tool call: {tool_call}")
                continue

            logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")

            # Execute tool - use MCP if available, otherwise embedded
            # Execute tool - embedded only (MCP removed)
            if False:  # Placeholder for removed MCP logic
                try:
                    # MCP client should already be connected and tools retrieved
                    # Execute tool via MCP
                    tool_result_raw = await mcp_client.execute_tool(
                        tool_name, arguments
                    )

                    # Convert MCP result to expected format
                    if isinstance(tool_result_raw, dict):
                        tool_result = tool_result_raw
                    elif isinstance(tool_result_raw, str):
                        # Check if it's JSON
                        try:
                            import json

                            tool_result = json.loads(tool_result_raw)
                        except (json.JSONDecodeError, ValueError):
                            # Not JSON, wrap in content field
                            tool_result = {"content": tool_result_raw}
                    else:
                        tool_result = {"content": str(tool_result_raw)}

                except Exception as e:
                    logger.warning(
                        f"MCP tool execution failed: {e}, falling back to embedded tools"
                    )
                    # Fallback to embedded tools
                    tool_result = await execute_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        project_path=project_path,
                        state=state,
                    )
            else:
                # Use embedded tools
                # Create observation for audit trail
                role_name = llm_state.get("metadata", {}).get("role", "unknown_agent")
                
                # Wrap tool execution in Langfuse observation
                observation = create_observation(
                    name=f"tool_{tool_name}",
                    input_data=arguments,
                    metadata={
                        "role": role_name,
                        "tool_name": tool_name,
                        "iteration": iteration
                    },
                    level="DEFAULT"
                )
                
                # Use context manager if observation was created
                if observation:
                    with observation:
                        tool_result = await execute_tool(
                            tool_name=tool_name,
                            arguments=arguments,
                            project_path=project_path,
                            state=state,
                        )
                        # Update observation with result
                        try:
                            # Sanitize result for metadata if it's too large
                            result_preview = str(tool_result)[:1000]
                            observation.update(output=result_preview)
                        except Exception:
                            pass
                else:
                    # Fallback if observation creation failed
                    tool_result = await execute_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        project_path=project_path,
                        state=state,
                    )

            # Record in history
            tool_call_record = {
                "tool_name": tool_name,
                "arguments": arguments,
                "result": tool_result,
                "via_mcp": mcp_client is not None and not use_embedded_tools,
            }
            tool_calls_history.append(tool_call_record)
            state[StateKeys.TOOL_CALLS_HISTORY].append(tool_call_record)

            # Add tool result to messages
            # Format: ToolMessage with tool_call_id
            tool_call_id = tool_call.get("id", f"call_{tool_calls_made}")

            # Extract content from tool_result
            if isinstance(tool_result, dict):
                if "content" in tool_result:
                    content = tool_result["content"]
                elif "error" in tool_result:
                    content = f"Error: {tool_result['error']}"
                else:
                    content = str(tool_result)
            else:
                content = str(tool_result)

            tool_results.append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": content}
            )

        # Add tool results to message history
        current_messages.extend(
            [
                {
                    "role": "assistant",
                    "content": llm_result.get("last_response", ""),
                    "tool_calls": tool_calls,
                }
            ]
        )
        current_messages.extend(tool_results)

        # SANDWICH STRATEGY: Re-inject original context to prevent attention drift
        # Find original user message (the one with the code chunk)
        original_user_msg = next(
            (m for m in llm_state["messages"] if m["role"] == "user"), None
        )
        if original_user_msg and original_user_msg.get("content"):
            logger.debug(
                f"Applying Sandwich Strategy: Re-injecting original context ({len(original_user_msg['content'])} chars)"
            )
            current_messages.append(
                {
                    "role": "user",
                    "content": (
                        "SYSTEM_INJECTION: Tool outputs received.\n"
                        "--------------------------------------------------\n"
                        "CONTEXT RESTORATION (SANDWICH STRATEGY):\n"
                        "To prevent context loss, here is the ORIGINAL code chunk/request you are analyzing.\n"
                        "You must synthesize the tool outputs above to answer THIS specific request.\n"
                        "--------------------------------------------------\n\n"
                        f"{original_user_msg['content']}"
                    ),
                }
            )

        # Continue loop to get final response
        logger.debug(
            f"Tool calls executed, continuing analysis (calls made: {tool_calls_made}/{max_tool_calls})"
        )

    # This should never be reached, but just in case
    # Cleanup MCP client (Removed)
    pass
    return llm_result



from .security import build_agent_system_prompt

@validate_state_fields(["selected_roles", "generated_prompts"], "spawn_agents")
def spawn_agents_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Spawn agents for each role.

    Args:
        state: Current swarm analysis state

    Returns:
        Updated state with agents list
    """
    selected_roles: List[Dict[str, Any]] = state.get(StateKeys.SELECTED_ROLES, [])
    generated_prompts: Dict[str, str] = state.get(StateKeys.GENERATED_PROMPTS, {})

    if not selected_roles or not generated_prompts:
        error_msg = (
            "selected_roles and generated_prompts are required for agent spawning. "
            "Please ensure roles were selected and prompts were generated successfully."
        )
        state[StateKeys.ERROR] = error_msg
        state[StateKeys.ERROR_STAGE] = "spawn_agents"
        logger.error(error_msg)
        return state

    try:
        # TODO: Create actual agent objects
        # For now, store agent metadata
        agents: List[Dict[str, Any]] = []
        for role_obj in selected_roles:
            role_name = (
                role_obj["name"] if isinstance(role_obj, dict) else str(role_obj)
            )

            # Phase 2: Enhanced Prompt Versioning
            # Try to fetch latest production prompt from Langfuse
            langfuse_prompt = None
            prompt_source = "generated"
            try:
                from utils.langfuse.prompts import get_langfuse_prompt_by_name

                # Normalize names for lookup (role_architecture_goal)
                # This logic should match create_langfuse_prompt in utils/langfuse/prompts.py
                from utils.langfuse.prompts import normalize_label

                architecture_type = state.get(StateKeys.ARCHITECTURE_MODEL, {}).get(
                    "system_type", "unknown"
                )
                goal_category = (
                    "general"  # Default, would need robust logic to match exactly
                )

                # Construct name to search for
                prompt_name = f"{normalize_label(role_name)}_{normalize_label(architecture_type)}_{normalize_label(goal_category)}"

                # Try to get prompt (optional: filter by label "production")
                # For now, we prefer the GENERATED prompt from this session over stored ones
                # UNLESS we explicitly want to test stored prompts (future feature)
                # So we stick to generated_prompts.get(role_name)
                pass
            except Exception:
                pass

            prompt = generated_prompts.get(role_name, "")

            # Store agent info (would create actual agent objects)
            agent_info = {
                "role": role_name,
                "prompt": prompt,
                "agent_id": f"swarm_agent_{role_name.lower().replace(' ', '_')}",
            }
            agents.append(agent_info)


        # Update state
        updated_state = state.copy()
        updated_state[StateKeys.AGENTS] = agents
        
        # TRUST BOUNDARY ENFORCEMENT:
        # Whitelist the authorized roles for this execution run.
        # Only roles created here are allowed to execute.
        authorized_roles = [agent["role"] for agent in agents]
        updated_state[StateKeys.AUTHORIZED_ROLES] = authorized_roles

        logger.info(f"Spawned {len(agents)} agents. Authorized roles: {authorized_roles}")
        return updated_state

    except Exception as e:
        error_context = create_error_context(
            "spawn_agents",
            state,
            {"role_count": len(selected_roles), "prompt_count": len(generated_prompts)},
        )
        error_msg = (
            f"Failed to spawn agents: {str(e)}. "
            f"Attempted to spawn {len(selected_roles)} agents with {len(generated_prompts)} prompts. "
            f"Check that roles and prompts are valid."
        )
        logger.error(error_msg, extra=error_context, exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = error_msg
        updated_state[StateKeys.ERROR_STAGE] = "spawn_agents"
        updated_state[StateKeys.ERROR_CONTEXT] = error_context
        return updated_state


@validate_state_fields(["agents"], "dispatch_agents")
def dispatch_agents_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Dispatch agents node - prepares state for parallel execution.

    This node validates agents and prepares state. The actual Send objects
    are created in the conditional edge function (route_after_dispatch).

    Args:
        state: Current swarm analysis state

    Returns:
        Updated state (agents should already be set by spawn_agents_node)
    """
    agents: List[Dict[str, Any]] = state.get(StateKeys.AGENTS, [])

    if not agents:
        error_msg = (
            "agents are required for agent dispatch. "
            "Please ensure agents were spawned successfully before dispatching."
        )
        state[StateKeys.ERROR] = error_msg
        state[StateKeys.ERROR_STAGE] = "dispatch_agents"
        logger.error(error_msg)
        return state

    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    if stream_callback:
        stream_callback(
            "agent_execution_start",
            {
                "message": "Dispatching agents for parallel execution...",
                "agent_count": len(agents),
            },
        )

    # Just validate and return state - Send objects are created in conditional edge
    logger.info(f"Prepared {len(agents)} agents for parallel execution")

    # Return state unchanged (agents are already in state from spawn_agents_node)
    return state



def _enrich_result_with_tool_metadata(agent_result: Dict[str, Any], state: SwarmAnalysisState) -> None:
    """Helper to add tool calling metadata and file cache utilization to agent result."""
    if StateKeys.TOOL_CALLS_HISTORY in state and state[StateKeys.TOOL_CALLS_HISTORY]:
        agent_result["tool_calls_made"] = len(state[StateKeys.TOOL_CALLS_HISTORY])
    if StateKeys.FILE_CACHE in state and state[StateKeys.FILE_CACHE]:
        agent_result["files_requested"] = len(state[StateKeys.FILE_CACHE])


async def _analyze_chunk(
    chunk_content: str,
    chunk_num: int,
    total_chunks: int,
    role_name: str,
    prompt: str,
    architecture_model: Any,
    architecture_hash: str,
    langfuse_prompt_id: Optional[str],
    model_name: str,
    project_path: str,
    state: Dict[str, Any],
    config: Optional[Dict[str, Any]],
    stream_callback: Optional[Any] = None,
    primary_file: str = "multiple files",
    analysis_mode: str = "chunked",
) -> Dict[str, Any]:
    """
    Analyzes a single chunk of code using the specified agent role and model.

    This helper function centralizes the repetitive logic required for chunk processing:
    1. Emits lifecycle events (chunk_start/chunk_complete) to the stream callback.
    2. Validates the chunk content for syntax and sanity.
    3. Checks for tool-calling support on the target model.
    4. Constructs the final user message and system prompt.
    5. Executes the LLM call, handling tool-calling loops if enabled.
    6. Extracts the Langfuse trace ID and analysis response.

    Args:
        chunk_content: The actual code content/context to be analyzed.
        chunk_num: Current chunk index (1-based) for logging and UI.
        total_chunks: Total number of chunks in the workload.
        role_name: The name of the agent role (e.g., "Performance Engineer").
        prompt: The role-specific prompt instructions.
        architecture_model: The project's architecture model for context.
        architecture_hash: Hash of the architecture model for prompt resolution.
        langfuse_prompt_id: ID of the Langfuse prompt being used.
        model_name: Name of the LLM model to invoke.
        project_path: Root path of the project.
        state: The current workflow state dictionary.
        config: Optional LangGraph configuration.
        stream_callback: Optional callback for real-time event streaming.
        primary_file: Filename to display in the UI for this chunk.
        analysis_mode: Either "direct" (single-pass) or "chunked".

    Returns:
        A dictionary containing:
        - "analysis_text": The raw analysis report from the LLM.
        - "trace_id": The Langfuse trace ID for observability.
        - "llm_result": The full result object from the LLM invocation.
    """
    # 1. Notify UI/Streams that the agent has started working on this specific chunk
    if stream_callback:
        stream_callback(
            "agent_chunk_start",
            {
                "agent": role_name,
                "agent_id": f"agent_{role_name.lower().replace(' ', '_')}",
                "chunk_index": chunk_num,
                "total_chunks": total_chunks,
                "file": primary_file,
                "status": "processing",
            },
        )

    # 2. Safety Check: Validate chunk content to catch empty/corrupt data early
    metrics = validate_chunk_content(chunk_content, chunk_num, total_chunks, role_name)
    logger.debug(f"Chunk metrics for {role_name} (Mode: {analysis_mode}): {metrics}")

    # 3. Dynamic Configuration: Resolve tool settings and model capabilities
    enable_tool_calling = state.get(StateKeys.ENABLE_TOOL_CALLING, True)
    max_tool_calls = state.get(StateKeys.MAX_TOOL_CALLS, 10)
    max_iterations = state.get(StateKeys.MAX_ITERATIONS, 15)
    max_duplicate_iterations = state.get(StateKeys.MAX_DUPLICATE_ITERATIONS, 3)
    llm_call_timeout = state.get(StateKeys.LLM_CALL_TIMEOUT, 60.0)
    tools_available = False
    tools = None

    # Check if the model supports tool-calling (e.g., OpenAI, Anthropic, some Gemini models)
    if enable_tool_calling:
        tools_available = check_tool_calling_support(model_name)
        if tools_available:
            tools = get_tool_definitions()
            logger.debug(f"Tool calling enabled for chunk analysis ({role_name})")

    # 4. Message Construction: Combine system instructions with user requirements
    user_message = build_chunk_analysis_message(
        chunk_content=chunk_content,
        chunk_num=chunk_num,
        total_chunks=total_chunks,
        prompt=prompt,
        architecture_model=architecture_model,
        tools_available=tools_available,
    )

    # Prepare the payload for the LLM node
    llm_state = {
        "messages": [
            {
                "role": "system",
                "content": build_agent_system_prompt(
                    role_name, architecture_hash, architecture_model
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "model": model_name,
        "temperature": 0.7,
        "metadata": {
            "role": role_name,
            "chunk_num": chunk_num,
            "total_chunks": total_chunks,
            "langfuse_prompt_id": langfuse_prompt_id,
            "analysis_mode": analysis_mode,
        },
    }

    # Inject tool definitions if supported and enabled
    if tools_available and tools:
        llm_state["tools"] = tools
        llm_state["tool_choice"] = "auto"

    # 5. Execution: Dispatch the LLM call
    # If tools are available, we use the tool-calling executor which handles the loop
    if tools_available and tools:
        llm_result = await _execute_with_tool_calling(
            llm_state=llm_state,
            project_path=project_path,
            state=state,
            config=config,
            max_tool_calls=max_tool_calls,
            max_iterations=max_iterations,
            max_duplicate_iterations=max_duplicate_iterations,
            llm_call_timeout=llm_call_timeout,
        )
    else:
        # Standard stateless LLM call
        llm_result = await llm_node(llm_state, config=config)

    # 6. Post-processing: Extract content and trace information
    analysis_text = llm_result.get("last_response", "")
    metadata = llm_result.get("metadata", {})
    captured_trace_id = metadata.get("langfuse_trace_id") if isinstance(metadata, dict) else None

    # Notify UI/Streams that this chunk is done
    if stream_callback:
        stream_callback(
            "agent_chunk_complete",
            {
                "agent": role_name,
                "agent_id": f"agent_{role_name.lower().replace(' ', '_')}",
                "chunk_index": chunk_num,
                "total_chunks": total_chunks,
                "file": primary_file,
                "findings": 0, # Note: Current implementation uses raw response; extraction happens later
                "status": "completed",
            },
        )

    return {
        "analysis_text": analysis_text,
        "trace_id": captured_trace_id,
        "llm_result": llm_result,
    }


@validate_state_fields(["agent_info"], "execute_single_agent")
@observe(as_type="span", name="execute_single_agent")
async def execute_single_agent_node(
    state: SwarmAnalysisState, config: Optional[Dict[str, Any]] = None
) -> SwarmAnalysisState:
    """
    Execute a single agent's analysis.

    This node processes one agent's analysis including chunking, LLM calls, and synthesis.
    Designed to be called in parallel via Send from dispatch_agents_node.

    Args:
        state: State containing agent_info and all necessary context
        config: Optional LangGraph config

    Returns:
        Updated state with agent result in agent_results_list
    """
    # Extract agent info from state (set by Send)
    agent_info = state.get(StateKeys.AGENT_INFO)
    if not agent_info:
        # Fallback: try to get from agents list (for backward compatibility)
        agents = state.get(StateKeys.AGENTS, [])
        if agents:
            agent_info = agents[0]
        else:
            state[StateKeys.ERROR] = "agent_info is required"
            state[StateKeys.ERROR_STAGE] = "execute_single_agent"
            return state

    role_name = agent_info.get("role", "Unknown")
    
    # TRUST BOUNDARY ENFORCEMENT:
    # Verify that the agent role is authorized to execute.
    authorized_roles = state.get(StateKeys.AUTHORIZED_ROLES, [])
    if authorized_roles and role_name not in authorized_roles:
        # Security violation: Unauthorized agent detected
        error_msg = (
            f"SECURITY VIOLATION: Unauthorized agent execution attempt. "
            f"Role '{role_name}' is not in the authorized_roles whitelist. "
            f"Execution blocked."
        )
        logger.critical(error_msg)
        state[StateKeys.ERROR] = error_msg
        state[StateKeys.ERROR_STAGE] = "execute_single_agent_security_check"
        return state
        
    prompt = agent_info.get("prompt", "")

    # Get configuration from state
    project_path = state.get(StateKeys.PROJECT_PATH)
    all_files = state.get(StateKeys.FILES, [])
    model_name = state.get(StateKeys.MODEL_NAME)
    max_tokens_per_chunk = state.get(StateKeys.MAX_TOKENS_PER_CHUNK, 100000)
    enable_chunking = state.get(StateKeys.ENABLE_CHUNKING, True)
    chunking_strategy_str = state.get(StateKeys.CHUNKING_STRATEGY, "STANDARD")
    architecture_model = state.get(StateKeys.ARCHITECTURE_MODEL)
    architecture_hash = state.get(StateKeys.ARCHITECTURE_HASH)
    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)
    enable_dynamic_file_selection = state.get(StateKeys.ENABLE_DYNAMIC_FILE_SELECTION, False)

    # Track trace ID captured during agent execution
    captured_trace_id = None

    # Check if dynamic file selection is enabled and agent has selected files
    files = all_files  # Default to all files
    if enable_dynamic_file_selection:
        # Try to get from agent_info first (from Send payload, if file selection was skipped)
        selected_files = agent_info.get("selected_files")

        # Fallback: get from per-role dictionary in state (stored by file_selection_node)
        if not selected_files:
            selected_files_dict = state.get(StateKeys.SELECTED_FILES, {})
            selected_files = selected_files_dict.get(role_name)

        if selected_files:
            # Filter files to only selected ones
            from .file_selection_utils import filter_files_by_selection

            files = filter_files_by_selection(all_files, selected_files)
            logger.info(
                f"Agent {role_name} using {len(files)} selected files "
                f"(from {len(all_files)} total files)"
            )
        else:
            logger.warning(
                f"Dynamic file selection enabled but no selected_files for {role_name}, "
                f"using all {len(all_files)} files"
            )

    # Debug information available via logger.debug() if needed
    files_with_content = sum(1 for f in files if f.get("content")) if files else 0
    logger.debug(
        f"Executing agent {role_name}: {len(files) if files else 0} files, {files_with_content} with content"
    )

    # Get Langfuse prompt ID for this role (for trace linking)
    langfuse_prompt_ids = state.get(StateKeys.LANGFUSE_PROMPT_IDS, {})
    langfuse_prompt_id = langfuse_prompt_ids.get(role_name)

    if stream_callback:
        stream_callback("agent_started", {"agent": role_name})
        # Emit agent execution start with full details
        stream_callback(
            "agent_execution_start",
            {
                "agent": role_name,
                "agent_id": f"agent_{role_name.lower().replace(' ', '_')}",
                "total_chunks": 0,  # Will update later
                "status": "active",
            },
        )

    try:
        if enable_chunking and files:
            # Chunking enabled - create chunks and analyze per agent
            from chunking import ChunkManager, ChunkingStrategy
            from scanners.project_scanner import FileInfo

            # Convert strategy string to enum
            strategy_map = {
                "NONE": ChunkingStrategy.NONE,
                "STANDARD": ChunkingStrategy.STANDARD,
                "AGGRESSIVE": ChunkingStrategy.AGGRESSIVE,
            }
            chunking_strategy = strategy_map.get(
                chunking_strategy_str.upper(), ChunkingStrategy.STANDARD
            )

            # Convert files dict to FileInfo objects
            file_objects = []
            for f in files:
                file_info = FileInfo(
                    path=Path(f.get("path", "")),
                    relative_path=f.get("relative_path", ""),
                    content=f.get("content", ""),
                    size=f.get("size", 0),
                    line_count=f.get("line_count", 0),
                    encoding=f.get("encoding", "utf-8"),
                )
                file_objects.append(file_info)

            # Validate content preservation
            for i, file_info in enumerate(file_objects):
                original_content = files[i].get("content", "")
                if not file_info.content and original_content:
                    logger.warning(
                        f"Content lost during FileInfo conversion for {file_info.relative_path}. "
                        f"Original length: {len(original_content)}"
                    )
                    # Restore content from original dict
                    file_info.content = original_content
                elif not file_info.content:
                    logger.warning(f"No content found for {file_info.relative_path}")

            # Diagnose content issues
            diagnostic_stats = diagnose_content_issues(files, file_objects)
            logger.debug(
                f"Content diagnostic stats for agent {role_name}: {diagnostic_stats}"
            )

            # Hybrid chunking: Check if selected files fit in token limit
            # If they do, analyze directly without chunking
            from chunking.token_counter import TokenCounter

            token_counter = TokenCounter(model_name=model_name)

            # Calculate total tokens for selected files
            total_tokens = sum(
                token_counter.count_tokens(f.get("content", "")) for f in files
            )

            logger.debug(
                f"Agent {role_name}: {len(files)} files, "
                f"{total_tokens:,} tokens, limit: {max_tokens_per_chunk:,}"
            )

            # If files fit in token limit, analyze directly (no chunking)
            if total_tokens <= max_tokens_per_chunk and len(files) > 0:
                logger.info(
                    f"Agent {role_name}: Selected files fit in token limit "
                    f"({total_tokens:,} <= {max_tokens_per_chunk:,}), analyzing directly"
                )

                # Prepare all file content for direct analysis
                from chunking import Chunk
                from chunking.chunk_manager import FileChunk

                # Create a single "chunk" with all files for direct analysis
                file_chunks = [
                    FileChunk(file_info=file_info) for file_info in file_objects
                ]
                single_chunk = Chunk(
                    chunk_index=0,
                    files=file_chunks,
                    total_tokens=total_tokens,
                    total_files=len(files),
                    total_lines=sum(f.line_count for f in file_objects),
                )

                # Prepare content
                chunk_content = prepare_chunk_content(single_chunk, files)

                # Process single chunk using helper
                chunk_analysis = await _analyze_chunk(
                    chunk_content=chunk_content,
                    chunk_num=1,
                    total_chunks=1,
                    role_name=role_name,
                    prompt=prompt,
                    architecture_model=architecture_model,
                    architecture_hash=architecture_hash,
                    langfuse_prompt_id=langfuse_prompt_id,
                    model_name=model_name,
                    project_path=project_path,
                    state=state,
                    config=config,
                    stream_callback=stream_callback,
                    primary_file="multiple files",
                    analysis_mode="direct"
                )

                analysis_text = chunk_analysis["analysis_text"]
                captured_trace_id = chunk_analysis["trace_id"]

                # Store result (no synthesis needed for direct analysis)
                agent_result = {
                    "role_name": role_name,
                    "synthesized_report": analysis_text,
                    "status": "completed",
                    "chunks_analyzed": 1,
                    "files_analyzed": len(files),
                    "tokens_used": total_tokens,
                    "analysis_mode": "direct",
                }

                # Add tool calling metadata to result if available
                _enrich_result_with_tool_metadata(agent_result, state)

                if stream_callback:
                    # Emit agent execution complete
                    stream_callback(
                        "agent_execution_complete",
                        {
                            "agent": role_name,
                            "agent_id": f"agent_{role_name.lower().replace(' ', '_')}",
                            "total_chunks": 1,
                            "status": "completed",
                        },
                    )

                logger.info(
                    f"Agent {role_name} completed direct analysis (via helper) of {len(files)} files"
                )

                # Return ONLY agent_results_list (has Annotated reducer for parallel updates)
                return {"agent_results_list": [agent_result]}


            # Files exceed token limit - proceed with chunking
            logger.info(
                f"Agent {role_name}: Selected files exceed token limit "
                f"({total_tokens:,} > {max_tokens_per_chunk:,}), using chunking"
            )

            # Create chunk manager for this agent
            chunk_manager = ChunkManager(
                max_tokens=max_tokens_per_chunk,
                model_name=model_name,
                chunking_strategy=chunking_strategy,
            )

            # Create chunks for this agent (only selected files)
            chunks = chunk_manager.create_chunks(file_objects)

            if stream_callback:
                stream_callback(
                    "chunks_created", {"agent": role_name, "chunk_count": len(chunks)}
                )

            # Route based on chunk count
            if len(chunks) > 1:
                # Multiple chunks - handle chunking internally to avoid state conflicts
                # When agents execute in parallel, we can't store chunks in shared state

                # Process all chunks sequentially within this node
                chunk_results = []
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_num = chunk_idx + 1
                    total_chunks_count = len(chunks)

                    # Get primary file for display
                    primary_file = "multiple files"
                    if hasattr(chunk, "files") and chunk.files:
                        primary_file = chunk.files[0].file_info.relative_path
                        if len(chunk.files) > 1:
                            primary_file += f" (+{len(chunk.files)-1} others)"

                    # Prepare chunk content
                    chunk_content = prepare_chunk_content(chunk, files)

                    # Process chunk using helper
                    chunk_analysis = await _analyze_chunk(
                        chunk_content=chunk_content,
                        chunk_num=chunk_num,
                        total_chunks=total_chunks_count,
                        role_name=role_name,
                        prompt=prompt,
                        architecture_model=architecture_model,
                        architecture_hash=architecture_hash,
                        langfuse_prompt_id=langfuse_prompt_id,
                        model_name=model_name,
                        project_path=project_path,
                        state=state,
                        config=config,
                        stream_callback=stream_callback,
                        primary_file=primary_file,
                        analysis_mode="chunked"
                    )

                    analysis_text = chunk_analysis["analysis_text"]
                    # For multi-chunk, we typically keep the trace ID of the LAST chunk or the synthesis trace ID.
                    # Here we capture it to ensure we have it if needed.
                    captured_trace_id = chunk_analysis["trace_id"]

                    # Store chunk result
                    chunk_results.append(
                        {
                            "chunk_index": chunk.chunk_index,
                            "chunk_num": chunk_num,
                            "total_chunks": total_chunks_count,
                            "files_in_chunk": chunk.total_files,
                            "tokens_in_chunk": chunk.total_tokens,
                            "analysis": analysis_text,
                        }
                    )

                # Synthesize chunk results if multiple chunks
                from .chunking import synthesize_chunk_results

                chunk_result_dicts = [
                    {
                        "chunk_num": cr["chunk_num"],
                        "chunk_index": cr["chunk_index"],
                        "analysis": cr["analysis"],
                        "role_name": role_name,
                    }
                    for cr in chunk_results
                ]

                synthesized_report = await synthesize_chunk_results(
                    chunk_results=chunk_result_dicts,
                    role_name=role_name,
                    prompt=prompt,
                    model_name=model_name,
                    architecture_model=architecture_model,
                    langfuse_prompt_id=langfuse_prompt_id,
                    config=config,
                )

                agent_result = {
                    "role_name": role_name,
                    "synthesized_report": synthesized_report,
                    "status": "completed",
                    "chunks_analyzed": len(chunks),
                    "chunk_results": chunk_results,
                }

                # Add tool calling metadata
                _enrich_result_with_tool_metadata(agent_result, state)

                logger.info(
                    f"Agent {role_name} completed inline processing of {len(chunks)} chunks"
                )

                if stream_callback:
                    stream_callback(
                        "agent_execution_complete",
                        {
                            "agent": role_name,
                            "agent_id": f"agent_{role_name.lower().replace(' ', '_')}",
                            "total_chunks": len(chunks),
                            "status": "completed",
                        },
                    )

                # Return agent result directly
                return {"agent_results_list": [agent_result]}

            elif len(chunks) == 1:
                # Single chunk - analyze directly (no need for parallelization)
                chunk = chunks[0]
                chunk_num = 1
                total_chunks = 1

                # Prepare chunk content
                chunk_content = prepare_chunk_content(chunk, files)

                # Process single chunk using helper
                chunk_analysis = await _analyze_chunk(
                    chunk_content=chunk_content,
                    chunk_num=1,
                    total_chunks=1,
                    role_name=role_name,
                    prompt=prompt,
                    architecture_model=architecture_model,
                    architecture_hash=architecture_hash,
                    langfuse_prompt_id=langfuse_prompt_id,
                    model_name=model_name,
                    project_path=project_path,
                    state=state,
                    config=config,
                    stream_callback=stream_callback,
                    primary_file=chunk.files[0].file_info.relative_path if chunk.files else "unknown",
                    analysis_mode="chunked"
                )

                analysis_text = chunk_analysis["analysis_text"]
                captured_trace_id = chunk_analysis["trace_id"]

                # Single chunk - use analysis directly
                synthesized_report = analysis_text

                agent_result = {
                    "role_name": role_name,
                    "synthesized_report": synthesized_report,
                    "status": "completed",
                    "chunks_analyzed": 1,
                    "chunk_results": [
                        {
                            "chunk_index": chunk.chunk_index,
                            "chunk_num": chunk_num,
                            "total_chunks": total_chunks,
                            "files_in_chunk": chunk.total_files,
                            "tokens_in_chunk": chunk.total_tokens,
                            "analysis": analysis_text,
                        }
                    ],
                }

                # Add tool calling metadata to result
                _enrich_result_with_tool_metadata(agent_result, state)

                # Phase 2: Automated Scoring
                try:
                    from utils.langfuse.evaluation import evaluate_prompt_effectiveness

                    # Evaluate and attach scores to the current trace
                    # We use the captured_trace_id if available, otherwise it falls back to current context
                    logger.info(f"Calculating automated scores for agent {role_name}")
                    evaluate_prompt_effectiveness(
                        trace_id=captured_trace_id,
                        role_name=role_name,
                        # Pass validation results if we had them (from a previous step? or calculate now?)
                        # For now we don't have per-agent validation steps inside this node
                        agent_result=agent_result,
                        prompt_id=langfuse_prompt_id,
                    )
                except Exception as score_error:
                    logger.warning(
                        f"Failed to perform automated scoring for {role_name}: {score_error}"
                    )
            else:
                # No chunks - error
                state[StateKeys.ERROR] = f"No chunks created for agent {role_name}"
                state[StateKeys.ERROR_STAGE] = "execute_single_agent"
                return state
        else:
            # Chunking disabled - send full content for all files
            logger.warning(
                f"Chunking is disabled for agent {role_name}. "
                "This may result in incomplete analysis for large projects. "
                "Consider enabling chunking for better results."
            )

            # Validate content exists
            valid_files = [f for f in files if f.get("content")]
            logger.debug(
                f"Agent {role_name}: {len(valid_files)} valid files out of {len(files) if files else 0} total"
            )

            if not valid_files:
                logger.error(f"No files with content found for agent {role_name}")
                # Return error result
                error_result = {
                    "role_name": role_name,
                    "error": "No files with content found",
                    "status": "error",
                }
                return {"agent_results_list": [error_result]}

            # Send full content for all files (not truncated, not limited to 10)
            file_samples = "\n\n".join(
                [
                    f"# File: {f.get('relative_path', 'unknown')}\n\n{f.get('content', '')}"
                    for f in valid_files
                ]
            )

            logger.info(
                f"Prepared {len(valid_files)} files for agent {role_name} (chunking disabled)"
            )

            # Prepare date context for the report
            current_date = datetime.now()
            date_context = f"""
## Report Date Context

IMPORTANT: Use the following date when generating your analysis report:
- Current Date: {current_date.strftime("%B %d, %Y")}
- Current Date (ISO): {current_date.strftime("%Y-%m-%d")}

When referencing dates in your report, use: {current_date.strftime("%B %d, %Y")}
"""

            # Check if tool calling is enabled and model supports it
            enable_tool_calling = state.get(StateKeys.ENABLE_TOOL_CALLING, True)
            max_tool_calls = state.get(StateKeys.MAX_TOOL_CALLS, 10)
            max_iterations = state.get(StateKeys.MAX_ITERATIONS, 15)
            max_duplicate_iterations = state.get(StateKeys.MAX_DUPLICATE_ITERATIONS, 3)
            llm_call_timeout = state.get(StateKeys.LLM_CALL_TIMEOUT, 60.0)
            tools_available = False
            tools = None

            if enable_tool_calling:
                tools_available = check_tool_calling_support(model_name)
                if tools_available:
                    tools = get_tool_definitions()
                    logger.info(
                        f"Tool calling enabled for agent {role_name} (model: {model_name})"
                    )
                else:
                    logger.info(
                        f"Tool calling disabled for agent {role_name} (model {model_name} doesn't support it)"
                    )

            # Add tools section to prompt if available
            tools_section = ""
            if tools_available:
                tools_section = """

## Available Tools

You have access to tools to read additional files if needed:
- **read_file(file_path)**: Read the content of any file in the project by its relative path (e.g., 'utils/experience_storage.py')
- **list_directory(directory_path)**: List files in a directory to explore the project structure
- **get_file_info(file_path)**: Get metadata about a file (size, line count, encoding)
"""

            analysis_prompt = f"""{prompt}{tools_section}

Project Path: {project_path}
Files to Analyze:
{file_samples}

{date_context}

Analyze the codebase according to your role and provide a comprehensive analysis report.
"""

            # Calculate actual input tokens that will be sent to the API
            # This includes system prompt + user prompt (which includes file content)
            system_prompt_content = build_agent_system_prompt(
                role_name, architecture_hash, architecture_model
            )

            # Count tokens for both system and user prompts
            from chunking.token_counter import TokenCounter

            token_counter = TokenCounter(model_name=model_name)
            system_tokens = token_counter.count_tokens(system_prompt_content)
            user_tokens = token_counter.count_tokens(analysis_prompt)
            total_input_tokens = system_tokens + user_tokens

            logger.debug(
                f"Agent {role_name}: Input tokens - system: {system_tokens:,}, "
                f"user: {user_tokens:,}, total: {total_input_tokens:,}"
            )

            # Calculate safe max_tokens based on available context
            # max_tokens = context_window - input_tokens - safety_margin
            output_max_tokens = None
            try:
                from llm.config import ConfigManager

                config_manager = ConfigManager()
                model_config = config_manager.get_model_config(model_name)
                if model_config and model_config.max_input_tokens:
                    # Reserve 10% of context as safety margin
                    safety_margin = model_config.max_input_tokens // 10
                    available_for_output = (
                        model_config.max_input_tokens
                        - total_input_tokens
                        - safety_margin
                    )

                    # Ensure we have at least some room for output (min 4K), but cap at reasonable max (32K)
                    output_max_tokens = max(4000, min(available_for_output, 32000))

                    logger.debug(
                        f"Agent {role_name}: Context window: {model_config.max_input_tokens:,}, "
                        f"input: {total_input_tokens:,}, safety: {safety_margin:,}, "
                        f"available for output: {available_for_output:,}, max_tokens: {output_max_tokens:,}"
                    )
                else:
                    # Fallback to safe default if model config not found
                    output_max_tokens = 16000
            except Exception as e:
                logger.debug(f"Error calculating max_tokens: {e}")
                output_max_tokens = 16000  # Safe default

            # Use llm_node for agent analysis
            llm_state = {
                "messages": [
                    {"role": "system", "content": system_prompt_content},
                    {"role": "user", "content": analysis_prompt},
                ],
                "model": model_name,
                "temperature": 0.7,
                "metadata": {
                    "role": role_name,
                    "langfuse_prompt_id": langfuse_prompt_id,  # Always include, even if None
                },
            }

            # Set max_tokens based on available context
            # This ensures input_tokens + max_tokens never exceeds context_window
            if output_max_tokens and output_max_tokens > 0:
                llm_state["max_tokens"] = output_max_tokens
            else:
                logger.warning(
                    f"Agent {role_name}: Calculated max_tokens is {output_max_tokens}, "
                    "input may be too large. Consider reducing max_tokens_per_chunk."
                )

            # Add tools if available
            if tools_available and tools:
                llm_state["tools"] = tools
                llm_state["tool_choice"] = "auto"

            # Call llm_node with tool calling support
            if tools_available and tools:
                llm_result = await _execute_with_tool_calling(
                    llm_state=llm_state,
                    project_path=project_path,
                    state=state,
                    config=config,
                    max_tool_calls=max_tool_calls,
                    max_iterations=max_iterations,
                    max_duplicate_iterations=max_duplicate_iterations,
                    llm_call_timeout=llm_call_timeout,
                )
            else:
                llm_result = await llm_node(llm_state, config=config)

            # Extract trace ID from llm_result metadata
            llm_metadata = llm_result.get("metadata", {})
            if isinstance(llm_metadata, dict):
                extracted_trace_id = llm_metadata.get("langfuse_trace_id")
                if extracted_trace_id:
                    captured_trace_id = extracted_trace_id
                    logger.debug(
                        f"Captured trace ID for {role_name} (no chunking): {captured_trace_id}"
                    )

            # Extract analysis result
            analysis_result = llm_result.get("last_response", "")

            agent_result = {
                "role_name": role_name,
                "synthesized_report": analysis_result,
                "status": "completed",
                "chunks_analyzed": 1,
            }

            # Add tool calling metadata to result
            _enrich_result_with_tool_metadata(agent_result, state)

        if stream_callback:
            stream_callback(
                "agent_chunk_complete",
                {
                    "agent": role_name,
                    "agent_id": f"agent_{role_name.lower().replace(' ', '_')}",
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "file": chunk.files[0].file_info.relative_path if 'chunk' in locals() and chunk.files else "single chunk",
                    "findings": 0,
                    "status": "completed",
                },
            )
            stream_callback(
                "agent_execution_complete",
                {
                    "agent": role_name,
                    "agent_id": f"agent_{role_name.lower().replace(' ', '_')}",
                    "total_chunks": 1,
                    "status": "completed",
                },
            )

        # Return ONLY the fields that should be updated (agent_results_list and trace_ids)
        # Do NOT return the entire state copy - that causes concurrent update errors
        # The reducer (operator.add) will aggregate agent_results_list from all parallel nodes
        logger.info(f"Completed agent execution: {role_name}")

        return_dict = {StateKeys.AGENT_RESULTS_LIST: [agent_result]}

        # Store trace ID if captured (from either single-chunk or no-chunking path)
        if captured_trace_id:
            return_dict[StateKeys.TRACE_IDS] = {role_name: captured_trace_id}
            logger.debug(f"Stored trace ID for {role_name}: {captured_trace_id}")

        return return_dict

    except Exception as e:
        error_context = create_error_context(
            "execute_single_agent",
            state,
            {
                "role_name": role_name,
                "file_count": len(files) if files else 0,
                "enable_chunking": enable_chunking,
            },
        )
        error_msg = (
            f"Failed to execute agent '{role_name}': {str(e)}. "
            f"Processing {len(files) if files else 0} files with chunking {'enabled' if enable_chunking else 'disabled'}. "
            f"Check that files are valid and the LLM model is accessible."
        )
        logger.error(error_msg, extra=error_context, exc_info=True)
        # Return error result in agent_results_list
        # Only return the fields that should be updated, not the entire state
        error_result = {
            "role_name": role_name,
            "error": error_msg,
            "status": "error",
            "error_context": error_context,
        }
        return {"agent_results_list": [error_result]}


@validate_state_fields([], "collect_agent_results")  # agent_results_list may be empty
def collect_agent_results_node(state: SwarmAnalysisState) -> SwarmAnalysisState:
    """
    Collect and aggregate results from parallel agent executions.

    Converts agent_results_list (populated by parallel nodes) into agent_results Dict format.
    Calculates total chunks_analyzed and validates all agents completed.

    Args:
        state: Current swarm analysis state with agent_results_list

    Returns:
        Updated state with agent_results Dict and aggregated metadata
    """
    agent_results_list: List[Dict[str, Any]] = state.get(StateKeys.AGENT_RESULTS_LIST, [])
    agents: List[Dict[str, Any]] = state.get(StateKeys.AGENTS, [])

    if not agent_results_list:
        # No results collected - check if we have agents
        if agents:
            error_msg = (
                f"No agent results collected from parallel execution. "
                f"Expected results from {len(agents)} agents. "
                f"Check that agents executed successfully and results were properly collected."
            )
            state[StateKeys.ERROR] = error_msg
            state[StateKeys.ERROR_STAGE] = "collect_agent_results"
            logger.error(error_msg)
            return state
        else:
            # No agents to execute - this is fine
            updated_state = state.copy()
            updated_state[StateKeys.AGENT_RESULTS] = {}
            updated_state[StateKeys.AGENT_STATUS] = {}
            updated_state[StateKeys.CHUNKS_ANALYZED] = 0
            updated_state[StateKeys.INDIVIDUAL_AGENT_RESULTS] = {}
            return updated_state

    from workflows.swarm_analysis.utils import get_stream_callback
    stream_callback = get_stream_callback(state)

    try:
        # Convert list to Dict format
        agent_results: Dict[str, Dict[str, Any]] = {}
        agent_status: Dict[str, str] = {}
        total_chunks: int = 0
        successful_agents: int = 0
        failed_agents: int = 0

        for result in agent_results_list:
            role_name = result.get("role_name", "Unknown")

            if result.get("status") == "error":
                # Error case
                agent_results[role_name] = {
                    "error": result.get("error", "Unknown error"),
                    "status": "error",
                }
                agent_status[role_name] = "error"
                failed_agents += 1
            else:
                # Success case
                agent_results[role_name] = {
                    "synthesized_report": result.get("synthesized_report", ""),
                    "status": "completed",
                    "chunks_analyzed": result.get("chunks_analyzed", 0),
                }
                if "chunk_results" in result:
                    agent_results[role_name]["chunk_results"] = result["chunk_results"]

                agent_status[role_name] = "completed"
                total_chunks += result.get("chunks_analyzed", 0)
                successful_agents += 1

        # Validate all agents completed
        expected_agent_count = len(agents) if agents else len(agent_results_list)
        actual_agent_count = len(agent_results_list)

        if actual_agent_count < expected_agent_count:
            logger.warning(
                f"Only {actual_agent_count} of {expected_agent_count} agents completed. "
                f"Some agents may have failed silently."
            )

        # Set workflow-level error only if all agents failed
        if failed_agents > 0 and successful_agents == 0:
            logger.error("All agents failed during execution")
            # Don't set error here - let synthesize_results_node handle it
            # But log the issue

        # Update state
        updated_state = state.copy()
        updated_state[StateKeys.AGENT_RESULTS] = agent_results
        updated_state[StateKeys.AGENT_STATUS] = agent_status
        updated_state[StateKeys.CHUNKS_ANALYZED] = total_chunks
        updated_state[StateKeys.INDIVIDUAL_AGENT_RESULTS] = agent_results

        # Ensure trace_ids are properly merged (in case reducer didn't handle it)
        # trace_ids should already be in state from parallel executions, but verify
        existing_trace_ids = state.get(StateKeys.TRACE_IDS, {})
        if existing_trace_ids:
            # trace_ids should already be merged by LangGraph reducer, but log for debugging
            logger.debug(
                f"trace_ids in state after agent execution: {existing_trace_ids}"
            )
        else:
            logger.debug("No trace_ids found in state after agent execution")

        if stream_callback:
            stream_callback(
                "agent_execution_complete",
                {
                    "message": "All agents completed",
                    "successful": successful_agents,
                    "failed": failed_agents,
                    "total_chunks": total_chunks,
                },
            )

        logger.info(
            f"Collected results from {actual_agent_count} agents: "
            f"{successful_agents} successful, {failed_agents} failed, "
            f"{total_chunks} total chunks analyzed"
        )

        return updated_state

    except Exception as e:
        error_context = create_error_context(
            "collect_agent_results",
            state,
            {
                "agent_results_count": len(agent_results_list),
                "expected_agent_count": len(agents) if agents else 0,
            },
        )
        error_msg = (
            f"Failed to collect agent results: {str(e)}. "
            f"Received {len(agent_results_list)} results from {len(agents) if agents else 0} expected agents. "
            f"Check that all agents completed successfully."
        )
        logger.error(error_msg, extra=error_context, exc_info=True)
        updated_state = state.copy()
        updated_state[StateKeys.ERROR] = error_msg
        updated_state[StateKeys.ERROR_STAGE] = "collect_agent_results"
        updated_state[StateKeys.ERROR_CONTEXT] = error_context
        if stream_callback:
            stream_callback("swarm_analysis_error", {"error": error_msg})
        return updated_state
