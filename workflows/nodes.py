"""
Reusable node functions for LangGraph workflows.

This module provides common node implementations including LLM nodes,
tool-calling nodes, and conditional routing nodes.
"""

from typing import Dict, Any, Callable, Optional, List
import logging

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from llm.factory import get_llm
from llm.config import ConfigManager
from workflows.state_keys import StateKeys
from workflows.swarm_analysis_langfuse import get_node_phase

logger = logging.getLogger(__name__)


async def llm_node(state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    LLM node that uses the centralized ChatLiteLLM factory.
    
    Expects state to have:
    - messages: List of message dictionaries or LangChain messages
    - model: Optional model name (uses default from config if not provided)
    
    Returns updated state with:
    - messages: Updated with LLM response
    - last_response: The LLM response content
    
    Args:
        state: Current graph state
        config: Optional LangGraph config (contains callbacks from graph invocation)
        
    Returns:
        Updated state with LLM response
    """
    try:
        config_manager = ConfigManager()
        llm_config = config_manager.load_config()
        
        # Get messages from state
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("State must contain 'messages' list")
        
        # Get model name (use default if not provided)
        model_name = state.get("model", llm_config.default_model)
        # Format model name for LiteLLM (openai prefix for custom endpoints)
        if not model_name.startswith("openai/"):
            model_name = f"openai/{model_name}"
        
        # Extract context from state for rich metadata
        # (This metadata helps debug traces but LangChain callback handles the main tracing)
        metadata = {
            "node_type": state.get("node_type", "llm_call"),
            "node_name": state.get("node_name") or state.get("node_type", "llm_call"),
            "phase": get_node_phase(state.get("node_name")),
            "project_path": state.get("project_path"),
            "goal": state.get("goal")
        }
        
        if state.get("role_name"):
            metadata["role_name"] = state.get("role_name")
            
        # Add metadata from architecture model if present
        arch_model = state.get("architecture_model", {})
        if arch_model:
            metadata["architecture_type"] = arch_model.get("system_type")
            metadata["architecture_pattern"] = arch_model.get("architecture_pattern")
            
        # Langfuse prompt linking
        state_metadata = state.get(StateKeys.METADATA, {})
        prompt_id = state_metadata.get("langfuse_prompt_id") or state.get("langfuse_prompt_id")  # Nested in metadata
        prompt_version = state_metadata.get("langfuse_prompt_version")
        if prompt_version is None:
            prompt_version = state.get(StateKeys.LANGFUSE_PROMPT_VERSIONS, {}).get(state.get(StateKeys.ROLE_NAME)) if state.get(StateKeys.ROLE_NAME) else None
            
        if prompt_id:
            metadata["langfuse_prompt_id"] = prompt_id
        if prompt_version is not None:
             metadata["langfuse_prompt_version"] = prompt_version
             
        # Add chunk info
        chunk_info = state.get(StateKeys.CHUNK_INFO, {})
        if chunk_info:
            metadata.update({
                "chunk_index": chunk_info.get("chunk_index"),
                "chunk_num": chunk_info.get("chunk_num"),
                "total_chunks": chunk_info.get("total_chunks")
            })

        # Convert messages to LangChain format if needed
        langchain_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
                
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                     langchain_messages.append(AIMessage(content=content, tool_calls=tool_calls or []))
                elif role == "tool":
                    # Tool messages need tool_call_id
                    langchain_messages.append(ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", "")))
                else:
                    langchain_messages.append(HumanMessage(content=content))
            else:
                # Already a LangChain message
                langchain_messages.append(msg)
                
        # Prepare tools if provided
        kwargs = {}
        tools = state.get("tools")  # Not in StateKeys - generic LLM node state
        if tools:
            # If tools are functions/pydantic models, ChatLiteLLM via LangChain handles bind_tools usually
            # But the previous implementation passed them directly.
            # ChatLiteLLM supports `bind_tools`.
            kwargs["tools"] = tools
            kwargs["tool_choice"] = state.get("tool_choice", "auto")  # Not in StateKeys - generic LLM node state

        # Get LLM instance
        # Pass config callbacks if available to ensure LangChain/Langfuse callbacks propagate
        callbacks = config.get("callbacks", []) if config else []
        
        llm = get_llm(
            model_name=model_name,
            temperature=state.get("temperature"),  # Not in StateKeys - generic LLM node state
            max_tokens=state.get("max_tokens"),  # Not in StateKeys - generic LLM node state
            callbacks=callbacks,
            metadata=metadata, # Pass metadata for tracing
        )
        
        # If tools are present, we should allow binding them.
        # ChatLiteLLM wraps .bind_tools().
        if "tools" in kwargs:
             llm = llm.bind_tools(kwargs["tools"], tool_choice=kwargs.get("tool_choice"))

        # Invoke LLM
        # Pass config to ainvoke is critical for callbacks to work in LangGraph
        logger.debug(f"Invoking LLM {model_name} with metadata: {metadata}")
        response = await llm.ainvoke(langchain_messages, config=config)
        
        # Extract content
        content = response.content if hasattr(response, 'content') else str(response)
        tool_calls = getattr(response, 'tool_calls', [])
        
        # Update state
        updated_state = state.copy()
        updated_state["messages"] = langchain_messages + [response]
        updated_state["last_response"] = content
        updated_state["tool_calls"] = tool_calls
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in LLM node: {e}", exc_info=True)
        updated_state = state.copy()
        updated_state["error"] = str(e)  # Generic error field, not SwarmAnalysisState
        return updated_state


def tool_node(tool_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable:
    """
    Create a tool node that wraps a tool function.
    
    Args:
        tool_func: Function that takes state and returns updated state
        
    Returns:
        Node function for use in LangGraph
    """
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = tool_func(state)
            updated_state = state.copy()
            updated_state.update(result)
            return updated_state
        except Exception as e:
            logger.error(f"Error in tool node: {e}", exc_info=True)
            updated_state = state.copy()
            updated_state["error"] = str(e)  # Generic error field, not SwarmAnalysisState
            return updated_state
    
    return node


def conditional_node(
    condition_func: Callable[[Dict[str, Any]], str]
) -> Callable:
    """
    Create a conditional routing node.
    
    Args:
        condition_func: Function that takes state and returns next node name
        
    Returns:
        Conditional routing function for use in LangGraph
    """
    def node(state: Dict[str, Any]) -> str:
        try:
            return condition_func(state)
        except Exception as e:
            logger.error(f"Error in conditional node: {e}", exc_info=True)
            return "error"
    
    return node
