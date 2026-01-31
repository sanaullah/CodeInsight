"""
Graph builder utilities for creating LangGraph workflows.

This module provides factory functions and utilities for building
LangGraph workflows with integration to LiteLLM and Langfuse.
"""

from typing import Dict, Any, Optional, Callable, List
import logging
from llm.config import ConfigManager, ensure_env_ready
from .checkpoints import setup_checkpoints
from .integration import setup_langfuse_callbacks

logger = logging.getLogger(__name__)

# Note: Callbacks are now passed directly in config when invoking graphs
# LangGraph automatically propagates them to nodes via run_manager
# No global store or threading locks needed


class GraphBuilder:
    """
    Builder class for creating LangGraph workflows.
    
    This class provides a convenient interface for building graphs
    with automatic integration to LiteLLM and Langfuse.
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load_config()
        
        # Ensure LiteLLM is ready
        ensure_env_ready()
        
        # Configure Langfuse for LiteLLM auto-tracking
        try:
            from utils import configure_litellm_for_langfuse
            configure_litellm_for_langfuse()
        except Exception as e:
            logger.warning(f"Failed to configure Langfuse for LiteLLM: {e}")
        
        # Setup checkpoints if enabled
        self.checkpoint_adapter = setup_checkpoints()
        
        # Note: Langfuse callbacks are now passed directly in config when invoking graphs
        # No need to store them here - they're passed at invocation time
    
    def create_graph(
        self,
        nodes: Dict[str, Callable],
        edges: List[tuple],
        entry_point: str = "start",
        state_schema: Optional[type] = None,
        trace_name: Optional[str] = None
    ) -> Any:
        """
        Create a compiled LangGraph from nodes and edges with Langfuse integration.
        
        Args:
            nodes: Dictionary mapping node names to node functions
            edges: List of tuples (from_node, to_node) or (from_node, condition_func, to_node)
            entry_point: Name of the entry node
            state_schema: Optional state schema (TypedDict)
            trace_name: Optional name for Langfuse trace
        
        Returns:
            Compiled LangGraph instance that automatically uses Langfuse callbacks when invoked
        """
        try:
            # Import from the actual LangGraph library
            # No path manipulation needed since we're now in workflows/ directory
            try:
                from langgraph.graph import StateGraph, END
            except ImportError:
                # If that fails, try importing from langgraph_core
                try:
                    from langgraph_core.graph import StateGraph, END
                except ImportError:
                    raise ImportError(
                        "LangGraph not installed. Install with: pip install langgraph>=0.2.0"
                    )
            
            # Create state graph
            if state_schema:
                graph = StateGraph(state_schema)
            else:
                from .state import StateSchema
                graph = StateGraph(StateSchema)
            
            # Add nodes (no wrapping needed - callbacks are injected via config)
            for node_name, node_func in nodes.items():
                graph.add_node(node_name, node_func)
            
            # Add edges
            graph.set_entry_point(entry_point)
            
            for edge in edges:
                if len(edge) == 2:
                    # Simple edge: (from, to)
                    from_node, to_node = edge
                    if to_node == "END":
                        graph.add_edge(from_node, END)
                    else:
                        graph.add_edge(from_node, to_node)
                elif len(edge) == 3:
                    # Conditional edge: (from, condition, to)
                    from_node, condition_func, to_node = edge
                    graph.add_conditional_edges(
                        from_node,
                        condition_func,
                        {to_node: to_node} if to_node != "END" else {END: END}
                    )
            
            # Compile graph with checkpoints if enabled
            if self.checkpoint_adapter:
                compiled_graph = graph.compile(checkpointer=self.checkpoint_adapter)
            else:
                compiled_graph = graph.compile()
            
            # Return compiled graph directly - callbacks are passed in config when invoking
            logger.info(f"✅ Graph compiled with {len(nodes)} nodes")
            logger.info("✅ Langfuse callbacks should be passed in config when invoking graph")
            return compiled_graph
            
        except ImportError:
            raise ImportError(
                "LangGraph not installed. Install with: pip install langgraph>=0.2.0"
            )
        except Exception as e:
            logger.error(f"Error creating graph: {e}", exc_info=True)
            raise


def create_graph(
    nodes: Dict[str, Callable],
    edges: List[tuple],
    entry_point: str = "start",
    state_schema: Optional[type] = None,
    use_checkpoints: Optional[bool] = None,
    use_langfuse: Optional[bool] = None,
    trace_name: Optional[str] = None
) -> Any:
    """
    Convenience function to create a compiled LangGraph with Langfuse integration.
    
    Args:
        nodes: Dictionary mapping node names to node functions
        edges: List of tuples (from_node, to_node) or (from_node, condition_func, to_node)
        entry_point: Name of the entry node
        state_schema: Optional state schema (TypedDict)
        use_checkpoints: Whether to use checkpoints (uses config if None)
        use_langfuse: Whether to use Langfuse (uses config if None)
        trace_name: Optional name for Langfuse trace
        
    Returns:
        Compiled LangGraph instance with Langfuse callbacks integrated
    
    Example:
        def node1(state):
            return {"result": "done"}
        
        graph = create_graph(
            nodes={"node1": node1},
            edges=[("start", "node1"), ("node1", "END")],
            entry_point="start",
            trace_name="my_workflow"
        )
        
        # Execute graph - automatically traced in Langfuse
        result = graph.invoke({"input": "test"})
    """
    builder = GraphBuilder()
    return builder.create_graph(nodes, edges, entry_point, state_schema, trace_name)

