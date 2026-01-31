# Workflows Module

This module provides comprehensive LangGraph support for building stateful, multi-step agent workflows with full integration to LiteLLM and Langfuse observability.

## Quick Start

### Basic Usage

```python
from workflows import create_graph, llm_node
from llm.config import ensure_env_ready
from utils import configure_litellm_for_langfuse

# Setup
ensure_env_ready()
configure_litellm_for_langfuse()

# Define nodes
def process_node(state):
    return {"messages": [{"role": "user", "content": state["input"]}]}

# Create graph
graph = create_graph(
    nodes={
        "process": process_node,
        "llm": llm_node,
    },
    edges=[
        ("process", "llm"),
        ("llm", "END"),
    ],
    entry_point="process"
)

# Execute
result = graph.invoke({"input": "Hello!"})
print(result["last_response"])
```

## Features

### 1. Graph Building

The `GraphBuilder` class and `create_graph()` function provide easy graph creation:

```python
from workflows import create_graph

graph = create_graph(
    nodes={"node1": func1, "node2": func2},
    edges=[("node1", "node2"), ("node2", "END")],
    entry_point="node1"
)
```

### 2. State Management

Define custom state schemas using TypedDict:

```python
from typing import TypedDict
from workflows.state import StateSchema

class MyState(StateSchema):
    user_input: str
    result: str
    step_count: int

# Use in graph
graph = create_graph(..., state_schema=MyState)
```

### 3. LLM Integration

Use `llm_node` for LLM calls that automatically use your LiteLLM configuration:

```python
from workflows import llm_node

def my_node(state):
    return {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "qwen/qwen3-coder"  # Optional, uses default from config
    }

# llm_node will call LiteLLM and add response to state
```

### 4. Checkpoint Persistence

Enable checkpoints in `config.yaml`:

```yaml
workflows:
  checkpoint:
    enabled: true
    type: "sqlite"  # or "memory"
    path: "checkpoints.db"
```

Checkpoints allow you to:
- Resume interrupted workflows
- Track state across executions
- Implement human-in-the-loop with state persistence

### 5. Human-in-the-Loop

Add approval and feedback nodes:

```python
from workflows import human_approval_node, human_feedback_node

approval = human_approval_node(
    timeout=60.0,
    default_action="reject"
)

feedback = human_feedback_node(
    feedback_prompt="Please provide feedback:"
)
```

Enable in `config.yaml`:

```yaml
workflows:
  human_in_loop:
    enabled: true
    timeout: 300.0
    approval_required: true
    default_action: "reject"
```

### 6. Stream Callback Registry

The callback registry enables checkpoint serialization by storing callback IDs instead of callable objects in state.

**Usage**:

```python
from workflows.callback_registry import get_callback_registry

# Get singleton registry
registry = get_callback_registry()

# Register a callback
def my_callback(event_type, data):
    print(f"{event_type}: {data}")

callback_id = registry.register(my_callback)

# Store ID in state (not the callable)
state = {
    "stream_callback_id": callback_id,
    # ... other state fields
}

# Retrieve callback later
callback = registry.get(callback_id)
if callback:
    callback("test_event", {"key": "value"})

# Unregister when done
registry.unregister(callback_id)
```

**Features**:
- Thread-safe operations
- TTL-based expiration (default 1 hour)
- Automatic cleanup of expired callbacks
- Singleton pattern for global access

**Location**: `workflows/callback_registry.py`

### 7. Streaming

Stream graph execution in real-time:

```python
from workflows.streaming import stream_graph_sync, format_stream_event

for event in stream_graph_sync(graph, initial_state, stream_mode="updates"):
    print(format_stream_event(event))
```

Enable in `config.yaml`:

```yaml
workflows:
  streaming:
    enabled: true
    stream_mode: "updates"  # "values", "updates", or "debug"
    include_events: true
```

### 7. Langfuse Observability

LangGraph execution is automatically tracked in Langfuse when enabled:

- Graph runs create traces
- Node executions create spans
- State transitions are logged
- Errors are tracked

No additional code needed - just enable Langfuse in `config.yaml`.

## Configuration

All workflow settings are in `config.yaml`:

```yaml
workflows:
  enabled: true
  checkpoint:
    enabled: false
    type: "memory"  # "memory" or "sqlite"
    path: "checkpoints.db"
  human_in_loop:
    enabled: false
    timeout: 300.0
    approval_required: true
    default_action: "reject"
  streaming:
    enabled: false
    stream_mode: "values"
    include_events: true
```

## Examples

See the `examples/` directory for complete examples:

- `langgraph_basic.py` - Simple linear workflow
- `langgraph_multi_step.py` - Multi-step workflow with state
- `langgraph_conditional.py` - Conditional routing
- `langgraph_human_loop.py` - Human-in-the-loop approval
- `langgraph_streaming.py` - Streaming execution

## Node Types

### LLM Nodes

```python
from workflows import llm_node

# Automatically uses LiteLLM and config settings
state = llm_node({
    "messages": [{"role": "user", "content": "Hello"}],
    "model": "qwen/qwen3-coder"  # Optional
})
```

### Tool Nodes

```python
from workflows import tool_node

def my_tool(state):
    result = do_something(state["input"])
    return {"result": result}

tool_node_func = tool_node(my_tool)
```

### Conditional Nodes

```python
from workflows import conditional_node

def route(state):
    if state.get("condition"):
        return "path_a"
    return "path_b"

route_func = conditional_node(route)
```

## Advanced Usage

### Custom State Schemas

```python
from typing import TypedDict
from typing_extensions import NotRequired

class WorkflowState(TypedDict):
    input: str
    step: int
    result: NotRequired[str]
    error: NotRequired[str]

graph = create_graph(
    ...,
    state_schema=WorkflowState
)
```

### Manual Checkpoint Management

```python
from workflows.checkpoints import get_checkpoint_adapter

adapter = get_checkpoint_adapter("sqlite", "my_checkpoints.db")
graph = graph.compile(checkpointer=adapter)

# Resume from checkpoint
result = graph.invoke(
    {"input": "continue"},
    config={"configurable": {"thread_id": "thread-123"}}
)
```

### Custom Langfuse Callbacks

```python
from workflows.integration import LangGraphLangfuseCallback

callback = LangGraphLangfuseCallback(trace_name="my_workflow")
callback.on_graph_start(input_state)
# ... execute graph ...
callback.on_graph_end(final_state)
```

## Integration with Framework

Workflows integrate seamlessly with the framework:

- **LiteLLM**: All LLM calls use your configured endpoint and models
- **Langfuse**: Automatic observability tracking
- **Config**: All settings in `config.yaml`
- **State**: TypedDict-based state management

## Best Practices

1. **Start Simple**: Begin with basic linear graphs, add complexity gradually
2. **Use State Schemas**: Define TypedDict schemas for type safety
3. **Enable Checkpoints**: Use checkpoints for production workflows
4. **Monitor with Langfuse**: Always enable Langfuse for observability
5. **Handle Errors**: Always include error handling in nodes
6. **Test Nodes**: Test individual nodes before building graphs

## Troubleshooting

### Graph Not Executing

- Check that all nodes are defined in the `nodes` dict
- Verify edges reference existing nodes
- Ensure entry point is set correctly

### LLM Calls Failing

- Verify LiteLLM is configured: `ensure_env_ready()`
- Check `config.yaml` has correct API settings
- Ensure model name is correct

### Checkpoints Not Working

- Verify checkpoint is enabled in `config.yaml`
- Check file permissions for SQLite checkpoints
- Ensure checkpoint adapter is passed to `compile()`

### Langfuse Not Tracking

- Verify Langfuse is enabled in `config.yaml`
- Check Langfuse credentials are correct
- Ensure Langfuse server is running

## API Reference

See the individual module files for detailed API documentation:

- `graph_builder.py` - Graph creation utilities
- `nodes.py` - Node implementations
- `state.py` - State schema utilities
- `checkpoints.py` - Checkpoint persistence
- `streaming.py` - Streaming utilities
- `callback_registry.py` - Stream callback registry for checkpoint serialization
- `human_in_loop.py` - Human-in-the-loop nodes
- `integration.py` - Langfuse integration

