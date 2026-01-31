# LLM Framework - LiteLLM + Langfuse Integration

A minimal, reusable framework for using LiteLLM with Langfuse observability. This framework provides configuration management and automatic Langfuse tracking for LiteLLM calls.

## Quick Start - Copying to New Project

To use this framework in a new project, copy:
- `llm/` folder (exclude `__pycache__/`)
- `utils/` folder (exclude `__pycache__/`)
- `config.yaml` (update with your settings)
- `requirements.txt` (install dependencies)

See `docs/SETUP_GUIDE.md` for detailed instructions.

## Overview

This framework simplifies the integration of LiteLLM with Langfuse by:
- Managing configuration via YAML files
- Automatically setting up LiteLLM environment variables
- Enabling Langfuse auto-tracking for all LiteLLM calls
- Providing utilities for manual trace creation

## Installation

Install the required dependencies:

```bash
pip install litellm>=1.0.0 langfuse>=2.0.0 pydantic>=2.0.0 pyyaml>=6.0
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Configuration

**Important**: This framework uses `config.yaml` as the single source of truth for all configuration. It does NOT use `.env` files.

The `.env` file in the `langfuse/` directory is only for Docker Compose to run the Langfuse server itself, not for the framework.

Create a `config.yaml` file in your project root:

```yaml
# LLM Configuration
api_base: "https://nano-gpt.com/api/v1"  # Your OpenAI-compatible endpoint
api_key: "your-api-key-here"             # Your API key
default_model: "qwen/qwen3-coder"        # Default model to use
timeout: 30.0                             # Request timeout in seconds
max_retries: 3                            # Maximum number of retries

# Langfuse Observability Configuration
langfuse:
  enabled: true                           # Enable/disable Langfuse tracking
  public_key: "pk-lf-..."                 # Langfuse public key
  secret_key: "sk-lf-..."                 # Langfuse secret key
  host: "http://localhost:3000"           # Langfuse server URL
  flush_interval: 10                      # Flush interval in seconds (0 = immediate)
  timeout: 5.0                            # Request timeout in seconds
  enabled_for_litellm: true              # Enable LiteLLM auto-tracking
  enabled_for_langchain: true            # Enable LangChain callback tracking
  default_tags: []                       # Default tags for traces
```

## Basic Usage

### Step 1: Setup

```python
from llm.config import ConfigManager, ensure_env_ready
from utils import configure_litellm_for_langfuse

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Setup LiteLLM environment (sets OPENAI_API_KEY, OPENAI_API_BASE, etc.)
ensure_env_ready()

# Configure Langfuse auto-tracking for LiteLLM
configure_litellm_for_langfuse()
```

### Step 2: Use LiteLLM

```python
import litellm

# For custom OpenAI-compatible endpoints, use the custom_openai/ prefix
response = litellm.completion(
    model="custom_openai/qwen/qwen3-coder",  # Use custom_openai/ prefix
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ],
    temperature=0.7,
    max_tokens=100
)

# Access response
content = response.choices[0].message.content
print(content)
```

## Complete Example

```python
"""
Complete example showing LiteLLM + Langfuse integration.
"""

from llm.config import ConfigManager, ensure_env_ready
from utils import configure_litellm_for_langfuse
import litellm


def main():
    # Step 1: Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Step 2: Setup LiteLLM environment
    ensure_env_ready()
    
    # Step 3: Configure Langfuse auto-tracking
    configure_litellm_for_langfuse()
    
    # Step 4: Make LLM calls
    # Format model name with custom_openai prefix for custom endpoints
    model_name = config.default_model
    if not model_name.startswith("custom_openai/"):
        model_name = f"custom_openai/{model_name}"
    
    response = litellm.completion(
        model=model_name,
        messages=[
            {"role": "user", "content": "Tell me a joke"}
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    print(response.choices[0].message.content)
    
    # Check Langfuse dashboard for automatic trace


if __name__ == "__main__":
    main()
```

## Advanced Usage

### Manual Langfuse Traces

Create custom traces for tracking specific operations:

```python
from utils import create_langfuse_trace

# Create a trace
trace = create_langfuse_trace(
    name="custom_operation",
    user_id="user123",
    session_id="session456",
    metadata={"operation": "data_processing"},
    tags=["custom", "processing"]
)

# Your code here...

# Trace is automatically sent to Langfuse
```

### Accessing Langfuse Client Directly

```python
from utils import get_langfuse_client

client = get_langfuse_client()
if client:
    # Use Langfuse client directly
    trace = client.trace(name="my_trace")
    span = trace.span(name="my_span")
    span.end()
    client.flush()
```

### Streaming Responses

```python
import litellm

response = litellm.completion(
    model="custom_openai/qwen/qwen3-coder",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Function Calling

```python
import litellm

response = litellm.completion(
    model="custom_openai/qwen/qwen3-coder",
    messages=[{"role": "user", "content": "What's the weather in New York?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    tool_choice="auto"
)
```

## API Reference

### ConfigManager

Manages configuration loading from YAML files.

```python
from llm.config import ConfigManager

config_manager = ConfigManager(config_path="config.yaml")  # Optional: custom path
config = config_manager.load_config()

# Access configuration
print(config.api_base)
print(config.api_key)
print(config.default_model)
print(config.langfuse.enabled)
```

### ensure_env_ready()

Sets up LiteLLM environment variables from configuration.

```python
from llm.config import ensure_env_ready

ensure_env_ready()
# Sets:
# - OPENAI_API_KEY
# - OPENAI_API_BASE
# - LITELLM_LOG
```

### configure_litellm_for_langfuse()

Configures LiteLLM to automatically send traces to Langfuse.

```python
from utils import configure_litellm_for_langfuse

configure_litellm_for_langfuse()
# Sets:
# - LANGFUSE_PUBLIC_KEY
# - LANGFUSE_SECRET_KEY
# - LANGFUSE_HOST
# - litellm.success_callback = ["langfuse"]
# - litellm.failure_callback = ["langfuse"]
```

### get_langfuse_client()

Get or create a Langfuse client instance (singleton).

```python
from utils import get_langfuse_client

client = get_langfuse_client()
# Returns None if Langfuse is disabled or not configured
```

### create_langfuse_trace()

Create a manual Langfuse trace.

```python
from utils import create_langfuse_trace

trace = create_langfuse_trace(
    name="operation_name",
    user_id="user123",
    session_id="session456",
    metadata={"key": "value"},
    tags=["tag1", "tag2"]
)
```

## Model Name Format

For custom OpenAI-compatible endpoints, use the `custom_openai/` prefix:

- ✅ `custom_openai/qwen/qwen3-coder`
- ✅ `custom_openai/Qwen/Qwen3-Next-80B-A3B-Instruct`
- ❌ `qwen/qwen3-coder` (without prefix, may not work with custom endpoints)

## Langfuse Dashboard

After making LLM calls, check your Langfuse dashboard at the configured host (default: `http://localhost:3000`) to see:
- Automatic traces from LiteLLM calls
- Token usage and costs
- Latency metrics
- Error tracking

## Troubleshooting

### Langfuse not tracking calls

1. Check that `langfuse.enabled: true` in config.yaml
2. Verify `langfuse.enabled_for_litellm: true`
3. Ensure Langfuse credentials are correct
4. Check that `configure_litellm_for_langfuse()` was called before making LLM calls

### Model not found errors

- Ensure you're using the `custom_openai/` prefix for custom endpoints
- Verify your API endpoint is accessible
- Check that the model name matches what your endpoint expects

### Configuration not loading

- Ensure `config.yaml` exists in the project root
- Check file permissions
- Verify YAML syntax is correct

## Project Structure

```
Framework/
├── llm/
│   ├── __init__.py              # Exports ConfigManager, ensure_env_ready
│   ├── config.py                 # Configuration management
│   └── README.md                 # This file
├── utils/
│   ├── __init__.py               # Exports Langfuse utilities
│   └── langfuse_integration.py   # Langfuse integration
├── config.yaml                   # Configuration file
├── requirements.txt              # Dependencies
└── examples/example.py           # Usage example
```

## Migration from Custom Client

If migrating from the previous custom client implementation:

1. Replace `from llm import OpenAI` with `import litellm`
2. Replace `client.create_chat_completion()` with `litellm.completion()`
3. Add `ensure_env_ready()` and `configure_litellm_for_langfuse()` setup
4. Update model names to use `custom_openai/` prefix
5. Remove custom retry/rate limiting code (LiteLLM handles this)

## Best Practices

1. **Call setup functions once** at application startup:
   ```python
   ensure_env_ready()
   configure_litellm_for_langfuse()
   ```

2. **Use config.yaml** as the single source of truth for configuration

3. **Check Langfuse dashboard regularly** to monitor usage and costs

4. **Handle errors gracefully** - LiteLLM will retry automatically, but you should still handle exceptions

5. **Use appropriate model names** - Always use `custom_openai/` prefix for custom endpoints

## License

MIT License

## Support

For issues and questions:
- Check the examples/example.py file for usage patterns
- Review the configuration in config.yaml
- Consult LiteLLM documentation: https://docs.litellm.ai
- Consult Langfuse documentation: https://langfuse.com/docs
