"""
Configuration management for the LLM package.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    name: str
    max_input_tokens: int
    max_output_tokens: Optional[int] = None
    input_price_per_1k: Optional[float] = None
    output_price_per_1k: Optional[float] = None


class RetryConfig(BaseModel):
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    burst_size: int = 10


class PromptTrackingConfig(BaseModel):
    """Prompt tracking configuration for Langfuse."""
    enabled: bool = False  # Default: disabled to avoid breaking existing observability
    store_in_langfuse: bool = False  # Store prompts in Langfuse
    store_in_postgresql: bool = True  # Store prompts in PostgreSQL (always enabled as fallback)


class LangfuseConfig(BaseModel):
    """Langfuse observability configuration."""
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"
    flush_interval: int = 10
    timeout: float = 5.0
    enabled_for_litellm: bool = True
    enabled_for_langchain: bool = True
    enabled_for_custom_client: bool = True
    default_tags: List[str] = Field(default_factory=list)
    prompt_tracking: PromptTrackingConfig = Field(default_factory=PromptTrackingConfig)  # Prompt tracking config


class CheckpointConfig(BaseModel):
    """Checkpoint persistence configuration for workflows."""
    enabled: bool = False
    type: str = "memory"  # "memory" or "sqlite"
    path: str = "checkpoints.db"  # Path for SQLite checkpoints


class HumanInLoopConfig(BaseModel):
    """Human-in-the-loop configuration for workflows."""
    enabled: bool = False
    timeout: float = 300.0  # Timeout in seconds
    approval_required: bool = True
    default_action: str = "reject"  # "approve" or "reject"


class StreamingConfig(BaseModel):
    """Streaming configuration for workflows."""
    enabled: bool = False
    stream_mode: str = "values"  # "values", "updates", or "debug"
    include_events: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    enabled: bool = True
    log_dir: str = "logs"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    use_json_format: bool = False  # JSON for files (false = readable format)
    date_based_files: bool = True  # app_YYYY-MM-DD.log vs app.log
    rotation: Dict[str, Any] = Field(default_factory=lambda: {
        "max_bytes": 10485760,  # 10MB default
        "backup_count": 5,
        "retention_days": 30,
        "enable_time_rotation": True
    })
    module_loggers: Optional[Dict[str, Any]] = Field(default_factory=lambda: {"enabled": True, "modules": ["langfuse", "workflows"]})


class WorkflowsConfig(BaseModel):
    """Workflows configuration."""
    enabled: bool = True
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    human_in_loop: HumanInLoopConfig = Field(default_factory=HumanInLoopConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)


class LLMConfig(BaseModel):
    """Main configuration for the LLM client."""
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    default_model: str = "gpt-3.5-turbo"
    timeout: float = 30.0
    max_retries: int = 3
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    rate_limit_config: RateLimitConfig = Field(default_factory=RateLimitConfig)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)
    fetch_models_on_init: bool = False
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    workflows: WorkflowsConfig = Field(default_factory=WorkflowsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    role_validation: Optional[Dict[str, Any]] = Field(default=None)


class ConfigManager:
    """
    Manages configuration loading and validation.
    
    This class reads configuration from config.yaml files, not from .env files.
    The .env file in langfuse/ directory is only for Docker Compose to run the Langfuse server.
    """
    
    # Built-in model registry with Deepseek, GLM, Qwen, Kimi models
    # Sorted by context window size (ascending)
    DEFAULT_MODELS = {
        "deepseek-coder-6.7b": ModelConfig(
            name="deepseek-coder-6.7b",
            max_input_tokens=16384,
            max_output_tokens=4096
        ),
        "deepseek-coder-33b": ModelConfig(
            name="deepseek-coder-33b", 
            max_input_tokens=16384,
            max_output_tokens=4096
        ),
        "glm-4-9b": ModelConfig(
            name="glm-4-9b",
            max_input_tokens=32768,
            max_output_tokens=8192
        ),
        "qwen2.5-7b": ModelConfig(
            name="qwen2.5-7b",
            max_input_tokens=32768,
            max_output_tokens=8192
        ),
        "qwen2.5-14b": ModelConfig(
            name="qwen2.5-14b",
            max_input_tokens=32768,
            max_output_tokens=8192
        ),
        "qwen2.5-32b": ModelConfig(
            name="qwen2.5-32b",
            max_input_tokens=32768,
            max_output_tokens=8192
        ),
        "kimi-8b": ModelConfig(
            name="kimi-8b",
            max_input_tokens=128000,
            max_output_tokens=16384
        ),
        "deepseek-v2.5": ModelConfig(
            name="deepseek-v2.5",
            max_input_tokens=128000,
            max_output_tokens=16384
        ),
        "qwen2.5-72b": ModelConfig(
            name="qwen2.5-72b",
            max_input_tokens=128000,
            max_output_tokens=16384
        ),
        "glm-4-9b-1m": ModelConfig(
            name="glm-4-9b-1m",
            max_input_tokens=1000000,
            max_output_tokens=16384
        ),
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.yaml"
        self._config: Optional[LLMConfig] = None
    
    def load_config(self) -> LLMConfig:
        """Load configuration from YAML file with fallback to defaults."""
        if self._config is not None:
            return self._config
            
        config_data = self._load_yaml_config()
        
        # Merge with defaults
        default_config = LLMConfig()
        default_config.models = self.DEFAULT_MODELS.copy()
        
        if config_data:
            # Update with loaded config
            for key, value in config_data.items():
                if hasattr(default_config, key):
                    if key == "models" and isinstance(value, dict):
                        # Merge model configs
                        for model_name, model_data in value.items():
                            if isinstance(model_data, dict):
                                default_config.models[model_name] = ModelConfig(**model_data)
                    elif key == "retry_config" and isinstance(value, dict):
                        default_config.retry_config = RetryConfig(**value)
                    elif key == "rate_limit_config" and isinstance(value, dict):
                        default_config.rate_limit_config = RateLimitConfig(**value)
                    elif key == "langfuse" and isinstance(value, dict):
                        default_config.langfuse = LangfuseConfig(**value)
                    elif key == "workflows" and isinstance(value, dict):
                        # Handle nested workflows config
                        workflows_data = value.copy()
                        if "checkpoint" in workflows_data and isinstance(workflows_data["checkpoint"], dict):
                            workflows_data["checkpoint"] = CheckpointConfig(**workflows_data["checkpoint"])
                        if "human_in_loop" in workflows_data and isinstance(workflows_data["human_in_loop"], dict):
                            workflows_data["human_in_loop"] = HumanInLoopConfig(**workflows_data["human_in_loop"])
                        if "streaming" in workflows_data and isinstance(workflows_data["streaming"], dict):
                            workflows_data["streaming"] = StreamingConfig(**workflows_data["streaming"])
                        default_config.workflows = WorkflowsConfig(**workflows_data)
                    elif key == "logging" and isinstance(value, dict):
                        default_config.logging = LoggingConfig(**value)
                    elif key == "role_validation" and isinstance(value, dict):
                        # Store role_validation config as dict (no Pydantic model needed for now)
                        default_config.role_validation = value
                    elif key == "prompt_validation" and isinstance(value, dict):
                        # Store prompt_validation config as dict (no Pydantic model needed for now)
                        default_config.prompt_validation = value
                    else:
                        setattr(default_config, key, value)
        
        self._config = default_config
        
        # Override with environment variables
        self._apply_env_overrides()
        
        return self._config
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to the configuration."""
        if not self._config:
            return
            
        # API Configuration
        if os.environ.get("OPENAI_API_BASE"):
            self._config.api_base = os.environ["OPENAI_API_BASE"]
            
        if os.environ.get("OPENAI_API_KEY"):
            self._config.api_key = os.environ["OPENAI_API_KEY"]
            
        if os.environ.get("DEFAULT_MODEL"):
            self._config.default_model = os.environ["DEFAULT_MODEL"]
            
        # Langfuse Configuration
        if os.environ.get("LANGFUSE_PUBLIC_KEY"):
            self._config.langfuse.public_key = os.environ["LANGFUSE_PUBLIC_KEY"]
            
        if os.environ.get("LANGFUSE_SECRET_KEY"):
            self._config.langfuse.secret_key = os.environ["LANGFUSE_SECRET_KEY"]
            
        if os.environ.get("LANGFUSE_HOST"):
            self._config.langfuse.host = os.environ["LANGFUSE_HOST"]
    
    def _load_yaml_config(self) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            return None
            
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_path}: {e}")
            return None
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        config = self.load_config()
        return config.models.get(model_name)
    
    def validate_model_tokens(self, model_name: str, input_tokens: int) -> bool:
        """Validate if input tokens are within model limits."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            return True  # Allow if model not in registry
        
        return input_tokens <= model_config.max_input_tokens
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        config = self.load_config()
        return list(config.models.keys())


def ensure_env_ready() -> None:
    """
    Ensure LiteLLM environment is properly configured.
    Sets environment variables for LiteLLM compatibility.
    
    Note: This function reads from config.yaml (via ConfigManager), not from .env files.
    The .env file in langfuse/ directory is only for Docker Compose, not for this framework.
    """
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Set environment variables for LiteLLM
    if config.api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = config.api_key
    
    if config.api_base and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = config.api_base
    
    # Set LiteLLM log level to DEBUG for better diagnostics
    if not os.environ.get("LITELLM_LOG"):
        os.environ["LITELLM_LOG"] = "DEBUG"
    
    # Setup logging if configured
    if hasattr(config, 'logging') and config.logging.enabled:
        try:
            from utils.logging_config import setup_logging_from_config
            setup_logging_from_config()
        except Exception as e:
            # Don't fail if logging setup fails
            print(f"Warning: Failed to setup logging: {e}")
    
    print(f"LiteLLM environment ready - API Base: {config.api_base[:50]}..., Model: {config.default_model}")
