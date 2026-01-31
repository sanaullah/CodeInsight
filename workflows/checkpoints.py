"""
Checkpoint persistence for LangGraph workflows.

This module provides utilities for setting up checkpoint adapters
for state persistence and graph resumption.
"""

from typing import Optional, Any
import logging
from llm.config import ConfigManager

logger = logging.getLogger(__name__)


def get_checkpoint_adapter(
    checkpoint_type: str = "memory",
    path: Optional[str] = None
) -> Any:
    """
    Get a checkpoint adapter for LangGraph.
    
    Args:
        checkpoint_type: Type of checkpoint ("memory" or "sqlite")
        path: Path for SQLite checkpoints (required if type is "sqlite")
        
    Returns:
        Checkpoint adapter instance
        
    Raises:
        ValueError: If invalid checkpoint type or missing path for SQLite
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        raise ImportError(
            "LangGraph checkpoint modules not found. "
            "Install with: pip install langgraph>=0.2.0"
        )
    
    if checkpoint_type == "memory":
        return MemorySaver()
    elif checkpoint_type == "sqlite":
        if not path:
            raise ValueError("Path is required for SQLite checkpoints")
        return SqliteSaver.from_conn_string(path)
    else:
        raise ValueError(f"Invalid checkpoint type: {checkpoint_type}. Use 'memory' or 'sqlite'")


def setup_checkpoints(
    checkpoint_config: Optional[Any] = None
) -> Optional[Any]:
    """
    Setup checkpoints based on configuration.
    
    Args:
        checkpoint_config: CheckpointConfig from LLMConfig (optional)
                          If None, loads from ConfigManager
        
    Returns:
        Checkpoint adapter if enabled, None otherwise
    """
    if checkpoint_config is None:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        checkpoint_config = config.workflows.checkpoint
    
    if not checkpoint_config.enabled:
        logger.debug("Checkpoints disabled in configuration")
        return None
    
    try:
        adapter = get_checkpoint_adapter(
            checkpoint_type=checkpoint_config.type,
            path=checkpoint_config.path if checkpoint_config.type == "sqlite" else None
        )
        logger.info(f"✅ Checkpoints enabled: {checkpoint_config.type}")
        return adapter
    except Exception as e:
        logger.error(f"Error setting up checkpoints: {e}", exc_info=True)
        return None

