"""
Error formatting utilities for user-friendly error messages.

Provides functions to convert technical exceptions to user-friendly messages
while preserving technical details in logs.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import exception types
from services.storage.base_storage import (
    StorageError,
    StorageNotFoundError,
    StorageValidationError,
    StorageConnectionError,
    DatabaseConnectionError,
    ConfigurationError,
    UserFacingError,
    AuthenticationError
)


def format_user_error(exception: Exception) -> str:
    """
    Convert technical exceptions to user-friendly messages.
    
    Maps specific exception types to appropriate user-facing messages
    while hiding technical implementation details.
    
    Args:
        exception: The exception to format
        
    Returns:
        User-friendly error message string
    """
    # Check if exception is already user-facing
    if isinstance(exception, UserFacingError):
        return exception.user_message
    
    # Map specific exception types to user-friendly messages
    if isinstance(exception, (StorageConnectionError, DatabaseConnectionError)):
        return "Unable to connect to analysis database. Please try again later."
    
    if isinstance(exception, StorageNotFoundError):
        return "The requested information could not be found."
    
    if isinstance(exception, StorageValidationError):
        return "Invalid data provided. Please check your input."
    
    if isinstance(exception, ConfigurationError):
        return "System configuration error - please contact support"
    
    if isinstance(exception, AuthenticationError):
        return "Your session has expired. Please log in again."
    
    # Generic fallback for unknown exceptions
    return "An unexpected error occurred. Please try again or contact support."


def is_user_facing(exception: Exception) -> bool:
    """
    Check if an exception is safe to show to users.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if exception is user-facing, False otherwise
    """
    return isinstance(exception, (
        UserFacingError,
        AuthenticationError,
        ConfigurationError,
        StorageNotFoundError,
        StorageValidationError,
        StorageConnectionError,
        DatabaseConnectionError
    ))


def log_technical_error(exception: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log technical error details with context.
    
    Logs the full exception with traceback and any additional context
    for debugging purposes. This should be called before displaying
    user-friendly error messages.
    
    Args:
        exception: The exception to log
        context: Optional dictionary with additional context information
    """
    error_msg = f"Technical error: {type(exception).__name__}: {str(exception)}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        error_msg = f"{error_msg} | Context: {context_str}"
    
    logger.error(error_msg, exc_info=True)

