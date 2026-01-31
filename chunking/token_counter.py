"""
Token counting utilities using LiteLLM with tiktoken fallback.

Uses LiteLLM's token_counter as the primary method for accurate token counting
across multiple models (including Qwen, Claude, etc.), with tiktoken as a fallback
for backward compatibility and reliability.
"""

import logging
from typing import Optional

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    litellm = None
    LITELLM_AVAILABLE = False
    logging.debug("LiteLLM not installed. Will use tiktoken for token counting.")

try:
    import tiktoken
except ImportError:
    tiktoken = None
    logging.warning("tiktoken not installed. Token counting will not work.")

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counter using LiteLLM with tiktoken fallback for accurate token estimation.
    
    Uses LiteLLM's token_counter as the primary method when a model_name is provided,
    which supports a wide range of models including Qwen, Claude, Anthropic, Cohere,
    Llama2, Llama3, and OpenAI models. Falls back to tiktoken for backward compatibility
    and reliability.
    
    Fallback chain: LiteLLM → tiktoken → character estimate
    
    Supports different encodings based on model type when using tiktoken fallback.
    """
    
    # Model to encoding mapping
    # Default to cl100k_base (used by GPT-4, GPT-3.5-turbo, etc.)
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "o1-preview": "cl100k_base",
        "o1-mini": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "code-davinci-002": "p50k_base",
    }
    
    def __init__(self, model_name: Optional[str] = None, encoding_name: Optional[str] = None):
        """
        Initialize token counter.
        
        Args:
            model_name: Name of the model (used for LiteLLM token counting and tiktoken encoding selection)
            encoding_name: Explicit encoding name (overrides model_name for tiktoken only)
        
        Raises:
            ImportError: If tiktoken is not installed (required for fallback)
            ValueError: If encoding_name is invalid or model_name is unknown
        """
        # Store model_name for LiteLLM usage
        self.model_name = model_name
        
        # Validate that at least tiktoken is available for fallback
        if tiktoken is None:
            raise ImportError("tiktoken is required for token counting. Install with: pip install tiktoken")
        
        # Determine encoding for tiktoken fallback
        if encoding_name is not None:
            # Validate encoding_name is a valid tiktoken encoding
            if not isinstance(encoding_name, str) or not encoding_name.strip():
                raise ValueError("encoding_name must be a non-empty string")
            
            # Check if encoding exists in tiktoken registry
            valid_encodings = tiktoken.list_encoding_names()
            if encoding_name not in valid_encodings:
                raise ValueError(
                    f"Invalid encoding: '{encoding_name}'. "
                    f"Valid encodings: {', '.join(sorted(valid_encodings))}"
                )
            
            self.encoding_name = encoding_name
            logger.info(f"TokenCounter initialized with explicit encoding '{self.encoding_name}'")
        elif model_name is not None:
            # Validate model_name type and format
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name must be a non-empty string")
            
            # Check if model is in mapping
            if model_name in self.MODEL_ENCODINGS:
                self.encoding_name = self.MODEL_ENCODINGS[model_name]
                logger.info(
                    f"TokenCounter initialized with model '{model_name}' -> encoding '{self.encoding_name}'"
                )
            else:
                # Unknown model - fall back to default encoding
                # This allows custom models (like Qwen, Claude, etc.) to work
                # Most modern models use cl100k_base or similar encodings
                self.encoding_name = "cl100k_base"
                
                # Only warn if LiteLLM is not available (LiteLLM will handle unknown models)
                if not LITELLM_AVAILABLE:
                    supported_models = sorted(self.MODEL_ENCODINGS.keys())
                    logger.warning(
                        f"Unknown model name: '{model_name}'. "
                        f"Falling back to default encoding 'cl100k_base'. "
                        f"Known models with specific encodings: {', '.join(supported_models)}. "
                        f"If token counts are inaccurate, specify encoding_name explicitly. "
                        f"Consider installing LiteLLM for better model support."
                    )
                else:
                    logger.debug(
                        f"Unknown model name: '{model_name}' not in tiktoken mapping. "
                        f"Will use LiteLLM token_counter (tiktoken 'cl100k_base' as fallback)."
                    )
        else:
            # Only default when both model_name and encoding_name are None
            self.encoding_name = "cl100k_base"
            logger.info(
                "TokenCounter initialized with default encoding 'cl100k_base'. "
                "Specify model_name or encoding_name to use a different encoding."
            )
        
        # Get tiktoken encoding for fallback
        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
        except Exception as e:
            logger.warning(
                f"Could not load encoding '{self.encoding_name}': {e}. "
                "Falling back to 'cl100k_base'. This may cause incorrect token counts."
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.encoding_name = "cl100k_base"
        
        # Log which method will be used
        if LITELLM_AVAILABLE and self.model_name:
            logger.debug(
                f"TokenCounter initialized with model '{self.model_name}'. "
                f"Will use LiteLLM token_counter (tiktoken '{self.encoding_name}' as fallback)."
            )
        else:
            logger.debug(
                f"TokenCounter initialized with tiktoken encoding: {self.encoding_name}. "
                f"LiteLLM {'not available' if not LITELLM_AVAILABLE else 'will not be used (no model_name)'}."
            )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Uses LiteLLM's token_counter as primary method when model_name is provided,
        falls back to tiktoken, then to character-based estimation if all else fails.
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        # Try LiteLLM first if available and model_name is provided
        if LITELLM_AVAILABLE and self.model_name:
            try:
                token_count = litellm.token_counter(model=self.model_name, text=text)
                logger.debug(f"Token count via LiteLLM for model '{self.model_name}': {token_count}")
                return token_count
            except Exception as e:
                # LiteLLM failed, fall back to tiktoken
                logger.debug(
                    f"LiteLLM token_counter failed for model '{self.model_name}': {e}. "
                    "Falling back to tiktoken."
                )
        
        # Fall back to tiktoken (existing logic)
        try:
            return len(self.encoding.encode(text))
        except (UnicodeDecodeError, UnicodeError) as e:
            # Non-UTF-8 or invalid Unicode content detected
            sample = repr(text[:50]) if len(text) > 50 else repr(text)
            logger.warning(
                f"Non-UTF-8 content detected (sample: {sample}); "
                f"falling back to char-based estimate (inaccurate). "
                f"Original error: {e}"
            )
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
        except Exception as e:
            # Other unexpected errors (preserve existing behavior for non-Unicode errors)
            logger.error(f"Error counting tokens: {e}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4
    
    def count_tokens_estimate(self, text: str) -> int:
        """
        Fast token estimation (less accurate but faster).
        
        Uses character-based estimation: 1 token ≈ 4 characters.
        
        Args:
            text: Text to estimate tokens for
        
        Returns:
            Estimated number of tokens
        """
        if not text:
            return 0
        return len(text) // 4


def estimate_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Convenience function to estimate tokens.
    
    Args:
        text: Text to count tokens for
        model_name: Optional model name for encoding selection
    
    Returns:
        Number of tokens
    """
    counter = TokenCounter(model_name=model_name)
    return counter.count_tokens(text)

