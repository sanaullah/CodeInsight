"""
Langfuse prompt management utilities.
"""

import logging
from typing import Optional, Dict, Any, List
from .client import get_langfuse_client

logger = logging.getLogger(__name__)


def _get_prompt_id_from_database(prompt_name: str, fallback_name: Optional[str] = None) -> Optional[str]:
    """
    Query Langfuse PostgreSQL database directly to get prompt ID.
    
    This avoids the SDK's get_prompt() which tries to fetch with labels
    that may not exist (like "production").
    
    Args:
        prompt_name: The normalized prompt name
        fallback_name: Optional fallback name to try
        
    Returns:
        Prompt ID if found, None otherwise
    """
    try:
        import psycopg2
        from llm.config import ConfigManager
        
        # Get Langfuse database config - use defaults from docker-compose.yml
        # In production, these should come from environment variables
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "postgres"
        }
        
        conn = psycopg2.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                # Query for the most recent version of the prompt with this name
                cursor.execute("""
                    SELECT id, name, version
                    FROM prompts
                    WHERE name = %s
                    ORDER BY version DESC, created_at DESC
                    LIMIT 1
                """, (prompt_name,))
                
                result = cursor.fetchone()
                if result:
                    prompt_id = result[0]
                    logger.debug(f"Found prompt ID from database: {prompt_id} for name: {prompt_name}")
                    return prompt_id
                
                # Try fallback name if provided
                if fallback_name and fallback_name != prompt_name:
                    cursor.execute("""
                        SELECT id, name, version
                        FROM prompts
                        WHERE name = %s
                        ORDER BY version DESC, created_at DESC
                        LIMIT 1
                    """, (fallback_name,))
                    
                    result = cursor.fetchone()
                    if result:
                        prompt_id = result[0]
                        logger.debug(f"Found prompt ID from database: {prompt_id} for fallback name: {fallback_name}")
                        return prompt_id
                
                return None
        finally:
            conn.close()
            
    except ImportError:
        # psycopg2 not available - skip database query
        logger.debug("psycopg2 not available, skipping database query for prompt ID")
        return None
    except Exception as e:
        # Non-blocking: log at debug level
        logger.debug(f"Could not query database for prompt ID: {e}")
        return None


def _get_prompt_version_from_database(prompt_name: str) -> Optional[int]:
    """
    Query Langfuse PostgreSQL database directly to get prompt version.
    
    Args:
        prompt_name: The normalized prompt name
        
    Returns:
        Prompt version number if found, None otherwise
    """
    try:
        import psycopg2
        from llm.config import ConfigManager
        
        # Get Langfuse database config - use defaults from docker-compose.yml
        # In production, these should come from environment variables
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "postgres"
        }
        
        conn = psycopg2.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                # Query for the latest version of the prompt
                cursor.execute("""
                    SELECT version
                    FROM prompts
                    WHERE name = %s
                    ORDER BY version DESC, created_at DESC
                    LIMIT 1
                """, (prompt_name,))
                
                result = cursor.fetchone()
                if result:
                    prompt_version = result[0]
                    logger.debug(f"Found prompt version from database: {prompt_version} for name: {prompt_name}")
                    return prompt_version
                return None
        finally:
            conn.close()
            
    except ImportError:
        # psycopg2 not available - skip database query
        logger.debug("psycopg2 not available, skipping database query for prompt version")
        return None
    except Exception as e:
        # Non-blocking: log at debug level
        logger.debug(f"Could not query database for prompt version: {e}")
        return None


def normalize_label(label: str, max_length: int = 36) -> str:
    """
    Normalize label to Langfuse format with length limit.
    
    Langfuse labels must be:
    - lowercase alphanumeric with optional underscores, hyphens, or periods
    - Maximum length: 36 characters
    """
    if not label:
        return ""
    # Convert to lowercase
    normalized = label.lower()
    # Replace spaces and other invalid chars with underscores
    normalized = "".join(
        c if c.isalnum() or c in ['_', '-', '.'] else '_'
        for c in normalized
    )
    # Remove consecutive underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    # Truncate to max_length (36 chars for Langfuse)
    if len(normalized) > max_length:
        # Smart truncation: try to preserve meaningful parts
        # Truncate and remove trailing underscore if present
        normalized = normalized[:max_length].rstrip('_')
    return normalized


def create_readable_prompt_name(
    role: str,
    architecture_type: str,
    goal_category: str,
    max_length: int = 200
) -> str:
    """
    Create a human-readable prompt name using underscores.
    
    Preserves role name readability while staying within Langfuse's 200-character limit.
    Format: {role}_{architecture_type}_{goal_category}
    
    Args:
        role: Role name (e.g., "LangGraph Workflow Orchestration Analyst")
        architecture_type: Architecture type (e.g., "library", "web_app")
        goal_category: Goal category (e.g., "general", "security", "performance")
        max_length: Maximum length for the prompt name (default: 200)
        
    Returns:
        Readable prompt name with underscores
        
    Example:
        >>> create_readable_prompt_name(
        ...     "LangGraph Workflow Orchestration Analyst",
        ...     "library",
        ...     "general"
        ... )
        'LangGraph_Workflow_Orchestration_Analyst_library_general'
    """
    if not role:
        role = "unknown"
    if not architecture_type:
        architecture_type = "unknown"
    if not goal_category:
        goal_category = "general"
    
    # Sanitize role name: preserve readable format, convert spaces to underscores
    # Keep original case (don't lowercase) for readability
    sanitized_role = "".join(
        c if c.isalnum() or c in [' ', '-', '_'] else '_'
        for c in role
    )
    # Convert spaces to underscores and clean up
    sanitized_role = sanitized_role.replace(' ', '_')
    # Remove consecutive underscores
    while '__' in sanitized_role:
        sanitized_role = sanitized_role.replace('__', '_')
    sanitized_role = sanitized_role.strip('_')
    
    # Sanitize architecture type (lowercase, underscores)
    sanitized_arch = architecture_type.lower().replace(' ', '_')
    # Remove invalid chars
    sanitized_arch = "".join(
        c if c.isalnum() or c == '_' else '_'
        for c in sanitized_arch
    )
    while '__' in sanitized_arch:
        sanitized_arch = sanitized_arch.replace('__', '_')
    sanitized_arch = sanitized_arch.strip('_')
    
    # Sanitize goal category (lowercase, underscores)
    sanitized_goal = goal_category.lower().replace(' ', '_')
    sanitized_goal = "".join(
        c if c.isalnum() or c == '_' else '_'
        for c in sanitized_goal
    )
    while '__' in sanitized_goal:
        sanitized_goal = sanitized_goal.replace('__', '_')
    sanitized_goal = sanitized_goal.strip('_')
    
    # Build name with underscores
    prompt_name = f"{sanitized_role}_{sanitized_arch}_{sanitized_goal}"
    
    # Truncate if needed (but preserve structure)
    if len(prompt_name) > max_length:
        # Calculate available space for role (keep arch and goal)
        arch_goal_length = len(f"_{sanitized_arch}_{sanitized_goal}")
        max_role_length = max_length - arch_goal_length
        
        if max_role_length > 20:  # Ensure minimum role name length
            # Truncate role name, preserving word boundaries if possible
            truncated_role = sanitized_role[:max_role_length].rstrip('_')
            prompt_name = f"{truncated_role}_{sanitized_arch}_{sanitized_goal}"
        else:
            # Fallback: use normalized version if role is too long
            normalized_role = normalize_label(role)
            prompt_name = f"{normalized_role}_{sanitized_arch}_{sanitized_goal}"
    
    return prompt_name


def create_langfuse_prompt(
    name: str,
    prompt: str,
    role: str,
    architecture_type: str,
    goal: Optional[str] = None,
    validation_metadata: Optional[Dict[str, Any]] = None,
    labels: Optional[List[str]] = None,
    architecture_hash: Optional[str] = None,
    file_hash: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Create or update prompt in Langfuse.
    
    Uses consistent naming pattern for grouping similar prompts:
    {role}_{architecture_type}_{goal_category}
    
    Args:
        name: Prompt name (will be normalized to consistent pattern)
        prompt: Prompt content
        role: Role name
        architecture_type: Architecture type (e.g., "web_app", "api")
        goal: Optional analysis goal
        validation_metadata: Optional validation results metadata
        labels: Optional list of labels for grouping
        architecture_hash: Optional SHA256 hash of architecture model
        file_hash: Optional SHA256 hash of file structure
        
    Returns:
        Dictionary with 'id' and 'version' keys, or None if error/Langfuse unavailable
        Format: {"id": str, "version": int} or None
    """
    # Check if prompt tracking is enabled in config (default: disabled)
    try:
        from llm.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Check if Langfuse is enabled
        if not config.langfuse.enabled:
            logger.debug("Langfuse is disabled, skipping prompt creation")
            return None
        
        # Check prompt_tracking config (default: disabled to avoid breaking observability)
        prompt_tracking_enabled = False
        if hasattr(config.langfuse, 'prompt_tracking'):
            prompt_tracking_config = config.langfuse.prompt_tracking
            # Handle both Pydantic model and dict
            if hasattr(prompt_tracking_config, 'enabled'):
                prompt_tracking_enabled = prompt_tracking_config.enabled
            elif isinstance(prompt_tracking_config, dict):
                prompt_tracking_enabled = prompt_tracking_config.get('enabled', False)
        
        if not prompt_tracking_enabled:
            logger.debug("Langfuse prompt tracking is disabled in config (default: disabled)")
            return None
    except Exception as config_error:
        # If config check fails, default to disabled (safer - don't break observability)
        logger.debug(f"Could not check prompt tracking config: {config_error}, defaulting to disabled")
        return None
    
    client = get_langfuse_client()
    if client is None:
        logger.debug(f"Cannot create Langfuse prompt '{name}': client not initialized")
        return None
    
    try:
        # Normalize goal to category for consistent naming
        goal_category = "general"
        if goal:
            goal_lower = goal.lower()
            if "security" in goal_lower:
                goal_category = "security"
            elif "performance" in goal_lower:
                goal_category = "performance"
            elif "quality" in goal_lower or "review" in goal_lower:
                goal_category = "quality"
            else:
                goal_category = "custom"
        
        # Create consistent prompt name for grouping
        # Use readable prompt name that preserves role information
        prompt_name = create_readable_prompt_name(
            role=role,
            architecture_type=architecture_type,
            goal_category=goal_category,
            max_length=200
        )
        
        # Build labels list with normalized values
        prompt_labels = [normalize_label(l) for l in labels] if labels else []
        prompt_labels.extend([
            normalize_label(role),
            normalize_label(architecture_type),
            "dynamic",
            "llm_generated",
            "production"  # Default label for SDK compatibility (get_prompt() defaults to "production")
        ])
        # Remove duplicates and empty strings while preserving order
        prompt_labels = list(dict.fromkeys([l for l in prompt_labels if l]))
        
        # Build config/metadata
        prompt_config = {
            "role": role,
            "architecture_type": architecture_type,
            "goal_category": goal_category,
            "original_name": name
        }
        if goal:
            prompt_config["goal"] = goal
        if validation_metadata:
            prompt_config["validation"] = validation_metadata
        if architecture_hash:
            prompt_config["architecture_hash"] = architecture_hash
            logger.debug(f"Adding architecture_hash to prompt config: {architecture_hash[:20]}...")
        else:
            logger.warning(f"architecture_hash is None/empty for prompt '{prompt_name}'")
        if file_hash:
            prompt_config["file_hash"] = file_hash
            logger.debug(f"Adding file_hash to prompt config: {file_hash[:20]}...")
        else:
            logger.warning(f"file_hash is None/empty for prompt '{prompt_name}'")
        
        logger.debug(f"Final prompt_config keys: {list(prompt_config.keys())}")
        logger.debug(f"prompt_config with hashes: architecture_hash={'present' if 'architecture_hash' in prompt_config else 'MISSING'}, file_hash={'present' if 'file_hash' in prompt_config else 'MISSING'}")
        
        # Create prompt using Langfuse SDK
        # Langfuse Python SDK uses create_prompt() with type="text" parameter
        try:
            # Use the correct Langfuse Python SDK API format
            # Based on Langfuse docs: client.create_prompt(name="...", type="text", prompt="...", labels=[...])
            if hasattr(client, 'create_prompt'):
                # Log what we're about to send to Langfuse
                logger.info(
                    f"Creating Langfuse prompt '{prompt_name}' with config keys: {list(prompt_config.keys())}"
                )
                
                # Use create_prompt with explicit type="text" parameter
                langfuse_prompt = client.create_prompt(
                    name=prompt_name,
                    type="text",  # Explicitly specify text prompt type (required)
                    prompt=prompt,  # String, not array
                    labels=prompt_labels,  # Array of normalized labels (max 36 chars each)
                    config=prompt_config  # This should include architecture_hash and file_hash if provided
                )
            else:
                # If create_prompt not available, log and return None
                logger.debug(f"Langfuse create_prompt() method not available")
                return None
            
            # Extract prompt ID and version
            prompt_id = None
            prompt_version = None
            
            # Check if it's a TextPromptClient (Langfuse v3 pattern)
            if hasattr(langfuse_prompt, '__class__') and 'TextPromptClient' in str(type(langfuse_prompt)):
                # Try to get prompt ID from TextPromptClient
                # TextPromptClient might have the ID in different places
                if hasattr(langfuse_prompt, 'id'):
                    prompt_id = langfuse_prompt.id
                elif hasattr(langfuse_prompt, 'prompt_id'):
                    prompt_id = langfuse_prompt.prompt_id
                elif hasattr(langfuse_prompt, 'name'):
                    # Fallback: Query database directly to get ID (avoids SDK label issues)
                    # The SDK's get_prompt() tries to fetch with "production" label which may not exist
                    prompt_id = _get_prompt_id_from_database(prompt_name, langfuse_prompt.name)
                    if not prompt_id:
                        # If database query fails, log at debug level (non-blocking)
                        logger.debug(f"Could not extract prompt ID for '{langfuse_prompt.name}' - prompt was created but ID not immediately available")
                
                # Try to get version from TextPromptClient
                if hasattr(langfuse_prompt, 'version'):
                    prompt_version = langfuse_prompt.version
                elif hasattr(langfuse_prompt, '__dict__'):
                    prompt_version = langfuse_prompt.__dict__.get('version')
            elif hasattr(langfuse_prompt, 'id'):
                prompt_id = langfuse_prompt.id
                if hasattr(langfuse_prompt, 'version'):
                    prompt_version = langfuse_prompt.version
            elif isinstance(langfuse_prompt, dict):
                prompt_id = langfuse_prompt.get('id')
                prompt_version = langfuse_prompt.get('version')
            elif isinstance(langfuse_prompt, str):
                prompt_id = langfuse_prompt
            else:
                # Last resort: Query database directly to get ID
                prompt_id = _get_prompt_id_from_database(prompt_name)
                if not prompt_id:
                    logger.debug(f"Could not extract prompt ID for '{prompt_name}' - prompt was created but ID not immediately available")
            
            # If version not available, try to get from database
            if not prompt_version and prompt_id:
                prompt_version = _get_prompt_version_from_database(prompt_name)
            
            if not prompt_id:
                # Enhanced logging for debugging
                available_attrs = 'N/A'
                if hasattr(langfuse_prompt, '__dict__'):
                    available_attrs = [attr for attr in dir(langfuse_prompt) if not attr.startswith('_')]
                elif hasattr(langfuse_prompt, '__dir__'):
                    try:
                        available_attrs = [attr for attr in dir(langfuse_prompt) if not attr.startswith('_')]
                    except Exception:
                        available_attrs = 'Could not list attributes'
                
                logger.warning(
                    f"Could not extract prompt ID from Langfuse response: {type(langfuse_prompt)}. "
                    f"Available attributes: {available_attrs}"
                )
                return None
            
            logger.info(f"Created Langfuse prompt '{prompt_name}' (id: {prompt_id}, version: {prompt_version})")
            return {
                "id": str(prompt_id),
                "version": prompt_version
            }
            
        except AttributeError as attr_error:
            # API method not available
            logger.debug(f"Langfuse prompt creation API not available: {attr_error}")
            logger.debug(f"Prompt will be stored in PostgreSQL only")
            return None
        except Exception as api_error:
            # Isolate prompt creation errors - don't let them affect observability
            # Log at WARNING level (not ERROR) to avoid alarming users
            error_msg = str(api_error)
            # Check if it's a validation error (label length, type, etc.)
            if "status_code" in error_msg or "400" in error_msg or "Bad request" in error_msg:
                logger.warning(f"Langfuse prompt creation validation error: {api_error}")
            else:
                logger.warning(f"Error creating Langfuse prompt via API: {api_error}")
            # Non-blocking: return None but don't raise exception
            # This ensures prompt creation errors don't affect trace/observation creation
            return None
            
    except Exception as e:
        # Isolate prompt creation errors - don't let them affect observability
        # Log at WARNING level (not ERROR) to avoid alarming users
        # This ensures prompt creation failures don't break trace/observation creation
        error_msg = str(e)
        if "status_code" in error_msg or "400" in error_msg or "Bad request" in error_msg:
            logger.warning(f"Langfuse prompt creation validation error (non-blocking): {e}")
        else:
            logger.warning(f"Error creating Langfuse prompt (non-blocking): {e}")
        # Return None instead of raising - ensures workflow continues
        # Prompt will still be stored in PostgreSQL as fallback
        return None


def get_langfuse_prompt_by_name(
    name: str,
    label: Optional[str] = None,
    include_version: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get prompt from Langfuse by name with optional label filter.
    
    Args:
        name: Prompt name
        label: Optional label to filter by (e.g., "production", "staging")
        include_version: Whether to include version information
        
    Returns:
        Prompt dictionary with id, name, prompt, version, labels, etc., or None if not found
    """
    client = get_langfuse_client()
    if client is None:
        logger.debug(f"Cannot get Langfuse prompt '{name}': client not initialized")
        return None
    
    try:
        # Try to get prompt using Langfuse SDK
        # Note: This depends on Langfuse API availability
        try:
            if hasattr(client, 'get_prompt'):
                # Try to get prompt by name (without label first)
                fetched_prompt = client.get_prompt(name)
                if fetched_prompt:
                    # Convert to dict if needed
                    if hasattr(fetched_prompt, '__dict__'):
                        prompt_dict = {
                            'id': getattr(fetched_prompt, 'id', None),
                            'name': getattr(fetched_prompt, 'name', name),
                            'prompt': getattr(fetched_prompt, 'prompt', None),
                            'labels': getattr(fetched_prompt, 'labels', []),
                        }
                        # Add version if requested
                        if include_version:
                            prompt_dict['version'] = getattr(fetched_prompt, 'version', None)
                            # If version not available from object, try database
                            if prompt_dict['version'] is None:
                                prompt_dict['version'] = _get_prompt_version_from_database(name)
                        return prompt_dict
                    elif isinstance(fetched_prompt, dict):
                        if include_version and 'version' not in fetched_prompt:
                            # Try to get version from database if not in dict
                            fetched_prompt['version'] = _get_prompt_version_from_database(name)
                        return fetched_prompt
                    else:
                        return None
                return None
            else:
                # Langfuse v3 might have different methods
                logger.debug(f"Langfuse get_prompt() method not available")
                return None
        except Exception as e:
            # Handle 404 errors gracefully - prompts might not exist yet
            error_str = str(e)
            if "404" in error_str or "not found" in error_str.lower() or "LangfuseNotFoundError" in error_str:
                # This is normal for new prompts - don't log as error
                logger.debug(f"Prompt '{name}' not found in Langfuse (this is normal for new prompts)")
            else:
                logger.debug(f"Error retrieving Langfuse prompt '{name}': {e}")
            return None
            
    except Exception as e:
        # Handle unexpected errors
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.debug(f"Prompt '{name}' not found in Langfuse")
        else:
            logger.debug(f"Error getting Langfuse prompt '{name}': {e}")
        return None


def get_langfuse_prompt_by_id(prompt_id: str) -> Optional[Any]:
    """
    Get Langfuse prompt object by ID.
    
    Langfuse SDK's get_prompt() requires a name, not an ID.
    This function queries the database to get the prompt name,
    then fetches the prompt object using the SDK.
    
    Args:
        prompt_id: Langfuse prompt ID (UUID string)
        
    Returns:
        Langfuse prompt object (TextPromptClient or similar) or None if not found
    """
    if not prompt_id:
        return None
    
    client = get_langfuse_client()
    if client is None:
        logger.debug(f"Cannot get Langfuse prompt by ID '{prompt_id}': client not initialized")
        return None
    
    try:
        # Query database to get prompt name from ID
        import psycopg2
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "postgres"
        }
        
        conn = psycopg2.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                # Query for prompt name and labels by ID
                # Langfuse stores labels in a JSONB column or separate table
                cursor.execute("""
                    SELECT name, labels
                    FROM prompts
                    WHERE id = %s
                    LIMIT 1
                """, (prompt_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.debug(f"Prompt ID '{prompt_id}' not found in database")
                    return None
                
                prompt_name = result[0]
                prompt_labels = result[1] if len(result) > 1 else None
                
                # Parse labels if it's a JSON/JSONB column
                if prompt_labels:
                    if isinstance(prompt_labels, str):
                        try:
                            import json as json_lib
                            prompt_labels = json_lib.loads(prompt_labels)
                        except:
                            prompt_labels = None
                    elif not isinstance(prompt_labels, list):
                        prompt_labels = None
                
                logger.debug(f"Found prompt name '{prompt_name}' for ID '{prompt_id}' with labels: {prompt_labels}")
        finally:
            conn.close()
        
        # Use SDK to get prompt object by name
        # Try different approaches: with labels, without label, or direct database query
        if hasattr(client, 'get_prompt'):
            # Try with each label from database, or without label
            labels_to_try = []
            if prompt_labels and isinstance(prompt_labels, list):
                labels_to_try = prompt_labels
            # Also try None (no label) and common defaults
            labels_to_try.extend([None, "production", "staging"])
            # Remove duplicates while preserving order
            labels_to_try = list(dict.fromkeys(labels_to_try))
            
            for label in labels_to_try:
                try:
                    # Try get_prompt with label parameter if label is not None
                    if label is not None:
                        # Check if get_prompt accepts label parameter
                        import inspect
                        sig = inspect.signature(client.get_prompt)
                        if 'label' in sig.parameters:
                            prompt_obj = client.get_prompt(prompt_name, label=label)
                        else:
                            # If label parameter not supported, try without
                            prompt_obj = client.get_prompt(prompt_name)
                    else:
                        # Try without label
                        prompt_obj = client.get_prompt(prompt_name)
                    
                    if prompt_obj:
                        logger.debug(f"Successfully fetched prompt object for ID '{prompt_id}' (name: '{prompt_name}', label: {label})")
                        return prompt_obj
                except Exception as e:
                    # Continue to next label if this one fails
                    error_str = str(e)
                    if "404" in error_str or "not found" in error_str.lower() or "LangfuseNotFoundError" in error_str:
                        continue  # Try next label
                    else:
                        continue  # Try next label
            
            # If all labels failed, construct a minimal prompt object from database data
            # This is needed because get_prompt() defaults to 'production' label which may not exist
            
            # Query database for full prompt data to construct object
            try:
                conn = psycopg2.connect(**db_config)
                try:
                    with conn.cursor() as cursor:
                        cursor.execute("""
                            SELECT id, name, prompt, version, labels, config
                            FROM prompts
                            WHERE id = %s
                            LIMIT 1
                        """, (prompt_id,))
                        result = cursor.fetchone()
                        if result:
                            # Construct a minimal prompt-like object
                            class MinimalPromptObject:
                                def __init__(self, prompt_id, name, prompt, version, labels, config):
                                    self.id = prompt_id
                                    self.name = name
                                    self.prompt = prompt
                                    self.version = version
                                    self.labels = labels or []
                                    self.config = config or {}
                            
                            prompt_obj = MinimalPromptObject(
                                prompt_id=result[0],
                                name=result[1],
                                prompt=result[2],
                                version=result[3],
                                labels=result[4] if len(result) > 4 else None,
                                config=result[5] if len(result) > 5 else None
                            )
                            
                            logger.debug(f"Constructed prompt object from database for ID '{prompt_id}'")
                            return prompt_obj
                finally:
                    conn.close()
            except Exception as db_error:
                logger.debug(f"Could not construct prompt object from database: {db_error}")
            
            logger.debug(f"Could not fetch prompt object for name '{prompt_name}' with any label")
            return None
        else:
            logger.debug(f"Langfuse get_prompt() method not available")
            return None
            
    except ImportError:
        # psycopg2 not available - skip database query
        logger.debug("psycopg2 not available, skipping database query for prompt name")
        return None
    except Exception as e:
        # Non-blocking: log at debug level
        logger.debug(f"Could not get prompt object by ID '{prompt_id}': {e}")
        return None


def get_langfuse_prompt_by_metadata(
    role: str,
    architecture_hash: str,
    file_hash: str,
    architecture_type: Optional[str] = None,
    label: Optional[str] = "production"
) -> Optional[Dict[str, Any]]:
    """
    Query Langfuse database for prompt matching role, architecture_hash, and file_hash.
    
    This uses direct database queries because Langfuse SDK doesn't support
    querying by config/metadata fields.
    
    Args:
        role: Role name
        architecture_hash: Architecture model hash (SHA256)
        file_hash: File structure hash (SHA256)
        architecture_type: Optional architecture type filter
        label: Label to filter by (default: "production")
        
    Returns:
        Dictionary with prompt data (id, name, prompt, version, config) or None
    """
    try:
        import psycopg2
        
        # Get Langfuse database config - use defaults from docker-compose.yml
        # In production, these should come from environment variables
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "postgres"
        }
        
        conn = psycopg2.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                # Query prompts table with config JSONB filter
                # Filter by role, architecture_hash, file_hash, and optionally label
                query = """
                    SELECT id, name, prompt, version, config, labels
                    FROM prompts
                    WHERE config->>'role' = %s
                      AND config->>'architecture_hash' = %s
                      AND config->>'file_hash' = %s
                """
                params = [role, architecture_hash, file_hash]
                
                if architecture_type:
                    query += " AND config->>'architecture_type' = %s"
                    params.append(architecture_type)
                
                # Filter by label if specified (labels is a JSONB array)
                if label:
                    query += " AND %s = ANY(labels)"
                    params.append(label)
                
                query += " ORDER BY version DESC, created_at DESC LIMIT 1"
                
                cursor.execute(query, params)
                result = cursor.fetchone()
                
                if result:
                    prompt_id = result[0]
                    prompt_name = result[1]
                    prompt_content = result[2]
                    prompt_version = result[3]
                    prompt_config = result[4] if len(result) > 4 else {}
                    prompt_labels = result[5] if len(result) > 5 else []
                    
                    # Parse JSONB config if it's a string
                    if isinstance(prompt_config, str):
                        import json
                        prompt_config = json.loads(prompt_config)
                    
                    logger.info(
                        f"Found existing Langfuse prompt by metadata: {prompt_name} "
                        f"(id: {prompt_id}, version: {prompt_version})"
                    )
                    
                    return {
                        "id": str(prompt_id),
                        "name": prompt_name,
                        "prompt": prompt_content,
                        "version": prompt_version,
                        "config": prompt_config,
                        "labels": prompt_labels
                    }
                
                return None
        finally:
            conn.close()
            
    except ImportError:
        logger.debug("psycopg2 not available, skipping database query")
        return None
    except Exception as e:
        logger.debug(f"Error querying prompts by metadata: {e}")
        return None


def update_langfuse_prompt_config(
    prompt_id: str,
    validation_metadata: Dict[str, Any]
) -> bool:
    """
    Update existing Langfuse prompt's config with validation metadata.
    
    This directly updates the config JSONB field in the Langfuse database
    without creating a new version. The validation scores (clarity, confidence,
    etc.) are merged into the existing config under the "validation" key.
    
    Args:
        prompt_id: Existing Langfuse prompt ID (UUID string)
        validation_metadata: Dictionary with validation scores:
            - is_valid: bool
            - confidence: float
            - clarity: float
            - completeness: float
            - relevance: float
            - accuracy: float
            - overall_score: float
            - feedback: str
        
    Returns:
        True if updated successfully, False otherwise
    """
    if not prompt_id or not validation_metadata:
        logger.debug("prompt_id and validation_metadata are required for update")
        return False
    
    # Check if prompt tracking is enabled
    try:
        from llm.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        if not config.langfuse.enabled:
            logger.debug("Langfuse is disabled, skipping prompt config update")
            return False
        
        prompt_tracking_enabled = False
        if hasattr(config.langfuse, 'prompt_tracking'):
            prompt_tracking_config = config.langfuse.prompt_tracking
            if hasattr(prompt_tracking_config, 'enabled'):
                prompt_tracking_enabled = prompt_tracking_config.enabled
            elif isinstance(prompt_tracking_config, dict):
                prompt_tracking_enabled = prompt_tracking_config.get('enabled', False)
        
        if not prompt_tracking_enabled:
            logger.debug("Langfuse prompt tracking is disabled, skipping update")
            return False
    except Exception as config_error:
        logger.debug(f"Could not check prompt tracking config: {config_error}")
        return False
    
    try:
        import psycopg2
        import json
        
        # Get Langfuse database config - use defaults from docker-compose.yml
        # In production, these should come from environment variables
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "postgres",
            "user": "postgres",
            "password": "postgres"
        }
        
        conn = psycopg2.connect(**db_config)
        try:
            with conn.cursor() as cursor:
                # Get the existing config
                cursor.execute("""
                    SELECT config
                    FROM prompts
                    WHERE id = %s
                    LIMIT 1
                """, (prompt_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.warning(f"Prompt ID '{prompt_id}' not found in database")
                    return False
                
                existing_config = result[0]
                
                # Parse existing config if it's a string
                if isinstance(existing_config, str):
                    try:
                        existing_config = json.loads(existing_config)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse existing config for prompt {prompt_id}")
                        existing_config = {}
                elif existing_config is None:
                    existing_config = {}
                
                # Merge validation metadata into existing config
                updated_config = existing_config.copy()
                updated_config["validation"] = validation_metadata
                
                # Update the config JSONB field
                cursor.execute("""
                    UPDATE prompts
                    SET config = %s::jsonb
                    WHERE id = %s
                """, (json.dumps(updated_config), prompt_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated Langfuse prompt {prompt_id} config with validation scores")
                    return True
                else:
                    logger.warning(f"No rows updated for prompt {prompt_id}")
                    return False
                    
        finally:
            conn.close()
            
    except ImportError:
        logger.debug("psycopg2 not available, skipping prompt config update")
        return False
    except Exception as e:
        logger.warning(f"Error updating Langfuse prompt config: {e}", exc_info=True)
        return False
