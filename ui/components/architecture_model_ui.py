"""
Architecture Model UI Component.

Reusable UI components for managing architecture models in Streamlit.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from agents.architecture_model_builder import ArchitectureModelBuilder
from agents.architecture_models import ArchitectureModel
from utils.project_root_detector import detect_project_root
from utils.knowledge_base import get_architecture_model
from utils.io.file_hash import compute_file_hash
from scanners import create_project_scanner

logger = logging.getLogger(__name__)


def format_model_age(created_at: datetime) -> str:
    """
    Format model age as human-readable string.
    
    Args:
        created_at: Model creation timestamp
        
    Returns:
        Human-readable age string (e.g., "2 hours ago", "3 days ago")
    """
    if not created_at:
        return "Unknown"
    
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except:
            return "Unknown"
    
    now = datetime.now(created_at.tzinfo) if created_at.tzinfo else datetime.now()
    age = now - created_at
    
    if age < timedelta(hours=1):
        minutes = int(age.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago" if minutes > 0 else "Just now"
    elif age < timedelta(days=1):
        hours = int(age.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif age < timedelta(days=30):
        days = age.days
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif age < timedelta(days=365):
        months = age.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = age.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"


def get_freshness_badge_color(age_days: float) -> str:
    """
    Get badge color based on model age.
    
    Args:
        age_days: Model age in days
        
    Returns:
        Color name for Streamlit badge: "green", "yellow", "orange", or "red"
    """
    if age_days < 1:
        return "green"
    elif age_days < 7:
        return "yellow"
    elif age_days < 30:
        return "orange"
    else:
        return "red"


def compare_file_hashes(current_hash: str, model_hash: str) -> Dict[str, Any]:
    """
    Compare file hashes and return comparison result.
    
    Args:
        current_hash: Current project file hash
        model_hash: Model's file hash
        
    Returns:
        Dictionary with comparison results:
        - matches: bool
        - current_hash_short: str (first 8 chars)
        - model_hash_short: str (first 8 chars)
        - status: str ("Matches Current Code" or "Outdated Model")
    """
    matches = current_hash == model_hash
    return {
        "matches": matches,
        "current_hash_short": current_hash[:8] if current_hash else "N/A",
        "model_hash_short": model_hash[:8] if model_hash else "N/A",
        "status": "Matches Current Code" if matches else "Outdated Model"
    }


def export_model_json(model: ArchitectureModel, filename: Optional[str] = None):
    """
    Export model as JSON bytes for download.
    
    Args:
        model: ArchitectureModel to export
        filename: Optional filename (will generate if not provided)
        
    Returns:
        Tuple of (JSON bytes, filename)
    """
    # Convert model to dict
    model_dict = model.model_dump() if hasattr(model, 'model_dump') else model.dict()
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        system_name = model.system_name.replace(" ", "_").replace("/", "_")
        filename = f"architecture_model_{system_name}_{timestamp}.json"
    
    # Serialize to JSON
    json_str = json.dumps(model_dict, indent=2, default=str, ensure_ascii=False)
    return json_str.encode('utf-8'), filename


def render_architecture_model_status(
    project_path: str, 
    file_hash: Optional[str] = None,
    knowledge_id: Optional[str] = None
) -> Optional[ArchitectureModel]:
    """
    Display current architecture model status and details with enhanced metadata.
    
    Args:
        project_path: Path to project directory
        file_hash: Optional file_hash to use for lookup (if None, computes from current files)
        knowledge_id: Optional knowledge_id to retrieve model directly (takes precedence over file_hash)
    
    Returns:
        ArchitectureModel (Pydantic) if exists, None otherwise
    """
    if not project_path:
        st.info("Enter a project path to check for architecture models.")
        return None
    
    try:
        from utils.knowledge_base import get_architecture_model_dict
        from services.storage.cached_knowledge_storage import CachedKnowledgeStorage
        
        # Get raw model dict with all metadata
        storage = CachedKnowledgeStorage()
        
        # If knowledge_id is provided, retrieve directly (for dropdown selections)
        if knowledge_id:
            model_data = storage.get(knowledge_id)
            # Extract current_file_hash from model_data when using knowledge_id
            current_file_hash = model_data.get('file_hash') if model_data else None
        else:
            # Use provided file_hash or compute from current files
            if file_hash:
                current_file_hash = file_hash
            else:
                # Scan files to compute file_hash for lookup
                scanner = create_project_scanner()
                files = scanner.scan_directory(project_path)
                
                if not files:
                    st.info("No files found in project directory.")
                    return None
                
                # Compute file_hash for unified storage lookup
                current_file_hash = compute_file_hash(files)
            
            # Get model matching file_hash
            model_data = storage.get_architecture_model(project_path, current_file_hash)
        
        if model_data:
            try:
                # Extract content (the actual model dict)
                model_dict = model_data.get('content', {})
                if not model_dict:
                    # Fallback: try getting via knowledge_obj (only if current_file_hash is available)
                    if current_file_hash:
                        knowledge_obj = get_architecture_model(project_path, current_file_hash)
                        if knowledge_obj:
                            model_dict = knowledge_obj.content if hasattr(knowledge_obj, 'content') else knowledge_obj
                
                # Reconstruct Pydantic model from dict
                model = ArchitectureModel.model_validate(model_dict)
                
                # Get model metadata from model_data dict
                model_file_hash = model_data.get('file_hash', current_file_hash)
                created_at = model_data.get('created_at') or model_data.get('timestamp')
                metadata = model_data.get('metadata', {})
                
                # Parse created_at if it's a string
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        created_at = None
                elif created_at and not isinstance(created_at, datetime):
                    try:
                        created_at = datetime.fromisoformat(str(created_at))
                    except:
                        created_at = None
                
                # Calculate model age
                age_str = format_model_age(created_at) if created_at else "Unknown"
                age_days = (datetime.now() - created_at).total_seconds() / 86400 if created_at else 999
                
                # Compare file hashes
                hash_comparison = compare_file_hashes(current_file_hash, model_file_hash or "")
                
                # Display success message with status
                st.success(f"✅ Architecture model exists for this project")
                
                # Enhanced Metadata Dashboard
                with st.container():
                    # Header Section
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        status_badge = "🟢 Current" if hash_comparison["matches"] else "🟡 Historical"
                        st.markdown(f"**Status:** {status_badge}")
                    
                    with col2:
                        freshness_color = get_freshness_badge_color(age_days)
                        st.markdown(f"**Freshness:** :{freshness_color}[{age_str}]")
                    
                    with col3:
                        if created_at:
                            st.caption(f"Created: {created_at.strftime('%Y-%m-%d %H:%M') if isinstance(created_at, datetime) else str(created_at)}")
                    
                    st.divider()
                    
                    # Statistics Section
                    st.markdown("### 📊 Model Statistics")
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("System Name", model.system_name)
                        st.metric("System Type", model.system_type)
                    
                    with stat_col2:
                        st.metric("Architecture Pattern", model.architecture_pattern or "N/A")
                        st.metric("Modules", len(model.modules))
                    
                    with stat_col3:
                        st.metric("Dependencies", len(model.dependencies))
                        st.metric("API Endpoints", len(model.api_endpoints))
                    
                    with stat_col4:
                        # Get model size from metadata if available
                        model_size = "N/A"
                        storage_location = "PostgreSQL"
                        if metadata and isinstance(metadata, dict):
                            size_bytes = metadata.get('size')
                            if size_bytes:
                                if size_bytes < 1024:
                                    model_size = f"{size_bytes} B"
                                elif size_bytes < 1024 * 1024:
                                    model_size = f"{size_bytes / 1024:.1f} KB"
                                else:
                                    model_size = f"{size_bytes / (1024 * 1024):.1f} MB"
                            storage_location = metadata.get('stored_in', 'postgresql').upper()
                        
                        st.metric("Model Size", model_size)
                        st.metric("Storage", storage_location)
                    
                    st.divider()
                    
                    # Model Age Section
                    st.markdown("### ⏰ Model Age")
                    age_col1, age_col2 = st.columns(2)
                    
                    with age_col1:
                        if created_at:
                            st.markdown(f"**Created:** {created_at.strftime('%Y-%m-%d %H:%M:%S') if isinstance(created_at, datetime) else str(created_at)}")
                        st.markdown(f"**Age:** {age_str}")
                    
                    with age_col2:
                        freshness_indicator = "🟢 Fresh" if age_days < 1 else "🟡 Recent" if age_days < 7 else "🟠 Old" if age_days < 30 else "🔴 Very Old"
                        st.markdown(f"**Status:** {freshness_indicator}")
                    
                    st.divider()
                    
                    # File Hash Status Section
                    st.markdown("### 🔐 File Hash Status")
                    hash_col1, hash_col2, hash_col3 = st.columns(3)
                    
                    with hash_col1:
                        st.markdown(f"**Current Hash:** `{hash_comparison['current_hash_short']}...`")
                    
                    with hash_col2:
                        st.markdown(f"**Model Hash:** `{hash_comparison['model_hash_short']}...`")
                    
                    with hash_col3:
                        match_status = "✅ " + hash_comparison['status'] if hash_comparison['matches'] else "⚠️ " + hash_comparison['status']
                        st.markdown(f"**Status:** {match_status}")
                    
                    st.divider()
                    
                    # Quick Actions Section
                    st.markdown("### 🛠️ Quick Actions")
                    action_col1, action_col2, action_col3 = st.columns(3)
                    
                    with action_col1:
                        # Toggle for viewing details
                        details_key = f"show_details_{project_path}"
                        show_details = st.checkbox("📋 View Full Details", value=st.session_state.get(details_key, False), key=details_key)
                    
                    with action_col2:
                        # Export functionality
                        json_bytes, filename = export_model_json(model)
                        st.download_button(
                            label="💾 Export Model",
                            data=json_bytes,
                            file_name=filename,
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with action_col3:
                        # Placeholder for future "Use This Model" functionality
                        st.info("💡 Model is automatically used by agents")
                    
                    # View Full Details Section (expandable)
                    if show_details:
                        st.divider()
                        st.markdown("### 📋 Full Model Details")
                        
                        # Complete model structure
                        detail_tabs = st.tabs(["Overview", "Modules", "Dependencies", "API Endpoints", "Technology Stack", "Patterns & Security"])
                        
                        with detail_tabs[0]:  # Overview
                            st.markdown(f"**System Name:** {model.system_name}")
                            st.markdown(f"**System Type:** {model.system_type}")
                            st.markdown(f"**Architecture Pattern:** {model.architecture_pattern or 'N/A'}")
                            st.markdown(f"**Version:** {model.version if hasattr(model, 'version') else 'N/A'}")
                            
                            if model.database_schema:
                                st.markdown("**Database Schema:**")
                                st.json(model.database_schema)
                        
                        with detail_tabs[1]:  # Modules
                            if model.modules:
                                for i, module in enumerate(model.modules, 1):
                                    with st.expander(f"Module {i}: {module.name if hasattr(module, 'name') else str(module)}"):
                                        if isinstance(module, dict):
                                            st.json(module)
                                        else:
                                            st.markdown(f"**Name:** {getattr(module, 'name', 'N/A')}")
                                            st.markdown(f"**Type:** {getattr(module, 'type', 'N/A')}")
                                            if hasattr(module, 'description'):
                                                st.markdown(f"**Description:** {module.description}")
                            else:
                                st.info("No modules defined")
                        
                        with detail_tabs[2]:  # Dependencies
                            if model.dependencies:
                                for module, deps in model.dependencies.items():
                                    with st.expander(f"{module}"):
                                        if isinstance(deps, list):
                                            for dep in deps:
                                                st.markdown(f"- {dep}")
                                        else:
                                            st.json(deps)
                            else:
                                st.info("No dependencies defined")
                        
                        with detail_tabs[3]:  # API Endpoints
                            if model.api_endpoints:
                                for i, endpoint in enumerate(model.api_endpoints, 1):
                                    with st.expander(f"Endpoint {i}: {getattr(endpoint, 'path', 'N/A') if hasattr(endpoint, 'path') else str(endpoint)}"):
                                        if isinstance(endpoint, dict):
                                            st.json(endpoint)
                                        else:
                                            st.markdown(f"**Path:** {getattr(endpoint, 'path', 'N/A')}")
                                            st.markdown(f"**Method:** {getattr(endpoint, 'method', 'N/A')}")
                                            if hasattr(endpoint, 'description'):
                                                st.markdown(f"**Description:** {endpoint.description}")
                            else:
                                st.info("No API endpoints defined")
                        
                        with detail_tabs[4]:  # Technology Stack
                            if model.tech_stack:
                                for category, items in model.tech_stack.items():
                                    if items:
                                        st.markdown(f"**{category}:**")
                                        for item in items:
                                            st.markdown(f"- {item}")
                            
                            if model.frameworks:
                                st.markdown("**Frameworks:**")
                                for framework in model.frameworks:
                                    st.markdown(f"- {framework}")
                            
                            if model.libraries:
                                st.markdown("**Libraries:**")
                                for library in model.libraries:
                                    st.markdown(f"- {library}")
                            
                            if not model.tech_stack and not model.frameworks and not model.libraries:
                                st.info("No technology stack information")
                        
                        with detail_tabs[5]:  # Patterns & Security
                            if model.design_patterns:
                                st.markdown("**Design Patterns:**")
                                for pattern in model.design_patterns:
                                    st.markdown(f"- {pattern}")
                            
                            if model.anti_patterns:
                                st.markdown("**Anti-Patterns:**")
                                for anti_pattern in model.anti_patterns:
                                    st.markdown(f"- {anti_pattern}")
                            
                            if model.architectural_smells:
                                st.markdown("**Architectural Smells:**")
                                for smell in model.architectural_smells:
                                    st.markdown(f"- {smell}")
                            
                            if model.security_architecture:
                                st.markdown("**Security Architecture:**")
                                st.json(model.security_architecture)
                            
                            if model.performance_characteristics:
                                st.markdown("**Performance Characteristics:**")
                                st.json(model.performance_characteristics)
                            
                            if not model.design_patterns and not model.anti_patterns and not model.security_architecture:
                                st.info("No patterns or security information")
                
                return model
            except Exception as e:
                logger.error(f"Error reconstructing architecture model: {e}", exc_info=True)
                st.error(f"Error loading model: {str(e)}")
                return None
        else:
            st.info("No architecture model found for this project. Build one to get started.")
            return None
            
    except Exception as e:
        logger.error(f"Error checking architecture model status: {e}", exc_info=True)
        st.error(f"Error checking model status: {str(e)}")
        return None


def render_build_controls(project_path: str, project_root: str) -> None:
    """
    Render action buttons for building, rebuilding, and deleting architecture models.
    
    Args:
        project_path: Original project path (for model lookup)
        project_root: Detected project root (for building)
    """
    if not project_root:
        st.warning("Please enter a valid project path.")
        return
    
    # Validate project root exists
    try:
        root_path = Path(project_root)
        if not root_path.exists():
            st.error(f"Project root does not exist: {project_root}")
            return
        if not root_path.is_dir():
            st.error(f"Project root is not a directory: {project_root}")
            return
    except Exception as e:
        logger.error(f"Error validating project root: {e}", exc_info=True)
        st.error(f"Error validating project root: {str(e)}")
        return
    
    # Check if model exists (need to compute file_hash for lookup)
    try:
        scanner = create_project_scanner()
        files = scanner.scan_directory(project_root)
        
        if files:
            file_hash = compute_file_hash(files)
            knowledge_obj = get_architecture_model(project_root, file_hash)
            model_exists = knowledge_obj is not None
        else:
            model_exists = False
    except Exception as e:
        logger.error(f"Error checking for existing model: {e}", exc_info=True)
        st.error(f"Error checking for existing model: {str(e)}")
        model_exists = False
    
    # Initialize session state for build status
    if "arch_model_building" not in st.session_state:
        st.session_state.arch_model_building = False
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        build_disabled = st.session_state.arch_model_building
        if st.button("Build Model", type="primary", disabled=build_disabled, use_container_width=True):
            st.session_state.arch_model_building = True
            try:
                result = build_architecture_model_async(project_root)
                if result.get("success"):
                    st.success("Architecture model built successfully!")
                    st.rerun()
                else:
                    error_msg = result.get('error', 'Unknown error')
                    st.error(f"Failed to build model: {error_msg}")
                    logger.error(f"Architecture model build failed: {error_msg}")
            except Exception as e:
                logger.error(f"Unexpected error during build: {e}", exc_info=True)
                st.error(f"Unexpected error: {str(e)}")
            finally:
                st.session_state.arch_model_building = False
    
    with col2:
        rebuild_disabled = st.session_state.arch_model_building or not model_exists
        if st.button("Rebuild Model", disabled=rebuild_disabled, use_container_width=True):
            st.session_state.arch_model_building = True
            try:
                # Delete existing model first
                if delete_architecture_model_sync(project_path):
                    # Then build new one
                    result = build_architecture_model_async(project_root, rebuild=True)
                    if result.get("success"):
                        st.success("Architecture model rebuilt successfully!")
                        st.rerun()
                    else:
                        error_msg = result.get('error', 'Unknown error')
                        st.error(f"Failed to rebuild model: {error_msg}")
                        logger.error(f"Architecture model rebuild failed: {error_msg}")
                else:
                    st.error("Failed to delete existing model before rebuild")
                    logger.error("Failed to delete architecture model before rebuild")
            except Exception as e:
                logger.error(f"Unexpected error during rebuild: {e}", exc_info=True)
                st.error(f"Unexpected error: {str(e)}")
            finally:
                st.session_state.arch_model_building = False
    
    with col3:
        delete_disabled = st.session_state.arch_model_building or not model_exists
        if st.button("Delete Model", disabled=delete_disabled, use_container_width=True):
            # Confirmation dialog
            if st.session_state.get("confirm_delete", False):
                try:
                    if delete_architecture_model_sync(project_path):
                        st.success("Architecture model deleted successfully!")
                        st.session_state.confirm_delete = False
                        st.rerun()
                    else:
                        st.error("Failed to delete architecture model")
                        logger.error("Failed to delete architecture model")
                        st.session_state.confirm_delete = False
                except Exception as e:
                    logger.error(f"Unexpected error during delete: {e}", exc_info=True)
                    st.error(f"Unexpected error: {str(e)}")
                    st.session_state.confirm_delete = False
            else:
                st.session_state.confirm_delete = True
                st.warning("Click Delete Model again to confirm deletion.")


def build_architecture_model_async(
    project_root: str,
    model_name: Optional[str] = None,
    rebuild: bool = False
) -> Dict[str, Any]:
    """
    Build architecture model asynchronously with progress updates.
    
    Args:
        project_root: Project root directory path
        model_name: Optional model name override
        rebuild: Whether this is a rebuild (use_cache=False)
        
    Returns:
        Dictionary with 'success' boolean and 'model' or 'error' key
    """
    # Validate project root before starting
    try:
        root_path = Path(project_root)
        if not root_path.exists():
            return {
                "success": False,
                "error": f"Project root does not exist: {project_root}"
            }
        if not root_path.is_dir():
            return {
                "success": False,
                "error": f"Project root is not a directory: {project_root}"
            }
    except Exception as e:
        logger.error(f"Error validating project root: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Error validating project root: {str(e)}"
        }
    
    try:
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.info("Starting architecture model build...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Initialize builder
            try:
                status_text.text("Initializing architecture model builder...")
                progress_bar.progress(10)
                builder = ArchitectureModelBuilder(model_name=model_name)
            except Exception as e:
                logger.error(f"Error initializing builder: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Failed to initialize builder: {str(e)}"
                }
            
            # Step 2: Scan project
            try:
                status_text.text("Scanning project files...")
                progress_bar.progress(30)
                scanner = create_project_scanner()
                files = scanner.scan_directory(project_root)
                
                if not files:
                    return {
                        "success": False,
                        "error": "No files found in project directory"
                    }
            except Exception as e:
                logger.error(f"Error scanning project: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Failed to scan project: {str(e)}"
                }
            
            status_text.text(f"Found {len(files)} files. Analyzing architecture...")
            progress_bar.progress(50)
            
            # Step 3: Build model (use_cache=False for rebuild, True for new build)
            try:
                status_text.text("Building architecture model with LLM...")
                progress_bar.progress(70)
                
                model = builder.build_architecture_model(
                    project_path=project_root,
                    files=files,
                    model_name=model_name,
                    use_cache=not rebuild  # Don't use cache if rebuilding
                )
                
                if not model:
                    return {
                        "success": False,
                        "error": "Failed to build architecture model (LLM returned no model)"
                    }
            except Exception as e:
                logger.error(f"Error building model: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Failed to build model: {str(e)}"
                }
            
            progress_bar.progress(100)
            status_text.text("Architecture model built and stored successfully!")
            
            return {
                "success": True,
                "model": model,
                "files_scanned": len(files)
            }
            
    except Exception as e:
        logger.error(f"Unexpected error building architecture model: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def delete_architecture_model_sync(project_path: str) -> bool:
    """
    Delete architecture model synchronously.
    
    Args:
        project_path: Path to project directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get all models for this project and delete them
        from services.storage.cached_knowledge_storage import CachedKnowledgeStorage
        from agents.collaboration_models import KnowledgeType
        
        storage = CachedKnowledgeStorage()
        
        # Query for all architecture models for this project
        # We need to delete by project_path, so we'll query first
        try:
            from services.db_config import get_db_config, CODELUMEN_DATABASE
            from services.postgresql_connection_pool import get_db_connection
            
            config = get_db_config()
            schema = config.postgresql.schema
            
            with get_db_connection(CODELUMEN_DATABASE) as conn:
                with conn.cursor() as cursor:
                    # Find all architecture models for this project
                    query = f"""
                        SELECT knowledge_id FROM {schema}.knowledge_base
                        WHERE knowledge_type = %s AND project_path = %s
                    """
                    cursor.execute(query, [KnowledgeType.ARCHITECTURE_MODEL.value, project_path])
                    rows = cursor.fetchall()
                    
                    # Delete each model
                    deleted_count = 0
                    for row in rows:
                        knowledge_id = row[0]
                        if storage.delete(knowledge_id):
                            deleted_count += 1
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted {deleted_count} architecture model(s) for {project_path}")
                        return True
                    else:
                        logger.warning(f"No architecture models found to delete for {project_path}")
                        return False
        except Exception as e:
            logger.error(f"Error querying/deleting architecture models: {e}", exc_info=True)
            return False
    except Exception as e:
        logger.error(f"Error deleting architecture model: {e}", exc_info=True)
        return False


def get_all_stored_architecture_models(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get all stored architecture models from knowledge_base.
    
    Args:
        limit: Optional limit on number of results (default: None for all)
        
    Returns:
        List of model dictionaries with project_path, file_hash, created_at, etc.
        Sorted by created_at DESC (newest first).
        Returns empty list on error.
    """
    try:
        from services.storage.cached_knowledge_storage import CachedKnowledgeStorage
        from agents.collaboration_models import KnowledgeType
        
        storage = CachedKnowledgeStorage()
        
        # Query all architecture models
        results = storage.query(
            filters={'knowledge_type': KnowledgeType.ARCHITECTURE_MODEL.value},
            limit=limit
        )
        
        # Sort by created_at DESC (newest first)
        if results:
            results.sort(
                key=lambda x: x.get('created_at') or x.get('timestamp') or datetime.min,
                reverse=True
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting all stored architecture models: {e}", exc_info=True)
        return []  # Graceful degradation


def format_model_for_dropdown(model: Dict[str, Any]) -> str:
    """
    Format model dict for dropdown display.
    
    Format: "Project Path (YYYY-MM-DD)"
    Example: "C:\\Projects\\MyApp (2025-01-20)"
    
    Args:
        model: Model dictionary from knowledge_base
        
    Returns:
        Formatted string for dropdown option
    """
    # Handle project_path (can be None if key exists but value is None)
    project_path = model.get('project_path') or 'Unknown Path'
    if not isinstance(project_path, str):
        project_path = 'Unknown Path'
    
    # Parse created_at (handle both datetime and string formats)
    created_at = model.get('created_at') or model.get('timestamp')
    date_str = "Unknown Date"
    
    if created_at:
        try:
            if isinstance(created_at, str):
                # Try parsing ISO format string
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    date_str = dt.strftime('%Y-%m-%d')
                except:
                    # Try just extracting date part if it's in YYYY-MM-DD format
                    if len(created_at) >= 10:
                        date_str = created_at[:10]
            elif isinstance(created_at, datetime):
                date_str = created_at.strftime('%Y-%m-%d')
            else:
                # Try converting to string and extracting date
                created_str = str(created_at)
                if len(created_str) >= 10:
                    date_str = created_str[:10]
        except Exception as e:
            logger.debug(f"Error parsing date for dropdown: {e}")
            date_str = "Unknown Date"
    
    # Truncate very long paths (optional: show ellipsis if > 80 chars)
    max_path_length = 80
    if project_path and len(project_path) > max_path_length:
        project_path = project_path[:max_path_length - 3] + "..."
    
    return f"{project_path} ({date_str})"

