"""
Scan history database management for CodeLumen.

This module provides a PostgreSQL-based API for managing scan history.
All data is stored in PostgreSQL with Redis caching.

Backward-compatible functions:
- save_scan()
- get_scan_by_id()
- get_all_scans()
- search_scans()
- delete_scan()
- get_scan_count()
- get_statistics()
- save_metrics()
- get_metrics_history()
- get_analysis_metrics()
- get_latest_swarm_analysis()
- init_database()
- serialize_planning_result()
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from services.storage.base_storage import (
    StorageError,
    StorageConnectionError,
    DatabaseConnectionError,
    ConfigurationError,
    UserFacingError
)

logger = logging.getLogger(__name__)

# Import storage
try:
    from services.storage.cached_scan_history_storage import CachedScanHistoryStorage
    _storage = None
    
    def _get_storage():
        """Get or create storage instance."""
        global _storage
        if _storage is None:
            _storage = CachedScanHistoryStorage()
        return _storage
except ImportError as e:
    logger.error(f"Failed to import CachedScanHistoryStorage: {e}")
    _storage = None
    
    def _get_storage():
        """Get storage instance."""
        return None


def init_database() -> None:
    """
    Initialize database.
    
    This is a no-op for PostgreSQL (handled by migrations),
    but kept for backward compatibility.
    """
    try:
        storage = _get_storage()
        if storage:
            # Storage initializes itself
            logger.debug("Database initialized via storage")
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.warning(f"Database connection warning during initialization: {e}")
    except ConfigurationError as e:
        logger.warning(f"Configuration warning during initialization: {e}")
    except Exception as e:
        logger.warning(f"Database initialization warning: {e}")


def save_scan(scan_data: Dict[str, Any]) -> Optional[int]:
    """
    Save analysis result to database.
    
    Args:
        scan_data: Dictionary containing:
            - project_path: Path to analyzed project
            - agent_name: Name of agent used
            - model_used: Model name used
            - files_scanned: Number of files scanned (optional)
            - chunks_analyzed: Number of chunks analyzed (optional)
            - result: Full analysis result dictionary
            - status: Status of scan (default: 'completed')
    
    Returns:
        ID of the saved scan, or None if error
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return None
        
        scan_id = storage.save(scan_data)
        if scan_id:
            return int(scan_id)
        return None
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error saving scan: {e}", exc_info=True)
        return None  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error saving scan: {e}", exc_info=True)
        return None
    except ConfigurationError as e:
        logger.error(f"Configuration error saving scan: {e}", exc_info=True)
        return None  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error saving scan: {e}", exc_info=True)
        return None  # Fallback


def get_scan_by_id(scan_id: int) -> Optional[Dict[str, Any]]:
    """
    Get scan by ID.
    
    Args:
        scan_id: ID of the scan to retrieve
    
    Returns:
        Dictionary with scan data including full result, or None if not found
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return None
        
        return storage.get(str(scan_id))
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting scan {scan_id}: {e}", exc_info=True)
        return None  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting scan {scan_id}: {e}", exc_info=True)
        return None
    except ConfigurationError as e:
        logger.error(f"Configuration error getting scan {scan_id}: {e}", exc_info=True)
        return None  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting scan {scan_id}: {e}", exc_info=True)
        return None  # Fallback


def get_all_scans(limit: Optional[int] = None, offset: int = 0, 
                  order_by: str = "timestamp DESC") -> List[Dict[str, Any]]:
    """
    Retrieve all scan history.
    
    Args:
        limit: Maximum number of scans to return (None for all)
        offset: Number of scans to skip
        order_by: SQL ORDER BY clause (default: "timestamp DESC")
    
    Returns:
        List of scan dictionaries
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return []
        
        return storage.query(limit=limit, offset=offset, order_by=order_by)
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error retrieving scans: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error retrieving scans: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error retrieving scans: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error retrieving scans: {e}", exc_info=True)
        return []  # Fallback


def search_scans(project_path: Optional[str] = None,
                agent_name: Optional[str] = None,
                status: Optional[str] = None,
                limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Search scans with filters.
    
    Args:
        project_path: Filter by project path (partial match)
        agent_name: Filter by agent name (exact match)
        status: Filter by status (exact match)
        limit: Maximum number of results
    
    Returns:
        List of matching scans
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return []
        
        filters = {}
        if project_path:
            filters["project_path"] = project_path
        if agent_name:
            filters["agent_name"] = agent_name
        if status:
            filters["status"] = status
        
        return storage.query(filters=filters, limit=limit)
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error searching scans: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error searching scans: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error searching scans: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error searching scans: {e}", exc_info=True)
        return []  # Fallback


def delete_scan(scan_id: int) -> bool:
    """
    Delete a scan from history.
    
    Args:
        scan_id: ID of the scan to delete
    
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return False
        
        return storage.delete(str(scan_id))
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error deleting scan {scan_id}: {e}", exc_info=True)
        return False  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error deleting scan {scan_id}: {e}", exc_info=True)
        return False
    except ConfigurationError as e:
        logger.error(f"Configuration error deleting scan {scan_id}: {e}", exc_info=True)
        return False  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error deleting scan {scan_id}: {e}", exc_info=True)
        return False  # Fallback


def get_scan_count() -> int:
    """
    Get total number of scans in database.
    
    Returns:
        Total count of scans
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return 0
        
        stats = storage.get_statistics()
        return stats.get("total_scans", 0)
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting scan count: {e}", exc_info=True)
        return 0  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting scan count: {e}", exc_info=True)
        return 0
    except ConfigurationError as e:
        logger.error(f"Configuration error getting scan count: {e}", exc_info=True)
        return 0  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting scan count: {e}", exc_info=True)
        return 0  # Fallback


def get_statistics() -> Dict[str, Any]:
    """
    Get statistics about scan history.
    
    Returns:
        Dictionary with statistics:
        - total_scans: Total number of scans
        - most_used_agent: Most frequently used agent
        - most_used_agent_count: Count for most used agent
        - average_files_scanned: Average files scanned per scan
        - recent_scans: Number of scans in last 7 days
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return {
                "total_scans": 0,
                "most_used_agent": None,
                "most_used_agent_count": 0,
                "average_files_scanned": 0,
                "recent_scans": 0
            }
        
        return storage.get_statistics()
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error getting statistics: {e}", exc_info=True)
        return {
            "total_scans": 0,
            "most_used_agent": None,
            "most_used_agent_count": 0,
            "average_files_scanned": 0,
            "recent_scans": 0
        }  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error getting statistics: {e}", exc_info=True)
        return {
            "total_scans": 0,
            "most_used_agent": None,
            "most_used_agent_count": 0,
            "average_files_scanned": 0,
            "recent_scans": 0
        }
    except ConfigurationError as e:
        logger.error(f"Configuration error getting statistics: {e}", exc_info=True)
        return {
            "total_scans": 0,
            "most_used_agent": None,
            "most_used_agent_count": 0,
            "average_files_scanned": 0,
            "recent_scans": 0
        }  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error getting statistics: {e}", exc_info=True)
        return {
            "total_scans": 0,
            "most_used_agent": None,
            "most_used_agent_count": 0,
            "average_files_scanned": 0,
            "recent_scans": 0
        }  # Fallback


def get_latest_swarm_analysis(project_path: str) -> Optional[Dict[str, Any]]:
    """
    Get the latest Swarm Analysis scan for a given project path.
    
    Args:
        project_path: Path to the project
    
    Returns:
        Dictionary with scan data including full result, or None if not found
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return None
        
        return storage.get_latest_swarm_analysis(project_path)
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error retrieving latest Swarm Analysis for {project_path}: {e}", exc_info=True)
        return None  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error retrieving latest Swarm Analysis for {project_path}: {e}", exc_info=True)
        return None
    except ConfigurationError as e:
        logger.error(f"Configuration error retrieving latest Swarm Analysis for {project_path}: {e}", exc_info=True)
        return None  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error retrieving latest Swarm Analysis for {project_path}: {e}", exc_info=True)
        return None  # Fallback


def save_metrics(metrics: List[Dict[str, Any]]) -> bool:
    """
    Save metrics to metrics_history table.
    
    Args:
        metrics: List of metric dictionaries
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return False
        
        return storage.save_metrics(metrics)
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error saving metrics: {e}", exc_info=True)
        return False  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error saving metrics: {e}", exc_info=True)
        return False
    except ConfigurationError as e:
        logger.error(f"Configuration error saving metrics: {e}", exc_info=True)
        return False  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error saving metrics: {e}", exc_info=True)
        return False  # Fallback


def get_metrics_history(
    metric_name: Optional[str] = None,
    project_path: Optional[str] = None,
    agent_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve metrics history with optional filters.
    
    Args:
        metric_name: Optional metric name filter
        project_path: Optional project path filter
        agent_name: Optional agent name filter
        start_time: Optional start time filter
        end_time: Optional end time filter
    
    Returns:
        List of metric dictionaries
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return []
        
        filters = {}
        if metric_name:
            filters["metric_name"] = metric_name
        if project_path:
            filters["project_path"] = project_path
        if agent_name:
            filters["agent_name"] = agent_name
        if start_time:
            filters["start_time"] = start_time
        if end_time:
            filters["end_time"] = end_time
        
        return storage.get_metrics_history(filters=filters)
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error retrieving metrics history: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error retrieving metrics history: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error retrieving metrics history: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error retrieving metrics history: {e}", exc_info=True)
        return []  # Fallback


def get_analysis_metrics(scan_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve analysis metrics for a specific scan.
    
    Args:
        scan_id: ID of the scan
    
    Returns:
        List of metric dictionaries for the scan
    """
    try:
        storage = _get_storage()
        if not storage:
            logger.error("Storage not available")
            return []
        
        return storage.get_analysis_metrics(scan_id)
        
    except (StorageConnectionError, DatabaseConnectionError) as e:
        logger.error(f"Database connection error retrieving analysis metrics for scan {scan_id}: {e}", exc_info=True)
        return []  # Graceful degradation
    except StorageError as e:
        logger.error(f"Storage error retrieving analysis metrics for scan {scan_id}: {e}", exc_info=True)
        return []
    except ConfigurationError as e:
        logger.error(f"Configuration error retrieving analysis metrics for scan {scan_id}: {e}", exc_info=True)
        return []  # Graceful degradation
    except Exception as e:
        logger.error(f"Unexpected error retrieving analysis metrics for scan {scan_id}: {e}", exc_info=True)
        return []  # Fallback


def serialize_planning_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize planning/execution result containing dataclasses to JSON-serializable dict.
    
    Handles:
    - ExecutionPlan objects
    - Task objects (nested in ExecutionPlan)
    - Findings objects
    - Issue, Warning, Recommendation objects (nested in Findings)
    - datetime objects → ISO format strings
    - Enum values → string values
    
    Args:
        result: Result dictionary that may contain dataclass objects
        
    Returns:
        Fully serialized dictionary ready for JSON encoding
    """
    try:
        # Import here to avoid circular dependencies
        from agents.planning_models import (
            ExecutionPlan, Task, Findings, Issue, Warning, Recommendation,
            Priority, TaskStatus, PlanStatus, Severity
        )
        from agents.context_analyzer import ProjectContext
        
        serialized = {}
        
        for key, value in result.items():
            if value is None:
                serialized[key] = None
            elif isinstance(value, ExecutionPlan):
                serialized[key] = _serialize_execution_plan(value)
            elif isinstance(value, Findings):
                serialized[key] = _serialize_findings(value)
            elif isinstance(value, ProjectContext):
                serialized[key] = _serialize_project_context(value)
            elif isinstance(value, (list, tuple)):
                # Handle lists that may contain dataclasses
                serialized[key] = [
                    _serialize_item(item) for item in value
                ]
            elif isinstance(value, dict):
                # Recursively serialize nested dictionaries
                serialized[key] = serialize_planning_result(value)
            else:
                serialized[key] = _serialize_value(value)
        
        return serialized
        
    except Exception as e:
        logger.error(f"Error serializing planning result: {e}", exc_info=True)
        # Try to serialize as much as possible, even if some parts fail
        # Return a dict with error message rather than non-serializable objects
        return {
            "_serialization_error": str(e),
            "_original_keys": list(result.keys()) if isinstance(result, dict) else []
        }


def _serialize_item(item: Any) -> Any:
    """Serialize a single item, handling dataclasses that might be in lists."""
    try:
        from agents.planning_models import (
            ExecutionPlan, Task, Findings, Issue, Warning, Recommendation
        )
        from agents.context_analyzer import ProjectContext
        
        if item is None:
            return None
        elif isinstance(item, ExecutionPlan):
            return _serialize_execution_plan(item)
        elif isinstance(item, Findings):
            return _serialize_findings(item)
        elif isinstance(item, ProjectContext):
            return _serialize_project_context(item)
        elif isinstance(item, Task):
            return _serialize_task(item)
        elif isinstance(item, (Issue, Warning, Recommendation)):
            if isinstance(item, Issue):
                return _serialize_issue(item)
            elif isinstance(item, Warning):
                return _serialize_warning(item)
            else:
                return _serialize_recommendation(item)
        else:
            return _serialize_value(item)
    except Exception:
        # If we can't identify the type, try basic serialization
        return _serialize_value(item)


def _serialize_value(value: Any) -> Any:
    """Serialize a single value (datetime, Enum, or return as-is)."""
    if value is None:
        return None
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, Enum):
        return value.value
    elif isinstance(value, (list, tuple)):
        return [_serialize_item(item) for item in value]
    elif isinstance(value, dict):
        # Recursively serialize nested dictionaries
        return serialize_planning_result(value)
    else:
        # Check if it's a dataclass we need to handle
        # This is a fallback in case isinstance checks fail or object is nested unexpectedly
        try:
            from agents.planning_models import (
                ExecutionPlan, Task, Findings, Issue, Warning, Recommendation
            )
            from agents.context_analyzer import ProjectContext
            
            if isinstance(value, ExecutionPlan):
                return _serialize_execution_plan(value)
            elif isinstance(value, Findings):
                return _serialize_findings(value)
            elif isinstance(value, ProjectContext):
                return _serialize_project_context(value)
            elif isinstance(value, Task):
                return _serialize_task(value)
            elif isinstance(value, Issue):
                return _serialize_issue(value)
            elif isinstance(value, Warning):
                return _serialize_warning(value)
            elif isinstance(value, Recommendation):
                return _serialize_recommendation(value)
        except (ImportError, TypeError):
            pass
        
        # If we can't serialize it, return as string representation to avoid JSON errors
        try:
            return str(value)
        except Exception:
            return f"<non-serializable: {type(value).__name__}>"


def _serialize_execution_plan(plan: 'ExecutionPlan') -> Dict[str, Any]:
    """Serialize ExecutionPlan to dictionary."""
    from agents.planning_models import ExecutionPlan
    
    return {
        "plan_id": plan.plan_id,
        "goal": plan.goal,
        "tasks": [_serialize_task(task) for task in plan.tasks],
        "dependencies": plan.dependencies,
        "estimated_duration": plan.estimated_duration,
        "resource_requirements": plan.resource_requirements,
        "created_at": plan.created_at.isoformat() if plan.created_at else None,
        "status": plan.status.value if hasattr(plan.status, 'value') else str(plan.status),
        "adapted_from": plan.adapted_from,
        "adaptation_reason": plan.adaptation_reason
    }


def _serialize_task(task: 'Task') -> Dict[str, Any]:
    """Serialize Task to dictionary."""
    from agents.planning_models import Task
    
    return {
        "task_id": task.task_id,
        "agent_type": task.agent_type,
        "goal": task.goal,
        "priority": task.priority.value if hasattr(task.priority, 'value') else str(task.priority),
        "dependencies": task.dependencies,
        "estimated_duration": task.estimated_duration,
        "resource_requirements": task.resource_requirements,
        "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
        "result": task.result,
        "error": task.error,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None
    }


def _serialize_findings(findings: 'Findings') -> Dict[str, Any]:
    """Serialize Findings to dictionary."""
    from agents.planning_models import Findings
    
    return {
        "critical_issues": [_serialize_issue(issue) for issue in findings.critical_issues],
        "warnings": [_serialize_warning(warning) for warning in findings.warnings],
        "recommendations": [_serialize_recommendation(rec) for rec in findings.recommendations],
        "adaptation_needed": findings.adaptation_needed,
        "adaptation_reason": findings.adaptation_reason,
        "summary": findings.summary
    }


def _serialize_issue(issue: 'Issue') -> Dict[str, Any]:
    """Serialize Issue to dictionary."""
    from agents.planning_models import Issue
    
    return {
        "issue_id": issue.issue_id,
        "title": issue.title,
        "description": issue.description,
        "severity": issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
        "category": issue.category,
        "file_path": issue.file_path,
        "line_number": issue.line_number,
        "agent_name": issue.agent_name,
        "metadata": issue.metadata
    }


def _serialize_warning(warning: 'Warning') -> Dict[str, Any]:
    """Serialize Warning to dictionary."""
    from agents.planning_models import Warning
    
    return {
        "warning_id": warning.warning_id,
        "title": warning.title,
        "description": warning.description,
        "severity": warning.severity.value if hasattr(warning.severity, 'value') else str(warning.severity),
        "category": warning.category,
        "file_path": warning.file_path,
        "line_number": warning.line_number,
        "agent_name": warning.agent_name,
        "metadata": warning.metadata
    }


def _serialize_recommendation(rec: 'Recommendation') -> Dict[str, Any]:
    """Serialize Recommendation to dictionary."""
    from agents.planning_models import Recommendation
    
    return {
        "recommendation_id": rec.recommendation_id,
        "title": rec.title,
        "description": rec.description,
        "category": rec.category,
        "priority": rec.priority.value if hasattr(rec.priority, 'value') else str(rec.priority),
        "agent_name": rec.agent_name,
        "metadata": rec.metadata
    }


def _serialize_project_context(context: 'ProjectContext') -> Dict[str, Any]:
    """Serialize ProjectContext to dictionary."""
    from agents.context_analyzer import ProjectContext
    
    return {
        "project_type": context.project_type.value if hasattr(context.project_type, 'value') else str(context.project_type),
        "complexity": context.complexity.value if hasattr(context.complexity, 'value') else str(context.complexity),
        "file_count": context.file_count,
        "total_lines": context.total_lines,
        "estimated_tokens": context.estimated_tokens,
        "technologies": list(context.technologies) if isinstance(context.technologies, set) else context.technologies,
        "has_security_concerns": context.has_security_concerns,
        "has_performance_concerns": context.has_performance_concerns,
        "project_structure": context.project_structure
    }

