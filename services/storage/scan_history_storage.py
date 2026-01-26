"""
Scan History Storage Service.

PostgreSQL-only storage implementation for scan history.
Provides CRUD operations for scan history, metrics, and related data.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager

from services.storage.base_storage import BaseStorage, StorageError, StorageNotFoundError
from services.db_config import get_db_config, CODELUMEN_DATABASE
from services.postgresql_connection_pool import get_db_connection, transaction

logger = logging.getLogger(__name__)

# Try to use orjson for faster JSON parsing if available
try:
    import orjson
    _json_loads = orjson.loads
    _json_dumps = lambda obj: orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode('utf-8')
    _USE_ORJSON = True
except ImportError:
    _json_loads = json.loads
    _json_dumps = lambda obj: json.dumps(obj, ensure_ascii=False, default=str)
    _USE_ORJSON = False


class ScanHistoryStorage(BaseStorage):
    """
    PostgreSQL storage for scan history.
    
    Implements BaseStorage interface and provides additional methods
    for backward compatibility with existing scan_history API.
    """
    
    def __init__(self):
        """Initialize scan history storage."""
        self.config = get_db_config()
        self.schema = self.config.postgresql.schema
        self._initialized = False
        super().__init__(db_identifier=CODELUMEN_DATABASE)
    
    def _init_database(self) -> None:
        """
        Initialize database schema.
        
        Creates schema, tables, and indexes if they don't exist.
        This is called automatically by BaseStorage.__init__.
        """
        if self._initialized:
            return
        
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Ensure schema exists
                    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
                    
                    # Tables should already exist from migration, but we check anyway
                    # The migration framework handles table creation
                    logger.debug(f"Database schema initialized for {self.schema}")
                
                conn.commit()
                self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise StorageError(f"Failed to initialize database: {e}") from e
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save scan to PostgreSQL.
        
        Args:
            data: Dictionary containing:
                - project_path: Path to analyzed project
                - agent_name: Name of agent used
                - model_used: Model name used
                - files_scanned: Number of files scanned (optional)
                - chunks_analyzed: Number of chunks analyzed (optional)
                - result: Full analysis result dictionary
                - status: Status of scan (default: 'completed')
        
        Returns:
            ID of saved scan as string
            
        Raises:
            StorageError: If save fails
            StorageValidationError: If data validation fails
        """
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise StorageError(f"Data validation failed: {error_msg}")
        
        try:
            # Extract data
            project_path = data.get("project_path", "")
            agent_name = data.get("agent_name", "Unknown")
            model_used = data.get("model_used", "Unknown")
            files_scanned = data.get("files_scanned", 0)
            chunks_analyzed = data.get("chunks_analyzed", 0)
            result = data.get("result", {})
            status = data.get("status", "completed")
            timestamp = datetime.now()
            
            # Serialize result to JSON (optimized with orjson if available)
            try:
                result_json = _json_dumps(result)
            except (TypeError, ValueError) as json_error:
                logger.error(f"Failed to serialize result to JSON: {json_error}", exc_info=True)
                # Try with serialization helper if available
                try:
                    from utils.scan_history import serialize_planning_result
                    serialized_result = serialize_planning_result(result) if result else {}
                    result_json = _json_dumps(serialized_result)
                except Exception as serialize_error:
                    logger.error(f"Failed to serialize result even with helper: {serialize_error}", exc_info=True)
                    raise StorageError(f"Failed to serialize result: {json_error}") from json_error
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Insert scan
                    cursor.execute(f"""
                        INSERT INTO {self.schema}.scan_history 
                        (project_path, agent_name, model_used, files_scanned, chunks_analyzed, 
                         timestamp, result_json, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (project_path, agent_name, model_used, files_scanned, chunks_analyzed,
                          timestamp, result_json, status))
                    
                    result_row = cursor.fetchone()
                    if not result_row:
                        raise StorageError("Failed to get scan ID after insert")
                    
                    scan_id = result_row[0]
                    
                    # Store metrics if present in result
                    if result and isinstance(result, dict):
                        metrics_to_store = []
                        
                        # Extract metrics from result
                        if "files_scanned" in result:
                            metrics_to_store.append({
                                "scan_id": scan_id,
                                "metric_name": "files_scanned",
                                "metric_value": float(result.get("files_scanned", 0)),
                                "timestamp": timestamp
                            })
                        
                        if "chunks_analyzed" in result:
                            metrics_to_store.append({
                                "scan_id": scan_id,
                                "metric_name": "chunks_analyzed",
                                "metric_value": float(result.get("chunks_analyzed", 0)),
                                "timestamp": timestamp
                            })
                        
                        if "token_usage" in result:
                            metrics_to_store.append({
                                "scan_id": scan_id,
                                "metric_name": "token_usage",
                                "metric_value": float(result.get("token_usage", 0)),
                                "timestamp": timestamp
                            })
                        
                        # Batch insert metrics
                        if metrics_to_store:
                            cursor.executemany(f"""
                                INSERT INTO {self.schema}.analysis_metrics 
                                (scan_id, metric_name, metric_value, timestamp)
                                VALUES (%s, %s, %s, %s)
                            """, [(m["scan_id"], m["metric_name"], m["metric_value"], m["timestamp"]) 
                                  for m in metrics_to_store])
                            logger.debug(f"Stored {len(metrics_to_store)} metrics for scan {scan_id}")
                
                conn.commit()
            
            logger.info(f"Saved scan {scan_id} (agent: {agent_name}, project: {project_path})")
            return str(scan_id)
            
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error saving scan: {e}", exc_info=True)
            raise StorageError(f"Failed to save scan: {e}") from e
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get scan by ID.
        
        Args:
            record_id: Scan ID as string
        
        Returns:
            Dictionary with scan data including parsed result, or None if not found
            
        Raises:
            StorageError: If get fails
        """
        try:
            scan_id = int(record_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid scan ID: {record_id}")
            return None
        
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT * FROM {self.schema}.scan_history 
                        WHERE id = %s
                    """, (scan_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    scan = dict(zip(columns, row))
                    
                    # Parse result JSON (optimized with orjson if available)
                    if "result_json" in scan and scan["result_json"]:
                        try:
                            if isinstance(scan["result_json"], str):
                                scan["result"] = _json_loads(scan["result_json"])
                            else:
                                scan["result"] = scan["result_json"]  # Already JSONB
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.warning(f"Failed to parse result_json for scan {scan_id}: {e}")
                            scan["result"] = {}
                    else:
                        scan["result"] = {}
                    
                    # Ensure created_at exists
                    if "created_at" not in scan or not scan["created_at"]:
                        scan["created_at"] = scan.get("timestamp", datetime.now())
            
            return scan
            
        except Exception as e:
            logger.error(f"Error getting scan {scan_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to get scan: {e}") from e
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None,
              order_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query scans with filters.
        
        Args:
            filters: Optional filter dictionary with keys:
                - project_path: Filter by project path (partial match with LIKE)
                - agent_name: Filter by agent name (exact match)
                - status: Filter by status (exact match)
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            order_by: Optional ORDER BY clause (default: "timestamp DESC")
        
        Returns:
            List of scan dictionaries
            
        Raises:
            StorageError: If query fails
        """
        try:
            filters = filters or {}
            order_by = order_by or "timestamp DESC"
            
            # Build WHERE clause
            where_clauses = []
            params = []
            
            if "project_path" in filters and filters["project_path"]:
                where_clauses.append("project_path LIKE %s")
                params.append(f"%{filters['project_path']}%")
            
            if "agent_name" in filters and filters["agent_name"]:
                where_clauses.append("agent_name = %s")
                params.append(filters["agent_name"])
            
            if "status" in filters and filters["status"]:
                where_clauses.append("status = %s")
                params.append(filters["status"])
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Build query with performance hints
            # Use explicit column selection for better performance on large tables
            query = f"""
                SELECT id, project_path, agent_name, model_used, files_scanned, 
                       chunks_analyzed, timestamp, result_json, status, created_at, updated_at
                FROM {self.schema}.scan_history 
                WHERE {where_clause}
                ORDER BY {order_by}
            """
            
            if limit:
                query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
            
            # Monitor query performance for slow queries
            start_time = time.time()
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    scans = []
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in rows:
                        scan = dict(zip(columns, row))
                        
                        # Parse result JSON (optimized with orjson if available)
                        if "result_json" in scan and scan["result_json"]:
                            try:
                                if isinstance(scan["result_json"], str):
                                    scan["result"] = _json_loads(scan["result_json"])
                                else:
                                    scan["result"] = scan["result_json"]
                            except (json.JSONDecodeError, TypeError):
                                scan["result"] = {}
                        else:
                            scan["result"] = {}
                        
                        # Ensure created_at exists
                        if "created_at" not in scan or not scan["created_at"]:
                            scan["created_at"] = scan.get("timestamp", datetime.now())
                        
                        scans.append(scan)
            
            # Log slow queries (> 1 second)
            query_time = time.time() - start_time
            if query_time > 1.0:
                logger.warning(f"Slow query detected: {query_time:.2f}s for query with filters: {filters}")
            
            return scans
            
        except Exception as e:
            logger.error(f"Error querying scans: {e}", exc_info=True)
            raise StorageError(f"Failed to query scans: {e}") from e
    
    def delete(self, record_id: str) -> bool:
        """
        Delete scan by ID.
        
        Args:
            record_id: Scan ID as string
        
        Returns:
            True if deleted, False if not found
            
        Raises:
            StorageError: If delete fails
        """
        try:
            scan_id = int(record_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid scan ID: {record_id}")
            return False
        
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Delete related analysis_metrics first (CASCADE should handle this, but explicit is safer)
                    cursor.execute(f"""
                        DELETE FROM {self.schema}.analysis_metrics 
                        WHERE scan_id = %s
                    """, (scan_id,))
                    metrics_deleted = cursor.rowcount
                    
                    # Delete the scan
                    cursor.execute(f"""
                        DELETE FROM {self.schema}.scan_history 
                        WHERE id = %s
                    """, (scan_id,))
                    deleted = cursor.rowcount > 0
                
                conn.commit()
            
            if deleted:
                logger.info(f"Deleted scan {scan_id} (and {metrics_deleted} related metrics)")
            else:
                logger.warning(f"Scan {scan_id} not found for deletion")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting scan {scan_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete scan: {e}") from e
    
    @contextmanager
    def transaction(self):
        """
        Transaction context manager.
        
        Usage:
            with storage.transaction():
                storage.save(data1)
                storage.save(data2)
        """
        with get_db_connection(self.db_identifier, read_only=False) as conn:
            with transaction(conn):
                yield
    
    def validate_data(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate data before saving.
        
        Args:
            data: Data dictionary to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        
        required_fields = ["project_path", "agent_name", "model_used", "result"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate status
        if "status" in data:
            valid_statuses = ["completed", "failed", "in_progress", "cancelled"]
            if data["status"] not in valid_statuses:
                return False, f"Invalid status: {data['status']}. Must be one of: {valid_statuses}"
        
        return True, None
    
    # Additional methods for backward compatibility
    
    def get_by_project_path(self, project_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get scans by project path.
        
        Args:
            project_path: Project path to filter by
            limit: Optional limit on number of results
        
        Returns:
            List of scan dictionaries
        """
        return self.query(filters={"project_path": project_path}, limit=limit, order_by="timestamp DESC")
    
    def get_by_agent_name(self, agent_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get scans by agent name.
        
        Args:
            agent_name: Agent name to filter by
            limit: Optional limit on number of results
        
        Returns:
            List of scan dictionaries
        """
        return self.query(filters={"agent_name": agent_name}, limit=limit, order_by="timestamp DESC")
    
    def get_latest_swarm_analysis(self, project_path: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest Swarm Analysis scan for a given project path.
        
        Args:
            project_path: Path to the project
        
        Returns:
            Dictionary with scan data including full result, or None if not found
        """
        scans = self.query(
            filters={"project_path": project_path, "agent_name": "Swarm Analysis"},
            limit=1,
            order_by="timestamp DESC"
        )
        return scans[0] if scans else None
    
    def get_statistics(self) -> Dict[str, Any]:
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
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    stats = {}
                    
                    # Total scans
                    cursor.execute(f"SELECT COUNT(*) FROM {self.schema}.scan_history")
                    stats["total_scans"] = cursor.fetchone()[0]
                    
                    # Most used agent
                    cursor.execute(f"""
                        SELECT agent_name, COUNT(*) as count 
                        FROM {self.schema}.scan_history 
                        GROUP BY agent_name 
                        ORDER BY count DESC 
                        LIMIT 1
                    """)
                    result = cursor.fetchone()
                    if result:
                        stats["most_used_agent"] = result[0]
                        stats["most_used_agent_count"] = result[1]
                    else:
                        stats["most_used_agent"] = None
                        stats["most_used_agent_count"] = 0
                    
                    # Average files scanned
                    cursor.execute(f"SELECT AVG(files_scanned) FROM {self.schema}.scan_history")
                    result = cursor.fetchone()
                    stats["average_files_scanned"] = round(float(result[0] or 0), 2)
                    
                    # Recent scans (last 7 days)
                    from datetime import timedelta
                    seven_days_ago = datetime.now() - timedelta(days=7)
                    
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {self.schema}.scan_history 
                        WHERE timestamp >= %s
                    """, (seven_days_ago,))
                    stats["recent_scans"] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}", exc_info=True)
            return {
                "total_scans": 0,
                "most_used_agent": None,
                "most_used_agent_count": 0,
                "average_files_scanned": 0,
                "recent_scans": 0
            }
    
    def save_metrics(self, metrics: List[Dict[str, Any]]) -> bool:
        """
        Save metrics to metrics_history table.
        
        Args:
            metrics: List of metric dictionaries, each containing:
                - metric_name: Name of the metric
                - metric_value: Value of the metric
                - project_path: Optional project path
                - agent_name: Optional agent name
                - timestamp: Optional timestamp (default: now)
                - metadata: Optional metadata dictionary
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            timestamp = datetime.now()
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    values = []
                    for metric in metrics:
                        metric_name = metric.get("metric_name", "")
                        metric_value = float(metric.get("metric_value", 0.0))
                        project_path = metric.get("project_path")
                        agent_name = metric.get("agent_name")
                        metric_timestamp = metric.get("timestamp", timestamp)
                        metadata = metric.get("metadata", {})
                        metadata_json = _json_dumps(metadata) if metadata else None
                        
                        values.append((project_path, agent_name, metric_name, metric_value, metric_timestamp, metadata_json))
                    
                    # Batch insert
                    if values:
                        cursor.executemany(f"""
                            INSERT INTO {self.schema}.metrics_history 
                            (project_path, agent_name, metric_name, metric_value, timestamp, metadata_json)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, values)
                
                conn.commit()
            
            logger.info(f"Saved {len(metrics)} metrics to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}", exc_info=True)
            return False
    
    def get_metrics_history(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve metrics history with optional filters.
        
        Args:
            filters: Optional filter dictionary with keys:
                - metric_name: Optional metric name filter
                - project_path: Optional project path filter
                - agent_name: Optional agent name filter
                - start_time: Optional start time filter (datetime)
                - end_time: Optional end time filter (datetime)
        
        Returns:
            List of metric dictionaries
        """
        try:
            filters = filters or {}
            
            # Build WHERE clause
            where_clauses = []
            params = []
            
            if "metric_name" in filters and filters["metric_name"]:
                where_clauses.append("metric_name = %s")
                params.append(filters["metric_name"])
            
            if "project_path" in filters and filters["project_path"]:
                where_clauses.append("project_path = %s")
                params.append(filters["project_path"])
            
            if "agent_name" in filters and filters["agent_name"]:
                where_clauses.append("agent_name = %s")
                params.append(filters["agent_name"])
            
            if "start_time" in filters and filters["start_time"]:
                where_clauses.append("timestamp >= %s")
                params.append(filters["start_time"])
            
            if "end_time" in filters and filters["end_time"]:
                where_clauses.append("timestamp <= %s")
                params.append(filters["end_time"])
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            query = f"""
                SELECT * FROM {self.schema}.metrics_history 
                WHERE {where_clause}
                ORDER BY timestamp ASC
            """
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    metrics = []
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in rows:
                        metric = dict(zip(columns, row))
                        
                        # Parse metadata JSON (optimized with orjson if available)
                        if "metadata_json" in metric and metric["metadata_json"]:
                            try:
                                if isinstance(metric["metadata_json"], str):
                                    metric["metadata"] = _json_loads(metric["metadata_json"])
                                else:
                                    metric["metadata"] = metric["metadata_json"]
                            except (json.JSONDecodeError, TypeError):
                                metric["metadata"] = None
                        else:
                            metric["metadata"] = None
                        
                        metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving metrics history: {e}", exc_info=True)
            return []
    
    def get_analysis_metrics(self, scan_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve analysis metrics for a specific scan.
        
        Args:
            scan_id: ID of the scan
        
        Returns:
            List of metric dictionaries for the scan
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT * FROM {self.schema}.analysis_metrics 
                        WHERE scan_id = %s
                        ORDER BY timestamp ASC
                    """, (scan_id,))
                    
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    metrics = []
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in rows:
                        metric = dict(zip(columns, row))
                        # Ensure created_at exists
                        if "created_at" not in metric or not metric["created_at"]:
                            metric["created_at"] = metric.get("timestamp", datetime.now())
                        metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving analysis metrics for scan {scan_id}: {e}", exc_info=True)
            return []

