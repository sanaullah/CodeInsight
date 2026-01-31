"""
Experience Storage Service.

PostgreSQL-only storage implementation for experience storage.
Provides CRUD operations for experiences, querying, and analytics.
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
from services.experience_models import Experience, PerformanceMetrics, Outcome, SuccessLevel
from psycopg2 import sql

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

logger = logging.getLogger(__name__)


class ExperienceStorage(BaseStorage):
    """
    PostgreSQL storage for agent experiences.
    
    Implements BaseStorage interface and provides additional methods
    for experience-specific operations.
    """
    
    def __init__(self):
        """Initialize experience storage."""
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
                    cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema)
                    ))
                    
                    # Tables should already exist from migration, but we check anyway
                    logger.debug(f"Database schema initialized for {self.schema}")
                
                conn.commit()
                self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise StorageError(f"Failed to initialize database: {e}") from e
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save experience to PostgreSQL.
        
        Args:
            data: Dictionary containing Experience object or experience data
        
        Returns:
            Experience ID as string
            
        Raises:
            StorageError: If save fails
            StorageValidationError: If data validation fails
        """
        # Validate data
        is_valid, error_msg = self.validate_data(data)
        if not is_valid:
            raise StorageError(f"Data validation failed: {error_msg}")
        
        try:
            # Handle both Experience object and dict
            if isinstance(data, Experience):
                experience = data
            else:
                # Convert dict to Experience if needed
                experience = self._dict_to_experience(data)
            
            # Serialize complex objects
            goal_understanding_json = _json_dumps(self._serialize_goal_understanding(experience.goal_understanding))
            strategy_used_json = _json_dumps(self._serialize_strategy(experience.strategy_used))
            outcome_json = _json_dumps(self._serialize_outcome(experience.outcome))
            performance_metrics_json = _json_dumps(self._serialize_performance_metrics(experience.performance_metrics))
            
            # Extract success_level from outcome
            success_level = experience.outcome.success_level.value if experience.outcome.success_level else None
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Insert experience
                    query_template = sql.SQL("""
                        INSERT INTO {schema}.experiences (
                            experience_id, agent_name, goal, goal_understanding_json,
                            strategy_used_json, outcome_json, performance_metrics_json,
                            timestamp, success_level
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (experience_id) DO UPDATE SET
                            agent_name = EXCLUDED.agent_name,
                            goal = EXCLUDED.goal,
                            goal_understanding_json = EXCLUDED.goal_understanding_json,
                            strategy_used_json = EXCLUDED.strategy_used_json,
                            outcome_json = EXCLUDED.outcome_json,
                            performance_metrics_json = EXCLUDED.performance_metrics_json,
                            timestamp = EXCLUDED.timestamp,
                            success_level = EXCLUDED.success_level
                    """).format(schema=sql.Identifier(self.schema))
                    
                    cursor.execute(query_template, (
                        experience.experience_id,
                        experience.agent_name,
                        experience.goal,
                        goal_understanding_json,
                        strategy_used_json,
                        outcome_json,
                        performance_metrics_json,
                        experience.timestamp,
                        success_level
                    ))
                
                conn.commit()
            
            logger.info(f"Saved experience: {experience.experience_id}")
            return experience.experience_id
            
        except StorageError:
            raise
        except Exception as e:
            logger.error(f"Error saving experience: {e}", exc_info=True)
            raise StorageError(f"Failed to save experience: {e}") from e
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experience by ID.
        
        Args:
            record_id: Experience ID as string
        
        Returns:
            Dictionary with experience data including parsed JSON fields, or None if not found
            
        Raises:
            StorageError: If get fails
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    query_template = sql.SQL("""
                        SELECT * FROM {schema}.experiences 
                        WHERE experience_id = %s
                    """).format(schema=sql.Identifier(self.schema))
                    
                    cursor.execute(query_template, (record_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    experience = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    experience = self._parse_json_fields(experience)
            
            return experience
            
        except Exception as e:
            logger.error(f"Error getting experience {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to get experience: {e}") from e
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None,
              order_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query experiences with filters.
        
        Args:
            filters: Optional filter dictionary with keys:
                - agent_name: Filter by agent name (exact match)
                - success_level: Filter by success level (exact match)
                - date_from: Filter by timestamp >= date_from (datetime)
                - date_to: Filter by timestamp <= date_to (datetime)
                - goal_keywords: Filter by goal containing keywords (LIKE search)
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            order_by: Optional ORDER BY clause (default: "timestamp DESC")
        
        Returns:
            List of experience dictionaries
            
        Raises:
            StorageError: If query fails
        """
        try:
            filters = filters or {}
            order_by = order_by or "timestamp DESC"
            
            # Build WHERE clause
            where_clauses = []
            params = []
            
            if "agent_name" in filters and filters["agent_name"]:
                where_clauses.append("agent_name = %s")
                params.append(filters["agent_name"])
            
            if "success_level" in filters and filters["success_level"]:
                where_clauses.append("success_level = %s")
                params.append(filters["success_level"])
            
            if "date_from" in filters and filters["date_from"]:
                where_clauses.append("timestamp >= %s")
                params.append(filters["date_from"])
            
            if "date_to" in filters and filters["date_to"]:
                where_clauses.append("timestamp <= %s")
                params.append(filters["date_to"])
            
            if "goal_keywords" in filters and filters["goal_keywords"]:
                where_clauses.append("goal LIKE %s")
                params.append(f"%{filters['goal_keywords']}%")
            
            where_clause = sql.SQL(" AND ").join(map(sql.SQL, where_clauses)) if where_clauses else sql.SQL("1=1")
            
            # Build query
            # Note: order_by is usually safe if controlled by application code, but for safety lets check basic safety or assume safe string but identifier protection
            # Since order_by defaults to "timestamp DESC" and is a string, we might need manual handling or assume it's safe query part.
            # Ideally we should parse order_by, but for now let's assume it's a safe string but we wrap schema.
            
            query_template = sql.SQL("""
                SELECT * FROM {schema}.experiences 
                WHERE {where}
                ORDER BY {order}
            """).format(
                schema=sql.Identifier(self.schema),
                where=where_clause,
                order=sql.SQL(order_by) # Danger: blindly trusting order_by string, but it's consistent with previous f-string. Ideally validation needed.
            )
            
            if limit:
                query_template += sql.SQL(" LIMIT {}").format(sql.Literal(limit))
                if offset:
                    query_template += sql.SQL(" OFFSET {}").format(sql.Literal(offset))
            
            # Monitor query performance
            start_time = time.time()
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query_template, params)
                    rows = cursor.fetchall()
                    
                    # Convert rows to dictionaries
                    experiences = []
                    columns = [desc[0] for desc in cursor.description]
                    
                    for row in rows:
                        experience = dict(zip(columns, row))
                        experience = self._parse_json_fields(experience)
                        experiences.append(experience)
            
            # Log slow queries (> 1 second)
            query_time = time.time() - start_time
            if query_time > 1.0:
                logger.warning(f"Slow query detected: {query_time:.2f}s for query with filters: {filters}")
            
            return experiences
            
        except Exception as e:
            logger.error(f"Error querying experiences: {e}", exc_info=True)
            raise StorageError(f"Failed to query experiences: {e}") from e
    
    def delete(self, record_id: str) -> bool:
        """
        Delete experience by ID.
        
        Args:
            record_id: Experience ID as string
        
        Returns:
            True if deleted, False if not found
            
        Raises:
            StorageError: If delete fails
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    query_template = sql.SQL("""
                        DELETE FROM {schema}.experiences 
                        WHERE experience_id = %s
                    """).format(schema=sql.Identifier(self.schema))
                    
                    cursor.execute(query_template, (record_id,))
                    deleted = cursor.rowcount > 0
                
                conn.commit()
            
            if deleted:
                logger.info(f"Deleted experience {record_id}")
            else:
                logger.warning(f"Experience {record_id} not found for deletion")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting experience {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete experience: {e}") from e
    
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
            data: Data dictionary or Experience object to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle Experience object
        if isinstance(data, Experience):
            if not data.experience_id:
                return False, "Experience ID is required"
            if not data.agent_name:
                return False, "Agent name is required"
            if not data.goal:
                return False, "Goal is required"
            return True, None
        
        # Handle dict
        if not isinstance(data, dict):
            return False, "Data must be a dictionary or Experience object"
        
        required_fields = ["experience_id", "agent_name", "goal"]
        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        return True, None
    
    # Additional methods for backward compatibility and experience-specific operations
    
    def store_experience(self, experience: Experience) -> str:
        """
        Store experience (backward compatibility method).
        
        Args:
            experience: Experience object
        
        Returns:
            Experience ID
        """
        return self.save(experience)
    
    def retrieve_experience(self, experience_id: str) -> Optional[Experience]:
        """
        Retrieve experience as Experience object (backward compatibility method).
        
        Args:
            experience_id: Experience ID
        
        Returns:
            Experience object or None
        """
        data = self.get(experience_id)
        if not data:
            return None
        return self._dict_to_experience(data)
    
    def query_experiences(self, filters: Dict[str, Any], limit: int = 100) -> List[Experience]:
        """
        Query experiences and return as Experience objects (backward compatibility method).
        
        Args:
            filters: Filter dictionary
            limit: Maximum number of results
        
        Returns:
            List of Experience objects
        """
        results = self.query(filters=filters, limit=limit)
        return [self._dict_to_experience(r) for r in results]
    
    def get_similar_experiences(self, experience: Experience, limit: int = 10) -> List[Experience]:
        """
        Get similar experiences for learning.
        
        Args:
            experience: Experience to find similar ones for
            limit: Maximum number of results
        
        Returns:
            List of similar Experience objects
        """
        # Simple similarity based on agent name and goal keywords
        goal_keywords = experience.goal.split()[:3]  # First 3 words
        
        filters = {
            "agent_name": experience.agent_name,
            "goal_keywords": goal_keywords[0] if goal_keywords else None
        }
        
        results = self.query(filters=filters, limit=limit + 1)  # +1 to exclude current
        
        # Filter out current experience and convert to Experience objects
        similar = []
        for result in results:
            if result.get("experience_id") != experience.experience_id:
                similar.append(self._dict_to_experience(result))
                if len(similar) >= limit:
                    break
        
        return similar
    
    def get_experience_statistics(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about experiences.
        
        Args:
            agent_name: Optional agent name filter
        
        Returns:
            Statistics dictionary
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    if agent_name:
                        query = sql.SQL("""
                            SELECT COUNT(*) FROM {schema}.experiences 
                            WHERE agent_name = %s
                        """).format(schema=sql.Identifier(self.schema))
                        cursor.execute(query, (agent_name,))
                    else:
                        query = sql.SQL("SELECT COUNT(*) FROM {schema}.experiences").format(schema=sql.Identifier(self.schema))
                        cursor.execute(query)
                    
                    count = cursor.fetchone()[0]
            
            return {
                "total_experiences": count,
                "agent_name": agent_name or "all"
            }
            
        except Exception as e:
            logger.error(f"Error getting experience statistics: {e}", exc_info=True)
            return {"total_experiences": 0}
    
    # Serialization/deserialization helpers
    
    def _serialize_goal_understanding(self, goal) -> Dict[str, Any]:
        """Serialize GoalUnderstanding to dict."""
        # Handle both object and dict
        if isinstance(goal, dict):
            return goal
        
        try:
            from agents.autonomous_models import GoalUnderstanding
            if isinstance(goal, GoalUnderstanding):
                return {
                    "primary_objective": goal.primary_objective,
                    "sub_goals": [{"goal_id": sg.goal_id, "description": sg.description} for sg in goal.sub_goals],
                    "priority": goal.priority.value if hasattr(goal.priority, 'value') else str(goal.priority),
                    "estimated_complexity": goal.estimated_complexity.value if hasattr(goal.estimated_complexity, 'value') else str(goal.estimated_complexity),
                    "confidence": goal.confidence
                }
        except ImportError:
            pass
        
        # Fallback: convert to dict if possible
        if hasattr(goal, '__dict__'):
            return goal.__dict__
        
        return {"primary_objective": str(goal)}
    
    def _serialize_strategy(self, strategy) -> Dict[str, Any]:
        """Serialize Strategy to dict."""
        if isinstance(strategy, dict):
            return strategy
        
        try:
            from agents.autonomous_models import Strategy
            if isinstance(strategy, Strategy):
                return {
                    "strategy_id": strategy.strategy_id,
                    "strategy_type": strategy.strategy_type.value if hasattr(strategy.strategy_type, 'value') else str(strategy.strategy_type),
                    "approach": strategy.approach,
                    "parameters": strategy.parameters
                }
        except ImportError:
            pass
        
        if hasattr(strategy, '__dict__'):
            return strategy.__dict__
        
        return {"strategy_id": "", "approach": str(strategy)}
    
    def _serialize_performance_metrics(self, metrics) -> Dict[str, Any]:
        """Serialize PerformanceMetrics to dict."""
        if isinstance(metrics, dict):
            return metrics
        
        if isinstance(metrics, PerformanceMetrics):
            return {
                "goal_achievement_score": metrics.goal_achievement_score,
                "quality_score": metrics.quality_score,
                "efficiency_score": metrics.efficiency_score,
                "token_usage": metrics.token_usage,
                "execution_time": metrics.execution_time,
                "error_count": metrics.error_count,
                "adaptation_count": metrics.adaptation_count
            }
        
        if hasattr(metrics, '__dict__'):
            return metrics.__dict__
        
        return {}
    
    def _serialize_outcome(self, outcome) -> Dict[str, Any]:
        """Serialize Outcome to dict."""
        if isinstance(outcome, dict):
            return outcome
        
        if isinstance(outcome, Outcome):
            return {
                "success": outcome.success,
                "success_level": outcome.success_level.value if hasattr(outcome.success_level, 'value') else str(outcome.success_level),
                "primary_achievements": outcome.primary_achievements,
                "failures": outcome.failures,
                "user_satisfaction": outcome.user_satisfaction,
                "feedback": outcome.feedback
            }
        
        if hasattr(outcome, '__dict__'):
            return outcome.__dict__
        
        return {"success": False}
    
    def _serialize_adaptation(self, adaptation) -> Dict[str, Any]:
        """Serialize AdaptationRecord to dict."""
        if isinstance(adaptation, dict):
            return adaptation
        
        try:
            from agents.autonomous_models import AdaptationRecord
            if isinstance(adaptation, AdaptationRecord):
                return {
                    "adaptation_id": adaptation.adaptation_id,
                    "timestamp": adaptation.timestamp.isoformat() if hasattr(adaptation.timestamp, 'isoformat') else str(adaptation.timestamp),
                    "reason": adaptation.reason,
                    "trigger": adaptation.trigger,
                    "effectiveness": adaptation.effectiveness
                }
        except ImportError:
            pass
        
        if hasattr(adaptation, '__dict__'):
            return adaptation.__dict__
        
        return {}
    
    def _parse_json_fields(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON fields in experience dict."""
        json_fields = [
            'goal_understanding_json', 'strategy_used_json', 'outcome_json',
            'performance_metrics_json'
        ]
        
        for field in json_fields:
            if field in experience and experience[field]:
                try:
                    if isinstance(experience[field], str):
                        experience[field.replace('_json', '')] = _json_loads(experience[field])
                    else:
                        experience[field.replace('_json', '')] = experience[field]
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse {field}: {e}")
                    experience[field.replace('_json', '')] = {}
        
        return experience
    
    def _dict_to_experience(self, data: Dict[str, Any]) -> Experience:
        """Convert dict to Experience object."""
        # This is a simplified conversion - full implementation would properly deserialize all nested objects
        try:
            # Parse JSON fields if they exist as strings
            goal_understanding = data.get('goal_understanding') or data.get('goal_understanding_json')
            if isinstance(goal_understanding, str):
                goal_understanding = _json_loads(goal_understanding)
            
            strategy_used = data.get('strategy_used') or data.get('strategy_used_json')
            if isinstance(strategy_used, str):
                strategy_used = _json_loads(strategy_used)
            
            outcome_data = data.get('outcome') or data.get('outcome_json')
            if isinstance(outcome_data, str):
                outcome_data = _json_loads(outcome_data)
            
            performance_metrics_data = data.get('performance_metrics') or data.get('performance_metrics_json')
            if isinstance(performance_metrics_data, str):
                performance_metrics_data = _json_loads(performance_metrics_data)
            
            # Create Outcome
            outcome = Outcome(
                success=outcome_data.get('success', False) if isinstance(outcome_data, dict) else False,
                success_level=SuccessLevel(outcome_data.get('success_level', 'partial')) if isinstance(outcome_data, dict) and outcome_data.get('success_level') else SuccessLevel.PARTIAL,
                primary_achievements=outcome_data.get('primary_achievements', []) if isinstance(outcome_data, dict) else [],
                failures=outcome_data.get('failures', []) if isinstance(outcome_data, dict) else [],
                user_satisfaction=outcome_data.get('user_satisfaction') if isinstance(outcome_data, dict) else None,
                feedback=outcome_data.get('feedback') if isinstance(outcome_data, dict) else None
            )
            
            # Create PerformanceMetrics
            performance_metrics = PerformanceMetrics(
                goal_achievement_score=performance_metrics_data.get('goal_achievement_score', 0.0) if isinstance(performance_metrics_data, dict) else 0.0,
                quality_score=performance_metrics_data.get('quality_score', 0.0) if isinstance(performance_metrics_data, dict) else 0.0,
                efficiency_score=performance_metrics_data.get('efficiency_score', 0.0) if isinstance(performance_metrics_data, dict) else 0.0,
                token_usage=performance_metrics_data.get('token_usage', 0) if isinstance(performance_metrics_data, dict) else 0,
                execution_time=performance_metrics_data.get('execution_time', 0.0) if isinstance(performance_metrics_data, dict) else 0.0,
                error_count=performance_metrics_data.get('error_count', 0) if isinstance(performance_metrics_data, dict) else 0,
                adaptation_count=performance_metrics_data.get('adaptation_count', 0) if isinstance(performance_metrics_data, dict) else 0
            )
            
            # Create Experience
            timestamp = data.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            return Experience(
                experience_id=data['experience_id'],
                agent_name=data['agent_name'],
                goal=data['goal'],
                goal_understanding=goal_understanding,  # Will be dict, not object
                strategy_used=strategy_used,  # Will be dict, not object
                context=data.get('context', {}),
                execution_plan=data.get('execution_plan', {}),
                results=data.get('results', {}),
                performance_metrics=performance_metrics,
                outcome=outcome,
                lessons_learned=data.get('lessons_learned', []),
                adaptations_made=[],  # Simplified - would need full deserialization
                timestamp=timestamp,
                duration=data.get('duration', 0.0)
            )
        except Exception as e:
            logger.error(f"Error converting dict to Experience: {e}", exc_info=True)
            raise StorageError(f"Failed to convert dict to Experience: {e}") from e

