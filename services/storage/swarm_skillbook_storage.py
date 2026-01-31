"""
Swarm Skillbook Storage Service.

PostgreSQL-only storage implementation for swarm skillbook.
Provides CRUD operations for skills, reflections, and skill usage tracking.
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from contextlib import contextmanager

from services.storage.base_storage import BaseStorage, StorageError, StorageNotFoundError
from services.db_config import get_db_config, CODELUMEN_DATABASE
from services.postgresql_connection_pool import get_db_connection, transaction
from agents.swarm_skillbook_models import SwarmSkill, SwarmReflection

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


class SwarmSkillbookStorage(BaseStorage):
    """
    PostgreSQL storage for swarm skillbook.
    
    Implements BaseStorage interface and provides additional methods
    for skill and reflection operations.
    """
    
    def __init__(self):
        """Initialize swarm skillbook storage."""
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
                    logger.debug(f"Database schema initialized for {self.schema}")
                
                conn.commit()
                self._initialized = True
        except Exception as e:
            logger.error(f"Error initializing database schema: {e}", exc_info=True)
            raise StorageError(f"Failed to initialize database: {e}") from e
    
    def save(self, data: Dict[str, Any]) -> str:
        """
        Save skill to PostgreSQL (implements BaseStorage interface).
        
        Args:
            data: Dictionary containing SwarmSkill object or skill data
        
        Returns:
            Skill ID as string
        """
        # Handle both SwarmSkill object and dict
        if isinstance(data, SwarmSkill):
            skill = data
        elif 'skill_id' in data or 'skill_type' in data:
            # Convert dict to SwarmSkill
            skill = self._dict_to_skill(data)
        else:
            raise StorageError("Invalid data format for skill save")
        
        return self.save_skill(skill)
    
    def save_skill(self, skill: SwarmSkill) -> str:
        """
        Save or update a skill with atomic operations.
        
        Args:
            skill: SwarmSkill object
            
        Returns:
            Skill ID
        """
        try:
            # Extract architecture_type from context if available
            architecture_type = skill.context.get("architecture_type", "") if skill.context else ""
            
            # Serialize JSON fields
            context_json = _json_dumps(skill.context) if skill.context else "{}"
            metadata_json = _json_dumps(skill.metadata) if skill.metadata else None
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Check if skill exists
                    cursor.execute(f"""
                        SELECT skill_id FROM {self.schema}.swarm_skills 
                        WHERE skill_id = %s
                    """, (skill.skill_id,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing skill
                        cursor.execute(f"""
                            UPDATE {self.schema}.swarm_skills SET
                                skill_type = %s,
                                skill_category = %s,
                                content = %s,
                                context_json = %s,
                                confidence = %s,
                                usage_count = %s,
                                success_rate = %s,
                                last_used = %s,
                                metadata_json = %s,
                                architecture_type = %s
                            WHERE skill_id = %s
                        """, (
                            skill.skill_type,
                            skill.skill_category,
                            skill.content,
                            context_json,
                            skill.confidence,
                            skill.usage_count,
                            skill.success_rate,
                            skill.last_used,
                            metadata_json,
                            architecture_type,
                            skill.skill_id
                        ))
                    else:
                        # Insert new skill
                        cursor.execute(f"""
                            INSERT INTO {self.schema}.swarm_skills (
                                skill_id, skill_type, skill_category, content, context_json,
                                confidence, usage_count, success_rate, created_at, last_used,
                                metadata_json, architecture_type
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            skill.skill_id,
                            skill.skill_type,
                            skill.skill_category,
                            skill.content,
                            context_json,
                            skill.confidence,
                            skill.usage_count,
                            skill.success_rate,
                            skill.created_at,
                            skill.last_used,
                            metadata_json,
                            architecture_type
                        ))
                
                conn.commit()
            
            logger.debug(f"Saved skill: {skill.skill_id}")
            return skill.skill_id
            
        except Exception as e:
            logger.error(f"Error saving skill: {e}", exc_info=True)
            raise StorageError(f"Failed to save skill: {e}") from e
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get skill by ID (implements BaseStorage interface).
        
        Args:
            record_id: Skill ID as string
        
        Returns:
            Dictionary with skill data, or None if not found
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT * FROM {self.schema}.swarm_skills 
                        WHERE skill_id = %s
                    """, (record_id,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    skill_dict = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    skill_dict = self._parse_skill_json_fields(skill_dict)
            
            return skill_dict
            
        except Exception as e:
            logger.error(f"Error getting skill {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to get skill: {e}") from e
    
    def get_skill(self, skill_id: str) -> Optional[SwarmSkill]:
        """
        Get skill by ID as SwarmSkill object.
        
        Args:
            skill_id: Skill ID
            
        Returns:
            SwarmSkill object or None if not found
        """
        skill_dict = self.get(skill_id)
        if not skill_dict:
            return None
        
        return self._dict_to_skill(skill_dict)
    
    def query(self, filters: Optional[Dict[str, Any]] = None, 
              limit: Optional[int] = None, 
              offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query skills with filters (implements BaseStorage interface).
        
        Args:
            filters: Optional filter dictionary
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            
        Returns:
            List of skill dictionaries
        """
        skills = self.get_skills(
            skill_type=filters.get('skill_type') if filters else None,
            skill_category=filters.get('skill_category') if filters else None,
            context=filters.get('context') if filters else None
        )
        
        # Convert to dicts
        return [skill.to_dict() for skill in skills]
    
    def get_skills(
        self,
        skill_type: Optional[str] = None,
        skill_category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SwarmSkill]:
        """
        Get skills with optional filters.
        
        Args:
            skill_type: Optional skill type filter
            skill_category: Optional category filter (helpful, harmful, neutral)
            context: Optional context filter
            
        Returns:
            List of SwarmSkill objects
        """
        try:
            conditions = []
            params = []
            
            if skill_type:
                conditions.append("skill_type = %s")
                params.append(skill_type)
            
            if skill_category:
                conditions.append("skill_category = %s")
                params.append(skill_category)
            
            if context and "architecture_type" in context:
                conditions.append("architecture_type = %s")
                params.append(context["architecture_type"])
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f"""
                SELECT * FROM {self.schema}.swarm_skills
                WHERE {where_clause}
                ORDER BY success_rate DESC, usage_count DESC, created_at DESC
            """
            
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
            
            skills = []
            for row in rows:
                try:
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    skill_dict = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    skill_dict = self._parse_skill_json_fields(skill_dict)
                    
                    skill = self._dict_to_skill(skill_dict)
                    if skill:
                        skills.append(skill)
                except Exception as e:
                    logger.warning(f"Error deserializing skill: {e}")
            
            return skills
            
        except Exception as e:
            logger.error(f"Error getting skills: {e}", exc_info=True)
            return []
    
    def get_relevant_skills(
        self,
        architecture_type: str,
        goal: Optional[str] = None
    ) -> Dict[str, List[SwarmSkill]]:
        """
        Get relevant skills organized by skill type.
        
        Args:
            architecture_type: Architecture type to match
            goal: Optional goal for additional filtering
            
        Returns:
            Dictionary mapping skill_type -> List[SwarmSkill]
        """
        try:
            # Get skills matching architecture type
            skills = self.get_skills(
                context={"architecture_type": architecture_type}
            )
            
            # Also get general skills (no architecture type specified)
            general_skills = self.get_skills()
            general_skills = [s for s in general_skills if not (s.context and s.context.get("architecture_type"))]
            
            # Combine and organize by skill type
            all_skills = skills + general_skills
            organized = {}
            
            for skill in all_skills:
                if skill.skill_type not in organized:
                    organized[skill.skill_type] = []
                organized[skill.skill_type].append(skill)
            
            # Sort each list by success rate and usage
            for skill_type in organized:
                organized[skill_type].sort(
                    key=lambda s: (s.success_rate, s.usage_count),
                    reverse=True
                )
            
            return organized
            
        except Exception as e:
            logger.error(f"Error getting relevant skills: {e}", exc_info=True)
            return {}
    
    def update_skill_usage(self, skill_id: str, effectiveness: float, analysis_id: str) -> None:
        """
        Update skill usage statistics with atomic operations.
        
        Args:
            skill_id: Skill ID
            effectiveness: Effectiveness score (0.0 to 1.0)
            analysis_id: Analysis ID that used this skill
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Get current skill stats
                    cursor.execute(f"""
                        SELECT usage_count, success_rate FROM {self.schema}.swarm_skills 
                        WHERE skill_id = %s
                    """, (skill_id,))
                    skill_row = cursor.fetchone()
                    
                    if not skill_row:
                        logger.warning(f"Skill {skill_id} not found for usage update")
                        return
                    
                    current_usage = skill_row[0] or 0
                    current_success_rate = skill_row[1] or 0.0
                    
                    # Atomic increment usage_count using SQL
                    cursor.execute(f"""
                        UPDATE {self.schema}.swarm_skills SET
                            usage_count = usage_count + 1,
                            last_used = %s
                        WHERE skill_id = %s
                    """, (datetime.now(), skill_id))
                    
                    # Get updated usage_count
                    cursor.execute(f"""
                        SELECT usage_count FROM {self.schema}.swarm_skills 
                        WHERE skill_id = %s
                    """, (skill_id,))
                    updated_row = cursor.fetchone()
                    new_usage = updated_row[0] if updated_row else current_usage + 1
                    
                    # Update success rate (weighted average) atomically
                    new_success_rate = (
                        (current_success_rate * (new_usage - 1) + effectiveness) / new_usage
                        if new_usage > 0 else effectiveness
                    )
                    
                    cursor.execute(f"""
                        UPDATE {self.schema}.swarm_skills SET
                            success_rate = %s
                        WHERE skill_id = %s
                    """, (new_success_rate, skill_id))
                    
                    # Record usage history
                    usage_id = str(uuid.uuid4())
                    cursor.execute(f"""
                        INSERT INTO {self.schema}.skill_usage_history (
                            usage_id, skill_id, analysis_id, effectiveness, timestamp
                        ) VALUES (%s, %s, %s, %s, %s)
                    """, (usage_id, skill_id, analysis_id, effectiveness, datetime.now()))
                
                conn.commit()
            
            logger.debug(f"Updated skill usage: {skill_id}")
            
        except Exception as e:
            logger.error(f"Error updating skill usage: {e}", exc_info=True)
    
    def save_reflection(self, reflection: SwarmReflection) -> str:
        """
        Save a reflection.
        
        Args:
            reflection: SwarmReflection object
            
        Returns:
            Reflection ID
        """
        try:
            # Serialize JSON fields
            roles_effectiveness_json = _json_dumps(reflection.roles_effectiveness)
            prompt_quality_json = _json_dumps(reflection.prompt_quality)
            metadata_json = _json_dumps(reflection.metadata) if reflection.metadata else None
            
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        INSERT INTO {self.schema}.swarm_reflections (
                            reflection_id, analysis_id, architecture_type, roles_selected,
                            roles_effectiveness_json, prompt_quality_json, synthesis_quality,
                            token_efficiency, key_insights, helpful_patterns, harmful_patterns,
                            recommendations, timestamp, metadata_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        reflection.reflection_id,
                        reflection.analysis_id,
                        reflection.architecture_type,
                        reflection.roles_selected,
                        roles_effectiveness_json,
                        prompt_quality_json,
                        reflection.synthesis_quality,
                        reflection.token_efficiency,
                        reflection.key_insights,
                        reflection.helpful_patterns,
                        reflection.harmful_patterns,
                        reflection.recommendations,
                        reflection.timestamp,
                        metadata_json
                    ))
                
                conn.commit()
            
            logger.debug(f"Saved reflection: {reflection.reflection_id}")
            return reflection.reflection_id
            
        except Exception as e:
            logger.error(f"Error saving reflection: {e}", exc_info=True)
            raise StorageError(f"Failed to save reflection: {e}") from e
    
    def get_similar_reflections(
        self,
        architecture_type: str,
        limit: int = 10
    ) -> List[SwarmReflection]:
        """
        Get similar reflections for a given architecture type.
        
        Args:
            architecture_type: Architecture type to match
            limit: Maximum number of results
            
        Returns:
            List of SwarmReflection objects
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT * FROM {self.schema}.swarm_reflections
                        WHERE architecture_type = %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (architecture_type, limit))
                    rows = cursor.fetchall()
            
            reflections = []
            for row in rows:
                try:
                    # Convert row to dict
                    columns = [desc[0] for desc in cursor.description]
                    reflection_dict = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    reflection_dict = self._parse_reflection_json_fields(reflection_dict)
                    
                    reflection = self._dict_to_reflection(reflection_dict)
                    if reflection:
                        reflections.append(reflection)
                except Exception as e:
                    logger.warning(f"Error deserializing reflection: {e}")
            
            return reflections
            
        except Exception as e:
            logger.error(f"Error getting similar reflections: {e}", exc_info=True)
            return []
    
    def delete(self, record_id: str) -> bool:
        """
        Delete skill by ID (implements BaseStorage interface).
        
        Args:
            record_id: Skill ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        DELETE FROM {self.schema}.swarm_skills 
                        WHERE skill_id = %s
                    """, (record_id,))
                    
                    deleted = cursor.rowcount > 0
                
                conn.commit()
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting skill {record_id}: {e}", exc_info=True)
            raise StorageError(f"Failed to delete skill: {e}") from e
    
    def deduplicate_skills(self) -> Dict[str, int]:
        """
        Deduplicate similar skills by merging them.
        
        Returns:
            Dictionary with deduplication statistics
        """
        try:
            with get_db_connection(self.db_identifier, read_only=False) as conn:
                with conn.cursor() as cursor:
                    # Get all skills grouped by type and category
                    cursor.execute(f"""
                        SELECT skill_id, skill_type, skill_category, content, context_json
                        FROM {self.schema}.swarm_skills
                        ORDER BY skill_type, skill_category, created_at
                    """)
                    skills_by_type = cursor.fetchall()
                
                merged_count = 0
                deleted_count = 0
                
                # Simple deduplication: merge skills with same type, category, and similar content
                seen = {}
                for row in skills_by_type:
                    skill_id, skill_type, skill_category, content, context_json = row
                    key = (skill_type, skill_category, content[:100])  # Use first 100 chars as key
                    
                    if key in seen:
                        # Merge with existing skill
                        existing_id = seen[key]
                        with conn.cursor() as cursor:
                            # Get both skills
                            cursor.execute(f"""
                                SELECT usage_count, success_rate FROM {self.schema}.swarm_skills 
                                WHERE skill_id = %s
                            """, (existing_id,))
                            existing_skill = cursor.fetchone()
                            
                            cursor.execute(f"""
                                SELECT usage_count, success_rate FROM {self.schema}.swarm_skills 
                                WHERE skill_id = %s
                            """, (skill_id,))
                            current_skill = cursor.fetchone()
                            
                            if existing_skill and current_skill:
                                total_usage = existing_skill[0] + current_skill[0]
                                if total_usage > 0:
                                    combined_success = (
                                        (existing_skill[1] * existing_skill[0] + current_skill[1] * current_skill[0]) / total_usage
                                    )
                                else:
                                    combined_success = max(existing_skill[1], current_skill[1])
                                
                                cursor.execute(f"""
                                    UPDATE {self.schema}.swarm_skills SET
                                        usage_count = %s,
                                        success_rate = %s
                                    WHERE skill_id = %s
                                """, (total_usage, combined_success, existing_id))
                                
                                # Delete duplicate
                                cursor.execute(f"""
                                    DELETE FROM {self.schema}.swarm_skills 
                                    WHERE skill_id = %s
                                """, (skill_id,))
                                deleted_count += 1
                                merged_count += 1
                        
                        conn.commit()
                    else:
                        seen[key] = skill_id
            
            logger.info(f"Deduplicated skills: {merged_count} merged, {deleted_count} deleted")
            
            return {
                "merged": merged_count,
                "deleted": deleted_count
            }
            
        except Exception as e:
            logger.error(f"Error deduplicating skills: {e}", exc_info=True)
            return {"merged": 0, "deleted": 0}
    
    def get_skill_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about skill usage and effectiveness.
        
        Returns:
            Dictionary with statistics
        """
        try:
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    # Total skills
                    cursor.execute(f"SELECT COUNT(*) FROM {self.schema}.swarm_skills")
                    total_skills = cursor.fetchone()[0] or 0
                    
                    # Skills by type
                    cursor.execute(f"""
                        SELECT skill_type, COUNT(*) as count
                        FROM {self.schema}.swarm_skills
                        GROUP BY skill_type
                    """)
                    skills_by_type = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Skills by category
                    cursor.execute(f"""
                        SELECT skill_category, COUNT(*) as count
                        FROM {self.schema}.swarm_skills
                        GROUP BY skill_category
                    """)
                    skills_by_category = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    # Average success rate
                    cursor.execute(f"""
                        SELECT AVG(success_rate) FROM {self.schema}.swarm_skills 
                        WHERE usage_count > 0
                    """)
                    avg_success = cursor.fetchone()[0] or 0.0
                    
                    # Total reflections
                    cursor.execute(f"SELECT COUNT(*) FROM {self.schema}.swarm_reflections")
                    total_reflections = cursor.fetchone()[0] or 0
            
            return {
                "total_skills": total_skills,
                "skills_by_type": skills_by_type,
                "skills_by_category": skills_by_category,
                "average_success_rate": float(avg_success),
                "total_reflections": total_reflections
            }
            
        except Exception as e:
            logger.error(f"Error getting skill statistics: {e}", exc_info=True)
            return {}
    
    def get_improvement_metrics(
        self,
        time_period: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Get improvement metrics over time.
        
        Args:
            time_period: Tuple of (start_time, end_time)
            
        Returns:
            Dictionary with improvement metrics
        """
        try:
            start_time, end_time = time_period
            
            # Get reflections in time period
            with get_db_connection(self.db_identifier, read_only=True) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        SELECT synthesis_quality, token_efficiency, timestamp
                        FROM {self.schema}.swarm_reflections
                        WHERE timestamp >= %s AND timestamp <= %s
                        ORDER BY timestamp
                    """, (start_time, end_time))
                    reflections = cursor.fetchall()
            
            if not reflections:
                return {
                    "period": (start_time.isoformat(), end_time.isoformat()),
                    "reflection_count": 0,
                    "quality_trend": "stable",
                    "efficiency_trend": "stable"
                }
            
            # Calculate trends
            first_half = reflections[:len(reflections)//2]
            second_half = reflections[len(reflections)//2:]
            
            if first_half and second_half:
                avg_quality_first = sum(r[0] for r in first_half) / len(first_half)
                avg_quality_second = sum(r[0] for r in second_half) / len(second_half)
                quality_change = ((avg_quality_second - avg_quality_first) / avg_quality_first * 100) if avg_quality_first > 0 else 0
                
                avg_efficiency_first = sum(r[1] for r in first_half) / len(first_half)
                avg_efficiency_second = sum(r[1] for r in second_half) / len(second_half)
                efficiency_change = ((avg_efficiency_second - avg_efficiency_first) / avg_efficiency_first * 100) if avg_efficiency_first > 0 else 0
                
                quality_trend = "improving" if quality_change > 5 else ("declining" if quality_change < -5 else "stable")
                efficiency_trend = "improving" if efficiency_change > 5 else ("declining" if efficiency_change < -5 else "stable")
            else:
                quality_trend = "stable"
                efficiency_trend = "stable"
                quality_change = 0
                efficiency_change = 0
            
            return {
                "period": (start_time.isoformat(), end_time.isoformat()),
                "reflection_count": len(reflections),
                "quality_trend": quality_trend,
                "quality_change_percent": quality_change,
                "efficiency_trend": efficiency_trend,
                "efficiency_change_percent": efficiency_change
            }
            
        except Exception as e:
            logger.error(f"Error getting improvement metrics: {e}", exc_info=True)
            return {}
    
    @contextmanager
    def transaction(self):
        """Transaction context manager."""
        with get_db_connection(self.db_identifier, read_only=False) as conn:
            with transaction(conn):
                yield
    
    # Helper methods
    
    def _parse_skill_json_fields(self, skill_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON fields in skill dictionary."""
        if 'context_json' in skill_dict and skill_dict['context_json']:
            if isinstance(skill_dict['context_json'], str):
                skill_dict['context'] = _json_loads(skill_dict['context_json'])
            else:
                skill_dict['context'] = skill_dict['context_json']
        
        if 'metadata_json' in skill_dict and skill_dict['metadata_json']:
            if isinstance(skill_dict['metadata_json'], str):
                skill_dict['metadata'] = _json_loads(skill_dict['metadata_json'])
            else:
                skill_dict['metadata'] = skill_dict['metadata_json']
        
        return skill_dict
    
    def _parse_reflection_json_fields(self, reflection_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Parse JSON fields in reflection dictionary."""
        if 'roles_effectiveness_json' in reflection_dict and reflection_dict['roles_effectiveness_json']:
            if isinstance(reflection_dict['roles_effectiveness_json'], str):
                reflection_dict['roles_effectiveness'] = _json_loads(reflection_dict['roles_effectiveness_json'])
            else:
                reflection_dict['roles_effectiveness'] = reflection_dict['roles_effectiveness_json']
        
        if 'prompt_quality_json' in reflection_dict and reflection_dict['prompt_quality_json']:
            if isinstance(reflection_dict['prompt_quality_json'], str):
                reflection_dict['prompt_quality'] = _json_loads(reflection_dict['prompt_quality_json'])
            else:
                reflection_dict['prompt_quality'] = reflection_dict['prompt_quality_json']
        
        if 'metadata_json' in reflection_dict and reflection_dict['metadata_json']:
            if isinstance(reflection_dict['metadata_json'], str):
                reflection_dict['metadata'] = _json_loads(reflection_dict['metadata_json'])
            else:
                reflection_dict['metadata'] = reflection_dict['metadata_json']
        
        return reflection_dict
    
    def _dict_to_skill(self, data: Dict[str, Any]) -> SwarmSkill:
        """Convert dictionary to SwarmSkill object."""
        # Handle datetime conversion
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('last_used'), str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        elif data.get('last_used') is None:
            data['last_used'] = None
        
        # Ensure context and metadata are dicts
        if 'context' not in data and 'context_json' in data:
            data['context'] = _json_loads(data['context_json']) if isinstance(data['context_json'], str) else data['context_json']
        if 'metadata' not in data and 'metadata_json' in data:
            data['metadata'] = _json_loads(data['metadata_json']) if isinstance(data['metadata_json'], str) else (data['metadata_json'] or {})
            
        # Remove DB-specific fields not in SwarmSkill model
        for field in ['context_json', 'metadata_json', 'architecture_type']:
            if field in data:
                del data[field]
        
        return SwarmSkill.from_dict(data)
    
    def _dict_to_reflection(self, data: Dict[str, Any]) -> SwarmReflection:
        """Convert dictionary to SwarmReflection object."""
        # Handle datetime conversion
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Parse JSON fields if needed
        if 'roles_effectiveness' not in data and 'roles_effectiveness_json' in data:
            data['roles_effectiveness'] = _json_loads(data['roles_effectiveness_json']) if isinstance(data['roles_effectiveness_json'], str) else data['roles_effectiveness_json']
        if 'prompt_quality' not in data and 'prompt_quality_json' in data:
            data['prompt_quality'] = _json_loads(data['prompt_quality_json']) if isinstance(data['prompt_quality_json'], str) else data['prompt_quality_json']
        if 'metadata' not in data and 'metadata_json' in data:
            data['metadata'] = _json_loads(data['metadata_json']) if isinstance(data['metadata_json'], str) else (data['metadata_json'] or {})
            
        # Remove DB-specific fields not in SwarmReflection model
        for field in ['roles_effectiveness_json', 'prompt_quality_json', 'metadata_json']:
            if field in data:
                del data[field]
        
        return SwarmReflection.from_dict(data)










