"""
Cached Swarm Skillbook Storage.

Redis-cached wrapper for SwarmSkillbookStorage providing automatic caching
with invalidation on writes and configurable TTL based on skill confidence.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from services.storage.swarm_skillbook_storage import SwarmSkillbookStorage
from services.storage.cached_storage import CachedStorage
from services.cache_config import get_ttl_for_cache_type
from services.redis_cache import get_cache
from services.cache_utils import validate_cache_key
from agents.swarm_skillbook_models import SwarmSkill, SwarmReflection

logger = logging.getLogger(__name__)


class CachedSwarmSkillbookStorage(CachedStorage):
    """
    Cached swarm skillbook storage with Redis.
    
    Wraps SwarmSkillbookStorage with automatic caching:
    - Individual skills: TTL 6 hours (high confidence >=0.8) or 1 hour (low confidence)
    - Query results: TTL 15 minutes
    - Reflections: TTL 1 hour
    - Invalidates cache on skill updates, usage updates
    """
    
    def __init__(self):
        """
        Initialize cached swarm skillbook storage.
        
        Creates underlying SwarmSkillbookStorage and wraps it with caching.
        """
        storage = SwarmSkillbookStorage()
        super().__init__(
            storage=storage,
            service_name="swarm_skillbook",
            get_ttl=get_ttl_for_cache_type("swarm_skillbook"),  # 1 hour default
            query_ttl=get_ttl_for_cache_type("query_result"),  # 15 min default
            enable_query_cache=True
        )
        self.cache = get_cache()
        logger.debug("CachedSwarmSkillbookStorage initialized")
    
    def get(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Get skill with caching and dynamic TTL based on confidence.
        
        Args:
            record_id: Skill ID
        
        Returns:
            Skill dictionary or None if not found
        """
        cache_key = self._get_cache_key(record_id)
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for skill: {record_id}")
                return cached
        except Exception as e:
            logger.warning(f"Cache get failed for {record_id}: {e}")
        
        # Cache miss, get from storage
        logger.debug(f"Cache miss for skill: {record_id}")
        skill = self.storage.get(record_id)
        
        # Cache result if found with appropriate TTL
        if skill is not None:
            try:
                ttl = self._get_ttl_for_skill(skill)
                self.cache.set(cache_key, skill, ttl=ttl)
                logger.debug(f"Cached skill: {record_id} with TTL {ttl}")
            except Exception as e:
                logger.warning(f"Cache set failed for {record_id}: {e}")
        
        return skill
    
    def get_skill(self, skill_id: str) -> Optional[SwarmSkill]:
        """Get skill as SwarmSkill object (delegates to storage)."""
        skill_dict = self.get(skill_id)
        if skill_dict:
            return self.storage._dict_to_skill(skill_dict)
        return None
    
    def get_skills(
        self,
        skill_type: Optional[str] = None,
        skill_category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SwarmSkill]:
        """
        Get skills with optional filters (delegates to storage).
        
        Args:
            skill_type: Optional skill type filter
            skill_category: Optional category filter
            context: Optional context filter
            
        Returns:
            List of SwarmSkill objects
        """
        return self.storage.get_skills(
            skill_type=skill_type,
            skill_category=skill_category,
            context=context
        )
    
    def get_relevant_skills(
        self,
        architecture_type: str,
        goal: Optional[str] = None
    ) -> Dict[str, List[SwarmSkill]]:
        """
        Get relevant skills organized by skill type (delegates to storage).
        
        Args:
            architecture_type: Architecture type to match
            goal: Optional goal for additional filtering
            
        Returns:
            Dictionary mapping skill_type -> List[SwarmSkill]
        """
        return self.storage.get_relevant_skills(architecture_type, goal)
    
    def save_skill(self, skill: SwarmSkill) -> str:
        """
        Save skill and invalidate cache.
        
        Args:
            skill: SwarmSkill object
            
        Returns:
            Skill ID
        """
        skill_id = self.storage.save_skill(skill)
        
        # Invalidate cache
        self._invalidate_record(skill_id)
        
        return skill_id
    
    def update_skill_usage(self, skill_id: str, effectiveness: float, analysis_id: str) -> None:
        """
        Update skill usage and invalidate cache.
        
        Args:
            skill_id: Skill ID
            effectiveness: Effectiveness score
            analysis_id: Analysis ID
        """
        self.storage.update_skill_usage(skill_id, effectiveness, analysis_id)
        
        # Invalidate cache for this skill
        self._invalidate_record(skill_id)
    
    def save_reflection(self, reflection: SwarmReflection) -> str:
        """
        Save reflection with caching.
        
        Args:
            reflection: SwarmReflection object
            
        Returns:
            Reflection ID
        """
        reflection_id = self.storage.save_reflection(reflection)
        
        # Cache reflection with 1 hour TTL
        cache_key = f"{self.service_name}:reflection:{reflection_id}"
        # Validate the cache key
        try:
            cache_key = validate_cache_key(cache_key)
        except ValueError as e:
            logger.warning(f"Invalid cache key for reflection: {e}, using sanitized key")
            # Re-raise to fail fast - caller should handle invalid keys
            raise
        
        try:
            self.cache.set(cache_key, reflection.to_dict(), ttl=3600)
        except Exception as e:
            logger.warning(f"Cache set failed for reflection: {e}")
        
        return reflection_id
    
    def get_similar_reflections(
        self,
        architecture_type: str,
        limit: int = 10
    ) -> List[SwarmReflection]:
        """
        Get similar reflections with caching.
        
        Args:
            architecture_type: Architecture type to match
            limit: Maximum number of results
            
        Returns:
            List of SwarmReflection objects
        """
        # Build cache key
        cache_key = f"{self.service_name}:reflections:{architecture_type}:{limit}"
        # Validate the cache key
        try:
            cache_key = validate_cache_key(cache_key)
        except ValueError as e:
            logger.warning(f"Invalid cache key for reflections: {e}, using sanitized key")
            # Re-raise to fail fast - caller should handle invalid keys
            raise
        
        # Try to get from cache
        try:
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for reflections: {architecture_type}")
                return [SwarmReflection.from_dict(r) for r in cached]
        except Exception as e:
            logger.warning(f"Cache get failed for reflections: {e}")
        
        # Cache miss, get from storage
        reflections = self.storage.get_similar_reflections(architecture_type, limit)
        
        # Cache results with 1 hour TTL
        try:
            reflection_dicts = [r.to_dict() for r in reflections]
            self.cache.set(cache_key, reflection_dicts, ttl=3600)
        except Exception as e:
            logger.warning(f"Cache set failed for reflections: {e}")
        
        return reflections
    
    def deduplicate_skills(self) -> Dict[str, int]:
        """Deduplicate skills (delegates to storage, invalidates cache)."""
        result = self.storage.deduplicate_skills()
        
        # Invalidate all skill caches
        try:
            pattern = f"{self.service_name}:skill:*"
            self.cache.delete_pattern(pattern)
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
        
        return result
    
    def get_skill_statistics(self) -> Dict[str, Any]:
        """Get skill statistics (delegates to storage, not cached)."""
        return self.storage.get_skill_statistics()
    
    def get_improvement_metrics(
        self,
        time_period: Tuple[Any, Any]
    ) -> Dict[str, Any]:
        """Get improvement metrics (delegates to storage, not cached)."""
        return self.storage.get_improvement_metrics(time_period)
    
    def _get_ttl_for_skill(self, skill: Dict[str, Any]) -> int:
        """
        Get TTL for skill based on confidence.
        
        Args:
            skill: Skill dictionary
            
        Returns:
            TTL in seconds
        """
        confidence = skill.get('confidence', 0.0)
        
        # High confidence (>=0.8): 6 hours
        if confidence >= 0.8:
            return 21600  # 6 hours
        
        # Low confidence: 1 hour
        return 3600  # 1 hour










