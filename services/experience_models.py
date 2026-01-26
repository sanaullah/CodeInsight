"""
Data models for experience storage and learning.

Defines structured data models for storing agent experiences,
performance metrics, and outcomes for learning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

# Note: This import may need to be adjusted based on where autonomous_models is located
# For now, we'll use a try/except to handle if it doesn't exist yet
try:
    from agents.autonomous_models import GoalUnderstanding, Strategy, AdaptationRecord
except ImportError:
    # Create stub types if autonomous_models doesn't exist yet
    GoalUnderstanding = Any
    Strategy = Any
    AdaptationRecord = Any


class SuccessLevel(Enum):
    """Levels of success achievement."""
    FULL = "full"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class PerformanceMetrics:
    """Performance metrics for an experience."""
    goal_achievement_score: float = 0.0  # 0.0 to 1.0
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    token_usage: int = 0
    execution_time: float = 0.0  # seconds
    error_count: int = 0
    adaptation_count: int = 0


@dataclass
class Outcome:
    """Outcome of an experience."""
    success: bool = False
    success_level: SuccessLevel = SuccessLevel.PARTIAL
    primary_achievements: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    user_satisfaction: Optional[float] = None  # 0.0 to 1.0 if available
    feedback: Optional[str] = None


@dataclass
class Experience:
    """Agent experience record."""
    experience_id: str
    agent_name: str
    goal: str
    goal_understanding: GoalUnderstanding
    strategy_used: Strategy
    context: Dict[str, Any]  # ProjectContext serialized
    execution_plan: Dict[str, Any]  # ExecutionPlan serialized
    results: Dict[str, Any]
    performance_metrics: PerformanceMetrics
    outcome: Outcome
    lessons_learned: List[str] = field(default_factory=list)
    adaptations_made: List[AdaptationRecord] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    duration: float = 0.0  # seconds










