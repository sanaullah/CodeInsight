"""
Data models for Swarm Analysis skillbook and reflection system.

Defines structured data models for storing swarm-level skills,
reflections, and learning patterns based on ACE framework.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid


@dataclass
class SwarmSkill:
    """Represents a learned skill/pattern for swarm analysis."""
    skill_id: str
    skill_type: str  # "role_selection", "prompt_generation", "synthesis", etc.
    skill_category: str  # "helpful", "harmful", "neutral"
    content: str  # The actual skill/pattern
    context: Dict[str, Any]  # When this skill applies
    confidence: float  # 0.0 to 1.0
    usage_count: int = 0
    success_rate: float = 0.0  # 0.0 to 1.0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate skill_id if not provided."""
        if not self.skill_id:
            self.skill_id = str(uuid.uuid4())
        if self.last_used is None:
            self.last_used = self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "skill_id": self.skill_id,
            "skill_type": self.skill_type,
            "skill_category": self.skill_category,
            "content": self.content,
            "context": self.context,
            "confidence": self.confidence,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmSkill':
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('last_used'), str):
            data['last_used'] = datetime.fromisoformat(data['last_used'])
        elif data.get('last_used') is None:
            data['last_used'] = None
        
        return cls(**data)


@dataclass
class SwarmReflection:
    """Represents reflection on a swarm analysis effectiveness."""
    reflection_id: str
    analysis_id: str  # Links to swarm analysis
    architecture_type: str
    roles_selected: List[str]
    roles_effectiveness: Dict[str, float]  # role -> effectiveness score
    prompt_quality: Dict[str, float]  # role -> prompt quality score
    synthesis_quality: float
    token_efficiency: float
    key_insights: List[str] = field(default_factory=list)
    helpful_patterns: List[str] = field(default_factory=list)
    harmful_patterns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate reflection_id if not provided."""
        if not self.reflection_id:
            self.reflection_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "reflection_id": self.reflection_id,
            "analysis_id": self.analysis_id,
            "architecture_type": self.architecture_type,
            "roles_selected": self.roles_selected,
            "roles_effectiveness": self.roles_effectiveness,
            "prompt_quality": self.prompt_quality,
            "synthesis_quality": self.synthesis_quality,
            "token_efficiency": self.token_efficiency,
            "key_insights": self.key_insights,
            "helpful_patterns": self.helpful_patterns,
            "harmful_patterns": self.harmful_patterns,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmReflection':
        """Create from dictionary."""
        # Convert ISO string back to datetime
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)










