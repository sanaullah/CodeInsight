"""
Data models for agent collaboration.

Defines structured data models for knowledge sharing, coordination,
and conflict resolution between agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional


class KnowledgeType(Enum):
    """Types of knowledge that can be shared."""
    FINDING = "finding"
    INSIGHT = "insight"
    PATTERN = "pattern"
    SOLUTION = "solution"
    ARCHITECTURE_MODEL = "architecture_model"


class CoordinationType(Enum):
    """Types of coordination requests."""
    SHARE_RESOURCE = "share_resource"
    COORDINATE_ANALYSIS = "coordinate_analysis"
    MERGE_FINDINGS = "merge_findings"
    DELEGATE_TASK = "delegate_task"


class ConflictType(Enum):
    """Types of conflicts between agents."""
    CONTRADICTORY_FINDINGS = "contradictory_findings"
    RESOURCE_CONTENTION = "resource_contention"
    STRATEGY_MISMATCH = "strategy_mismatch"


class ConflictSeverity(Enum):
    """Severity levels for conflicts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    MAJORITY_VOTE = "majority_vote"
    EXPERT_AUTHORITY = "expert_authority"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    CONSENSUS_BUILDING = "consensus_building"
    ESCALATION = "escalation"


@dataclass
class Knowledge:
    """Shared knowledge item."""
    knowledge_id: str
    agent_name: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    relevance_tags: List[str] = field(default_factory=list)
    confidence: float = 0.5  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class CoordinationRequest:
    """Request for coordination between agents."""
    request_id: str
    from_agent: str
    to_agent: str
    request_type: CoordinationType
    description: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    priority: str = "medium"  # low, medium, high, critical
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, accepted, rejected, completed


@dataclass
class CoordinationProposal:
    """Proposal for coordination."""
    proposal_id: str
    from_agent: str
    to_agents: List[str]
    proposal_type: CoordinationType
    description: str
    benefits: List[str] = field(default_factory=list)
    requirements: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Conflict:
    """Conflict between agent findings or actions."""
    conflict_id: str
    conflict_type: ConflictType
    agents_involved: List[str]
    description: str
    severity: ConflictSeverity
    findings: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class ConflictResolution:
    """Resolution of a conflict."""
    resolution_id: str
    conflict_id: str
    resolution_strategy: ConflictResolutionStrategy
    decision: Dict[str, Any]
    reasoning: str
    authoritative_agent: Optional[str] = None
    resolved_findings: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)










