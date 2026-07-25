"""Versioned contracts for the native CodeInsight workflow."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Strict base model used at durable and model-facing boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: int = Field(default=1, ge=1)


class AnalysisMode(StrEnum):
    QUICK = "quick"
    DEEP = "deep"
    SECURITY = "security"
    CHANGE_SET = "change-set"


class RunStage(StrEnum):
    DISCOVER = "discover"
    SNAPSHOT = "snapshot"
    INDEX = "index"
    PLAN_WAVE = "plan_wave"
    DISPATCH_TASKS = "dispatch_tasks"
    VERIFY_EVIDENCE = "verify_evidence"
    DEDUPLICATE_AND_CORRELATE = "deduplicate_and_correlate"
    ASSESS_COVERAGE = "assess_coverage"
    PLAN_FOLLOW_UP_WAVE = "plan_follow_up_wave"
    SYNTHESIZE = "synthesize"
    COMPLETE = "complete"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    RUNNING = "running"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunBudget(ContractModel):
    max_files: int = Field(default=2_000, ge=1)
    max_symbols: int = Field(default=20_000, ge=1)
    max_graph_nodes: int = Field(default=50_000, ge=1)
    max_specialists: int = Field(default=8, ge=1, le=64)
    max_tasks: int = Field(default=100, ge=1)
    max_waves: int = Field(default=2, ge=1, le=10)
    max_tokens: int = Field(default=1_000_000, ge=1)
    max_cost_usd: float = Field(default=25.0, ge=0)
    max_elapsed_seconds: int = Field(default=3_600, ge=1)
    reserved_follow_up_fraction: float = Field(default=0.2, ge=0, le=0.8)


class RepositorySnapshot(ContractModel):
    snapshot_id: str
    project_id: str
    canonical_path: str
    identity_hash: str
    configuration_hash: str
    scanner_version: str
    created_at: datetime
    git_repository: str | None = None
    base_commit: str | None = None
    head_commit: str | None = None
    dirty: bool = False
    included_paths: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()


class RoleSpec(ContractModel):
    role_id: str
    name: str
    mission: str
    rationale: str
    coverage_targets: tuple[str, ...]
    required_capabilities: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    input_artifact_ids: tuple[str, ...] = ()
    model_policy: str
    token_budget: int = Field(ge=1)
    time_budget_seconds: int = Field(ge=1)
    completion_criteria: tuple[str, ...]
    depends_on_role_ids: tuple[str, ...] = ()


class AnalysisTask(ContractModel):
    task_id: str
    run_id: str
    wave_id: str
    role_id: str
    task_type: str
    priority: int = Field(default=100, ge=0)
    immutable_input_ids: tuple[str, ...]
    configuration_hash: str
    idempotency_key: str
    model_policy: str
    token_budget: int = Field(ge=0)
    cost_budget_usd: float = Field(ge=0)
    time_budget_seconds: int = Field(ge=1)
    tool_call_budget: int = Field(ge=0)
    max_attempts: int = Field(default=3, ge=1)
    attempt_count: int = Field(default=0, ge=0)
    status: TaskStatus = TaskStatus.QUEUED
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancellation_requested: bool = False


class WavePlan(ContractModel):
    wave_id: str
    run_id: str
    wave_number: int = Field(ge=1)
    rationale: str
    roles: tuple[RoleSpec, ...]
    tasks: tuple[AnalysisTask, ...]
    coverage_targets: tuple[str, ...]
    reserved_follow_up: bool = False


class EvidenceRef(ContractModel):
    evidence_id: str
    snapshot_id: str
    file_id: str
    content_hash: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    excerpt_hash: str
    evidence_kind: str
    provenance: dict[str, Any]
    collected_at: datetime
    start_byte: int | None = Field(default=None, ge=0)
    end_byte: int | None = Field(default=None, ge=0)
    symbol_ids: tuple[str, ...] = ()
    graph_node_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_span(self) -> EvidenceRef:
        if self.end_line < self.start_line:
            raise ValueError("end_line cannot precede start_line")
        if (
            self.start_byte is not None
            and self.end_byte is not None
            and self.end_byte < self.start_byte
        ):
            raise ValueError("end_byte cannot precede start_byte")
        return self


class FindingCandidate(ContractModel):
    candidate_id: str
    producer_task_id: str
    producer_role_id: str
    category: str
    concept_id: str
    title: str
    claim: str
    evidence_ids: tuple[str, ...]
    preconditions: tuple[str, ...] = ()
    affected_path: str | None = None
    impact: str
    proposed_severity: Literal["info", "low", "medium", "high", "critical"]
    proposed_confidence: float = Field(ge=0, le=1)
    uncertainty: str | None = None
    missing_context: tuple[str, ...] = ()
    recommendation: str
    fingerprint_inputs: tuple[str, ...]


class FindingVerdict(ContractModel):
    verdict_id: str
    candidate_id: str
    disposition: Literal[
        "accepted", "rejected", "needs-more-evidence", "suggestion-only"
    ]
    evidence_integrity: Literal["valid", "invalid", "incomplete"]
    contradiction_result: str
    reachability_result: str
    calibrated_confidence: float = Field(ge=0, le=1)
    calibrated_severity: Literal["info", "low", "medium", "high", "critical"]
    verifier: str
    verifier_version: str
    rationale: str


class CanonicalFinding(ContractModel):
    finding_id: str
    fingerprint: str
    title: str
    claim: str
    severity: Literal["info", "low", "medium", "high", "critical"]
    confidence: float = Field(ge=0, le=1)
    candidate_ids: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]
    conflicting_candidate_ids: tuple[str, ...] = ()
    recommendation: str


class CoverageAssessment(ContractModel):
    assessment_id: str
    run_id: str
    wave_id: str
    measured: dict[str, float]
    exclusions: tuple[str, ...] = ()
    unsupported_areas: tuple[str, ...] = ()
    unresolved_uncertainty: tuple[str, ...] = ()
    remaining_gaps: tuple[str, ...] = ()
    proposed_follow_up_task_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    follow_up_decision: Literal["launch", "defer", "complete"]
    decision_rationale: str
