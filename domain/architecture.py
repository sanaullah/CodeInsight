"""Typed contracts for evidence-derived semantic repository architecture."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from .contracts import ContractModel


class ArchitectureSupportTier(StrEnum):
    EXACT = "exact"
    INFERRED = "inferred"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ArchitectureCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ComponentKind(StrEnum):
    SERVICE = "service"
    DATASTORE = "datastore"
    EXTERNAL_SYSTEM = "external_system"
    LIBRARY = "library"
    QUEUE = "queue"
    UNKNOWN = "unknown"


class RelationKind(StrEnum):
    REQUEST = "request"
    EVENT = "event"
    DATA_ACCESS = "data_access"
    DEPENDENCY = "dependency"
    CALL = "call"
    UNKNOWN = "unknown"


class ProvenanceSpan(ContractModel):
    provenance_id: str
    entity_kind: Literal[
        "component", "boundary", "membership", "resource", "endpoint", "relation"
    ]
    entity_id: str
    file_id: str
    relative_path: str | None = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    derivation: str
    extractor_version: str
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_span(self) -> ProvenanceSpan:
        if self.end_line < self.start_line:
            raise ValueError("end_line cannot precede start_line")
        return self


class SemanticComponent(ContractModel):
    component_id: str
    snapshot_id: str
    stable_key: str
    name: str
    component_kind: ComponentKind
    support_tier: ArchitectureSupportTier
    completeness: ArchitectureCompleteness
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticBoundary(ContractModel):
    boundary_id: str
    snapshot_id: str
    stable_key: str
    name: str
    boundary_kind: str
    support_tier: ArchitectureSupportTier
    completeness: ArchitectureCompleteness
    confidence: float = Field(ge=0, le=1)
    component_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticMembership(ContractModel):
    membership_id: str
    snapshot_id: str
    component_id: str
    file_id: str | None = None
    symbol_id: str | None = None
    membership_kind: str
    confidence: float = Field(ge=0, le=1)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_target(self) -> SemanticMembership:
        if self.file_id is None and self.symbol_id is None:
            raise ValueError("membership requires a file_id or symbol_id")
        return self


class SemanticResource(ContractModel):
    resource_id: str
    snapshot_id: str
    component_id: str | None = None
    stable_key: str
    resource_kind: str
    name: str
    locator: str | None = None
    support_tier: ArchitectureSupportTier
    completeness: ArchitectureCompleteness
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticEndpoint(ContractModel):
    endpoint_id: str
    snapshot_id: str
    component_id: str | None = None
    stable_key: str
    protocol: str
    method: str | None = None
    route: str
    direction: Literal["inbound", "outbound"]
    support_tier: ArchitectureSupportTier
    completeness: ArchitectureCompleteness
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticRelation(ContractModel):
    relation_id: str
    snapshot_id: str
    source_component_id: str
    target_component_id: str
    stable_key: str
    relation_kind: RelationKind
    transport: str | None = None
    is_async: bool = False
    support_tier: ArchitectureSupportTier
    completeness: ArchitectureCompleteness
    confidence: float = Field(ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticArchitectureProjection(ContractModel):
    snapshot_id: str
    extractor_version: str
    components: tuple[SemanticComponent, ...]
    memberships: tuple[SemanticMembership, ...] = ()
    boundaries: tuple[SemanticBoundary, ...] = ()
    resources: tuple[SemanticResource, ...] = ()
    endpoints: tuple[SemanticEndpoint, ...] = ()
    relations: tuple[SemanticRelation, ...] = ()
    provenance: tuple[ProvenanceSpan, ...] = ()
