"""Typed request and response models for the CodeInsight API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from domain.contracts import AnalysisMode, RunBudget


def utc_now() -> datetime:
    return datetime.now(UTC)


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NEEDS_ATTENTION = "needs_attention"


class AnalysisRequest(BaseModel):
    """A bounded request for the native durable analysis engine."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    RETIRED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "auto_detect_languages",
            "chunking_strategy",
            "enable_chunking",
            "enable_dynamic_file_selection",
            "enable_tool_calling",
            "max_tokens_per_chunk",
        }
    )

    project_path: str = Field(min_length=1, max_length=4096)
    goal: str | None = Field(default=None, max_length=4000)
    model_name: str | None = Field(default=None, max_length=300)
    max_agents: int = Field(default=4, ge=1, le=12)
    file_extensions: list[str] | None = Field(default=None, max_length=50)
    selected_directories: list[str] | None = Field(default=None, max_length=100)
    mode: AnalysisMode = AnalysisMode.DEEP
    max_waves: int = Field(default=2, ge=1, le=10)
    max_tasks: int = Field(default=100, ge=1, le=500)
    max_total_tokens: int = Field(default=1_000_000, ge=1)
    max_cost_usd: float = Field(default=25, ge=0)
    max_elapsed_seconds: int = Field(default=3_600, ge=1)

    @classmethod
    def from_persisted(cls, value: dict[str, Any]) -> AnalysisRequest:
        """Read pre-native ledger requests without reopening the public API."""

        sanitized = {
            key: item for key, item in value.items() if key not in cls.RETIRED_FIELDS
        }
        return cls.model_validate(sanitized)

    def run_budget(self) -> RunBudget:
        return RunBudget(
            max_specialists=self.max_agents,
            max_tasks=self.max_tasks,
            max_waves=self.max_waves,
            max_tokens=self.max_total_tokens,
            max_cost_usd=self.max_cost_usd,
            max_elapsed_seconds=self.max_elapsed_seconds,
        )

    @field_validator("file_extensions")
    @classmethod
    def normalize_extensions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        for extension in value:
            extension = extension.strip().lower()
            if not extension:
                continue
            if any(character in extension for character in ("/", "\\", "*", "?")):
                raise ValueError("file extensions cannot contain paths or wildcards")
            normalized.append(extension if extension.startswith(".") else f".{extension}")
        return sorted(set(normalized)) or None

    @field_validator("selected_directories")
    @classmethod
    def validate_selected_directories(
        cls, value: list[str] | None
    ) -> list[str] | None:
        if value is None:
            return None
        normalized: list[str] = []
        for directory in value:
            directory = directory.strip()
            if not directory:
                continue
            posix_path = PurePosixPath(directory.replace("\\", "/"))
            windows_path = PureWindowsPath(directory)
            if (
                posix_path.is_absolute()
                or windows_path.is_absolute()
                or windows_path.drive
                or ".." in posix_path.parts
            ):
                raise ValueError(
                    "selected directories must stay relative to the project root"
                )
            normalized.append(posix_path.as_posix())
        return sorted(set(normalized)) or None


class AnalysisEvent(BaseModel):
    sequence: int
    event_type: str
    timestamp: datetime
    data: dict[str, Any] = Field(default_factory=dict)


class AnalysisRun(BaseModel):
    run_id: str
    status: AnalysisStatus
    request: AnalysisRequest
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    events: list[AnalysisEvent] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    current_stage: str | None = None
    snapshot_id: str | None = None
    mode: AnalysisMode = AnalysisMode.DEEP


class AnalysisAccepted(BaseModel):
    run_id: str
    status: AnalysisStatus
    status_url: str


class RuntimeIdentity(BaseModel):
    application_server: Literal["FastAPI"] = "FastAPI"
    environment_manager: Literal["uv"] = "uv"
    database_engine: Literal["SQLite"] = "SQLite"
    database_journal_mode: Literal["WAL"] = "WAL"
    database_schema_version: int = Field(ge=1)
    artifact_store: Literal["filesystem"] = "filesystem"
    api_docs_url: str = "/api/docs"
    read_only_analysis: Literal[True] = True
    build_commit: str | None = None
    build_time: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    active_analyses: int
    max_concurrent_analyses: int
    runtime: RuntimeIdentity


class LanguageCapability(BaseModel):
    language: str
    display_name: str
    support_level: Literal["parsed", "dependency-aware", "discovery"]
    extensions: list[str]


class CapabilitiesResponse(BaseModel):
    languages: list[LanguageCapability]
    default_max_agents: int = 4
    max_agents: int = 12
    analysis_modes: list[AnalysisMode] = Field(
        default_factory=lambda: list(AnalysisMode)
    )
    native_durable_workflow: bool = True
    model_provider_configured: bool = False
    langfuse_enabled: bool = False


class RecoveryResponse(BaseModel):
    recovered_runs: int
    recovered_tasks: int
    scheduled_runs: int


class AnalysisIntelligence(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    status: str
    current_stage: str | None = None
    snapshot_id: str | None = None
    waves: list[dict[str, Any]] = Field(default_factory=list)
    roles: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    coverage: list[dict[str, Any]] = Field(default_factory=list)
    model_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, int | float] = Field(default_factory=dict)


FindingReviewState = Literal[
    "new", "validated", "acknowledged", "reviewed", "dismissed", "reopened", "resolved"
]


class FindingReviewUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    review_state: FindingReviewState
    note: str | None = Field(default=None, max_length=2000)
    expected_version: int | None = Field(default=None, ge=0)


class FindingPage(BaseModel):
    items: list[dict[str, Any]]
    next_cursor: str | None = None
    counts_by_severity: dict[str, int] = Field(default_factory=dict)


class LocalSettings(BaseModel):
    """Durable non-secret defaults for local analysis runs."""

    model_config = ConfigDict(extra="forbid")

    default_mode: AnalysisMode = AnalysisMode.DEEP
    default_max_agents: int = Field(default=4, ge=1, le=12)
    default_max_waves: int = Field(default=2, ge=1, le=10)
    default_max_tasks: int = Field(default=100, ge=1, le=500)
    default_max_total_tokens: int = Field(default=1_000_000, ge=1)
    default_max_cost_usd: float = Field(default=25, ge=0)
    default_max_elapsed_seconds: int = Field(default=3_600, ge=1)
    evidence_excerpt_enabled: bool = True
    retention_days: int = Field(default=90, ge=1, le=3650)


class SettingsResponse(BaseModel):
    settings: LocalSettings
    version: int = Field(ge=0)
    updated_at: datetime | None = None


class SettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: LocalSettings
    expected_version: int = Field(ge=0)


class PresetRequest(BaseModel):
    """A reusable review shape that intentionally excludes repository paths and secrets."""

    model_config = ConfigDict(extra="forbid")

    goal: str | None = Field(default=None, max_length=4000)
    model_name: str | None = Field(default=None, max_length=300)
    max_agents: int = Field(default=4, ge=1, le=12)
    file_extensions: list[str] | None = Field(default=None, max_length=50)
    selected_directories: list[str] | None = Field(default=None, max_length=100)
    mode: AnalysisMode = AnalysisMode.DEEP
    max_waves: int = Field(default=2, ge=1, le=10)
    max_tasks: int = Field(default=100, ge=1, le=500)
    max_total_tokens: int = Field(default=1_000_000, ge=1)
    max_cost_usd: float = Field(default=25, ge=0)
    max_elapsed_seconds: int = Field(default=3_600, ge=1)


class PresetCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    request: PresetRequest


class PresetUpdate(PresetCreate):
    expected_version: int = Field(ge=1)


class PresetResponse(BaseModel):
    preset_id: str
    name: str
    request: PresetRequest
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class ProviderTestResponse(BaseModel):
    configured: bool
    reachable: bool
    status: str
    latency_ms: int | None = Field(default=None, ge=0)
