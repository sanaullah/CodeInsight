"""Typed request and response models for the CodeInsight API."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisRequest(BaseModel):
    """A bounded request for the existing swarm-analysis engine."""

    model_config = ConfigDict(str_strip_whitespace=True)

    project_path: str = Field(min_length=1, max_length=4096)
    goal: str | None = Field(default=None, max_length=4000)
    model_name: str | None = Field(default=None, max_length=300)
    max_agents: int = Field(default=4, ge=1, le=12)
    max_tokens_per_chunk: int = Field(default=50_000, ge=4_000, le=200_000)
    enable_chunking: bool = True
    chunking_strategy: Literal["NONE", "STANDARD", "AGGRESSIVE"] = "STANDARD"
    auto_detect_languages: bool = True
    file_extensions: list[str] | None = Field(default=None, max_length=50)
    selected_directories: list[str] | None = Field(default=None, max_length=100)
    enable_dynamic_file_selection: bool = True
    enable_tool_calling: bool = False

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


class AnalysisAccepted(BaseModel):
    run_id: str
    status: AnalysisStatus
    status_url: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    active_analyses: int
    max_concurrent_analyses: int


class LanguageCapability(BaseModel):
    language: str
    display_name: str
    support_level: Literal["parsed", "dependency-aware", "discovery"]
    extensions: list[str]


class CapabilitiesResponse(BaseModel):
    languages: list[LanguageCapability]
    default_max_agents: int = 4
    max_agents: int = 12
