"""Non-blocking observability boundary; durable events remain authoritative."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    run_id: str
    stage: str | None = None
    wave_id: str | None = None
    role_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    attempt_number: int | None = None
    model_call_id: str | None = None
    prompt_artifact_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class TraceExporter(Protocol):
    def emit(self, event: TraceEvent) -> None: ...

    def close(self) -> None: ...


class NullTraceExporter:
    def emit(self, event: TraceEvent) -> None:
        del event

    def close(self) -> None:
        return None
