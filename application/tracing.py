"""Non-blocking observability boundary; durable events remain authoritative."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    run_id: str
    wave_id: str | None = None
    task_id: str | None = None
    model_call_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class TraceExporter(Protocol):
    def emit(self, event: TraceEvent) -> None: ...

    def close(self) -> None: ...


class NullTraceExporter:
    def emit(self, event: TraceEvent) -> None:
        del event

    def close(self) -> None:
        return None
