"""In-process analysis run coordination behind a stable API boundary."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from api.models import (
    AnalysisEvent,
    AnalysisRequest,
    AnalysisRun,
    AnalysisStatus,
    utc_now,
)

EventSink = Callable[[str, dict[str, Any]], None]


class AnalysisExecutor(Protocol):
    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]: ...


class SwarmAnalysisExecutor:
    """Adapter from the HTTP boundary to the existing LangGraph orchestrator."""

    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
        # Lazy import keeps API health, capabilities and the frontend available even
        # when optional model or observability configuration is incomplete.
        from analysis.agents.swarm_analysis_orchestrator import SwarmAnalysisOrchestrator

        orchestrator = SwarmAnalysisOrchestrator(
            model_name=request.model_name,
            auto_detect_languages=request.auto_detect_languages,
            file_extensions=request.file_extensions,
        )
        return await orchestrator.analyze(
            project_path=request.project_path,
            goal=request.goal,
            model_name=request.model_name,
            max_agents=request.max_agents,
            stream_callback=event_sink,
            auto_detect_languages=request.auto_detect_languages,
            file_extensions=request.file_extensions,
            selected_directories=request.selected_directories,
            max_tokens_per_chunk=request.max_tokens_per_chunk,
            enable_chunking=request.enable_chunking,
            chunking_strategy=request.chunking_strategy,
            enable_dynamic_file_selection=request.enable_dynamic_file_selection,
            enable_tool_calling=request.enable_tool_calling,
        )


class AnalysisService:
    """Owns bounded run state while persistence is introduced in a later phase."""

    def __init__(
        self,
        executor: AnalysisExecutor | None = None,
        *,
        max_concurrent: int = 2,
        event_history_limit: int = 200,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be greater than zero")
        if event_history_limit < 1:
            raise ValueError("event_history_limit must be greater than zero")
        self.executor = executor or SwarmAnalysisExecutor()
        self.max_concurrent = max_concurrent
        self.event_history_limit = event_history_limit
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._runs: dict[str, AnalysisRun] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    @property
    def active_count(self) -> int:
        return sum(
            run.status in {AnalysisStatus.QUEUED, AnalysisStatus.RUNNING}
            for run in self._runs.values()
        )

    async def submit(self, request: AnalysisRequest) -> AnalysisRun:
        normalized_request = request.model_copy(
            update={"project_path": self._resolve_project_path(request.project_path)}
        )
        run = AnalysisRun(
            run_id=uuid4().hex,
            status=AnalysisStatus.QUEUED,
            request=normalized_request,
            created_at=utc_now(),
        )
        async with self._lock:
            self._runs[run.run_id] = run
            task = asyncio.create_task(
                self._execute_run(run.run_id), name=f"analysis-{run.run_id}"
            )
            self._tasks[run.run_id] = task
            task.add_done_callback(
                lambda _task, run_id=run.run_id: self._tasks.pop(run_id, None)
            )
        return self._snapshot(run)

    async def get(self, run_id: str) -> AnalysisRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            return self._snapshot(run) if run else None

    async def list(self, limit: int = 20) -> list[AnalysisRun]:
        async with self._lock:
            runs = sorted(
                self._runs.values(), key=lambda item: item.created_at, reverse=True
            )
            return [self._snapshot(run) for run in runs[:limit]]

    async def cancel(self, run_id: str) -> AnalysisRun | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            task = self._tasks.get(run_id)
            if run.status in {AnalysisStatus.QUEUED, AnalysisStatus.RUNNING}:
                run.status = AnalysisStatus.CANCELLED
                run.completed_at = utc_now()
                self._append_event(run, "analysis_cancelled", {})
                if task:
                    task.cancel()
            return self._snapshot(run)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_run(self, run_id: str) -> None:
        try:
            async with self._semaphore:
                async with self._lock:
                    run = self._runs[run_id]
                    if run.status == AnalysisStatus.CANCELLED:
                        return
                    run.status = AnalysisStatus.RUNNING
                    run.started_at = utc_now()
                    self._append_event(run, "analysis_started", {})

                def event_sink(event_type: str, data: dict[str, Any]) -> None:
                    # Existing callbacks are synchronous. Each analysis task owns its
                    # run, so this bounded append cannot race with another writer.
                    current_run = self._runs.get(run_id)
                    if current_run is not None:
                        self._append_event(current_run, event_type, deepcopy(data))

                result = await self.executor.execute(run.request, event_sink)
                async with self._lock:
                    run = self._runs[run_id]
                    if run.status != AnalysisStatus.CANCELLED:
                        run.status = AnalysisStatus.SUCCEEDED
                        run.result = deepcopy(result)
                        run.completed_at = utc_now()
                        self._append_event(run, "analysis_completed", {})
        except asyncio.CancelledError:
            async with self._lock:
                run = self._runs.get(run_id)
                if run and run.status != AnalysisStatus.CANCELLED:
                    run.status = AnalysisStatus.CANCELLED
                    run.completed_at = utc_now()
                    self._append_event(run, "analysis_cancelled", {})
            raise
        except Exception as exc:
            async with self._lock:
                run = self._runs[run_id]
                run.status = AnalysisStatus.FAILED
                run.error = str(exc)
                run.completed_at = utc_now()
                self._append_event(run, "analysis_failed", {"error": str(exc)})

    def _append_event(
        self, run: AnalysisRun, event_type: str, data: dict[str, Any]
    ) -> None:
        history = deque(run.events, maxlen=self.event_history_limit)
        next_sequence = history[-1].sequence + 1 if history else 1
        history.append(
            AnalysisEvent(
                sequence=next_sequence,
                event_type=event_type,
                timestamp=utc_now(),
                data=data,
            )
        )
        run.events = list(history)

    @staticmethod
    def _resolve_project_path(project_path: str) -> str:
        try:
            resolved = Path(project_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Project path does not exist: {project_path}") from exc
        if not resolved.is_dir():
            raise ValueError(f"Project path is not a directory: {project_path}")
        return str(resolved)

    @staticmethod
    def _snapshot(run: AnalysisRun) -> AnalysisRun:
        return run.model_copy(deep=True)
