"""Durable analysis run coordination behind the HTTP boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from api.config import default_database_path
from api.models import AnalysisRequest, AnalysisRun
from infrastructure.db.database import checkpoint_database
from infrastructure.db.run_ledger import SqliteRunLedger

EventSink = Callable[[str, dict[str, Any]], None]


class AnalysisExecutor(Protocol):
    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]: ...


class SwarmAnalysisExecutor:
    """Temporary adapter to the legacy engine during native-workflow migration."""

    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
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
    """Runs bounded work while SQLite remains the authoritative lifecycle state."""

    def __init__(
        self,
        executor: AnalysisExecutor | None = None,
        *,
        database_path: str | Path | None = None,
        max_concurrent: int = 2,
        event_history_limit: int = 200,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be greater than zero")
        if event_history_limit < 1:
            raise ValueError("event_history_limit must be greater than zero")
        self.executor = executor or SwarmAnalysisExecutor()
        self.database_path = Path(database_path or default_database_path()).resolve()
        self.max_concurrent = max_concurrent
        self.event_history_limit = event_history_limit
        self._ledger: SqliteRunLedger | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._task_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closing = False

    def _get_ledger(self) -> SqliteRunLedger:
        if self._ledger is None:
            self._ledger = SqliteRunLedger(
                self.database_path,
                event_history_limit=self.event_history_limit,
            )
        return self._ledger

    @property
    def active_count(self) -> int:
        return self._get_ledger().count_active()

    async def start(self) -> int:
        """Initialize storage, recover interrupted work, and resume queued runs."""

        async with self._start_lock:
            if self._started:
                return 0
            self._closing = False
            ledger = self._get_ledger()
            recovered = ledger.recover_interrupted()
            self._started = True
            for run_id in ledger.list_queued_run_ids():
                await self._schedule(run_id)
            return recovered

    async def submit(self, request: AnalysisRequest) -> AnalysisRun:
        await self.start()
        normalized_request = request.model_copy(
            update={"project_path": self._resolve_project_path(request.project_path)}
        )
        run_id = uuid4().hex
        record = self._get_ledger().create_run(
            run_id=run_id,
            submission_key=run_id,
            request=normalized_request.model_dump(mode="json"),
        )
        await self._schedule(record["run_id"])
        return self._to_model(record)

    async def get(self, run_id: str) -> AnalysisRun | None:
        await self.start()
        record = self._get_ledger().get_run(run_id)
        return self._to_model(record) if record else None

    async def list(self, limit: int = 20) -> list[AnalysisRun]:
        await self.start()
        return [
            self._to_model(record) for record in self._get_ledger().list_runs(limit)
        ]

    async def cancel(self, run_id: str) -> AnalysisRun | None:
        await self.start()
        ledger = self._get_ledger()
        if ledger.get_run(run_id) is None:
            return None
        ledger.cancel(run_id)
        async with self._task_lock:
            task = self._tasks.get(run_id)
            if task is not None:
                task.cancel()
        record = ledger.get_run(run_id)
        return self._to_model(record) if record else None

    async def close(self) -> None:
        """Stop local workers while leaving unfinished work recoverable."""

        self._closing = True
        async with self._task_lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._ledger is not None:
            self._ledger.close()
            checkpoint_database(self.database_path)
            self._ledger = None
        self._started = False

    async def _schedule(self, run_id: str) -> None:
        async with self._task_lock:
            existing = self._tasks.get(run_id)
            if existing is not None and not existing.done():
                return
            task = asyncio.create_task(
                self._execute_run(run_id), name=f"analysis-{run_id}"
            )
            self._tasks[run_id] = task
            task.add_done_callback(
                lambda _task, scheduled_run_id=run_id: self._tasks.pop(
                    scheduled_run_id, None
                )
            )

    async def _execute_run(self, run_id: str) -> None:
        ledger = self._get_ledger()
        try:
            async with self._semaphore:
                if not ledger.mark_running(run_id):
                    return
                record = ledger.get_run(run_id)
                if record is None:
                    return
                request = AnalysisRequest.model_validate(record["request"])

                def event_sink(event_type: str, data: dict[str, Any]) -> None:
                    ledger.append_event(run_id, event_type, data)

                result = await self.executor.execute(request, event_sink)
                ledger.succeed(run_id, result)
        except asyncio.CancelledError:
            if self._closing:
                ledger.requeue_interrupted(run_id)
            else:
                ledger.cancel(run_id)
            raise
        except Exception as exc:
            ledger.fail(run_id, str(exc))

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
    def _to_model(record: dict[str, Any]) -> AnalysisRun:
        return AnalysisRun.model_validate(record)
