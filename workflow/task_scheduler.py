"""Framework-free bounded scheduler for durable analysis tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from infrastructure.db.task_repository import SqliteTaskRepository, TaskLease

TaskHandler = Callable[[TaskLease, "TaskContext"], Awaitable["TaskResult"]]
SchedulerEventSink = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class TaskResult:
    output_artifact_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class TaskContext:
    """Cancellation boundary supplied to native task handlers."""

    def __init__(self, repository: SqliteTaskRepository, task_id: str) -> None:
        self._repository = repository
        self.task_id = task_id

    @property
    def cancellation_requested(self) -> bool:
        return self._repository.is_cancellation_requested(self.task_id)

    def raise_if_cancelled(self) -> None:
        if self.cancellation_requested:
            raise asyncio.CancelledError


class NativeTaskScheduler:
    """Lease and execute durable tasks with bounded local concurrency."""

    def __init__(
        self,
        repository: SqliteTaskRepository,
        handlers: Mapping[str, TaskHandler],
        *,
        max_concurrent: int = 4,
        lease_seconds: float = 300,
        retry_delay_seconds: float = 0,
        cancellation_poll_seconds: float = 0.05,
        worker_id: str | None = None,
        event_sink: SchedulerEventSink | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be greater than zero")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be greater than zero")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        if cancellation_poll_seconds <= 0:
            raise ValueError("cancellation_poll_seconds must be greater than zero")
        self.repository = repository
        self.handlers = dict(handlers)
        self.max_concurrent = max_concurrent
        self.lease_seconds = lease_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.cancellation_poll_seconds = cancellation_poll_seconds
        self.worker_id = worker_id or f"local-{uuid4().hex}"
        self.event_sink = event_sink or (lambda _event, _data: None)

    async def run_until_idle(self, *, run_id: str | None = None) -> dict[str, int]:
        """Drain runnable work, including immediate retries, then return counts."""

        self.repository.recover_expired()
        while True:
            leases = self.repository.lease(
                worker_id=self.worker_id,
                limit=self.max_concurrent,
                lease_seconds=self.lease_seconds,
                run_id=run_id,
            )
            if not leases:
                break
            await asyncio.gather(*(self._execute(lease) for lease in leases))
        return self.repository.task_counts(run_id) if run_id is not None else {}

    async def _execute(self, lease: TaskLease) -> None:
        handler = self.handlers.get(lease.task_type)
        if handler is None:
            status = self.repository.fail(
                lease,
                error=f"no native handler registered for task type {lease.task_type!r}",
                retry_delay_seconds=self.retry_delay_seconds,
            )
            self._emit("task_failed", lease, {"status": status, "reason": "no_handler"})
            return

        self._emit(
            "task_started",
            lease,
            {"attempt": lease.attempt_number, "worker_id": self.worker_id},
        )
        context = TaskContext(self.repository, lease.task_id)
        execution = asyncio.create_task(handler(lease, context))
        cancellation_monitor = asyncio.create_task(self._wait_for_cancellation(context))
        timeout = float(lease.budget.get("time_budget_seconds", self.lease_seconds))
        try:
            done, _pending = await asyncio.wait(
                {execution, cancellation_monitor},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if execution in done:
                cancellation_monitor.cancel()
                await asyncio.gather(cancellation_monitor, return_exceptions=True)
                try:
                    result = await execution
                except asyncio.CancelledError:
                    if context.cancellation_requested:
                        self.repository.cancel_task(
                            lease, reason="cancellation requested"
                        )
                        self._emit(
                            "task_cancelled",
                            lease,
                            {"reason": "cancellation requested"},
                        )
                        return
                    raise
                if not self.repository.succeed(
                    lease,
                    output_artifact_id=result.output_artifact_id,
                    usage=result.usage,
                ):
                    raise RuntimeError("task lease became stale before completion")
                self._emit("task_succeeded", lease, {"attempt": lease.attempt_number})
                return
            execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            reason = (
                "cancellation requested"
                if cancellation_monitor in done
                else f"task exceeded {timeout:g}s time budget"
            )
            if cancellation_monitor not in done:
                cancellation_monitor.cancel()
                await asyncio.gather(cancellation_monitor, return_exceptions=True)
                status = self.repository.fail(
                    lease,
                    error=reason,
                    retry_delay_seconds=self.retry_delay_seconds,
                )
                self._emit("task_failed", lease, {"status": status, "reason": "timeout"})
            else:
                self.repository.cancel_task(lease, reason=reason)
                self._emit("task_cancelled", lease, {"reason": reason})
        except asyncio.CancelledError:
            execution.cancel()
            cancellation_monitor.cancel()
            await asyncio.gather(
                execution, cancellation_monitor, return_exceptions=True
            )
            self.repository.interrupt_task(lease, reason="scheduler interrupted")
            self._emit("task_interrupted", lease, {"reason": "scheduler interrupted"})
            raise
        except Exception as exc:
            cancellation_monitor.cancel()
            await asyncio.gather(cancellation_monitor, return_exceptions=True)
            status = self.repository.fail(
                lease,
                error=str(exc),
                retry_delay_seconds=self.retry_delay_seconds,
            )
            self._emit("task_failed", lease, {"status": status, "reason": str(exc)})

    async def _wait_for_cancellation(self, context: TaskContext) -> None:
        # SQLite is the cross-process cancellation source, so bounded polling is
        # intentional; an in-memory Event would miss cancellation from other workers.
        while not context.cancellation_requested:  # noqa: ASYNC110
            await asyncio.sleep(self.cancellation_poll_seconds)

    def _emit(self, event_type: str, lease: TaskLease, data: dict[str, Any]) -> None:
        self.event_sink(
            event_type,
            {
                "run_id": lease.run_id,
                "wave_id": lease.wave_id,
                "task_id": lease.task_id,
                **data,
            },
        )
