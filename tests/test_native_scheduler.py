from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from domain.contracts import AnalysisTask, RoleSpec
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.task_repository import SqliteTaskRepository
from workflow.task_scheduler import NativeTaskScheduler, TaskContext, TaskResult


def _role() -> RoleSpec:
    return RoleSpec(
        role_id="role-1",
        name="Architecture specialist",
        mission="Inspect module boundaries",
        rationale="Architecture is in scope",
        coverage_targets=("architecture",),
        model_policy="balanced",
        token_budget=10_000,
        time_budget_seconds=30,
        completion_criteria=("Return typed findings",),
    )


def test_native_workflow_import_does_not_load_legacy_framework_modules() -> None:
    import workflow

    assert workflow.NativeTaskScheduler is NativeTaskScheduler
    assert "workflow.nodes" not in sys.modules
    with pytest.raises(AttributeError):
        workflow.__getattr__("not_an_export")


def _task(
    task_id: str,
    *,
    task_type: str = "inspect",
    max_attempts: int = 3,
    time_budget_seconds: int = 5,
) -> AnalysisTask:
    return AnalysisTask(
        task_id=task_id,
        run_id="run-1",
        wave_id="wave-1",
        role_id="role-1",
        task_type=task_type,
        immutable_input_ids=("snapshot-1",),
        configuration_hash="config-1",
        idempotency_key=f"idem-{task_id}",
        model_policy="balanced",
        token_budget=1_000,
        cost_budget_usd=1,
        time_budget_seconds=time_budget_seconds,
        tool_call_budget=5,
        max_attempts=max_attempts,
    )


@asynccontextmanager
async def _repository(
    tmp_path: Path,
) -> AsyncIterator[tuple[SqliteRunLedger, SqliteTaskRepository]]:
    ledger = SqliteRunLedger(tmp_path / "application.sqlite3")
    ledger.create_run(
        run_id="run-1",
        submission_key="submission-1",
        request={"project_path": str(tmp_path)},
    )
    assert ledger.mark_running("run-1")
    repository = SqliteTaskRepository(ledger)
    repository.create_wave(
        run_id="run-1",
        wave_id="wave-1",
        wave_number=1,
        rationale="initial coverage",
        budget={"max_tasks": 10},
    )
    assert repository.add_role(run_id="run-1", wave_id="wave-1", role=_role())
    try:
        yield ledger, repository
    finally:
        ledger.close()


@pytest.mark.asyncio
async def test_scheduler_bounds_parallel_execution_and_persists_success(
    tmp_path: Path,
) -> None:
    async with _repository(tmp_path) as (_ledger, repository):
        for number in range(6):
            assert repository.enqueue(_task(f"task-{number}"))

        active = 0
        maximum_active = 0

        async def handler(_lease: Any, _context: TaskContext) -> TaskResult:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return TaskResult(usage={"tokens": 10})

        scheduler = NativeTaskScheduler(
            repository,
            {"inspect": handler},
            max_concurrent=2,
            cancellation_poll_seconds=0.001,
        )
        counts = await scheduler.run_until_idle(run_id="run-1")

        assert counts == {"succeeded": 6}
        assert maximum_active == 2


@pytest.mark.asyncio
async def test_scheduler_retries_then_succeeds(tmp_path: Path) -> None:
    async with _repository(tmp_path) as (_ledger, repository):
        repository.enqueue(_task("retry-task"))
        attempts = 0

        async def flaky(_lease: Any, _context: TaskContext) -> TaskResult:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient provider error")
            return TaskResult()

        scheduler = NativeTaskScheduler(
            repository,
            {"inspect": flaky},
            cancellation_poll_seconds=0.001,
        )
        counts = await scheduler.run_until_idle(run_id="run-1")

        assert counts == {"succeeded": 1}
        assert attempts == 2
        assert repository.get_task("retry-task")["attempt_count"] == 2


@pytest.mark.asyncio
async def test_scheduler_exhausts_attempts_for_missing_handler(tmp_path: Path) -> None:
    async with _repository(tmp_path) as (_ledger, repository):
        repository.enqueue(_task("unknown-task", task_type="unknown", max_attempts=2))

        scheduler = NativeTaskScheduler(repository, {}, max_concurrent=1)
        counts = await scheduler.run_until_idle(run_id="run-1")

        assert counts == {"failed": 1}
        task = repository.get_task("unknown-task")
        assert task["attempt_count"] == 2
        assert "no native handler" in task["error"]["message"]


@pytest.mark.asyncio
async def test_scheduler_times_out_and_records_failure(tmp_path: Path) -> None:
    async with _repository(tmp_path) as (_ledger, repository):
        repository.enqueue(
            _task("slow-task", max_attempts=1, time_budget_seconds=1)
        )

        async def slow(_lease: Any, _context: TaskContext) -> TaskResult:
            await asyncio.sleep(5)
            return TaskResult()

        scheduler = NativeTaskScheduler(
            repository,
            {"inspect": slow},
            cancellation_poll_seconds=0.001,
        )
        counts = await scheduler.run_until_idle(run_id="run-1")

        assert counts == {"failed": 1}
        assert "exceeded 1s" in repository.get_task("slow-task")["error"]["message"]


@pytest.mark.asyncio
async def test_run_cancellation_stops_active_and_queued_tasks(tmp_path: Path) -> None:
    async with _repository(tmp_path) as (_ledger, repository):
        repository.enqueue(_task("active-task"))
        repository.enqueue(_task("queued-task"))
        started = asyncio.Event()

        async def waiting(_lease: Any, _context: TaskContext) -> TaskResult:
            started.set()
            await asyncio.Event().wait()
            return TaskResult()

        scheduler = NativeTaskScheduler(
            repository,
            {"inspect": waiting},
            max_concurrent=1,
            cancellation_poll_seconds=0.001,
        )
        draining = asyncio.create_task(scheduler.run_until_idle(run_id="run-1"))
        await asyncio.wait_for(started.wait(), timeout=1)
        assert repository.request_run_cancellation("run-1") == 2
        counts = await asyncio.wait_for(draining, timeout=1)

        assert counts == {"cancelled": 2}
        assert repository.get_task("active-task")["cancellation_requested"]
        assert repository.get_task("queued-task")["cancellation_requested"]


@pytest.mark.asyncio
async def test_scheduler_interruption_releases_task_for_resume(tmp_path: Path) -> None:
    async with _repository(tmp_path) as (_ledger, repository):
        repository.enqueue(_task("interrupted-task"))
        started = asyncio.Event()

        async def waiting(_lease: Any, _context: TaskContext) -> TaskResult:
            started.set()
            await asyncio.Event().wait()
            return TaskResult()

        scheduler = NativeTaskScheduler(
            repository,
            {"inspect": waiting},
            cancellation_poll_seconds=0.001,
        )
        draining = asyncio.create_task(scheduler.run_until_idle(run_id="run-1"))
        await asyncio.wait_for(started.wait(), timeout=1)
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining

        task = repository.get_task("interrupted-task")
        assert task["status"] == "retry_wait"
        assert task["lease_owner"] is None


def test_expired_lease_recovery_requeues_then_exhausts(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _repository(tmp_path) as (_ledger, repository):
            repository.enqueue(_task("expired-task", max_attempts=2))
            lease_time = datetime.now(UTC)
            first = repository.lease(
                worker_id="dead-worker",
                limit=1,
                lease_seconds=1,
                run_id="run-1",
                now=lease_time,
            )
            assert len(first) == 1
            assert (
                repository.recover_expired(now=lease_time + timedelta(seconds=2)) == 1
            )
            assert repository.get_task("expired-task")["status"] == "retry_wait"

            second = repository.lease(
                worker_id="dead-worker-2",
                limit=1,
                lease_seconds=1,
                run_id="run-1",
                now=lease_time + timedelta(seconds=3),
            )
            assert len(second) == 1
            assert (
                repository.recover_expired(now=lease_time + timedelta(seconds=5)) == 1
            )
            assert repository.get_task("expired-task")["status"] == "failed"

    asyncio.run(scenario())


def test_concurrent_workers_never_receive_the_same_lease(tmp_path: Path) -> None:
    async def setup() -> None:
        async with _repository(tmp_path) as (_ledger, repository):
            for number in range(10):
                repository.enqueue(_task(f"concurrent-{number}"))

    asyncio.run(setup())
    ledger_one = SqliteRunLedger(tmp_path / "application.sqlite3")
    ledger_two = SqliteRunLedger(tmp_path / "application.sqlite3")
    repository_one = SqliteTaskRepository(ledger_one)
    repository_two = SqliteTaskRepository(ledger_two)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_one = pool.submit(
                repository_one.lease,
                worker_id="worker-one",
                limit=10,
                lease_seconds=30,
                run_id="run-1",
            )
            future_two = pool.submit(
                repository_two.lease,
                worker_id="worker-two",
                limit=10,
                lease_seconds=30,
                run_id="run-1",
            )
            leases = [*future_one.result(), *future_two.result()]
        task_ids = [lease.task_id for lease in leases]
        assert len(task_ids) == 10
        assert len(set(task_ids)) == 10
    finally:
        ledger_one.close()
        ledger_two.close()


def test_enqueue_and_wave_creation_are_idempotent(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _repository(tmp_path) as (_ledger, repository):
            repository.create_wave(
                run_id="run-1",
                wave_id="different-id",
                wave_number=1,
                rationale="duplicate",
                budget={},
            )
            assert repository.enqueue(_task("task-1"))
            duplicate = _task("different-id").model_copy(
                update={"idempotency_key": "idem-task-1"}
            )
            assert not repository.enqueue(duplicate)
            assert repository.task_counts("run-1") == {"queued": 1}

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_scheduler_throughput_smoke(tmp_path: Path) -> None:
    async with _repository(tmp_path) as (_ledger, repository):
        for number in range(100):
            repository.enqueue(_task(f"throughput-{number}"))

        async def immediate(_lease: Any, _context: TaskContext) -> TaskResult:
            return TaskResult()

        scheduler = NativeTaskScheduler(
            repository,
            {"inspect": immediate},
            max_concurrent=8,
            cancellation_poll_seconds=0.001,
        )
        started = time.perf_counter()
        counts = await scheduler.run_until_idle(run_id="run-1")
        elapsed = time.perf_counter() - started

        assert counts == {"succeeded": 100}
        assert elapsed < 5
