from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from api.analysis_service import AnalysisService as LegacyAnalysisService
from api.models import AnalysisRequest, AnalysisStatus
from application.analysis_service import AnalysisService, EventSink
from infrastructure.db.run_ledger import SqliteRunLedger


def test_legacy_analysis_service_import_is_compatible() -> None:
    assert LegacyAnalysisService is AnalysisService


class FakeExecutor:
    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
        event_sink("scan_completed", {"project": request.project_path})
        await asyncio.sleep(0)
        return {"synthesized_report": "Verified report", "files_scanned": 3}


class BlockingExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_submit_executes_and_records_events(tmp_path: Path) -> None:
    service = AnalysisService(
        FakeExecutor(),
        database_path=tmp_path / "codeinsight.db",
        max_concurrent=1,
    )
    submitted = await service.submit(
        AnalysisRequest(project_path=str(tmp_path), goal="Review boundary behavior")
    )

    assert submitted.status == AnalysisStatus.QUEUED

    for _ in range(20):
        run = await service.get(submitted.run_id)
        assert run is not None
        if run.status == AnalysisStatus.SUCCEEDED:
            break
        await asyncio.sleep(0.01)

    assert run.status == AnalysisStatus.SUCCEEDED
    assert run.request.project_path == str(tmp_path.resolve())
    assert run.result == {
        "synthesized_report": "Verified report",
        "files_scanned": 3,
    }
    assert [event.event_type for event in run.events] == [
        "analysis_queued",
        "analysis_started",
        "scan_completed",
        "analysis_completed",
    ]
    await service.close()


@pytest.mark.asyncio
async def test_shutdown_requeues_and_restart_recovers_run(tmp_path: Path) -> None:
    database_path = tmp_path / "codeinsight.db"
    blocking = BlockingExecutor()
    first_service = AnalysisService(
        blocking, database_path=database_path, max_concurrent=1
    )
    submitted = await first_service.submit(
        AnalysisRequest(project_path=str(tmp_path))
    )
    await asyncio.wait_for(blocking.started.wait(), timeout=1)
    await first_service.close()

    interrupted = SqliteRunLedger(database_path).get_run(submitted.run_id)
    assert interrupted is not None
    assert interrupted["status"] == AnalysisStatus.QUEUED

    second_service = AnalysisService(
        FakeExecutor(), database_path=database_path, max_concurrent=1
    )
    recovered_count = await second_service.start()
    assert recovered_count == 0
    for _ in range(30):
        recovered = await second_service.get(submitted.run_id)
        assert recovered is not None
        if recovered.status == AnalysisStatus.SUCCEEDED:
            break
        await asyncio.sleep(0.01)

    assert recovered.status == AnalysisStatus.SUCCEEDED
    assert [event.event_type for event in recovered.events] == [
        "analysis_queued",
        "analysis_started",
        "analysis_interrupted",
        "analysis_started",
        "scan_completed",
        "analysis_completed",
    ]
    await second_service.close()


@pytest.mark.asyncio
async def test_cancel_persists_and_stops_running_executor(tmp_path: Path) -> None:
    blocking = BlockingExecutor()
    service = AnalysisService(
        blocking,
        database_path=tmp_path / "codeinsight.db",
        max_concurrent=1,
    )
    submitted = await service.submit(AnalysisRequest(project_path=str(tmp_path)))
    await asyncio.wait_for(blocking.started.wait(), timeout=1)

    cancelled = await service.cancel(submitted.run_id)
    assert cancelled is not None
    assert cancelled.status == AnalysisStatus.CANCELLED
    await asyncio.sleep(0)
    persisted = await service.get(submitted.run_id)
    assert persisted is not None
    assert persisted.status == AnalysisStatus.CANCELLED
    assert persisted.events[-1].event_type == "analysis_cancelled"
    await service.close()


@pytest.mark.asyncio
async def test_submit_rejects_missing_directory(tmp_path: Path) -> None:
    service = AnalysisService(
        FakeExecutor(), database_path=tmp_path / "codeinsight.db"
    )
    with pytest.raises(ValueError, match="does not exist"):
        await service.submit(
            AnalysisRequest(project_path=str(tmp_path / "missing"))
        )


def test_request_normalizes_extensions() -> None:
    request = AnalysisRequest(
        project_path=".",
        file_extensions=["py", ".TS", " py "],
    )
    assert request.file_extensions == [".py", ".ts"]


def test_request_rejects_extension_wildcards() -> None:
    with pytest.raises(ValueError, match="wildcards"):
        AnalysisRequest(project_path=".", file_extensions=["*.py"])


@pytest.mark.parametrize(
    "directory", ["../sibling", "/etc", r"C:\Users", r"src\..\secrets"]
)
def test_request_rejects_directories_outside_project(directory: str) -> None:
    with pytest.raises(ValueError, match="project root"):
        AnalysisRequest(project_path=".", selected_directories=[directory])
