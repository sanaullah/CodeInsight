from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from application.analysis_service import AnalysisService, EventSink
from api.analysis_service import AnalysisService as LegacyAnalysisService
from api.models import AnalysisRequest, AnalysisStatus


def test_legacy_analysis_service_import_is_compatible() -> None:
    assert LegacyAnalysisService is AnalysisService


class FakeExecutor:
    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
        event_sink("scan_completed", {"project": request.project_path})
        await asyncio.sleep(0)
        return {"synthesized_report": "Verified report", "files_scanned": 3}


@pytest.mark.asyncio
async def test_submit_executes_and_records_events(tmp_path: Path) -> None:
    service = AnalysisService(FakeExecutor(), max_concurrent=1)
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
        "analysis_started",
        "scan_completed",
        "analysis_completed",
    ]
    await service.close()


@pytest.mark.asyncio
async def test_submit_rejects_missing_directory(tmp_path: Path) -> None:
    service = AnalysisService(FakeExecutor())
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
