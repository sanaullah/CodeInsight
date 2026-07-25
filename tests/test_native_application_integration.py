from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import ApiSettings
from api.models import AnalysisRequest, AnalysisStatus
from application.analysis_service import AnalysisService
from application.model_gateway import ModelRequest, ModelResponse, ModelUsage
from infrastructure.db.database import database_connection


class EvidenceGateway:
    def __init__(self, *, delay: float = 0) -> None:
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.started = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        self.active += 1
        self.started.set()
        self.max_active = max(self.max_active, self.active)
        if self.delay:
            await asyncio.sleep(self.delay)
        payload = json.loads(request.user_prompt)
        file = payload["files"][0]
        self.active -= 1
        return ModelResponse(
            content={
                "analyzed_paths": [item["relative_path"] for item in payload["files"]],
                "evidence": [
                    {
                        "relative_path": file["relative_path"],
                        "start_line": 1,
                        "end_line": 1,
                    }
                ],
                "findings": [
                    {
                        "category": "quality",
                        "concept_id": "integration-proof",
                        "title": "Verified integration finding",
                        "claim": "The first assigned source line demonstrates the fixture.",
                        "evidence_indexes": [0],
                        "affected_path": file["relative_path"],
                        "impact": "Proves the native application path.",
                        "severity": "medium",
                        "confidence": 0.9,
                        "recommendation": "Keep the behavior covered.",
                        "fingerprint_inputs": ["integration-proof"],
                    }
                ],
                "unresolved_uncertainty": [],
                "usage": {},
            },
            provider="fixture",
            model=request.model,
            usage=ModelUsage(input_tokens=20, output_tokens=10, cost_usd=0.001),
        )


class FailingGateway:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, _request: ModelRequest) -> ModelResponse:
        self.calls += 1
        raise OSError("local provider unavailable")


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "tests").mkdir(parents=True)
    (root / "app.py").write_text(
        "from helper import calculate\n\ndef run():\n    return calculate(2)\n",
        encoding="utf-8",
    )
    (root / "helper.py").write_text(
        "def calculate(value):\n    return value + 1\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_app.py").write_text(
        "from app import run\n\ndef test_run():\n    assert run() == 3\n",
        encoding="utf-8",
    )
    return root


async def _wait_for_terminal(
    service: AnalysisService, run_id: str
) -> tuple[AnalysisStatus, object]:
    for _ in range(200):
        run = await service.get(run_id)
        assert run is not None
        if run.status in {
            AnalysisStatus.SUCCEEDED,
            AnalysisStatus.FAILED,
            AnalysisStatus.CANCELLED,
            AnalysisStatus.NEEDS_ATTENTION,
        }:
            return run.status, run
        await asyncio.sleep(0.01)
    raise AssertionError("analysis did not reach a terminal state")


@pytest.mark.asyncio
async def test_native_application_path_persists_full_intelligence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state" / "codeinsight.db"
    gateway = EvidenceGateway(delay=0.005)
    settings = ApiSettings(
        database_path=database_path,
        default_model="fixture-model",
        max_concurrent_model_calls=2,
    )
    service = AnalysisService(
        database_path=database_path,
        max_concurrent=1,
        settings=settings,
        gateway=gateway,
    )
    submitted = await service.submit(
        AnalysisRequest(
            project_path=str(_repository(tmp_path)),
            max_agents=3,
            max_waves=1,
            max_tasks=3,
            mode="deep",
        )
    )
    status, run = await _wait_for_terminal(service, submitted.run_id)
    assert status == AnalysisStatus.SUCCEEDED
    assert run.current_stage == "complete"
    assert run.snapshot_id
    assert run.result["provider_mode"] == "model-backed"
    assert run.result["findings"]

    intelligence = await service.intelligence(submitted.run_id)
    assert intelligence is not None
    assert intelligence.current_stage == "complete"
    assert len(intelligence.roles) == 3
    assert len(intelligence.tasks) == 3
    assert all(task["status"] == "succeeded" for task in intelligence.tasks)
    assert intelligence.findings
    assert intelligence.coverage
    assert len(intelligence.model_calls) == gateway.calls
    assert intelligence.usage["total_tokens"] == gateway.calls * 30
    assert gateway.max_active <= 2
    await service.close()


@pytest.mark.asyncio
async def test_unconfigured_provider_completes_truthful_index_only_run(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "codeinsight.db"
    service = AnalysisService(
        database_path=database_path,
        max_concurrent=1,
        settings=ApiSettings(database_path=database_path),
    )
    submitted = await service.submit(
        AnalysisRequest(
            project_path=str(_repository(tmp_path)),
            max_agents=2,
            max_waves=3,
        )
    )
    status, run = await _wait_for_terminal(service, submitted.run_id)
    assert status == AnalysisStatus.SUCCEEDED
    assert run.result["provider_mode"] == "index-only"
    assert run.result["wave_count"] == 1
    assert run.result["findings"] == []
    assert "No evidence-backed findings" in run.result["synthesized_report"]
    intelligence = await service.intelligence(submitted.run_id)
    assert intelligence is not None
    assert intelligence.model_calls
    assert all(
        item["usage"].get("total_tokens", 0) == 0
        for item in intelligence.model_calls
    )
    await service.close()


@pytest.mark.asyncio
async def test_provider_failures_are_bounded_and_surface_needs_attention(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "provider-failure.db"
    gateway = FailingGateway()
    service = AnalysisService(
        database_path=database_path,
        max_concurrent=1,
        settings=ApiSettings(database_path=database_path),
        gateway=gateway,
    )
    submitted = await service.submit(
        AnalysisRequest(
            project_path=str(_repository(tmp_path)),
            max_agents=1,
            max_tasks=1,
            max_waves=1,
        )
    )
    status, run = await _wait_for_terminal(service, submitted.run_id)
    assert status == AnalysisStatus.NEEDS_ATTENTION
    assert run.result["failed_task_count"] == 1
    assert run.events[-1].event_type == "analysis_needs_attention"
    intelligence = await service.intelligence(submitted.run_id)
    assert intelligence is not None
    assert intelligence.tasks[0]["status"] == "failed"
    assert intelligence.model_calls[0]["status"] == "failed"
    assert gateway.calls == 3
    await service.close()


@pytest.mark.asyncio
async def test_native_interrupted_wave_resumes_without_duplicate_plan(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "recovery.db"
    root = _repository(tmp_path)
    blocking_gateway = EvidenceGateway(delay=10)
    first = AnalysisService(
        database_path=database_path,
        max_concurrent=1,
        settings=ApiSettings(database_path=database_path),
        gateway=blocking_gateway,
    )
    submitted = await first.submit(
        AnalysisRequest(
            project_path=str(root),
            max_agents=1,
            max_tasks=1,
            max_waves=1,
        )
    )
    await asyncio.wait_for(blocking_gateway.started.wait(), timeout=2)
    await first.close()

    second = AnalysisService(
        database_path=database_path,
        max_concurrent=1,
        settings=ApiSettings(database_path=database_path),
        gateway=EvidenceGateway(),
    )
    assert await second.start() == 0
    status, run = await _wait_for_terminal(second, submitted.run_id)
    assert status == AnalysisStatus.SUCCEEDED
    assert sum(event.event_type == "wave_planned" for event in run.events) == 1
    assert any(event.event_type == "wave_resumed" for event in run.events)
    intelligence = await second.intelligence(submitted.run_id)
    assert intelligence is not None
    assert len(intelligence.waves) == 1
    assert len(intelligence.tasks) == 1
    assert intelligence.tasks[0]["attempt_count"] == 2
    await second.close()


def test_fastapi_exposes_native_status_intelligence_recovery_and_ui(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    database_path = tmp_path / "api.db"
    service = AnalysisService(
        database_path=database_path,
        max_concurrent=1,
        settings=ApiSettings(database_path=database_path),
        gateway=EvidenceGateway(),
    )
    with TestClient(create_app(service)) as client:
        capabilities = client.get("/api/v1/capabilities").json()
        assert capabilities["native_durable_workflow"]
        assert capabilities["model_provider_configured"]
        assert not capabilities["langfuse_enabled"]

        created = client.post(
            "/api/v1/analyses",
            json={
                "project_path": str(root),
                "max_agents": 2,
                "max_waves": 1,
                "mode": "quick",
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        for _ in range(100):
            status = client.get(f"/api/v1/analyses/{run_id}").json()
            if status["status"] == "succeeded":
                break
            time_response = client.get(f"/api/v1/analyses/{run_id}/intelligence")
            assert time_response.status_code == 200
        assert status["status"] == "succeeded"
        intelligence = client.get(
            f"/api/v1/analyses/{run_id}/intelligence"
        ).json()
        assert intelligence["roles"]
        assert intelligence["coverage"]
        assert intelligence["findings"]
        recovery = client.post("/api/v1/recovery")
        assert recovery.status_code == 200
        assert recovery.json()["scheduled_runs"] == 0
        frontend = client.get("/").text
        assert "Planned specialists" in frontend
        assert "Coverage and gaps" in frontend
        assert 'id="analysis-mode"' in frontend
        assert 'id="max-waves"' in frontend
        assert 'id="chunking-strategy"' not in frontend
        assert 'id="tool-calling"' not in frontend
        app_script = client.get("/assets/app.js").text
        assert 'cancelled: "Review cancelled"' in app_script

        retired = client.post(
            "/api/v1/analyses",
            json={"project_path": str(root), "enable_tool_calling": True},
        )
        assert retired.status_code == 422

    with database_connection(database_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
