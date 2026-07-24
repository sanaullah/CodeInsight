from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from api.analysis_service import AnalysisService, EventSink
from api.app import create_app
from api.models import AnalysisRequest


class FakeExecutor:
    async def execute(
        self, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
        event_sink("architecture_ready", {"message": "Architecture indexed"})
        return {"synthesized_report": "API report", "files_scanned": 1}


def test_health_frontend_and_analysis_lifecycle(tmp_path: Path) -> None:
    service = AnalysisService(FakeExecutor(), max_concurrent=1)
    app = create_app(service)

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["max_concurrent_analyses"] == 1

        capabilities = client.get("/api/v1/capabilities")
        assert capabilities.status_code == 200
        support = {
            language["language"]: language["support_level"]
            for language in capabilities.json()["languages"]
        }
        assert len(support) == 30
        assert support["python"] == "parsed"
        assert support["typescript"] == "dependency-aware"
        assert support["rust"] == "discovery"

        frontend = client.get("/")
        assert frontend.status_code == 200
        assert "Turn a codebase into a focused review" in frontend.text

        created = client.post(
            "/api/v1/analyses",
            json={"project_path": str(tmp_path), "max_agents": 2},
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        run = client.get(f"/api/v1/analyses/{run_id}")
        assert run.status_code == 200
        assert run.json()["status"] in {"queued", "running", "succeeded"}


def test_analysis_rejects_invalid_project_path(tmp_path: Path) -> None:
    app = create_app(AnalysisService(FakeExecutor()))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyses",
            json={"project_path": str(tmp_path / "missing")},
        )
    assert response.status_code == 422
    assert "does not exist" in response.json()["detail"]
