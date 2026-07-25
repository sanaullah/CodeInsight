from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from api.app import create_app
from api.models import AnalysisRequest
from application.analysis_service import AnalysisService, EventSink


class FakeExecutor:
    async def execute(
        self, run_id: str, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
        event_sink(
            "architecture_ready",
            {"message": "Architecture indexed", "run_id": run_id},
        )
        return {"synthesized_report": "API report", "files_scanned": 1}


def test_health_frontend_and_analysis_lifecycle(tmp_path: Path) -> None:
    service = AnalysisService(
        FakeExecutor(),
        database_path=tmp_path / "codeinsight.db",
        max_concurrent=1,
    )
    app = create_app(service)

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        health_body = health.json()
        assert health_body["max_concurrent_analyses"] == 1
        assert health_body["runtime"] == {
            "api_contract_version": 1,
            "application_server": "FastAPI",
            "environment_manager": "uv",
            "database_engine": "SQLite",
            "database_journal_mode": "WAL",
            "database_schema_version": 6,
            "artifact_store": "filesystem",
            "api_docs_url": "/api/docs",
            "read_only_analysis": True,
            "build_commit": None,
            "build_time": None,
        }

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
        assert "streamlit_fallback_available" not in capabilities.json()

        frontend = client.get("/")
        assert frontend.status_code == 200
        assert 'id="app-root"' in frontend.text
        assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', frontend.text)
        assert len(assets) == 2
        for asset in assets:
            response = client.get(asset)
            assert response.status_code == 200
        assert client.get("/new-review").text == frontend.text
        assert client.get("/reviews/deep-link").text == frontend.text
        assert client.get("/api/missing").status_code == 404

        created = client.post(
            "/api/v1/analyses",
            json={"project_path": str(tmp_path), "max_agents": 2},
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]

        run = client.get(f"/api/v1/analyses/{run_id}")
        assert run.status_code == 200
        assert run.json()["status"] in {"queued", "running", "succeeded"}
        rerun = client.post(f"/api/v1/analyses/{run_id}/rerun")
        assert rerun.status_code == 202
        assert rerun.json()["run_id"] != run_id
        assert rerun.json()["status_url"] == (
            f"/api/v1/analyses/{rerun.json()['run_id']}"
        )


def test_analysis_rejects_invalid_project_path(tmp_path: Path) -> None:
    app = create_app(
        AnalysisService(
            FakeExecutor(), database_path=tmp_path / "codeinsight.db"
        )
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/analyses",
            json={"project_path": str(tmp_path / "missing")},
        )
    assert response.status_code == 422
    assert "does not exist" in response.json()["detail"]


def test_missing_run_routes_and_list_limits_are_truthful(tmp_path: Path) -> None:
    app = create_app(
        AnalysisService(FakeExecutor(), database_path=tmp_path / "codeinsight.db")
    )
    with TestClient(app) as client:
        assert client.get("/api/v1/analyses/missing").status_code == 404
        assert (
            client.get("/api/v1/analyses/missing/intelligence").status_code == 404
        )
        assert client.delete("/api/v1/analyses/missing").status_code == 404
        assert client.post("/api/v1/analyses/missing/rerun").status_code == 404
        assert client.get("/api/v1/analyses?limit=0").status_code == 422
        assert client.get("/api/v1/analyses?limit=101").status_code == 422
        assert client.get("/api/v1/analyses?limit=1").json() == []
        assert client.get("/api").status_code == 404
