"""Opt-in smoke test against a real OpenAI-compatible provider.

This module is skipped unless the operator explicitly opts in. It never prints
credentials, endpoint URLs, prompts, source contents, or raw model responses.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.config import ApiSettings, load_environment
from application.analysis_service import AnalysisService
from infrastructure.llm.gateway import OpenAICompatibleGateway

pytestmark = pytest.mark.live_provider

_OPT_IN = "CODEINSIGHT_RUN_LIVE_PROVIDER_TEST"
_TERMINAL = {"succeeded", "failed", "cancelled", "needs_attention"}


def _enabled() -> bool:
    return os.getenv(_OPT_IN, "").strip().lower() in {"1", "true", "yes", "on"}


def _configured_settings(database_path: Path) -> ApiSettings:
    load_environment()
    if not _enabled():
        pytest.skip(f"set {_OPT_IN}=1 to authorize a real provider call")
    settings = ApiSettings.from_environment()
    if not settings.model_base_url:
        pytest.skip("a configured OpenAI-compatible base URL is required")
    if not settings.model_api_key:
        pytest.skip("explicit provider credentials are required")
    if not settings.default_model or settings.default_model == "local-model":
        pytest.skip("an explicit provider model is required")
    return replace(
        settings,
        database_path=database_path,
        max_concurrent_analyses=1,
        max_concurrent_model_calls=1,
        langfuse_enabled=False,
    )


def _fixture_repository(root: Path) -> Path:
    repository = root / "live-provider-fixture"
    repository.mkdir()
    (repository / "calculator.py").write_text(
        '"""Small read-only analysis fixture."""\n\n'
        "def safe_ratio(numerator: float, denominator: float) -> float:\n"
        '    """Return a ratio; callers must provide a non-zero denominator."""\n'
        "    return numerator / denominator\n",
        encoding="utf-8",
    )
    return repository


def test_real_provider_completes_bounded_durable_api_workflow(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "state" / "codeinsight.db"
    settings = _configured_settings(database_path)
    gateway = OpenAICompatibleGateway(
        base_url=settings.model_base_url or "",
        api_key=settings.model_api_key,
        timeout_cap_seconds=45,
        max_output_tokens_cap=2_500,
        prefer_strict_schema=False,
    )
    service = AnalysisService(
        database_path=database_path,
        max_concurrent=1,
        settings=settings,
        gateway=gateway,
    )

    with TestClient(create_app(service)) as client:
        submitted = client.post(
            "/api/v1/analyses",
            json={
                "project_path": str(_fixture_repository(tmp_path)),
                "goal": "Check this tiny fixture for one evidence-backed reliability risk.",
                "model_name": settings.default_model,
                "mode": "quick",
                "max_agents": 1,
                "max_waves": 1,
                "max_tasks": 1,
                "max_total_tokens": 8_000,
                "max_cost_usd": 0.10,
                "max_elapsed_seconds": 90,
            },
        )
        assert submitted.status_code == 202
        run_id = submitted.json()["run_id"]

        deadline = time.monotonic() + 90
        run: dict[str, object] = {}
        while time.monotonic() < deadline:
            response = client.get(f"/api/v1/analyses/{run_id}")
            assert response.status_code == 200
            run = response.json()
            if run["status"] in _TERMINAL:
                break
            time.sleep(0.25)
        else:
            client.delete(f"/api/v1/analyses/{run_id}")
            pytest.fail("live provider workflow exceeded its 90 second test deadline")

        assert run["status"] == "succeeded"
        assert isinstance(run["result"], dict)
        assert run["result"]["provider_mode"] == "model-backed"
        assert run["result"]["wave_count"] == 1
        intelligence = client.get(
            f"/api/v1/analyses/{run_id}/intelligence"
        ).json()
        assert len(intelligence["tasks"]) == 1
        assert intelligence["tasks"][0]["status"] == "succeeded"
        assert len(intelligence["model_calls"]) == 1
        assert intelligence["model_calls"][0]["status"] == "succeeded"
        assert intelligence["coverage"]
