"""Opt-in metadata-only smoke against an operator-managed self-hosted Langfuse."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import time
import urllib.parse
import urllib.request
from uuid import uuid4

import pytest

from application.tracing import TraceEvent
from infrastructure.observability.langfuse import LangfuseTraceExporter, _trace_id

pytestmark = pytest.mark.live_langfuse

_OPT_IN = "CODEINSIGHT_RUN_LANGFUSE_DOCKER_SMOKE"


def _enabled() -> bool:
    return os.getenv(_OPT_IN, "").strip().lower() in {"1", "true", "yes", "on"}


def test_self_hosted_langfuse_accepts_correlated_metadata_event() -> None:
    if not _enabled():
        pytest.skip(f"set {_OPT_IN}=1 to authorize the self-hosted smoke")
    if importlib.util.find_spec("langfuse") is None:
        pytest.skip("install the observability extra before running this smoke")
    host = os.getenv("LANGFUSE_HOST", "").rstrip("/")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not host or not public_key or not secret_key:
        pytest.skip("LANGFUSE_HOST and project API keys are required")

    run_id = f"docker-smoke-{uuid4().hex}"
    exporter = LangfuseTraceExporter(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    assert exporter.enabled
    exporter.emit(
        TraceEvent(
            name="codeinsight_self_hosted_smoke",
            run_id=run_id,
            stage="complete",
            wave_id="wave-smoke",
            role_id="role-smoke",
            task_id="task-smoke",
            attempt_id="attempt-smoke",
            attempt_number=1,
            model_call_id="model-call-smoke",
            prompt_artifact_id="prompt-artifact-smoke",
            attributes={"input_tokens": 1, "cost_usd": 0},
        )
    )
    exporter.close()

    encoded = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    query = urllib.parse.urlencode(
        {"traceId": _trace_id(run_id), "limit": 10}
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            f"{host}/api/public/v2/observations?{query}",
            headers={"Authorization": f"Basic {encoded}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
                payload = json.loads(response.read())
        except (OSError, ValueError):
            time.sleep(0.5)
            continue
        observations = payload.get("data", [])
        if any(
            item.get("name") == "codeinsight_self_hosted_smoke"
            for item in observations
        ):
            return
        time.sleep(0.5)
    pytest.fail("self-hosted Langfuse did not return the correlated smoke event")
