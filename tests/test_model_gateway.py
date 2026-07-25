from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from api.config import ApiSettings
from application.model_gateway import (
    BoundedModelGateway,
    ModelBudgetExceeded,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    OfflineModelGateway,
    parse_json_object,
)
from application.tracing import TraceEvent
from infrastructure.llm.gateway import OpenAICompatibleGateway
from infrastructure.observability.langfuse import LangfuseTraceExporter, _redact


def _request(run_id: str = "run-1") -> ModelRequest:
    return ModelRequest(
        run_id=run_id,
        wave_id="wave-1",
        task_id=f"task-{run_id}",
        model="fixture",
        system_prompt="system",
        user_prompt="{}",
        response_schema={},
        max_output_tokens=10,
        timeout_seconds=1,
        token_budget=10,
        cost_budget_usd=1,
    )


class CountingGateway:
    def __init__(self, *, delay: float = 0.01, tokens: int = 2) -> None:
        self.delay = delay
        self.tokens = tokens
        self.active = 0
        self.max_active = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return ModelResponse(
            content={},
            provider="fixture",
            model=request.model,
            usage=ModelUsage(input_tokens=self.tokens),
        )


@pytest.mark.asyncio
async def test_gateway_enforces_concurrency_and_run_token_budget() -> None:
    provider = CountingGateway()
    gateway = BoundedModelGateway(provider, max_concurrent=2)
    await gateway.configure_run_budget("run-1", max_tokens=4, max_cost_usd=1)
    await asyncio.gather(
        gateway.complete(_request()),
        gateway.complete(_request()),
    )
    assert provider.max_active == 2
    assert (await gateway.usage_for_run("run-1")).total_tokens == 4
    with pytest.raises(ModelBudgetExceeded, match="token budget"):
        await gateway.complete(_request())


@pytest.mark.asyncio
async def test_zero_cost_budget_allows_offline_but_blocks_configured_provider() -> None:
    request = _request()
    request = replace(request, cost_budget_usd=0)
    offline = BoundedModelGateway(OfflineModelGateway())
    await offline.complete(request)
    paid = BoundedModelGateway(CountingGateway())
    with pytest.raises(ModelBudgetExceeded, match="free providers"):
        await paid.complete(request)


def test_json_contract_parser_rejects_prose_and_arrays() -> None:
    assert parse_json_object('{"ok": true}') == {"ok": True}
    with pytest.raises(ValueError, match="valid JSON"):
        parse_json_object("Here is the result: {}")
    with pytest.raises(ValueError, match="JSON object"):
        parse_json_object("[]")


@pytest.mark.asyncio
async def test_openai_compatible_gateway_maps_json_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "id": "provider-1",
                    "model": "served-model",
                    "choices": [{"message": {"content": '{"findings": []}'}}],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 3,
                        "cost": 0.002,
                    },
                }
            ).encode()

    def urlopen(request: Any, timeout: float) -> Response:
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    response = await OpenAICompatibleGateway(
        base_url="http://127.0.0.1:1234/v1/",
        api_key="local-key",
    ).complete(_request())
    assert captured == {
        "url": "http://127.0.0.1:1234/v1/chat/completions",
        "authorization": "Bearer local-key",
        "timeout": 1,
    }
    assert response.content == {"findings": []}
    assert response.model == "served-model"
    assert response.usage.total_tokens == 11
    assert response.usage.cost_usd == 0.002


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not-json", "unavailable"),
        (
            json.dumps({"choices": []}).encode(),
            "invalid response",
        ),
    ],
)
async def test_openai_compatible_gateway_classifies_bad_responses(
    monkeypatch: pytest.MonkeyPatch, body: bytes, message: str
) -> None:
    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return body

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(RuntimeError, match=message):
        await OpenAICompatibleGateway(base_url="http://localhost/v1").complete(
            _request()
        )


def test_api_settings_make_provider_and_langfuse_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CODEINSIGHT_DATABASE_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.setenv("CODEINSIGHT_MODEL_BASE_URL", "http://127.0.0.1:1234/v1/")
    monkeypatch.setenv("CODEINSIGHT_DEFAULT_MODEL", "local-fixture")
    monkeypatch.setenv("CODEINSIGHT_LANGFUSE_ENABLED", "true")
    settings = ApiSettings.from_environment()
    assert settings.model_base_url == "http://127.0.0.1:1234/v1"
    assert settings.default_model == "local-fixture"
    assert settings.langfuse_enabled


def test_langfuse_export_is_optional_redacted_and_non_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeLangfuse:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_event(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            raise RuntimeError("collector unavailable")

        def flush(self) -> None:
            raise RuntimeError("collector unavailable")

    monkeypatch.setitem(sys.modules, "langfuse", types.SimpleNamespace(Langfuse=FakeLangfuse))
    exporter = LangfuseTraceExporter(
        public_key="public",
        secret_key="secret",
        host="http://localhost:3000",
    )
    started = time.perf_counter()
    exporter.emit(
        TraceEvent(
            name="model_call",
            run_id="run-1",
            attributes={"api_token": "secret", "prompt": "private", "count": 2},
        )
    )
    assert time.perf_counter() - started < 0.1
    exporter.close()
    for _ in range(20):
        if calls:
            break
        time.sleep(0.01)
    assert calls
    assert _redact(
        {"api_token": "secret", "prompt": "private", "count": 2}, False
    ) == {"api_token": "[redacted]", "prompt": "[omitted]", "count": 2}
