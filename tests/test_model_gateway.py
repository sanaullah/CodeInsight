from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


@contextmanager
def _openai_compatible_server(
    *,
    status: int = 200,
    response_body: dict[str, Any] | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    captured: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API
            content_length = int(self.headers["Content-Length"])
            captured.update(
                path=self.path,
                authorization=self.headers.get("Authorization"),
                content_type=self.headers.get("Content-Type"),
                body=json.loads(self.rfile.read(content_length)),
            )
            body = response_body or {
                "id": "local-request-1",
                "model": "local-served-model",
                "choices": [{"message": {"content": '{"findings": []}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            }
            encoded = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", captured
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


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
async def test_openai_compatible_gateway_uses_real_chat_completions_contract() -> None:
    with _openai_compatible_server() as (base_url, captured):
        response = await OpenAICompatibleGateway(
            base_url=base_url,
            api_key="local-secret",
        ).complete(_request())

    assert captured["path"] == "/v1/chat/completions"
    assert captured["authorization"] == "Bearer local-secret"
    assert captured["content_type"] == "application/json"
    assert captured["body"] == {
        "model": "fixture",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "{}"},
        ],
        "temperature": 0,
        "max_tokens": 10,
        "response_format": {"type": "json_object"},
    }
    assert response.provider == "openai-compatible"
    assert response.model == "local-served-model"
    assert response.provider_request_id == "local-request-1"
    assert response.usage.total_tokens == 9


@pytest.mark.asyncio
async def test_openai_compatible_gateway_classifies_real_http_failure() -> None:
    with _openai_compatible_server(status=503) as (base_url, _captured):
        with pytest.raises(RuntimeError, match="unavailable"):
            await OpenAICompatibleGateway(base_url=base_url).complete(_request())


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


@pytest.mark.asyncio
async def test_bounded_gateway_does_not_commit_failed_or_over_budget_usage() -> None:
    class FixedGateway:
        def __init__(self, response: ModelResponse | Exception) -> None:
            self.response = response

        async def complete(self, _request: ModelRequest) -> ModelResponse:
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    failed = BoundedModelGateway(FixedGateway(RuntimeError("provider failed")))
    with pytest.raises(RuntimeError, match="provider failed"):
        await failed.complete(_request())
    assert (await failed.usage_for_run("run-1")).total_tokens == 0

    over_tokens = BoundedModelGateway(
        FixedGateway(
            ModelResponse(
                content={},
                provider="fixture",
                model="fixture",
                usage=ModelUsage(input_tokens=11),
            )
        )
    )
    with pytest.raises(ModelBudgetExceeded, match="response exceeded.*token"):
        await over_tokens.complete(_request())
    assert (await over_tokens.usage_for_run("run-1")).total_tokens == 0

    over_cost = BoundedModelGateway(
        FixedGateway(
            ModelResponse(
                content={},
                provider="fixture",
                model="fixture",
                usage=ModelUsage(cost_usd=2),
            )
        )
    )
    with pytest.raises(ModelBudgetExceeded, match="response exceeded.*cost"):
        await over_cost.complete(_request())
    assert (await over_cost.usage_for_run("run-1")).cost_usd == 0


@pytest.mark.asyncio
async def test_bounded_gateway_timeout_and_run_budgets_are_isolated() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        BoundedModelGateway(CountingGateway(), max_concurrent=0)

    timed = BoundedModelGateway(CountingGateway(delay=0.05))
    with pytest.raises(TimeoutError):
        await timed.complete(replace(_request(), timeout_seconds=0.001))

    gateway = BoundedModelGateway(CountingGateway(tokens=2))
    await gateway.configure_run_budget("small", max_tokens=1, max_cost_usd=1)
    with pytest.raises(ModelBudgetExceeded):
        await gateway.complete(_request("small"))
    await gateway.complete(_request("independent"))
    assert (await gateway.usage_for_run("small")).total_tokens == 0
    assert (await gateway.usage_for_run("independent")).total_tokens == 2


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


def test_langfuse_import_or_configuration_failure_disables_exporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenLangfuse:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("invalid optional configuration")

    monkeypatch.setitem(
        sys.modules, "langfuse", types.SimpleNamespace(Langfuse=BrokenLangfuse)
    )
    exporter = LangfuseTraceExporter(
        public_key="",
        secret_key="",
        host="http://unavailable.invalid",
    )
    assert not exporter.enabled
    exporter.emit(TraceEvent(name="ignored", run_id="run-1"))
    exporter.close()


def test_langfuse_content_opt_in_preserves_content_but_redacts_credentials() -> None:
    assert _redact(
        {
            "prompt": "inspect repository",
            "completion": "finding",
            "source_content": "code",
            "api_key": "secret",
            "password": "secret",
        },
        True,
    ) == {
        "prompt": "inspect repository",
        "completion": "finding",
        "source_content": "code",
        "api_key": "[redacted]",
        "password": "[redacted]",
    }
