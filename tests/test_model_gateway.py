from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
import types
import urllib.error
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
    ProviderUnavailable,
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
        response_schema={
            "type": "object",
            "properties": {"findings": {"type": "array"}},
            "required": ["findings"],
            "additionalProperties": False,
        },
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


class RecordingModelTracer:
    def __init__(self) -> None:
        self.records: list[tuple[ModelRequest, ModelResponse | None, object | None]] = []

    def record_model_call(
        self,
        request: ModelRequest,
        *,
        response: ModelResponse | None = None,
        error: object | None = None,
    ) -> None:
        self.records.append((request, response, error))


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
async def test_bounded_gateway_records_successful_provider_exchange() -> None:
    tracer = RecordingModelTracer()
    gateway = BoundedModelGateway(CountingGateway(), tracer=tracer)

    response = await gateway.complete(_request())

    assert len(tracer.records) == 1
    request, traced_response, error = tracer.records[0]
    assert request.user_prompt == "{}"
    assert traced_response is response
    assert error is None


@pytest.mark.asyncio
async def test_bounded_gateway_records_malformed_provider_content() -> None:
    class MalformedGateway:
        async def complete(self, _request: ModelRequest) -> ModelResponse:
            raise ProviderUnavailable(
                "provider_protocol_error category=invalid_content_json",
                category="invalid_content_json",
                response_content="not valid JSON",
            )

    tracer = RecordingModelTracer()
    gateway = BoundedModelGateway(MalformedGateway(), tracer=tracer)

    with pytest.raises(ProviderUnavailable, match="invalid_content_json"):
        await gateway.complete(_request())

    assert len(tracer.records) == 1
    _request_value, response, error = tracer.records[0]
    assert response is None
    assert error is not None
    assert error.response_content == "not valid JSON"


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
    assert captured["body"]["model"] == "fixture"
    assert captured["body"]["messages"][1] == {"role": "user", "content": "{}"}
    assert captured["body"]["messages"][0]["content"].startswith("system")
    assert '"required":["findings"]' in captured["body"]["messages"][0]["content"]
    assert captured["body"]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "specialist_output",
            "strict": True,
            "schema": _request().response_schema,
        },
    }
    assert captured["body"]["temperature"] == 0
    assert captured["body"]["max_tokens"] == 10
    assert response.provider == "openai-compatible"
    assert response.model == "local-served-model"
    assert response.provider_request_id == "local-request-1"
    assert response.usage.total_tokens == 9


@pytest.mark.asyncio
async def test_openai_compatible_gateway_classifies_real_http_failure() -> None:
    with _openai_compatible_server(status=503) as (base_url, _captured):
        with pytest.raises(RuntimeError, match="provider_http_error status=503"):
            await OpenAICompatibleGateway(base_url=base_url).complete(_request())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not-json", "invalid_envelope_json"),
        (b"[]", "invalid_envelope_shape"),
        (
            json.dumps({"choices": []}).encode(),
            "invalid_response_shape",
        ),
        (
            json.dumps({"choices": [{"message": {"content": []}}]}).encode(),
            "invalid_content_type",
        ),
        (
            json.dumps({"choices": [{"message": {"content": "not-json"}}]}).encode(),
            "invalid_content_json",
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
    with pytest.raises(ProviderUnavailable, match=message) as raised:
        await OpenAICompatibleGateway(base_url="http://localhost/v1").complete(
            _request()
        )
    assert raised.value.response_content
    if message == "invalid_envelope_json":
        assert raised.value.response_content == "not-json"


@pytest.mark.asyncio
async def test_openai_compatible_gateway_falls_back_to_prompted_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, Any]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "model": "served-model",
                    "choices": [{"message": {"content": '{"findings":[]}'}}],
                }
            ).encode()

    def urlopen(request: Any, timeout: float) -> Response:
        del timeout
        payloads.append(json.loads(request.data))
        if len(payloads) == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "strict schema unsupported",
                {},
                io.BytesIO(b'{"error":"sensitive provider detail"}'),
            )
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    response = await OpenAICompatibleGateway(
        base_url="https://provider.invalid/v1"
    ).complete(_request())

    assert response.content == {"findings": []}
    assert [item["response_format"]["type"] for item in payloads] == [
        "json_schema",
        "json_object",
    ]
    assert payloads[0]["messages"] == payloads[1]["messages"]
    assert '"additionalProperties":false' in payloads[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_openai_compatible_gateway_can_use_profiled_json_object_mode() -> None:
    with _openai_compatible_server() as (base_url, captured):
        await OpenAICompatibleGateway(
            base_url=base_url,
            prefer_strict_schema=False,
        ).complete(_request())

    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert '"required":["findings"]' in captured["body"]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_openai_compatible_gateway_classifies_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError("private transport detail")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(RuntimeError, match="category=connection") as caught:
        await OpenAICompatibleGateway(
            base_url="https://provider.invalid/v1"
        ).complete(_request())
    assert "private transport detail" not in str(caught.value)


@pytest.mark.asyncio
async def test_openai_compatible_gateway_errors_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(request: Any, timeout: float) -> None:
        del timeout
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "secret diagnostic",
            {},
            io.BytesIO(b'{"error":"credential was abc123"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(RuntimeError) as caught:
        await OpenAICompatibleGateway(
            base_url="https://provider.invalid/v1",
            api_key="abc123",
        ).complete(_request())
    assert str(caught.value) == "provider_http_error status=401"
    assert "abc123" not in str(caught.value)
    assert caught.value.category == "http_error"
    assert caught.value.http_status == 401


def test_openai_compatible_gateway_validates_timeout_cap() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        OpenAICompatibleGateway(
            base_url="http://localhost/v1", timeout_cap_seconds=0
        )
    with pytest.raises(ValueError, match="greater than zero"):
        OpenAICompatibleGateway(
            base_url="http://localhost/v1", max_output_tokens_cap=0
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
    monkeypatch.setenv("CODEINSIGHT_LANGFUSE_CAPTURE_PROMPTS", "true")
    monkeypatch.delenv("CODEINSIGHT_LANGFUSE_CAPTURE_COMPLETIONS", raising=False)
    settings = ApiSettings.from_environment()
    assert settings.model_base_url == "http://127.0.0.1:1234/v1"
    assert settings.default_model == "local-fixture"
    assert settings.langfuse_enabled
    assert settings.langfuse_capture_prompts
    assert not settings.langfuse_capture_completions


def test_langfuse_export_is_optional_redacted_and_non_blocking(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeLangfuse:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_event(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            raise RuntimeError("collector unavailable with private payload")

        def flush(self) -> None:
            raise RuntimeError("flush unavailable with private payload")

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
            stage="dispatch_tasks",
            wave_id="wave-1",
            role_id="role-1",
            task_id="task-1",
            attempt_id="attempt-1",
            attempt_number=1,
            model_call_id="call-1",
            prompt_artifact_id="prompt-1",
            attributes={
                "api_token": "secret",
                "prompt_text": "private",
                "input_tokens": 2,
            },
        )
    )
    assert time.perf_counter() - started < 0.1
    exporter.close()
    for _ in range(20):
        if calls:
            break
        time.sleep(0.01)
    assert calls
    assert calls[0]["trace_context"]["trace_id"]
    assert len(calls[0]["trace_context"]["trace_id"]) == 32
    assert calls[0]["metadata"]["attempt_id"] == "attempt-1"
    assert calls[0]["metadata"]["prompt_artifact_id"] == "prompt-1"
    assert _redact(
        {"api_token": "secret", "prompt": "private", "input_tokens": 2}
    ) == {
        "api_token": "[redacted]",
        "prompt": "[omitted]",
        "input_tokens": 2,
    }
    assert "private payload" not in caplog.text


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


def test_langfuse_capture_controls_are_independent_and_recursive() -> None:
    assert _redact(
        {
            "prompt_text": "inspect repository",
            "model_completion": {"finding": "bounded"},
            "source_content": "code",
            "nested": {
                "api_key": "secret",
                "input_tokens": 7,
                "project_path": "C:/private",
            },
        },
        capture_prompts=True,
        capture_completions=False,
    ) == {
        "prompt_text": "inspect repository",
        "model_completion": "[omitted]",
        "source_content": "[omitted]",
        "nested": {
            "api_key": "[redacted]",
            "input_tokens": 7,
            "project_path": "[omitted]",
        },
    }


def test_langfuse_records_failed_provider_exchange_with_capture_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generations: list[dict[str, Any]] = []

    class FakeGeneration:
        def end(self) -> None:
            return None

    class FakeLangfuse:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def start_observation(self, **kwargs: Any) -> FakeGeneration:
            generations.append(kwargs)
            return FakeGeneration()

        def flush(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules, "langfuse", types.SimpleNamespace(Langfuse=FakeLangfuse)
    )
    exporter = LangfuseTraceExporter(
        public_key="public",
        secret_key="secret",
        host="http://localhost:3000",
        capture_prompts=True,
        capture_completions=True,
    )
    request = _request()
    error = ProviderUnavailable(
        "provider_protocol_error category=invalid_content_json",
        category="invalid_content_json",
        response_content="this is not valid JSON",
    )
    exporter._record_model_call_safely(
        request,
        {"system_prompt": request.system_prompt, "user_prompt": request.user_prompt},
        {"model_completion": error.response_content},
        {"status": "failed", "error_category": error.category},
        None,
        str(error),
    )
    exporter.close()

    assert len(generations) == 1
    generation = generations[0]
    assert generation["as_type"] == "generation"
    assert generation["name"] == "provider_model_call"
    assert generation["input"]["system_prompt"] == "system"
    assert generation["output"]["model_completion"] == "this is not valid JSON"
    assert generation["metadata"]["error_category"] == "invalid_content_json"
    assert generation["level"] == "ERROR"


def test_langfuse_queue_is_bounded_and_emit_after_close_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingLangfuse:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def create_event(self, **_kwargs: Any) -> None:
            nonlocal calls
            calls += 1
            entered.set()
            release.wait(timeout=1)

        def flush(self) -> None:
            return None

    monkeypatch.setitem(
        sys.modules, "langfuse", types.SimpleNamespace(Langfuse=BlockingLangfuse)
    )
    exporter = LangfuseTraceExporter(
        public_key="public",
        secret_key="secret",
        host="http://localhost:3000",
        max_pending_events=1,
    )
    exporter.emit(TraceEvent(name="first", run_id="run-1"))
    assert entered.wait(timeout=1)
    exporter.emit(TraceEvent(name="dropped", run_id="run-1"))
    assert exporter.dropped_events == 1
    release.set()
    exporter.close()
    exporter.emit(TraceEvent(name="after-close", run_id="run-1"))
    assert calls == 1
