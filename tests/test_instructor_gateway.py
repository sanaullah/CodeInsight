from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from analysis.native.harness import SpecialistOutput
from api.config import ApiSettings
from application.model_gateway import (
    BoundedModelGateway,
    ModelRequest,
    ProviderUnavailable,
)
from infrastructure.llm.instructor_gateway import (
    InstructorOpenAICompatibleGateway,
    provider_capability_profile,
)
from infrastructure.llm.pydantic_ai_gateway import PydanticAIOpenAICompatibleGateway


def _request(**changes: Any) -> ModelRequest:
    request = ModelRequest(
        run_id="run-1",
        wave_id="wave-1",
        task_id="task-1",
        model="fixture-model",
        system_prompt="Return the requested repository analysis.",
        user_prompt='{"files":[]}',
        response_schema=SpecialistOutput.model_json_schema(),
        response_model=SpecialistOutput,
        max_output_tokens=200,
        timeout_seconds=2,
        token_budget=1_000,
        cost_budget_usd=1,
    )
    return replace(request, **changes)


@contextmanager
def _provider(
    contents: list[str | dict[str, Any]], *, delay_seconds: float = 0
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    requests: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            if delay_seconds:
                time.sleep(delay_seconds)
            index = min(len(requests) - 1, len(contents) - 1)
            selected = contents[index]
            message = (
                selected
                if isinstance(selected, dict)
                else {"role": "assistant", "content": selected}
            )
            body = {
                "object": "chat.completion",
                "id": f"request-{len(requests)}",
                "model": "served-model",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.01,
                },
            }
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except BrokenPipeError:
                pass

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _valid_output() -> str:
    return json.dumps(
        {
            "analyzed_paths": [],
            "evidence": [],
            "findings": [],
            "unresolved_uncertainty": [],
            "usage": {},
        }
    )


@pytest.mark.asyncio
async def test_instructor_adapter_validates_exact_specialist_contract() -> None:
    with _provider([_valid_output()]) as (base_url, requests):
        gateway = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            api_key="fixture-secret",
            profile=provider_capability_profile("instructor-json"),
        )
        try:
            response = await gateway.complete(_request())
        finally:
            await gateway.aclose()

    assert response.content == SpecialistOutput.model_validate_json(
        _valid_output()
    ).model_dump(mode="json")
    assert response.provider == "instructor-openai-compatible:instructor-json"
    assert response.model == "served-model"
    assert response.provider_request_id == "request-1"
    assert response.usage.total_tokens == 15
    assert len(requests) == 1
    assert requests[0]["model"] == "fixture-model"
    assert requests[0]["max_tokens"] == 200


@pytest.mark.asyncio
async def test_pydantic_ai_adapter_validates_the_same_specialist_contract() -> None:
    with _provider([_valid_output()]) as (base_url, requests):
        gateway = PydanticAIOpenAICompatibleGateway(
            base_url=base_url,
            api_key="fixture-secret",
        )
        try:
            response = await gateway.complete(_request())
        finally:
            await gateway.aclose()

    assert response.content == SpecialistOutput.model_validate_json(
        _valid_output()
    ).model_dump(mode="json")
    assert response.provider == "pydantic-ai-openai-compatible"
    assert len(requests) == 1
    assert requests[0]["model"] == "fixture-model"


@pytest.mark.asyncio
async def test_instructor_tools_profile_accepts_standard_tool_call_envelope() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "SpecialistOutput",
                    "arguments": _valid_output(),
                },
            }
        ],
    }
    with _provider([message]) as (base_url, requests):
        gateway = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            profile=provider_capability_profile("instructor-tools"),
        )
        try:
            response = await gateway.complete(_request())
        finally:
            await gateway.aclose()

    assert response.content["analyzed_paths"] == []
    assert len(requests) == 1
    assert requests[0]["tools"][0]["function"]["name"] == "SpecialistOutput"


@pytest.mark.asyncio
async def test_instructor_adapter_makes_only_one_sanitized_correction_call() -> None:
    with _provider(["{}", _valid_output()]) as (base_url, requests):
        gateway = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            profile=provider_capability_profile("instructor-json"),
        )
        try:
            response = await gateway.complete(_request())
        finally:
            await gateway.aclose()

    assert len(requests) == 2
    assert response.usage.total_tokens == 30
    assert response.usage.cost_usd == 0.02
    messages = requests[1]["messages"]
    assert "did not satisfy the required output contract" in messages[-1]["content"]
    assert "{}" not in messages[-1]["content"]


@pytest.mark.asyncio
async def test_instructor_adapter_stops_after_two_invalid_responses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with _provider(["{}", "{}"]) as (base_url, requests):
        gateway = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            profile=provider_capability_profile("instructor-json"),
        )
        try:
            with pytest.raises(ProviderUnavailable) as caught:
                await gateway.complete(_request())
        finally:
            await gateway.aclose()

    assert len(requests) == 2
    assert caught.value.category == "response_schema_validation"
    assert not caught.value.retryable
    assert str(caught.value) == (
        "provider_contract_error category=response_schema_validation"
    )
    assert "analyzed_paths" not in str(caught.value)
    assert "{}" not in caplog.text


@pytest.mark.asyncio
async def test_failed_correction_usage_is_charged_to_durable_run_budget() -> None:
    with _provider(["{}", "{}"]) as (base_url, _requests):
        provider = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            profile=provider_capability_profile("instructor-json"),
        )
        gateway = BoundedModelGateway(provider)
        try:
            with pytest.raises(ProviderUnavailable):
                await gateway.complete(_request())
            usage = await gateway.usage_for_run("run-1")
        finally:
            await gateway.aclose()

    assert usage.total_tokens == 30
    assert usage.cost_usd == 0.02


@pytest.mark.asyncio
async def test_instructor_adapter_requires_typed_contract_without_calling_provider() -> None:
    with _provider([_valid_output()]) as (base_url, requests):
        gateway = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            profile=provider_capability_profile("instructor-json"),
        )
        try:
            with pytest.raises(ProviderUnavailable) as caught:
                await gateway.complete(_request(response_model=None))
        finally:
            await gateway.aclose()

    assert requests == []
    assert caught.value.category == "missing_response_model"


@pytest.mark.asyncio
async def test_instructor_adapter_total_timeout_is_sanitized() -> None:
    with _provider([_valid_output()], delay_seconds=0.2) as (base_url, requests):
        gateway = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            profile=provider_capability_profile("instructor-json"),
        )
        try:
            with pytest.raises(ProviderUnavailable) as caught:
                await gateway.complete(_request(timeout_seconds=0.05))
        finally:
            await gateway.aclose()

    assert len(requests) == 1
    assert caught.value.category == "timeout"
    assert str(caught.value) == "provider_timeout category=timeout"


@pytest.mark.asyncio
async def test_instructor_adapter_propagates_cancellation() -> None:
    with _provider([_valid_output()], delay_seconds=0.2) as (base_url, requests):
        gateway = InstructorOpenAICompatibleGateway(
            base_url=base_url,
            profile=provider_capability_profile("instructor-json"),
        )
        task = asyncio.create_task(gateway.complete(_request()))
        try:
            await asyncio.sleep(0.03)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            await gateway.aclose()
    assert len(requests) == 1


def test_provider_capability_profile_is_explicit_and_validated() -> None:
    assert not provider_capability_profile("direct").instructor_supported
    assert provider_capability_profile("instructor-json").instructor_supported
    assert provider_capability_profile("instructor-tools").instructor_supported
    with pytest.raises(ValueError, match="must be one of"):
        provider_capability_profile("automatic")


def test_provider_capability_profile_configuration_rejects_implicit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CODEINSIGHT_PROVIDER_CAPABILITY_PROFILE", "automatic")
    with pytest.raises(ValueError, match="must be one of"):
        ApiSettings.from_environment()


def test_pydantic_ai_is_the_default_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CODEINSIGHT_AGENT_RUNTIME", raising=False)
    assert ApiSettings.from_environment().agent_runtime == "pydantic-ai"

    monkeypatch.setenv("CODEINSIGHT_AGENT_RUNTIME", "not-a-runtime")
    with pytest.raises(ValueError, match="CODEINSIGHT_AGENT_RUNTIME"):
        ApiSettings.from_environment()
