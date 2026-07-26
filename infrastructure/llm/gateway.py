"""OpenAI-compatible adapter for the provider-neutral model gateway."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from application.model_gateway import (
    ModelGateway,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderUnavailable,
    parse_json_object,
)


class OpenAICompatibleGateway(ModelGateway):
    """Call a local or remote OpenAI-compatible chat-completions endpoint."""

    _SCHEMA_FALLBACK_STATUSES = frozenset({400, 415, 422, 501})

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        timeout_cap_seconds: float | None = None,
        max_output_tokens_cap: int | None = None,
        prefer_strict_schema: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        if timeout_cap_seconds is not None and timeout_cap_seconds <= 0:
            raise ValueError("timeout_cap_seconds must be greater than zero")
        if max_output_tokens_cap is not None and max_output_tokens_cap <= 0:
            raise ValueError("max_output_tokens_cap must be greater than zero")
        self.timeout_cap_seconds = timeout_cap_seconds
        self.max_output_tokens_cap = max_output_tokens_cap
        self.prefer_strict_schema = prefer_strict_schema

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: ModelRequest) -> ModelResponse:
        schema_json = json.dumps(
            request.response_schema,
            sort_keys=True,
            separators=(",", ":"),
        )
        schema_instruction = (
            "\n\nReturn exactly one JSON object conforming to this JSON Schema. "
            "Do not add markdown fences, prose, or fields not allowed by the schema.\n"
            f"{schema_json}"
        )
        messages = [
            {
                "role": "system",
                "content": request.system_prompt + schema_instruction,
            },
            {"role": "user", "content": request.user_prompt},
        ]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": (
                min(request.max_output_tokens, self.max_output_tokens_cap)
                if self.max_output_tokens_cap is not None
                else request.max_output_tokens
            ),
        }
        strict_schema = bool(request.response_schema) and self.prefer_strict_schema
        if strict_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "specialist_output",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        else:
            payload["response_format"] = {"type": "json_object"}
        try:
            body = self._post(payload, request.timeout_seconds)
        except urllib.error.HTTPError as exc:
            if strict_schema and exc.code in self._SCHEMA_FALLBACK_STATUSES:
                fallback_payload = {
                    **payload,
                    "response_format": {"type": "json_object"},
                }
                body = self._post_with_classified_errors(
                    fallback_payload, request.timeout_seconds
                )
            else:
                raise self._http_error(exc) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise ProviderUnavailable(
                "provider_transport_error category=connection",
                category="connection",
            ) from exc
        return self._map_response(body, request)

    def _post(self, payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        timeout = (
            min(timeout_seconds, self.timeout_cap_seconds)
            if self.timeout_cap_seconds is not None
            else timeout_seconds
        )
        with urllib.request.urlopen(  # noqa: S310 - configured provider URL
            http_request, timeout=timeout
        ) as response:
            raw_body = response.read()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ProviderUnavailable(
                "provider_protocol_error category=invalid_envelope_json",
                category="invalid_envelope_json",
                response_content=_response_text(raw_body),
            ) from exc
        if not isinstance(body, dict):
            raise ProviderUnavailable(
                "provider_protocol_error category=invalid_envelope_shape",
                category="invalid_envelope_shape",
                response_content=_json_text(body),
            )
        return body

    def _post_with_classified_errors(
        self, payload: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        try:
            return self._post(payload, timeout_seconds)
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
        except (OSError, urllib.error.URLError) as exc:
            raise ProviderUnavailable(
                "provider_transport_error category=connection",
                category="connection",
            ) from exc

    @staticmethod
    def _http_error(exc: urllib.error.HTTPError) -> ProviderUnavailable:
        try:
            response_content = _response_text(exc.read())
        except OSError:
            response_content = None
        return ProviderUnavailable(
            f"provider_http_error status={exc.code}",
            category="http_error",
            http_status=exc.code,
            response_content=response_content,
        )

    @staticmethod
    def _map_response(
        body: dict[str, Any], request: ModelRequest
    ) -> ModelResponse:
        try:
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ProviderUnavailable(
                    "provider_protocol_error category=invalid_content_type",
                    category="invalid_content_type",
                    response_content=_json_text(body),
                )
            usage = body.get("usage", {})
            mapped_usage = ModelUsage(
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                cost_usd=float(usage.get("cost", 0) or 0),
            )
            return ModelResponse(
                content=parse_json_object(content),
                provider="openai-compatible",
                model=str(body.get("model") or request.model),
                usage=mapped_usage,
                provider_request_id=body.get("id"),
            )
        except ProviderUnavailable:
            raise
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderUnavailable(
                "provider_protocol_error category=invalid_response_shape",
                category="invalid_response_shape",
                response_content=_json_text(body),
            ) from exc
        except ValueError as exc:
            raise ProviderUnavailable(
                "provider_protocol_error category=invalid_content_json",
                category="invalid_content_json",
                response_content=content,
                usage=mapped_usage,
            ) from exc


def _response_text(raw_body: bytes) -> str:
    return raw_body.decode("utf-8", errors="replace")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ","), default=str)
