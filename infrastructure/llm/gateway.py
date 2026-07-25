"""OpenAI-compatible adapter for the provider-neutral model gateway."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

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

    def __init__(self, *, base_url: str, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await asyncio.to_thread(self._complete_sync, request)

    def _complete_sync(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": 0,
            "max_tokens": request.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - configured provider URL
                http_request, timeout=request.timeout_seconds
            ) as response:
                body = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ProviderUnavailable(
                f"model provider at {self.base_url} is unavailable"
            ) from exc
        try:
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            return ModelResponse(
                content=parse_json_object(content),
                provider="openai-compatible",
                model=str(body.get("model") or request.model),
                usage=ModelUsage(
                    input_tokens=int(usage.get("prompt_tokens", 0)),
                    output_tokens=int(usage.get("completion_tokens", 0)),
                    cost_usd=float(usage.get("cost", 0) or 0),
                ),
                provider_request_id=body.get("id"),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderUnavailable("model provider returned an invalid response") from exc
