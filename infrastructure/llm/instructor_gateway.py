"""Instructor-backed structured output for OpenAI-compatible providers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from typing import Any

import instructor
import openai
from instructor.core import InstructorRetryException
from instructor.core.hooks import HookName, Hooks
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from application.model_gateway import (
    ModelBudgetExceeded,
    ModelGateway,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderUnavailable,
)

logging.getLogger("instructor").setLevel(logging.CRITICAL + 1)


class StructuredOutputMode(StrEnum):
    JSON = "json"
    JSON_SCHEMA = "json_schema"
    TOOLS = "tools"


@dataclass(frozen=True, slots=True)
class ProviderCapabilityProfile:
    """Explicitly declare whether an endpoint supports the Instructor adapter."""

    name: str
    instructor_supported: bool
    mode: StructuredOutputMode


PROVIDER_CAPABILITY_PROFILES = {
    "direct": ProviderCapabilityProfile(
        name="direct",
        instructor_supported=False,
        mode=StructuredOutputMode.JSON,
    ),
    "instructor-json": ProviderCapabilityProfile(
        name="instructor-json",
        instructor_supported=True,
        mode=StructuredOutputMode.JSON,
    ),
    "instructor-json-schema": ProviderCapabilityProfile(
        name="instructor-json-schema",
        instructor_supported=True,
        mode=StructuredOutputMode.JSON_SCHEMA,
    ),
    "instructor-tools": ProviderCapabilityProfile(
        name="instructor-tools",
        instructor_supported=True,
        mode=StructuredOutputMode.TOOLS,
    ),
}


def provider_capability_profile(name: str) -> ProviderCapabilityProfile:
    try:
        return PROVIDER_CAPABILITY_PROFILES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(PROVIDER_CAPABILITY_PROFILES))
        raise ValueError(
            f"CODEINSIGHT_PROVIDER_CAPABILITY_PROFILE must be one of: {allowed}"
        ) from exc


class InstructorOpenAICompatibleGateway(ModelGateway):
    """Validate one response and permit at most one feedback correction call."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        profile: ProviderCapabilityProfile,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not profile.instructor_supported:
            raise ValueError("Instructor gateway requires a supported capability profile")
        self.profile = profile
        self.gateway_identity = f"{type(self).__name__}:{profile.name}"
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            base_url=base_url.rstrip("/") + "/",
            api_key=api_key or "not-required",
            max_retries=0,
        )
        modes = {
            StructuredOutputMode.JSON: instructor.Mode.JSON,
            StructuredOutputMode.JSON_SCHEMA: instructor.Mode.JSON_SCHEMA,
            StructuredOutputMode.TOOLS: instructor.Mode.TOOLS,
        }
        mode = modes[profile.mode]
        self._instructor = instructor.from_openai(self._client, mode=mode)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response_model = request.response_model
        if response_model is None:
            raise ProviderUnavailable(
                "provider_contract_error category=missing_response_model",
                category="missing_response_model",
                retryable=False,
            )
        started = monotonic()
        usage = ModelUsage()
        messages: list[dict[str, str]] = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        try:
            async with asyncio.timeout(request.timeout_seconds):
                for attempt in range(2):
                    response_capture: list[Any] = []
                    hooks = Hooks()
                    hooks.on(
                        HookName.COMPLETION_RESPONSE,
                        lambda response, target=response_capture: target.append(response),
                    )
                    remaining_tokens = request.token_budget - usage.total_tokens
                    remaining_cost = request.cost_budget_usd - usage.cost_usd
                    if remaining_tokens <= 0:
                        raise _budget_error(usage, "token")
                    if request.cost_budget_usd > 0 and remaining_cost <= 0:
                        raise _budget_error(usage, "cost")
                    try:
                        result, completion = (
                            await self._instructor.chat.completions.create_with_completion(
                                response_model=response_model,
                                messages=messages,
                                model=request.model,
                                temperature=0,
                                max_tokens=min(
                                    request.max_output_tokens, remaining_tokens
                                ),
                                max_retries=0,
                                timeout=max(
                                    0.001,
                                    request.timeout_seconds - (monotonic() - started),
                                ),
                                hooks=hooks,
                            )
                        )
                    except InstructorRetryException as exc:
                        usage = _combine_usage(
                            usage, _usage_from_completion(_captured(response_capture, exc))
                        )
                        if attempt == 1:
                            raise _validation_error(exc, usage) from exc
                        messages.append(
                            {
                                "role": "user",
                                "content": _validation_feedback(exc),
                            }
                        )
                        continue
                    usage = _combine_usage(
                        usage, _usage_from_completion(completion)
                    )
                    return ModelResponse(
                        content=_model_content(result),
                        provider=f"instructor-openai-compatible:{self.profile.name}",
                        model=str(getattr(completion, "model", None) or request.model),
                        usage=usage,
                        provider_request_id=getattr(completion, "id", None),
                    )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise ProviderUnavailable(
                "provider_timeout category=timeout",
                category="timeout",
                usage=usage,
            ) from exc
        except ModelBudgetExceeded:
            raise
        except openai.APIStatusError as exc:
            raise ProviderUnavailable(
                f"provider_http_error status={exc.status_code}",
                category="http_error",
                http_status=exc.status_code,
                usage=usage,
            ) from exc
        except openai.APITimeoutError as exc:
            raise ProviderUnavailable(
                "provider_timeout category=timeout",
                category="timeout",
                usage=usage,
            ) from exc
        except openai.APIConnectionError as exc:
            raise ProviderUnavailable(
                "provider_transport_error category=connection",
                category="connection",
                usage=usage,
            ) from exc
        except InstructorRetryException as exc:
            raise _validation_error(exc, usage) from exc
        except (ValidationError, ValueError, TypeError) as exc:
            raise ProviderUnavailable(
                "provider_contract_error category=response_schema_validation",
                category="response_schema_validation",
                usage=usage,
            ) from exc
        raise ProviderUnavailable(
            "provider_contract_error category=response_schema_validation",
            category="response_schema_validation",
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()


def _captured(responses: list[Any], exc: InstructorRetryException) -> Any | None:
    return responses[-1] if responses else exc.last_completion


def _model_content(value: BaseModel) -> dict[str, Any]:
    return value.model_dump(mode="json")


def _usage_from_completion(completion: Any | None) -> ModelUsage:
    raw = getattr(completion, "usage", None)
    if raw is None:
        return ModelUsage()
    extras = getattr(raw, "model_extra", None) or {}
    return ModelUsage(
        input_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        cost_usd=float(extras.get("cost", 0) or 0),
    )


def _combine_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    return ModelUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cost_usd=left.cost_usd + right.cost_usd,
    )


def _validation_feedback(exc: InstructorRetryException) -> str:
    fields: set[str] = set()
    for attempt in exc.failed_attempts or []:
        error = attempt.exception
        if isinstance(error, ValidationError):
            fields.update(
                ".".join(str(part) for part in item["loc"])
                for item in error.errors(
                    include_url=False, include_context=False, include_input=False
                )
            )
    suffix = f" Invalid fields: {','.join(sorted(fields)[:8])}." if fields else ""
    return (
        "The previous response did not satisfy the required output contract."
        f"{suffix} Return one corrected JSON object only."
    )


def _validation_error(
    exc: InstructorRetryException, usage: ModelUsage
) -> ProviderUnavailable:
    del exc
    return ProviderUnavailable(
        "provider_contract_error category=response_schema_validation",
        category="response_schema_validation",
        usage=usage,
        retryable=False,
    )


def _budget_error(usage: ModelUsage, kind: str) -> ProviderUnavailable:
    return ProviderUnavailable(
        f"provider_budget_error category=task_{kind}_budget_exhausted",
        category=f"task_{kind}_budget_exhausted",
        usage=usage,
        retryable=False,
    )
