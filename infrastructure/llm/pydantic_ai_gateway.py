"""Pydantic AI runtime adapter for OpenAI-compatible CodeInsight providers."""

from __future__ import annotations

import asyncio
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, exceptions
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

from application.model_gateway import (
    ModelGateway,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderUnavailable,
)


class PydanticAIOpenAICompatibleGateway(ModelGateway):
    """Run CodeInsight's typed model contracts through Pydantic AI agents.

    The durable scheduler remains outside this adapter.  This class owns only
    the bounded, typed LLM interaction: structured output, one correction
    retry, usage limits and provider error normalization.
    """

    def __init__(
        self, *, base_url: str, api_key: str = "", client: AsyncOpenAI | None = None
    ) -> None:
        self.gateway_identity = type(self).__name__
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            base_url=base_url.rstrip("/") + "/",
            api_key=api_key or "not-required",
            max_retries=0,
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response_model = request.response_model
        if response_model is None:
            raise ProviderUnavailable(
                "provider_contract_error category=missing_response_model",
                category="missing_response_model",
                retryable=False,
            )
        try:
            async with asyncio.timeout(request.timeout_seconds):
                result = await self._agent(request, response_model).run(
                    request.user_prompt,
                    usage_limits=UsageLimits(
                        request_limit=2,
                        total_tokens_limit=request.token_budget,
                        output_tokens_limit=request.max_output_tokens,
                    ),
                )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise ProviderUnavailable(
                "provider_timeout category=timeout", category="timeout"
            ) from exc
        except exceptions.ModelHTTPError as exc:
            raise ProviderUnavailable(
                f"provider_http_error status={exc.status_code}",
                category="http_error",
                http_status=exc.status_code,
            ) from exc
        except exceptions.AgentRunError as exc:
            raise ProviderUnavailable(
                "provider_contract_error category=pydantic_ai_agent_run",
                category="pydantic_ai_agent_run",
                retryable=False,
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ProviderUnavailable(
                "provider_contract_error category=response_schema_validation",
                category="response_schema_validation",
                retryable=False,
            ) from exc

        output = result.output
        if not isinstance(output, BaseModel):
            raise ProviderUnavailable(
                "provider_contract_error category=response_schema_validation",
                category="response_schema_validation",
                retryable=False,
            )
        usage = result.usage
        return ModelResponse(
            content=output.model_dump(mode="json"),
            provider="pydantic-ai-openai-compatible",
            model=request.model,
            usage=ModelUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )

    def _agent(self, request: ModelRequest, output_type: type[BaseModel]) -> Agent[Any, Any]:
        model = OpenAIChatModel(
            request.model,
            provider=OpenAIProvider(openai_client=self._client),
        )
        return Agent(
            model,
            name="codeinsight_typed_worker",
            instructions=request.system_prompt,
            output_type=output_type,
            retries=1,
            model_settings={"temperature": 0},
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()
