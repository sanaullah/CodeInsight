"""Provider-neutral, budget-aware model boundary for native analysis."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel

from application.tracing import TraceExporter


@dataclass(frozen=True, slots=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int | float]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True, slots=True)
class ModelRequest:
    run_id: str
    wave_id: str
    task_id: str
    model: str
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, Any]
    max_output_tokens: int
    timeout_seconds: float
    token_budget: int
    cost_budget_usd: float
    response_model: type[BaseModel] | None = None
    correlation: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: dict[str, Any]
    provider: str
    model: str
    usage: ModelUsage = ModelUsage()
    provider_request_id: str | None = None


class ModelGateway(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...


class ModelBudgetExceeded(RuntimeError):
    """The durable run or task budget cannot admit another model call."""


class ProviderUnavailable(RuntimeError):
    """The configured provider could not complete a request."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_unavailable",
        http_status: int | None = None,
        usage: ModelUsage | None = None,
        response_content: str | None = None,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.usage = usage or ModelUsage()
        self.response_content = response_content
        self.retryable = retryable


class BoundedModelGateway:
    """Enforce provider concurrency and per-run token/cost budgets."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        max_concurrent: int = 2,
        tracer: TraceExporter | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be greater than zero")
        self.gateway = gateway
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._usage: dict[str, ModelUsage] = {}
        self._limits: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()
        self._tracer = tracer

    async def configure_run_budget(
        self, run_id: str, *, max_tokens: int, max_cost_usd: float
    ) -> None:
        async with self._lock:
            self._limits[run_id] = (max_tokens, max_cost_usd)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        async with self._lock:
            used = self._usage.get(request.run_id, ModelUsage())
            token_limit, cost_limit = self._limits.get(
                request.run_id, (request.token_budget, request.cost_budget_usd)
            )
            if used.total_tokens >= token_limit:
                raise ModelBudgetExceeded("run token budget is exhausted")
            if cost_limit == 0 and not isinstance(self.gateway, OfflineModelGateway):
                raise ModelBudgetExceeded("run cost budget permits only free providers")
            if cost_limit > 0 and used.cost_usd >= cost_limit:
                raise ModelBudgetExceeded("run cost budget is exhausted")
        async with self._semaphore:
            try:
                response = await asyncio.wait_for(
                    self.gateway.complete(request), timeout=request.timeout_seconds
                )
            except ProviderUnavailable as exc:
                if exc.usage.total_tokens or exc.usage.cost_usd:
                    await self._commit_usage(request, exc.usage)
                self._record_model_call(request, error=exc)
                raise
        await self._commit_usage(request, response.usage)
        self._record_model_call(request, response=response)
        return response

    def _record_model_call(
        self,
        request: ModelRequest,
        *,
        response: ModelResponse | None = None,
        error: ProviderUnavailable | None = None,
    ) -> None:
        record = getattr(self._tracer, "record_model_call", None)
        if callable(record):
            record(request, response=response, error=error)

    async def _commit_usage(
        self, request: ModelRequest, usage: ModelUsage
    ) -> None:
        async with self._lock:
            previous = self._usage.get(request.run_id, ModelUsage())
            combined = ModelUsage(
                input_tokens=previous.input_tokens + usage.input_tokens,
                output_tokens=previous.output_tokens + usage.output_tokens,
                cost_usd=previous.cost_usd + usage.cost_usd,
            )
            token_limit, cost_limit = self._limits.get(
                request.run_id, (request.token_budget, request.cost_budget_usd)
            )
            if combined.total_tokens > token_limit:
                raise ModelBudgetExceeded("provider response exceeded run token budget")
            if cost_limit >= 0 and combined.cost_usd > cost_limit:
                raise ModelBudgetExceeded("provider response exceeded run cost budget")
            self._usage[request.run_id] = combined

    async def usage_for_run(self, run_id: str) -> ModelUsage:
        async with self._lock:
            return self._usage.get(run_id, ModelUsage())

    async def aclose(self) -> None:
        close = getattr(self.gateway, "aclose", None)
        if close is not None:
            await close()


class OfflineModelGateway:
    """Deterministic no-network gateway used when no provider is configured."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        return ModelResponse(
            content={
                "analyzed_paths": [],
                "evidence": [],
                "findings": [],
                "unresolved_uncertainty": [
                    "model-provider:not-configured; repository was indexed but "
                    "no model-backed specialist analysis ran"
                ],
                "usage": {},
            },
            provider="offline",
            model="deterministic-index-only",
        )


def parse_json_object(value: str) -> dict[str, Any]:
    """Parse a provider response without accepting prose around the contract."""

    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("model response is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("model response must be a JSON object")
    return result
