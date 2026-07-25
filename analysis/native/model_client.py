"""Model-gateway adapter for the strict native specialist contract."""

from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from analysis.native.harness import SpecialistOutput, SpecialistRequest
from application.model_gateway import (
    ModelGateway,
    ModelRequest,
    ProviderUnavailable,
)
from application.tracing import TraceEvent, TraceExporter
from infrastructure.db.model_call_repository import SqliteModelCallRepository


class GatewaySpecialistClient:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        model: str,
        calls: SqliteModelCallRepository,
        traces: TraceExporter,
        max_output_tokens: int = 8_000,
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.calls = calls
        self.traces = traces
        self.max_output_tokens = max_output_tokens

    async def analyze(self, request: SpecialistRequest) -> SpecialistOutput:
        user_payload = {
            "role": request.role.model_dump(mode="json"),
            "files": [
                {
                    "relative_path": item.relative_path,
                    "language": item.language,
                    "classification": item.classification,
                    "support_tier": item.support_tier,
                    "content": item.content,
                }
                for item in request.files
            ],
        }
        user_prompt = json.dumps(user_payload, sort_keys=True)
        request_hash = hashlib.sha256(
            (
                "native-specialist-v1\0"
                + self.model
                + "\0"
                + user_prompt
            ).encode()
        ).hexdigest()
        model_call_id = hashlib.sha256(
            f"{request.task_id}\0{request_hash}".encode()
        ).hexdigest()
        correlation = {
            "run_id": request.run_id,
            "wave_id": request.wave_id,
            "task_id": request.task_id,
            "model_call_id": model_call_id,
        }
        provider_hint = type(getattr(self.gateway, "gateway", self.gateway)).__name__
        self.calls.start(
            model_call_id=model_call_id,
            run_id=request.run_id,
            wave_id=request.wave_id,
            task_id=request.task_id,
            provider=provider_hint,
            model=self.model,
            request_hash=request_hash,
            correlation=correlation,
        )
        self.traces.emit(
            TraceEvent(
                name="model_call_started",
                run_id=request.run_id,
                wave_id=request.wave_id,
                task_id=request.task_id,
                model_call_id=model_call_id,
                attributes={"model": self.model, "request_hash": request_hash},
            )
        )
        try:
            response = await self.gateway.complete(
                ModelRequest(
                    run_id=request.run_id,
                    wave_id=request.wave_id,
                    task_id=request.task_id,
                    model=self.model,
                    system_prompt=(
                        "You are a read-only repository specialist. Return only "
                        "JSON matching the supplied schema. Every finding must cite "
                        "one or more exact assigned-file line spans. Do not claim "
                        "runtime behavior that the supplied source cannot prove."
                    ),
                    user_prompt=user_prompt,
                    response_schema=SpecialistOutput.model_json_schema(),
                    max_output_tokens=min(
                        self.max_output_tokens, max(1, request.token_budget)
                    ),
                    timeout_seconds=float(request.time_budget_seconds),
                    token_budget=request.token_budget,
                    cost_budget_usd=request.cost_budget_usd,
                    correlation=correlation,
                )
            )
        except BaseException:
            self.calls.finish(model_call_id, status="failed")
            self.traces.emit(
                TraceEvent(
                    name="model_call_failed",
                    run_id=request.run_id,
                    wave_id=request.wave_id,
                    task_id=request.task_id,
                    model_call_id=model_call_id,
                )
            )
            raise
        try:
            content = dict(response.content)
            if response.provider == "offline":
                content["analyzed_paths"] = [
                    item.relative_path for item in request.files
                ]
            content["usage"] = response.usage.as_dict()
            output = SpecialistOutput.model_validate(content)
        except ValidationError as exc:
            self.calls.finish(model_call_id, status="failed")
            locations = sorted(
                {
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors(include_url=False, include_context=False)
                }
            )
            field_summary = ",".join(locations[:8])
            self.traces.emit(
                TraceEvent(
                    name="model_call_failed",
                    run_id=request.run_id,
                    wave_id=request.wave_id,
                    task_id=request.task_id,
                    model_call_id=model_call_id,
                    attributes={
                        "category": "response_schema_validation",
                        "error_count": exc.error_count(),
                    },
                )
            )
            raise ProviderUnavailable(
                "provider_contract_error "
                "category=response_schema_validation "
                f"errors={exc.error_count()} fields={field_summary}",
                category="response_schema_validation",
            ) from exc
        usage = response.usage.as_dict()
        self.calls.finish(model_call_id, status="succeeded", usage=usage)
        self.traces.emit(
            TraceEvent(
                name="model_call_completed",
                run_id=request.run_id,
                wave_id=request.wave_id,
                task_id=request.task_id,
                model_call_id=model_call_id,
                attributes={"provider": response.provider, **usage},
            )
        )
        return output
