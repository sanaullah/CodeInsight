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
from infrastructure.db.prompt_artifact_repository import SqlitePromptArtifactRepository

SPECIALIST_PROMPT_TEMPLATE = "native-specialist-system"
SPECIALIST_PROMPT_VERSION = 2


class GatewaySpecialistClient:
    def __init__(
        self,
        *,
        gateway: ModelGateway,
        model: str,
        calls: SqliteModelCallRepository,
        prompts: SqlitePromptArtifactRepository,
        traces: TraceExporter,
        max_output_tokens: int = 8_000,
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.calls = calls
        self.prompts = prompts
        self.traces = traces
        self.max_output_tokens = max_output_tokens

    async def analyze(self, request: SpecialistRequest) -> SpecialistOutput:
        system_prompt = _specialist_system_prompt(request)
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
        provider = getattr(self.gateway, "gateway", self.gateway)
        provider_hint = str(
            getattr(provider, "gateway_identity", type(provider).__name__)
        )
        schema_json = json.dumps(
            SpecialistOutput.model_json_schema(),
            sort_keys=True,
            separators=(",", ":"),
        )
        request_hash = hashlib.sha256(
            (
                "native-specialist-v2\0"
                + self.model
                + "\0"
                + provider_hint
                + "\0"
                + str(SPECIALIST_PROMPT_VERSION)
                + "\0"
                + schema_json
                + "\0"
                + user_prompt
            ).encode()
        ).hexdigest()
        model_call_id = hashlib.sha256(
            f"{request.task_id}\0{request_hash}".encode()
        ).hexdigest()
        prompt_artifact_id = hashlib.sha256(
            (
                f"{request.task_id}\0{SPECIALIST_PROMPT_TEMPLATE}\0"
                f"{SPECIALIST_PROMPT_VERSION}\0{request_hash}"
            ).encode()
        ).hexdigest()
        correlation = {
            "run_id": request.run_id,
            "wave_id": request.wave_id,
            "task_id": request.task_id,
            "attempt_id": request.attempt_id,
            "attempt_number": str(request.attempt_number),
            "model_call_id": model_call_id,
            "prompt_artifact_id": prompt_artifact_id,
        }
        self.prompts.record(
            prompt_artifact_id=prompt_artifact_id,
            run_id=request.run_id,
            wave_id=request.wave_id,
            role_id=request.role.role_id,
            task_id=request.task_id,
            prompt_template=SPECIALIST_PROMPT_TEMPLATE,
            prompt_version=SPECIALIST_PROMPT_VERSION,
            prompt_text=system_prompt,
            request_hash=request_hash,
        )
        self.calls.start(
            model_call_id=model_call_id,
            run_id=request.run_id,
            wave_id=request.wave_id,
            task_id=request.task_id,
            attempt_id=request.attempt_id,
            provider=provider_hint,
            model=self.model,
            request_hash=request_hash,
            correlation=correlation,
        )
        self.traces.emit(
            TraceEvent(
                name="prompt_artifact_recorded",
                run_id=request.run_id,
                wave_id=request.wave_id,
                role_id=request.role.role_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                attempt_number=request.attempt_number,
                model_call_id=model_call_id,
                prompt_artifact_id=prompt_artifact_id,
                attributes={
                    "prompt_template": SPECIALIST_PROMPT_TEMPLATE,
                    "prompt_version": SPECIALIST_PROMPT_VERSION,
                    "request_hash": request_hash,
                    "prompt_text": system_prompt,
                },
            )
        )
        self.traces.emit(
            TraceEvent(
                name="model_call_started",
                run_id=request.run_id,
                wave_id=request.wave_id,
                role_id=request.role.role_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                attempt_number=request.attempt_number,
                model_call_id=model_call_id,
                prompt_artifact_id=prompt_artifact_id,
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
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_schema=SpecialistOutput.model_json_schema(),
                    response_model=SpecialistOutput,
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
                    role_id=request.role.role_id,
                    task_id=request.task_id,
                    attempt_id=request.attempt_id,
                    attempt_number=request.attempt_number,
                    model_call_id=model_call_id,
                    prompt_artifact_id=prompt_artifact_id,
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
                    role_id=request.role.role_id,
                    task_id=request.task_id,
                    attempt_id=request.attempt_id,
                    attempt_number=request.attempt_number,
                    model_call_id=model_call_id,
                    prompt_artifact_id=prompt_artifact_id,
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
                role_id=request.role.role_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                attempt_number=request.attempt_number,
                model_call_id=model_call_id,
                prompt_artifact_id=prompt_artifact_id,
                attributes={
                    "provider": response.provider,
                    "model_completion": response.content,
                    **usage,
                },
            )
        )
        return output


def _specialist_system_prompt(request: SpecialistRequest) -> str:
    """Render the exact provider system prompt without source bodies or credentials."""

    role = request.role
    prompt_contract = {
        "allowed_tools": list(role.allowed_tools),
        "completion_criteria": list(role.completion_criteria),
        "coverage_targets": list(role.coverage_targets),
        "mission": role.mission,
        "name": role.name,
        "rationale": role.rationale,
        "required_capabilities": list(role.required_capabilities),
        "role_id": role.role_id,
    }
    return (
        "You are a read-only repository specialist. Return only JSON matching "
        "the supplied schema. Every finding must cite one or more exact "
        "assigned-file line spans. Do not claim runtime behavior that the "
        "supplied source cannot prove.\n\nSpecialist contract:\n"
        + json.dumps(prompt_contract, indent=2, sort_keys=True)
        + "\n\nAssigned source files are supplied separately and are intentionally "
        "excluded from this versioned prompt artifact."
    )
