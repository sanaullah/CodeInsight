"""Bounded AI role proposals derived from one validated architecture model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from application.model_gateway import ModelGateway, ModelRequest
from domain.contracts import ArchitectureDiscovery, ContractModel, RunBudget

PROMPT_TEMPLATE = "architecture-role-proposals"
PROMPT_VERSION = 1
PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "prompts"
    / "core"
    / "phases"
    / "planning"
    / "role-proposals.md"
)


class ProposedRole(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    mission: str = Field(min_length=1, max_length=4_000)
    rationale: str = Field(min_length=1, max_length=4_000)
    coverage_targets: tuple[str, ...] = Field(min_length=1, max_length=20)
    required_capabilities: tuple[
        Literal[
            "architecture",
            "change-impact",
            "configuration",
            "language-analysis",
            "security",
            "test-quality",
            "verification",
        ],
        ...,
    ] = Field(min_length=1, max_length=8)
    focus_paths: tuple[str, ...] = Field(default=(), max_length=80)


class RoleProposalOutput(ContractModel):
    roles: tuple[ProposedRole, ...] = Field(min_length=1, max_length=12)
    unknowns: tuple[str, ...] = Field(max_length=40)


class AiRoleProposalService:
    """Ask the model for bounded role candidates; the host remains approver."""

    def __init__(self, *, gateway: ModelGateway | None, model: str) -> None:
        self.gateway = gateway
        self.model = model

    async def propose(
        self,
        *,
        run_id: str,
        discovery: ArchitectureDiscovery,
        budget: RunBudget,
    ) -> RoleProposalOutput | None:
        if self.gateway is None or discovery.status != "model-validated":
            return None
        prompt = role_proposal_system_prompt()
        request = ModelRequest(
            run_id=run_id,
            wave_id="role-proposals",
            task_id=_stable_id(run_id, discovery.result_hash),
            model=self.model,
            system_prompt=prompt,
            user_prompt=json.dumps(
                {
                    "architecture": discovery.result,
                    "architecture_result_hash": discovery.result_hash,
                    "max_roles": min(budget.max_specialists, 12),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            response_schema=RoleProposalOutput.model_json_schema(),
            response_model=RoleProposalOutput,
            max_output_tokens=min(3_000, budget.max_tokens),
            timeout_seconds=min(90, budget.max_elapsed_seconds),
            token_budget=min(8_000, budget.max_tokens),
            cost_budget_usd=min(0.50, budget.max_cost_usd),
            correlation={
                "kind": "role-proposals",
                "architecture_result_hash": discovery.result_hash,
            },
        )
        try:
            response = await self.gateway.complete(request)
            return RoleProposalOutput.model_validate(response.content)
        except (ValidationError, ValueError, RuntimeError, OSError):
            return None


def role_proposal_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8").strip() + "\n"
    except OSError as exc:
        raise RuntimeError(f"role proposal prompt unavailable: {PROMPT_PATH}") from exc


def _stable_id(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()
