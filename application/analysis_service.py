"""Native durable analysis coordination behind the HTTP boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen
from uuid import uuid4

from analysis.native.coordinator import NativeAnalysisCoordinator
from analysis.native.harness import TrustedSpecialistHarness
from analysis.native.model_client import GatewaySpecialistClient
from api.config import ApiSettings, default_database_path
from api.models import (
    AnalysisIntelligence,
    AnalysisRequest,
    AnalysisRun,
    RecoveryResponse,
)
from application.model_gateway import (
    BoundedModelGateway,
    ModelGateway,
    OfflineModelGateway,
)
from application.tracing import NullTraceExporter, TraceEvent, TraceExporter
from domain.contracts import RunStage
from indexing.repository_index import RepositoryIndexer
from infrastructure.artifacts.store import FilesystemArtifactStore
from infrastructure.db.analysis_repository import SqliteAnalysisRepository
from infrastructure.db.architecture_repository import SqliteArchitectureRepository
from infrastructure.db.artifact_repository import SqliteArtifactRepository
from infrastructure.db.component_annotation_repository import (
    SqliteComponentAnnotationRepository,
)
from infrastructure.db.database import checkpoint_database
from infrastructure.db.finding_repository import SqliteFindingRepository
from infrastructure.db.history_repository import SqliteHistoryRepository
from infrastructure.db.model_call_repository import SqliteModelCallRepository
from infrastructure.db.prompt_artifact_repository import SqlitePromptArtifactRepository
from infrastructure.db.run_ledger import SqliteRunLedger
from infrastructure.db.semantic_architecture_query import (
    SqliteSemanticArchitectureQuery,
)
from infrastructure.db.settings_repository import SqliteSettingsRepository
from infrastructure.db.snapshot_repository import SqliteSnapshotRepository
from infrastructure.db.task_repository import SqliteTaskRepository
from infrastructure.llm.gateway import OpenAICompatibleGateway
from infrastructure.llm.instructor_gateway import (
    InstructorOpenAICompatibleGateway,
    provider_capability_profile,
)
from infrastructure.observability.langfuse import LangfuseTraceExporter
from workflow.task_scheduler import NativeTaskScheduler

EventSink = Callable[[str, dict[str, Any]], None]


class AnalysisExecutor(Protocol):
    async def execute(
        self, run_id: str, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]: ...


class NativeAnalysisExecutor:
    """Compose the durable index, scheduler, specialist, and coverage pipeline."""

    def __init__(
        self,
        *,
        ledger: SqliteRunLedger,
        gateway: ModelGateway,
        default_model: str,
        tracer: TraceExporter,
        artifacts_path: Path,
        max_task_concurrency: int = 4,
        offline: bool = False,
    ) -> None:
        self.ledger = ledger
        self.gateway = gateway
        self.default_model = default_model
        self.tracer = tracer
        self.artifacts_path = artifacts_path
        self.max_task_concurrency = max_task_concurrency
        self.offline = offline

    async def execute(
        self, run_id: str, request: AnalysisRequest, event_sink: EventSink
    ) -> dict[str, Any]:
        snapshots = SqliteSnapshotRepository(self.ledger)
        analysis = SqliteAnalysisRepository(self.ledger)
        tasks = SqliteTaskRepository(self.ledger)
        artifacts = FilesystemArtifactStore(self.artifacts_path)
        artifact_records = SqliteArtifactRepository(self.ledger)

        self.ledger.set_stage(run_id, RunStage.DISCOVER)
        event_sink("repository_discovery_started", {"project": request.project_path})
        indexer = RepositoryIndexer(
            snapshots,
            artifacts,
            max_files=request.run_budget().max_files,
        )
        index = await asyncio.to_thread(
            indexer.build,
            request.project_path,
            selected_directories=tuple(request.selected_directories or ()),
            include_extensions=tuple(request.file_extensions or ()),
        )
        event_sink(
            "repository_index_ready",
            {
                "snapshot_id": index.snapshot.snapshot_id,
                "cached": index.cached,
                "file_count": index.file_count,
                "symbol_count": index.symbol_count,
                "edge_count": index.edge_count,
                "target_count": len(index.target_paths),
            },
        )
        budget = request.run_budget()
        if self.offline and budget.max_waves > 1:
            budget = budget.model_copy(update={"max_waves": 1})
        specialist = GatewaySpecialistClient(
            gateway=self.gateway,
            model=request.model_name or self.default_model,
            calls=SqliteModelCallRepository(self.ledger),
            prompts=SqlitePromptArtifactRepository(self.ledger),
            traces=self.tracer,
            max_output_tokens=min(8_000, budget.max_tokens),
        )
        harness = TrustedSpecialistHarness(
            client=specialist,
            snapshots=snapshots,
            analysis=analysis,
            artifacts=artifacts,
            artifact_records=artifact_records,
            event_sink=event_sink,
        )
        scheduler = NativeTaskScheduler(
            tasks,
            {"specialist_analysis": harness.execute},
            max_concurrent=min(self.max_task_concurrency, request.max_agents),
            event_sink=event_sink,
        )
        coordinator = NativeAnalysisCoordinator(
            ledger=self.ledger,
            tasks=tasks,
            snapshots=snapshots,
            analysis=analysis,
            scheduler=scheduler,
        )
        if isinstance(self.gateway, BoundedModelGateway):
            await self.gateway.configure_run_budget(
                run_id,
                max_tokens=budget.max_tokens,
                max_cost_usd=budget.max_cost_usd,
            )
        result = await coordinator.execute(
            run_id=run_id,
            snapshot=index.snapshot,
            target_paths=index.target_paths,
            mode=request.mode,
            budget=budget,
        )
        intelligence = analysis.run_intelligence(run_id) or {}
        findings = [item.model_dump(mode="json") for item in result.findings]
        coverage = [item.model_dump(mode="json") for item in result.coverage]
        return {
            "synthesized_report": _synthesize(findings, coverage),
            "snapshot": {
                "snapshot_id": index.snapshot.snapshot_id,
                "cached": index.cached,
                "file_count": index.file_count,
                "symbol_count": index.symbol_count,
                "edge_count": index.edge_count,
                "changed_paths": list(index.changed_paths),
                "target_paths": list(index.target_paths),
            },
            "findings": findings,
            "coverage": coverage,
            "wave_count": result.wave_count,
            "failed_task_count": result.failed_task_count,
            "usage": intelligence.get("usage", {}),
            "provider_mode": "index-only" if self.offline else "model-backed",
            "outcome": ("needs_attention" if result.failed_task_count else "succeeded"),
        }


class AnalysisService:
    """Own process-local workers while SQLite remains authoritative."""

    def __init__(
        self,
        executor: AnalysisExecutor | None = None,
        *,
        database_path: str | Path | None = None,
        max_concurrent: int = 2,
        event_history_limit: int = 200,
        settings: ApiSettings | None = None,
        gateway: ModelGateway | None = None,
        tracer: TraceExporter | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be greater than zero")
        if event_history_limit < 1:
            raise ValueError("event_history_limit must be greater than zero")
        self.settings = settings or ApiSettings(
            database_path=Path(database_path or default_database_path()).resolve(),
            max_concurrent_analyses=max_concurrent,
            event_history_limit=event_history_limit,
        )
        self.executor = executor
        self.database_path = Path(database_path or self.settings.database_path).resolve()
        self.max_concurrent = max_concurrent
        self.event_history_limit = event_history_limit
        self._configured_gateway = gateway
        self._tracer = tracer
        self._ledger: SqliteRunLedger | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._task_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._started = False
        self._closing = False
        self._startup_recovered = 0

    def _get_ledger(self) -> SqliteRunLedger:
        if self._ledger is None:
            self._ledger = SqliteRunLedger(
                self.database_path,
                event_history_limit=self.event_history_limit,
            )
        return self._ledger

    def _get_tracer(self) -> TraceExporter:
        if self._tracer is not None:
            return self._tracer
        if (
            self.settings.langfuse_enabled
            and self.settings.langfuse_public_key
            and self.settings.langfuse_secret_key
        ):
            self._tracer = LangfuseTraceExporter(
                public_key=self.settings.langfuse_public_key,
                secret_key=self.settings.langfuse_secret_key,
                host=self.settings.langfuse_host,
                capture_content=self.settings.langfuse_capture_content,
            )
        else:
            self._tracer = NullTraceExporter()
        return self._tracer

    def _get_executor(self) -> AnalysisExecutor:
        if self.executor is not None:
            return self.executor
        offline = self._configured_gateway is None and not self.settings.model_base_url
        profile = provider_capability_profile(
            self.settings.provider_capability_profile
        )
        if self._configured_gateway is not None:
            provider: ModelGateway = self._configured_gateway
        elif not self.settings.model_base_url:
            provider = OfflineModelGateway()
        elif profile.instructor_supported:
            provider = InstructorOpenAICompatibleGateway(
                base_url=self.settings.model_base_url,
                api_key=self.settings.model_api_key,
                profile=profile,
            )
        else:
            provider = OpenAICompatibleGateway(
                base_url=self.settings.model_base_url,
                api_key=self.settings.model_api_key,
            )
        bounded = BoundedModelGateway(
            provider,
            max_concurrent=self.settings.max_concurrent_model_calls,
        )
        self.executor = NativeAnalysisExecutor(
            ledger=self._get_ledger(),
            gateway=bounded,
            default_model=self.settings.default_model,
            tracer=self._get_tracer(),
            artifacts_path=self.database_path.parent / "artifacts",
            offline=offline,
        )
        return self.executor

    @property
    def active_count(self) -> int:
        return self._get_ledger().count_active()

    @property
    def provider_configured(self) -> bool:
        return self._configured_gateway is not None or bool(self.settings.model_base_url)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.settings.langfuse_enabled
            and self.settings.langfuse_public_key
            and self.settings.langfuse_secret_key
        )

    async def start(self) -> int:
        async with self._start_lock:
            if self._started:
                return 0
            self._closing = False
            ledger = self._get_ledger()
            recovered = ledger.recover_interrupted()
            SqliteTaskRepository(ledger).recover_expired()
            self._startup_recovered = recovered
            self._started = True
            for run_id in ledger.list_queued_run_ids():
                await self._schedule(run_id)
            return recovered

    async def submit(
        self,
        request: AnalysisRequest,
        *,
        source_run_id: str | None = None,
    ) -> AnalysisRun:
        await self.start()
        normalized_request = request.model_copy(
            update={"project_path": self._resolve_project_path(request.project_path)}
        )
        run_id = uuid4().hex
        budget = normalized_request.run_budget()
        record = self._get_ledger().create_run(
            run_id=run_id,
            submission_key=run_id,
            request=normalized_request.model_dump(mode="json"),
            mode=normalized_request.mode.value,
            budget=budget.model_dump(mode="json"),
        )
        if source_run_id is not None:
            self._get_ledger().append_event(
                record["run_id"],
                "analysis_rerun_requested",
                {
                    "source_run_id": source_run_id,
                    "input_policy": "latest-files",
                },
            )
        await self._schedule(record["run_id"])
        return self._to_model(record)

    async def rerun_latest(self, source_run_id: str) -> AnalysisRun | None:
        """Clone the expressed request and build a fresh snapshot from current files."""

        await self.start()
        source = self._get_ledger().get_run(source_run_id)
        if source is None:
            return None
        request = AnalysisRequest.from_persisted(source["request"])
        return await self.submit(request, source_run_id=source_run_id)

    async def get(self, run_id: str) -> AnalysisRun | None:
        await self.start()
        record = self._get_ledger().get_run(run_id)
        return self._to_model(record) if record else None

    async def intelligence(self, run_id: str) -> AnalysisIntelligence | None:
        await self.start()
        record = SqliteAnalysisRepository(self._get_ledger()).run_intelligence(run_id)
        return AnalysisIntelligence.model_validate(record) if record else None

    async def query_findings(self, **filters: Any) -> dict[str, Any]:
        await self.start()
        return await asyncio.to_thread(SqliteFindingRepository(self._get_ledger()).query, **filters)

    async def finding_detail(self, finding_id: str) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteFindingRepository(self._get_ledger()).detail, finding_id
        )

    async def update_finding_review(
        self,
        finding_id: str,
        *,
        review_state: str,
        note: str | None,
        expected_version: int | None,
    ) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteFindingRepository(self._get_ledger()).set_review_state,
            finding_id,
            review_state=review_state,
            note=note,
            expected_version=expected_version,
        )

    async def list_snapshots(self, limit: int = 20) -> list[dict[str, Any]]:
        await self.start()
        return await asyncio.to_thread(
            SqliteArchitectureRepository(self._get_ledger()).list_snapshots, limit
        )

    async def architecture_graph(self, snapshot_id: str, **options: Any) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteArchitectureRepository(self._get_ledger()).graph,
            snapshot_id,
            **options,
        )

    async def architecture_trace(self, snapshot_id: str, **options: Any) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteArchitectureRepository(self._get_ledger()).trace,
            snapshot_id,
            **options,
        )

    async def semantic_architecture_summary(
        self, snapshot_id: str
    ) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteSemanticArchitectureQuery(self._get_ledger()).summary,
            snapshot_id,
        )

    async def semantic_architecture_graph(
        self, snapshot_id: str, **options: Any
    ) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteSemanticArchitectureQuery(self._get_ledger()).graph,
            snapshot_id,
            **options,
        )

    async def semantic_component(
        self, snapshot_id: str, component_id: str
    ) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteSemanticArchitectureQuery(self._get_ledger()).component,
            snapshot_id,
            component_id,
        )

    async def update_component_annotation(
        self,
        snapshot_id: str,
        component_id: str,
        *,
        note: str,
        expected_version: int,
    ) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteComponentAnnotationRepository(self._get_ledger()).upsert,
            snapshot_id,
            component_id,
            note=note,
            expected_version=expected_version,
        )

    async def semantic_architecture_trace(
        self, snapshot_id: str, **options: Any
    ) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteSemanticArchitectureQuery(self._get_ledger()).trace,
            snapshot_id,
            **options,
        )

    async def query_history(self, **filters: Any) -> dict[str, Any]:
        await self.start()
        return await asyncio.to_thread(SqliteHistoryRepository(self._get_ledger()).query, **filters)

    async def get_settings(self) -> dict[str, Any]:
        await self.start()
        return await asyncio.to_thread(SqliteSettingsRepository(self._get_ledger()).get)

    async def update_settings(
        self, settings: dict[str, Any], expected_version: int
    ) -> dict[str, Any]:
        await self.start()
        return await asyncio.to_thread(
            SqliteSettingsRepository(self._get_ledger()).update,
            settings,
            expected_version,
        )

    async def list_presets(self) -> list[dict[str, Any]]:
        await self.start()
        return await asyncio.to_thread(
            SqliteSettingsRepository(self._get_ledger()).list_presets
        )

    async def save_preset(self, **values: Any) -> dict[str, Any]:
        await self.start()
        return await asyncio.to_thread(
            SqliteSettingsRepository(self._get_ledger()).save_preset,
            **values,
        )

    async def delete_preset(self, preset_id: str, expected_version: int) -> bool:
        await self.start()
        return await asyncio.to_thread(
            SqliteSettingsRepository(self._get_ledger()).delete_preset,
            preset_id,
            expected_version,
        )

    async def test_provider(self) -> dict[str, Any]:
        """Probe the configured OpenAI-compatible endpoint without exposing secrets."""

        if self._configured_gateway is not None:
            return {
                "configured": True,
                "reachable": True,
                "status": "Injected provider is available in this process.",
                "latency_ms": None,
            }
        if not self.settings.model_base_url:
            return {
                "configured": False,
                "reachable": False,
                "status": "Provider endpoint is not configured.",
                "latency_ms": None,
            }
        return await asyncio.to_thread(self._probe_provider)

    def _probe_provider(self) -> dict[str, Any]:
        from time import monotonic

        endpoint = f"{self.settings.model_base_url.rstrip('/')}/models"
        headers = {"Accept": "application/json"}
        if self.settings.model_api_key:
            headers["Authorization"] = f"Bearer {self.settings.model_api_key}"
        started = monotonic()
        try:
            with urlopen(UrlRequest(endpoint, headers=headers), timeout=3) as response:
                reachable = 200 <= response.status < 500
                status_message = f"Provider responded with HTTP {response.status}."
        except HTTPError as exc:
            reachable = exc.code < 500
            status_message = f"Provider responded with HTTP {exc.code}."
        except (URLError, TimeoutError, OSError):
            reachable = False
            status_message = "Provider could not be reached within the bounded probe."
        return {
            "configured": True,
            "reachable": reachable,
            "status": status_message,
            "latency_ms": round((monotonic() - started) * 1000),
        }

    async def compare_runs(self, baseline_run_id: str, target_run_id: str) -> dict[str, Any] | None:
        await self.start()
        return await asyncio.to_thread(
            SqliteHistoryRepository(self._get_ledger()).compare,
            baseline_run_id,
            target_run_id,
        )

    async def history_trends(self, days: int) -> dict[str, Any]:
        await self.start()
        return await asyncio.to_thread(SqliteHistoryRepository(self._get_ledger()).trends, days)

    async def list(self, limit: int = 20) -> list[AnalysisRun]:
        await self.start()
        return [self._to_model(record) for record in self._get_ledger().list_runs(limit)]

    async def cancel(self, run_id: str) -> AnalysisRun | None:
        await self.start()
        ledger = self._get_ledger()
        if ledger.get_run(run_id) is None:
            return None
        ledger.cancel(run_id)
        async with self._task_lock:
            task = self._tasks.get(run_id)
            if task is not None:
                task.cancel()
        record = ledger.get_run(run_id)
        return self._to_model(record) if record else None

    async def recover(self) -> RecoveryResponse:
        await self.start()
        repository = SqliteTaskRepository(self._get_ledger())
        recovered_tasks = repository.recover_expired()
        queued = self._get_ledger().list_queued_run_ids()
        for run_id in queued:
            await self._schedule(run_id)
        result = RecoveryResponse(
            recovered_runs=self._startup_recovered,
            recovered_tasks=recovered_tasks,
            scheduled_runs=len(queued),
        )
        self._startup_recovered = 0
        return result

    async def close(self) -> None:
        self._closing = True
        async with self._task_lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if isinstance(self.executor, NativeAnalysisExecutor):
            gateway = self.executor.gateway
            close = getattr(gateway, "aclose", None)
            if close is not None:
                await close()
        if self._tracer is not None:
            self._tracer.close()
        if self._ledger is not None:
            self._ledger.close()
            checkpoint_database(self.database_path)
            self._ledger = None
        self._started = False

    async def _schedule(self, run_id: str) -> None:
        async with self._task_lock:
            existing = self._tasks.get(run_id)
            if existing is not None and not existing.done():
                return
            task = asyncio.create_task(self._execute_run(run_id), name=f"analysis-{run_id}")
            self._tasks[run_id] = task
            task.add_done_callback(
                lambda _task, scheduled_run_id=run_id: self._tasks.pop(scheduled_run_id, None)
            )

    async def _execute_run(self, run_id: str) -> None:
        ledger = self._get_ledger()
        try:
            async with self._semaphore:
                if not ledger.mark_running(run_id):
                    return
                record = ledger.get_run(run_id)
                if record is None:
                    return
                request = AnalysisRequest.from_persisted(record["request"])

                def event_sink(event_type: str, data: dict[str, Any]) -> None:
                    ledger.append_event(run_id, event_type, data)
                    self._get_tracer().emit(
                        TraceEvent(
                            name=event_type,
                            run_id=run_id,
                            wave_id=data.get("wave_id"),
                            task_id=data.get("task_id"),
                            attributes=data,
                        )
                    )

                result = await self._get_executor().execute(run_id, request, event_sink)
                if result.get("outcome") == "needs_attention":
                    ledger.needs_attention(run_id, result)
                else:
                    ledger.succeed(run_id, result)
        except asyncio.CancelledError:
            if self._closing:
                ledger.requeue_interrupted(run_id)
            else:
                ledger.cancel(run_id)
            raise
        except Exception as exc:
            ledger.fail(run_id, str(exc))

    @staticmethod
    def _resolve_project_path(project_path: str) -> str:
        try:
            resolved = Path(project_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Project path does not exist: {project_path}") from exc
        if not resolved.is_dir():
            raise ValueError(f"Project path is not a directory: {project_path}")
        return str(resolved)

    @staticmethod
    def _to_model(record: dict[str, Any]) -> AnalysisRun:
        normalized = {
            **record,
            "request": AnalysisRequest.from_persisted(record["request"]),
        }
        return AnalysisRun.model_validate(normalized)


def _synthesize(findings: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> str:
    if not findings:
        gaps = coverage[-1].get("remaining_gaps", []) if coverage else []
        suffix = f" Remaining gaps: {', '.join(gaps)}." if gaps else ""
        return "No evidence-backed findings were accepted." + suffix
    lines = [f"{len(findings)} evidence-backed finding(s) accepted:"]
    for finding in findings:
        lines.append(f"- [{finding['severity'].upper()}] {finding['title']}: {finding['claim']}")
    return "\n".join(lines)
