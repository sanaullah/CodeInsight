"""FastAPI entry point and integrated frontend for CodeInsight."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from api.config import VERSION_STRING, ApiSettings, load_environment
from api.models import (
    AnalysisAccepted,
    AnalysisIntelligence,
    AnalysisRequest,
    AnalysisRun,
    CapabilitiesResponse,
    FindingPage,
    FindingReviewUpdate,
    HealthResponse,
    LanguageCapability,
    RecoveryResponse,
    RuntimeIdentity,
)
from application.analysis_service import AnalysisService
from infrastructure.db.database import SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_ROOT = PROJECT_ROOT / "web"


def _language_capabilities() -> list[LanguageCapability]:
    from indexing.scanners.language_config import (
        LANGUAGE_EXTENSIONS,
        Language,
        get_language_metadata,
    )

    dependency_aware = {Language.PYTHON, Language.JAVASCRIPT, Language.TYPESCRIPT}
    parsed = {Language.PYTHON}
    capabilities: list[LanguageCapability] = []
    for language in Language:
        metadata = get_language_metadata(language.value)
        level = (
            "parsed"
            if language in parsed
            else "dependency-aware"
            if language in dependency_aware
            else "discovery"
        )
        capabilities.append(
            LanguageCapability(
                language=language.value,
                display_name=metadata.name if metadata else language.value.title(),
                support_level=level,
                extensions=LANGUAGE_EXTENSIONS.get(language, []),
            )
        )
    return capabilities


def create_app(service: AnalysisService | None = None) -> FastAPI:
    load_environment()
    settings = ApiSettings.from_environment()
    analysis_service = service or AnalysisService(
        database_path=settings.database_path,
        max_concurrent=settings.max_concurrent_analyses,
        event_history_limit=settings.event_history_limit,
        settings=settings,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await analysis_service.start()
        yield
        await analysis_service.close()

    app = FastAPI(
        title="CodeInsight API",
        version=VERSION_STRING.removeprefix("v"),
        description="Stable HTTP boundary for CodeInsight analysis runs.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.state.analysis_service = analysis_service
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_ROOT / "assets"),
        name="frontend-assets",
    )

    @app.get("/", include_in_schema=False)
    async def frontend() -> FileResponse:
        return FileResponse(FRONTEND_ROOT / "index.html")

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        current_service: AnalysisService = request.app.state.analysis_service
        return HealthResponse(
            status="ok",
            version=VERSION_STRING,
            active_analyses=current_service.active_count,
            max_concurrent_analyses=current_service.max_concurrent,
            runtime=RuntimeIdentity(
                database_schema_version=SCHEMA_VERSION,
                build_commit=current_service.settings.build_commit,
                build_time=current_service.settings.build_time,
            ),
        )

    @app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            languages=_language_capabilities(),
            model_provider_configured=analysis_service.provider_configured,
            langfuse_enabled=analysis_service.langfuse_enabled,
        )

    @app.post(
        "/api/v1/analyses",
        response_model=AnalysisAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_analysis(payload: AnalysisRequest, request: Request) -> AnalysisAccepted:
        current_service: AnalysisService = request.app.state.analysis_service
        try:
            run = await current_service.submit(payload)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return AnalysisAccepted(
            run_id=run.run_id,
            status=run.status,
            status_url=f"/api/v1/analyses/{run.run_id}",
        )

    @app.get("/api/v1/analyses", response_model=list[AnalysisRun])
    async def list_analyses(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> list[AnalysisRun]:
        current_service: AnalysisService = request.app.state.analysis_service
        return await current_service.list(limit)

    @app.get("/api/v1/analyses/{run_id}", response_model=AnalysisRun)
    async def get_analysis(run_id: str, request: Request) -> AnalysisRun:
        current_service: AnalysisService = request.app.state.analysis_service
        run = await current_service.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        return run

    @app.get(
        "/api/v1/analyses/{run_id}/intelligence",
        response_model=AnalysisIntelligence,
    )
    async def get_analysis_intelligence(run_id: str, request: Request) -> AnalysisIntelligence:
        current_service: AnalysisService = request.app.state.analysis_service
        intelligence = await current_service.intelligence(run_id)
        if intelligence is None:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        return intelligence

    @app.delete("/api/v1/analyses/{run_id}", response_model=AnalysisRun)
    async def cancel_analysis(run_id: str, request: Request) -> AnalysisRun:
        current_service: AnalysisService = request.app.state.analysis_service
        run = await current_service.cancel(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        return run

    @app.get("/api/v1/findings", response_model=FindingPage)
    async def list_findings(
        request: Request,
        run_id: str | None = None,
        search: str | None = Query(default=None, max_length=200),
        severity: str | None = Query(default=None, pattern="^(info|low|medium|high|critical)$"),
        review_state: str | None = Query(
            default=None,
            pattern="^(new|validated|acknowledged|reviewed|dismissed|reopened|resolved)$",
        ),
        min_confidence: float | None = Query(default=None, ge=0, le=1),
        affected_path: str | None = Query(default=None, max_length=1000),
        cursor: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> FindingPage:
        current_service: AnalysisService = request.app.state.analysis_service
        result = await current_service.query_findings(
            run_id=run_id,
            search=search,
            severity=severity,
            review_state=review_state,
            min_confidence=min_confidence,
            affected_path=affected_path,
            cursor=cursor,
            limit=limit,
        )
        return FindingPage.model_validate(result)

    @app.get("/api/v1/findings/{finding_id}")
    async def get_finding(finding_id: str, request: Request) -> dict:
        detail = await request.app.state.analysis_service.finding_detail(finding_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Finding not found")
        return detail

    @app.put("/api/v1/findings/{finding_id}/review")
    async def update_finding_review(
        finding_id: str, payload: FindingReviewUpdate, request: Request
    ) -> dict:
        try:
            detail = await request.app.state.analysis_service.update_finding_review(
                finding_id,
                review_state=payload.review_state,
                note=payload.note,
                expected_version=payload.expected_version,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if detail is None:
            raise HTTPException(status_code=404, detail="Finding not found")
        return detail

    @app.get("/api/v1/findings-export")
    async def export_findings(
        request: Request,
        format: str = Query(default="json", pattern="^(json|csv)$"),
        run_id: str | None = None,
    ):
        items: list[dict] = []
        cursor: str | None = None
        while True:
            result = await request.app.state.analysis_service.query_findings(
                run_id=run_id, cursor=cursor, limit=200
            )
            items.extend(result["items"])
            cursor = result["next_cursor"]
            if cursor is None:
                break
        headers = {"Content-Disposition": f'attachment; filename="codeinsight-findings.{format}"'}
        if format == "json":
            return JSONResponse({"findings": items}, headers=headers)
        import csv
        import io

        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "finding_id",
                "run_id",
                "severity",
                "confidence",
                "review_state",
                "title",
                "claim",
                "recommendation",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(items)
        return PlainTextResponse(output.getvalue(), media_type="text/csv", headers=headers)

    @app.get("/api/v1/snapshots")
    async def list_snapshots(
        request: Request, limit: int = Query(default=20, ge=1, le=100)
    ) -> list[dict]:
        return await request.app.state.analysis_service.list_snapshots(limit)

    @app.get("/api/v1/snapshots/{snapshot_id}/architecture")
    async def architecture_graph(
        snapshot_id: str,
        request: Request,
        focus: str | None = Query(default=None, max_length=1000),
        depth: int = Query(default=1, ge=0, le=4),
        limit: int = Query(default=250, ge=1, le=1000),
        edge_kind: str | None = Query(default=None, pattern="^(imports|calls)$"),
    ) -> dict:
        graph = await request.app.state.analysis_service.architecture_graph(
            snapshot_id,
            focus=focus,
            depth=depth,
            limit=limit,
            edge_kind=edge_kind,
        )
        if graph is None:
            raise HTTPException(status_code=404, detail="Repository snapshot not found")
        return graph

    @app.get("/api/v1/snapshots/{snapshot_id}/trace")
    async def architecture_trace(
        snapshot_id: str,
        request: Request,
        source_id: str,
        target_id: str,
        max_hops: int = Query(default=8, ge=1, le=20),
    ) -> dict:
        trace = await request.app.state.analysis_service.architecture_trace(
            snapshot_id,
            source_id=source_id,
            target_id=target_id,
            max_hops=max_hops,
        )
        if trace is None:
            raise HTTPException(status_code=404, detail="Repository snapshot not found")
        return trace

    @app.get("/api/v1/history")
    async def analysis_history(
        request: Request,
        search: str | None = Query(default=None, max_length=200),
        status_filter: str | None = Query(
            default=None,
            alias="status",
            pattern="^(queued|running|succeeded|failed|cancelled|needs_attention)$",
        ),
        mode: str | None = Query(default=None, pattern="^(quick|deep|security|change-set)$"),
        cursor: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        return await request.app.state.analysis_service.query_history(
            search=search,
            status=status_filter,
            mode=mode,
            cursor=cursor,
            limit=limit,
        )

    @app.get("/api/v1/history/compare")
    async def compare_history(request: Request, baseline_run_id: str, target_run_id: str) -> dict:
        comparison = await request.app.state.analysis_service.compare_runs(
            baseline_run_id, target_run_id
        )
        if comparison is None:
            raise HTTPException(status_code=404, detail="Comparison run not found")
        return comparison

    @app.get("/api/v1/history/trends")
    async def history_trends(
        request: Request, days: int = Query(default=30, ge=1, le=3650)
    ) -> dict:
        return await request.app.state.analysis_service.history_trends(days)

    @app.post("/api/v1/recovery", response_model=RecoveryResponse)
    async def recover_analysis_work(request: Request) -> RecoveryResponse:
        current_service: AnalysisService = request.app.state.analysis_service
        return await current_service.recover()

    @app.get("/{frontend_path:path}", include_in_schema=False)
    async def frontend_route(frontend_path: str) -> FileResponse:
        if frontend_path == "api" or frontend_path.startswith(("api/", "assets/")):
            raise HTTPException(status_code=404, detail="Resource not found")
        return FileResponse(FRONTEND_ROOT / "index.html")

    return app


app = create_app()
