"""FastAPI entry point and integrated frontend for CodeInsight."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from application.analysis_service import AnalysisService
from api.config import ApiSettings
from api.models import (
    AnalysisAccepted,
    AnalysisRequest,
    AnalysisRun,
    CapabilitiesResponse,
    HealthResponse,
    LanguageCapability,
)
from infrastructure.utils.config.env_loader import load_env
from infrastructure.utils.version import VERSION_STRING

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
    load_env()
    settings = ApiSettings.from_environment()
    analysis_service = service or AnalysisService(
        max_concurrent=settings.max_concurrent_analyses,
        event_history_limit=settings.event_history_limit,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
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
        )

    @app.get("/api/v1/capabilities", response_model=CapabilitiesResponse)
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(languages=_language_capabilities())

    @app.post(
        "/api/v1/analyses",
        response_model=AnalysisAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_analysis(
        payload: AnalysisRequest, request: Request
    ) -> AnalysisAccepted:
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

    @app.delete("/api/v1/analyses/{run_id}", response_model=AnalysisRun)
    async def cancel_analysis(run_id: str, request: Request) -> AnalysisRun:
        current_service: AnalysisService = request.app.state.analysis_service
        run = await current_service.cancel(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        return run

    return app


app = create_app()

