"""Environment-backed settings for the FastAPI application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

VERSION_STRING = "v0.1.0-alpha"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment(env_file: str | Path | None = None) -> bool:
    """Load a local .env without overriding process environment values."""

    path = Path(env_file) if env_file else Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return False
    return bool(load_dotenv(path, override=False))


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _optional_url(name: str) -> str | None:
    value = os.getenv(name)
    return value.rstrip("/") if value and value.strip() else None


def _optional_text(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _provider_capability_profile() -> str:
    value = os.getenv("CODEINSIGHT_PROVIDER_CAPABILITY_PROFILE", "direct").strip()
    allowed = {
        "direct",
        "instructor-json",
        "instructor-json-schema",
        "instructor-tools",
    }
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(
            "CODEINSIGHT_PROVIDER_CAPABILITY_PROFILE must be one of: "
            f"{choices}"
        )
    return value


def _agent_runtime() -> str:
    value = os.getenv("CODEINSIGHT_AGENT_RUNTIME", "pydantic-ai").strip()
    allowed = {"pydantic-ai", "instructor", "direct"}
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"CODEINSIGHT_AGENT_RUNTIME must be one of: {choices}")
    return value


def default_data_directory() -> Path:
    configured = os.getenv("CODEINSIGHT_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / ".codeinsight"


def default_database_path() -> Path:
    """Return the one application ledger path."""

    configured = os.getenv("CODEINSIGHT_DATABASE_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return default_data_directory() / "codeinsight.db"


@dataclass(frozen=True)
class ApiSettings:
    """Settings that affect only the API process."""

    max_concurrent_analyses: int = 2
    event_history_limit: int = 200
    database_path: Path = default_database_path()
    model_base_url: str | None = None
    model_api_key: str = ""
    default_model: str = "local-model"
    provider_capability_profile: str = "direct"
    agent_runtime: str = "pydantic-ai"
    max_concurrent_model_calls: int = 2
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langfuse_capture_prompts: bool = False
    langfuse_capture_completions: bool = False
    build_commit: str | None = None
    build_time: str | None = None

    @classmethod
    def from_environment(cls) -> ApiSettings:
        return cls(
            max_concurrent_analyses=_positive_int(
                "CODEINSIGHT_MAX_CONCURRENT_ANALYSES", 2
            ),
            event_history_limit=_positive_int("CODEINSIGHT_EVENT_HISTORY_LIMIT", 200),
            database_path=default_database_path(),
            model_base_url=(
                _optional_url("CODEINSIGHT_MODEL_BASE_URL")
                or _optional_url("OPENAI_API_BASE")
            ),
            model_api_key=os.getenv(
                "CODEINSIGHT_MODEL_API_KEY", os.getenv("OPENAI_API_KEY", "")
            ),
            default_model=os.getenv(
                "CODEINSIGHT_DEFAULT_MODEL", os.getenv("DEFAULT_MODEL", "local-model")
            ),
            provider_capability_profile=_provider_capability_profile(),
            agent_runtime=_agent_runtime(),
            max_concurrent_model_calls=_positive_int(
                "CODEINSIGHT_MAX_CONCURRENT_MODEL_CALLS", 2
            ),
            langfuse_enabled=_enabled("CODEINSIGHT_LANGFUSE_ENABLED"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            langfuse_capture_prompts=_enabled(
                "CODEINSIGHT_LANGFUSE_CAPTURE_PROMPTS"
            ),
            langfuse_capture_completions=_enabled(
                "CODEINSIGHT_LANGFUSE_CAPTURE_COMPLETIONS"
            ),
            build_commit=_optional_text("CODEINSIGHT_BUILD_COMMIT"),
            build_time=_optional_text("CODEINSIGHT_BUILD_TIME"),
        )

