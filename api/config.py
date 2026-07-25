"""Environment-backed settings for the FastAPI application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def default_data_directory() -> Path:
    configured = os.getenv("CODEINSIGHT_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt" and os.getenv("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "CodeInsight"
    return Path.home() / ".local" / "share" / "codeinsight"


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
    max_concurrent_model_calls: int = 2
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    langfuse_capture_content: bool = False

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
            max_concurrent_model_calls=_positive_int(
                "CODEINSIGHT_MAX_CONCURRENT_MODEL_CALLS", 2
            ),
            langfuse_enabled=_enabled("CODEINSIGHT_LANGFUSE_ENABLED"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            langfuse_capture_content=_enabled("CODEINSIGHT_LANGFUSE_CAPTURE_CONTENT"),
        )

