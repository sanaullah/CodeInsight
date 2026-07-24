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

    @classmethod
    def from_environment(cls) -> ApiSettings:
        return cls(
            max_concurrent_analyses=_positive_int(
                "CODEINSIGHT_MAX_CONCURRENT_ANALYSES", 2
            ),
            event_history_limit=_positive_int("CODEINSIGHT_EVENT_HISTORY_LIMIT", 200),
            database_path=default_database_path(),
        )

