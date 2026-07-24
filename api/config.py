"""Environment-backed settings for the FastAPI application."""

from __future__ import annotations

import os
from dataclasses import dataclass


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


@dataclass(frozen=True)
class ApiSettings:
    """Settings that affect only the API process."""

    max_concurrent_analyses: int = 2
    event_history_limit: int = 200

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        return cls(
            max_concurrent_analyses=_positive_int(
                "CODEINSIGHT_MAX_CONCURRENT_ANALYSES", 2
            ),
            event_history_limit=_positive_int("CODEINSIGHT_EVENT_HISTORY_LIMIT", 200),
        )

