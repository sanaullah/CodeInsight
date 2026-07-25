"""Optional best-effort Langfuse exporter behind the tracing interface."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from application.tracing import TraceEvent

logger = logging.getLogger(__name__)


class LangfuseTraceExporter:
    """Export metadata asynchronously; failures never affect an analysis."""

    def __init__(
        self,
        *,
        public_key: str,
        secret_key: str,
        host: str,
        capture_content: bool = False,
    ) -> None:
        self.capture_content = capture_content
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="langfuse")
        self._client: Any | None = None
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
        except Exception as exc:  # optional dependency/configuration
            logger.info("Langfuse tracing disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def emit(self, event: TraceEvent) -> None:
        if self._client is None:
            return
        safe_attributes = _redact(event.attributes, self.capture_content)
        self._pool.submit(self._emit_safely, event, safe_attributes)

    def _emit_safely(
        self, event: TraceEvent, safe_attributes: dict[str, Any]
    ) -> None:
        try:
            self._client.create_event(
                name=event.name,
                metadata={
                    "run_id": event.run_id,
                    "wave_id": event.wave_id,
                    "task_id": event.task_id,
                    "model_call_id": event.model_call_id,
                    **safe_attributes,
                },
            )
        except Exception as exc:
            logger.debug("Langfuse export failed without blocking analysis: %s", exc)

    def close(self) -> None:
        if self._client is not None:
            self._pool.submit(self._flush_safely)
        self._pool.shutdown(wait=False, cancel_futures=False)

    def _flush_safely(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush failed: %s", exc)


def _redact(attributes: dict[str, Any], capture_content: bool) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        lowered = key.lower()
        if any(secret in lowered for secret in ("key", "token", "secret", "password")):
            safe[key] = "[redacted]"
        elif not capture_content and any(
            content in lowered for content in ("prompt", "completion", "source", "content")
        ):
            safe[key] = "[omitted]"
        else:
            safe[key] = value
    return safe
