"""Optional best-effort Langfuse exporter behind the tracing interface."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore, Lock
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
        capture_prompts: bool = False,
        capture_completions: bool = False,
        max_pending_events: int = 256,
    ) -> None:
        if max_pending_events < 1:
            raise ValueError("max_pending_events must be greater than zero")
        self.capture_prompts = capture_prompts
        self.capture_completions = capture_completions
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="langfuse")
        self._slots = BoundedSemaphore(max_pending_events)
        self._lock = Lock()
        self._closed = False
        self._dropped_events = 0
        self._client: Any | None = None
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=host,
            )
        except Exception as exc:  # optional dependency/configuration
            logger.info(
                "Langfuse tracing disabled: category=initialization class=%s",
                type(exc).__name__,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def dropped_events(self) -> int:
        return self._dropped_events

    def emit(self, event: TraceEvent) -> None:
        with self._lock:
            if self._client is None or self._closed:
                return
            if not self._slots.acquire(blocking=False):
                self._dropped_events += 1
                return
            safe_attributes = _redact(
                event.attributes,
                capture_prompts=self.capture_prompts,
                capture_completions=self.capture_completions,
            )
            try:
                future = self._pool.submit(
                    self._emit_safely, event, safe_attributes
                )
            except RuntimeError:
                self._slots.release()
                self._dropped_events += 1
                return
            future.add_done_callback(lambda _future: self._slots.release())

    def _emit_safely(
        self, event: TraceEvent, safe_attributes: dict[str, Any]
    ) -> None:
        try:
            self._client.create_event(
                name=event.name,
                trace_context={"trace_id": _trace_id(event.run_id)},
                level=_event_level(event.name),
                metadata={
                    "run_id": event.run_id,
                    "stage": event.stage,
                    "wave_id": event.wave_id,
                    "role_id": event.role_id,
                    "task_id": event.task_id,
                    "attempt_id": event.attempt_id,
                    "attempt_number": event.attempt_number,
                    "model_call_id": event.model_call_id,
                    "prompt_artifact_id": event.prompt_artifact_id,
                    **safe_attributes,
                },
            )
        except Exception as exc:
            logger.debug(
                "Langfuse export failed: category=send class=%s",
                type(exc).__name__,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._client is not None:
                try:
                    self._pool.submit(self._flush_safely)
                except RuntimeError:
                    pass
            self._pool.shutdown(wait=False, cancel_futures=False)

    def _flush_safely(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug(
                "Langfuse flush failed: category=flush class=%s",
                type(exc).__name__,
            )


def _redact(
    attributes: dict[str, Any],
    capture_prompts: bool = False,
    capture_completions: bool = False,
) -> dict[str, Any]:
    return {
        str(key): _redact_value(
            str(key),
            value,
            capture_prompts=capture_prompts,
            capture_completions=capture_completions,
            depth=0,
        )
        for key, value in attributes.items()
    }


def _redact_value(
    key: str,
    value: Any,
    *,
    capture_prompts: bool,
    capture_completions: bool,
    depth: int,
) -> Any:
    lowered = key.lower()
    if _secret_key(lowered):
        return "[redacted]"
    if any(
        marker in lowered
        for marker in (
            "source_content",
            "file_content",
            "source_body",
            "tool_output",
            "user_prompt",
        )
    ):
        return "[omitted]"
    if "path" in lowered or lowered == "project":
        return "[omitted]"
    if "prompt" in lowered and not capture_prompts:
        return "[omitted]"
    if "completion" in lowered and not capture_completions:
        return "[omitted]"
    if depth >= 5:
        return "[truncated]"
    if isinstance(value, dict):
        items = list(value.items())[:100]
        return {
            str(child_key): _redact_value(
                str(child_key),
                child_value,
                capture_prompts=capture_prompts,
                capture_completions=capture_completions,
                depth=depth + 1,
            )
            for child_key, child_value in items
        }
    if isinstance(value, (list, tuple)):
        return [
            _redact_value(
                key,
                item,
                capture_prompts=capture_prompts,
                capture_completions=capture_completions,
                depth=depth + 1,
            )
            for item in value[:100]
        ]
    if isinstance(value, str) and len(value) > 16_384:
        return value[:16_384] + "[truncated]"
    return value


def _secret_key(lowered: str) -> bool:
    if any(
        marker in lowered
        for marker in (
            "password",
            "secret",
            "authorization",
            "credential",
            "cookie",
            "api_key",
            "private_key",
            "access_token",
            "refresh_token",
            "auth_token",
            "api_token",
        )
    ):
        return True
    return lowered in {"key", "token"}


def _trace_id(run_id: str) -> str:
    return hashlib.sha256(f"codeinsight:run:{run_id}".encode()).hexdigest()[:32]


def _event_level(name: str) -> str:
    lowered = name.lower()
    if any(marker in lowered for marker in ("failed", "cancelled", "interrupted")):
        return "WARNING"
    return "DEFAULT"
