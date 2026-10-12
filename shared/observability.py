"""Small, dependency-light observability helpers shared by HTTP services."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request


_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_service_name: contextvars.ContextVar[str] = contextvars.ContextVar(
    "service_name", default="unknown"
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "service": getattr(record, "service", None) or _service_name.get(),
            "request_id": getattr(record, "request_id", None) or _request_id.get(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(service: str) -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    _service_name.set(service)
    return logging.getLogger(service)


def current_request_id() -> str | None:
    return _request_id.get()


def request_headers() -> dict[str, str]:
    request_id = current_request_id()
    return {"X-Request-ID": request_id} if request_id else {}


def _database_logging_enabled() -> bool:
    return os.getenv("DATABASE_EVENT_LOGGING", "true").lower() in {
        "1", "true", "yes", "on"
    }


def record_event(
    *,
    service: str,
    event_type: str,
    severity: str = "error",
    session_id: str | None = None,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> None:
    """Persist actionable events; never let telemetry break a user request."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url or not _database_logging_enabled():
        return
    try:
        import psycopg2

        with psycopg2.connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO operational_events (
                        service, request_id, severity, event_type, session_id, details
                    ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        service,
                        request_id or current_request_id(),
                        severity,
                        event_type,
                        session_id,
                        json.dumps(details or {}, default=str),
                    ),
                )
    except Exception:
        logging.getLogger(__name__).exception(
            "Could not persist operational event",
            extra={"event": {"event_type": "observability.persistence_failed"}},
        )


def install_http_observability(app: FastAPI, service: str) -> logging.Logger:
    logger = configure_logging(service)

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        supplied = request.headers.get("x-request-id", "").strip()
        request_id = supplied[:128] if supplied else uuid.uuid4().hex
        request_token = _request_id.set(request_id)
        service_token = _service_name.set(service)
        started = time.perf_counter()
        status_code = 500
        exception_recorded = False
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            details = {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
                "exception_type": type(exc).__name__,
                "error": str(exc)[:4000],
            }
            logger.exception(
                "Unhandled request failure",
                extra={"event": {"event_type": "http.request.exception", **details}},
            )
            record_event(
                service=service,
                event_type="http.request.exception",
                severity="error",
                details=details,
                request_id=request_id,
            )
            exception_recorded = True
            raise
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            event = {
                "event_type": "http.request.completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
            }
            logger.log(
                logging.ERROR if status_code >= 500 else logging.WARNING if status_code >= 400 else logging.INFO,
                "HTTP request completed",
                extra={"event": event},
            )
            if status_code >= 500 and not exception_recorded:
                record_event(
                    service=service,
                    event_type="http.request.failed",
                    severity="error",
                    details=event,
                    request_id=request_id,
                )
            _request_id.reset(request_token)
            _service_name.reset(service_token)

    return logger
