"""
shared_schemas/logging_config.py — Unified Structured Logging Config
=====================================================================
Configures structured JSON logging for every service in the AI CAD OS.

* Uses ``structlog`` when available; falls back to stdlib ``logging``.
* Injects ``correlation_id`` for end-to-end request tracing.
* Writes to both ``stderr`` and a rotating file under *log_dir*.
* **Replaces all ``print()`` debugging** — import and call
  ``setup_structured_logging()`` at service startup.

Usage::

    from shared_schemas.logging_config import setup_structured_logging

    logger = setup_structured_logging("geometry-engine")
    logger.info("pipeline.start", session_id="abc-123")
"""
import logging
import logging.handlers
import os
import sys
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

# Context variable holding the current request / pipeline correlation ID.
_correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> Optional[str]:
    """Return the current correlation ID, or ``None``."""
    return _correlation_id_var.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set (or generate) a correlation ID for the current context.

    Returns:
        The active correlation ID.
    """
    cid = cid or uuid.uuid4().hex
    _correlation_id_var.set(cid)
    return cid


# ---------------------------------------------------------------------------
# structlog-based setup
# ---------------------------------------------------------------------------

def _try_setup_structlog(
    service_name: str,
    log_level: str,
    log_dir: Optional[str],
) -> Any:
    """Attempt to configure ``structlog``.  Returns a bound logger or raises
    ``ImportError``.
    """
    import structlog  # type: ignore[import-untyped]

    # Shared processors applied to every log event.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # --- stdlib handler wiring ------------------------------------------------
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_path / f"{service_name}.log"),
            maxBytes=10 * 1024 * 1024,  # 10 MiB
            backupCount=5,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Attach the JSON renderer to all stdlib handlers via a formatter.
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    logger = structlog.get_logger(service_name)
    return logger.bind(service=service_name)


# ---------------------------------------------------------------------------
# stdlib fallback
# ---------------------------------------------------------------------------

def _setup_stdlib(
    service_name: str,
    log_level: str,
    log_dir: Optional[str],
) -> logging.Logger:
    """Pure-stdlib structured-ish JSON logging."""

    class _JsonFormatter(logging.Formatter):
        """Minimal JSON formatter without external dependencies."""

        def format(self, record: logging.LogRecord) -> str:
            import json as _json

            payload = {
                "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "service": service_name,
            }
            cid = get_correlation_id()
            if cid is not None:
                payload["correlation_id"] = cid
            if record.exc_info and record.exc_info[1] is not None:
                payload["exception"] = self.formatException(record.exc_info)
            return _json.dumps(payload)

    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(_JsonFormatter())
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(log_path / f"{service_name}.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(_JsonFormatter())
        logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_structured_logging(
    service_name: str,
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
) -> Any:
    """Configure and return a structured logger for *service_name*.

    Parameters:
        service_name: Logical name of the service (e.g. ``'geometry-engine'``).
        log_level: Python log level name (``'DEBUG'``, ``'INFO'``, …).
        log_dir: Optional directory for rotating log files.  Created
            automatically if it does not exist.

    Returns:
        A ``structlog.BoundLogger`` if structlog is installed, otherwise a
        stdlib ``logging.Logger``.  Both support ``.info()``, ``.warning()``,
        ``.error()``, ``.debug()`` etc.
    """
    try:
        return _try_setup_structlog(service_name, log_level, log_dir)
    except ImportError:
        return _setup_stdlib(service_name, log_level, log_dir)
