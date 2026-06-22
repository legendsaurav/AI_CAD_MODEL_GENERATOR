"""
Structured Logging Utility for Geometry Engine
================================================
Provides JSON-structured logging with colored console output,
context fields, and file handlers.
"""
import logging
import json
import sys
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines for structured log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, "context"):
            log_entry["context"] = record.context
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Console formatter with ANSI colors and timestamps."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = datetime.now().strftime("%H:%M:%S")
        return f"{color}[{ts}] {record.levelname:8s}{self.RESET} {record.name}: {record.getMessage()}"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    json_output: bool = False,
) -> logging.Logger:
    """
    Create and configure a structured logger.

    Args:
        name: Logger name (e.g., 'geometry_engine.heads.part').
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_dir: Directory for log files. If None, file logging is disabled.
        json_output: If True, console output is JSON-structured.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    if json_output:
        console.setFormatter(JSONFormatter())
    else:
        console.setFormatter(ColoredFormatter())
    logger.addHandler(console)

    # File handler (JSON lines)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        fh = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{date_str}.jsonl"),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(JSONFormatter())
        logger.addHandler(fh)

    return logger


class LogContext:
    """Context manager that adds fields to all log records within scope."""

    def __init__(self, logger: logging.Logger, **fields: Any):
        self.logger = logger
        self.fields = fields
        self._old_factory = None

    def __enter__(self):
        old = logging.getLogRecordFactory()
        self._old_factory = old
        fields = self.fields

        def factory(*args, **kwargs):
            record = old(*args, **kwargs)
            record.context = fields  # type: ignore[attr-defined]
            return record

        logging.setLogRecordFactory(factory)
        return self

    def __exit__(self, *exc):
        if self._old_factory:
            logging.setLogRecordFactory(self._old_factory)
