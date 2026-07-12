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


class ExperimentLogger:
    """Filesystem-backed logger for a single pipeline experiment run.

    Creates a timestamped experiment directory (with a ``plots`` subdir) plus
    a shared log directory tree, and provides helpers to persist configs,
    metrics, and GGL artifacts produced during a run.
    """

    def __init__(
        self,
        base_experiments_dir: str = "experiments",
        log_base_dir: str = "logs",
        experiment_name: Optional[str] = None,
    ) -> None:
        self._log_base_dir = log_base_dir

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = experiment_name or f"exp_{stamp}"
        self._exp_dir = os.path.join(base_experiments_dir, name)

        os.makedirs(self._exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self._exp_dir, "plots"), exist_ok=True)
        os.makedirs(log_base_dir, exist_ok=True)
        # Pre-create the default feature log directory used by the probing stage.
        os.makedirs(os.path.join(log_base_dir, "features"), exist_ok=True)

    def get_exp_dir(self) -> str:
        """Return the root directory for this experiment's artifacts."""
        return self._exp_dir

    def get_log_dir(self, name: str) -> str:
        """Return (creating if needed) a named subdirectory under the log root."""
        path = os.path.join(self._log_base_dir, name)
        os.makedirs(path, exist_ok=True)
        return path

    def log_config(self, config: Dict[str, Any], filename: str = "config.json") -> str:
        """Persist the run configuration as JSON in the experiment directory."""
        return self._write_json(config, filename)

    def log_metrics(
        self, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> str:
        """Append a metrics record (optionally tagged with a step) to metrics.jsonl."""
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "metrics": metrics,
        }
        path = os.path.join(self._exp_dir, "metrics.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return path

    def save_ggl(self, ggl_data: Dict[str, Any], filename: str = "ggl.json") -> str:
        """Persist a serialized GGL document as JSON in the experiment directory."""
        return self._write_json(ggl_data, filename)

    def _write_json(self, data: Dict[str, Any], filename: str) -> str:
        path = os.path.join(self._exp_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path


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
