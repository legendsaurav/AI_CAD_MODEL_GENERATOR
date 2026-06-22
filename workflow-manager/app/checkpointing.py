"""
app/checkpointing.py — Checkpoint management for pipeline state persistence.

Saves and loads per-session, per-stage checkpoints as JSON files on disk so
a pipeline can be resumed after a crash or restart.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class Checkpoint(BaseModel):
    """A single checkpoint snapshot."""
    session_id: str
    stage: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: Dict[str, Any] = Field(default_factory=dict)


class CheckpointManager:
    """
    Manages checkpoint files on local disk.

    Layout:
        <checkpoint_dir>/<session_id>/<stage>_<timestamp>.json

    The directory is configurable via the CHECKPOINT_DIR environment variable
    (defaults to ``./checkpoints``).
    """

    def __init__(self, checkpoint_dir: Optional[str] = None) -> None:
        self._dir = Path(
            checkpoint_dir
            or os.environ.get("CHECKPOINT_DIR", "./checkpoints")
        )
        self._log = structlog.get_logger(component="CheckpointManager")
        self._ensure_dir(self._dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        session_id: str,
        stage: str,
        data: Dict[str, Any],
    ) -> Path:
        """Persist a checkpoint and return the file path."""
        session_dir = self._dir / session_id
        self._ensure_dir(session_dir)

        checkpoint = Checkpoint(session_id=session_id, stage=stage, data=data)
        # Use a slug-safe timestamp for the filename
        ts_slug = checkpoint.timestamp.replace(":", "-").replace("+", "_")
        filename = f"{stage}_{ts_slug}.json"
        filepath = session_dir / filename

        filepath.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
        self._log.info(
            "checkpoint_saved",
            session_id=session_id,
            stage=stage,
            path=str(filepath),
        )
        return filepath

    def load_checkpoint(self, session_id: str) -> Optional[Checkpoint]:
        """Load the most recent checkpoint for a session (by file mtime)."""
        session_dir = self._dir / session_id
        if not session_dir.exists():
            return None

        files = sorted(session_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        if not files:
            return None

        raw = files[0].read_text(encoding="utf-8")
        checkpoint = Checkpoint.model_validate_json(raw)
        self._log.info(
            "checkpoint_loaded",
            session_id=session_id,
            stage=checkpoint.stage,
            path=str(files[0]),
        )
        return checkpoint

    def list_checkpoints(self, session_id: str) -> List[Checkpoint]:
        """Return all checkpoints for a session, oldest first."""
        session_dir = self._dir / session_id
        if not session_dir.exists():
            return []

        checkpoints: List[Checkpoint] = []
        for fp in sorted(session_dir.glob("*.json"), key=os.path.getmtime):
            raw = fp.read_text(encoding="utf-8")
            checkpoints.append(Checkpoint.model_validate_json(raw))
        return checkpoints

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
