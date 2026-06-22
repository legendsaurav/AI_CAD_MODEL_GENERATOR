"""
app/session.py — Session management for pipeline orchestration.

Manages lifecycle of pipeline sessions: creation, state tracking, event recording.
In-memory store for V1; swap to database-backed store for production.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class PipelineStage(str, Enum):
    """Ordered stages of the AI CAD OS pipeline."""
    PENDING = "pending"
    MODEL_GENERATOR = "model_generator_v2"
    GEOMETRY_ENGINE = "geometry_engine"
    GGL_OUTPUT = "ggl_output"
    CAD_PLANNER = "cad_planner"
    CAL_OUTPUT = "cal_output"
    DESKTOP_AGENT = "desktop_agent"
    CAD_EXECUTION = "cad_execution"
    VERIFICATION = "verification"
    REFINEMENT = "refinement"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStatus(str, Enum):
    """High-level status of a pipeline session."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineEvent(BaseModel):
    """An immutable event emitted during pipeline execution."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage: str
    event_type: str  # stage_started, stage_completed, stage_failed, retry, cancelled
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class SessionState(BaseModel):
    """Full state of a pipeline session."""
    session_id: str
    image_path: str
    current_stage: PipelineStage = PipelineStage.PENDING
    status: PipelineStatus = PipelineStatus.QUEUED
    events: List[PipelineEvent] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    stage_outputs: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    refinement_iteration: int = 0


class SessionNotFoundError(Exception):
    """Raised when a session_id does not exist in the store."""
    pass


class SessionManager:
    """
    In-memory session store (V1).

    NOTE: Replace with Redis / PostgreSQL-backed store for production
    deployments requiring persistence across restarts.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._log = structlog.get_logger(component="SessionManager")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self, image_path: str) -> str:
        """Create a new pipeline session, returning the session_id (UUID4)."""
        session_id = str(uuid.uuid4())
        session = SessionState(session_id=session_id, image_path=image_path)
        self._sessions[session_id] = session
        self._log.info(
            "session_created",
            session_id=session_id,
            image_path=image_path,
        )
        return session_id

    def get_session(self, session_id: str) -> SessionState:
        """Retrieve session state by id. Raises SessionNotFoundError if missing."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found")
        return session

    def update_session(
        self,
        session_id: str,
        stage: PipelineStage,
        status: PipelineStatus,
        *,
        error_message: Optional[str] = None,
        stage_output: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """Advance session to a new stage/status and record the transition."""
        session = self.get_session(session_id)
        session.current_stage = stage
        session.status = status
        session.updated_at = datetime.now(timezone.utc).isoformat()
        if error_message is not None:
            session.error_message = error_message
        if stage_output is not None:
            session.stage_outputs[stage.value] = stage_output
        self._log.info(
            "session_updated",
            session_id=session_id,
            stage=stage.value,
            status=status.value,
        )
        return session

    def add_event(self, session_id: str, event: PipelineEvent) -> None:
        """Append an event to the session's event log."""
        session = self.get_session(session_id)
        session.events.append(event)
        session.updated_at = datetime.now(timezone.utc).isoformat()

    def list_sessions(self) -> List[SessionState]:
        """Return all sessions (for metrics / admin)."""
        return list(self._sessions.values())

    @property
    def active_count(self) -> int:
        return sum(
            1 for s in self._sessions.values()
            if s.status in (PipelineStatus.QUEUED, PipelineStatus.RUNNING)
        )

    @property
    def completed_count(self) -> int:
        return sum(
            1 for s in self._sessions.values()
            if s.status == PipelineStatus.COMPLETED
        )

    @property
    def failed_count(self) -> int:
        return sum(
            1 for s in self._sessions.values()
            if s.status == PipelineStatus.FAILED
        )
