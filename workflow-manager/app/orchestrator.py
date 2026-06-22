"""
app/orchestrator.py — Core pipeline orchestration logic.

Sequences through all pipeline stages, calling the respective service
for each stage. Implements retry logic with exponential backoff and
event emission at each stage transition.

ARCHITECTURE RULE: The orchestrator performs orchestration ONLY.
It must NEVER perform geometry reasoning or CAD planning.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, Optional

import httpx
import structlog

from app.session import (
    PipelineEvent,
    PipelineStage,
    PipelineStatus,
    SessionManager,
)
from app.checkpointing import CheckpointManager

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration: service URLs (override via environment variables)
# ---------------------------------------------------------------------------

_SERVICE_URLS = {
    PipelineStage.MODEL_GENERATOR: os.environ.get(
        "MODEL_GENERATOR_URL", "http://localhost:8000"
    ),
    PipelineStage.GEOMETRY_ENGINE: os.environ.get(
        "GEOMETRY_ENGINE_URL", "http://localhost:8001"
    ),
    PipelineStage.CAD_PLANNER: os.environ.get(
        "CAD_PLANNER_URL", "http://localhost:8002"
    ),
    PipelineStage.DESKTOP_AGENT: os.environ.get(
        "DESKTOP_AGENT_URL", "http://localhost:8003"
    ),
}

# Pipeline stage ordering (linear for V1)
_STAGE_ORDER = [
    PipelineStage.MODEL_GENERATOR,
    PipelineStage.GEOMETRY_ENGINE,
    PipelineStage.GGL_OUTPUT,
    PipelineStage.CAD_PLANNER,
    PipelineStage.CAL_OUTPUT,
    PipelineStage.DESKTOP_AGENT,
    PipelineStage.CAD_EXECUTION,
    PipelineStage.VERIFICATION,
]

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
BACKOFF_MULTIPLIER = 2.0
REQUEST_TIMEOUT_S = 300.0


class PipelineOrchestrator:
    """
    Orchestrates the AI CAD OS pipeline from image to verified CAD model.

    Responsibilities:
    - Session management via SessionManager
    - Sequential stage execution with retry + exponential backoff
    - Checkpointing after each successful stage
    - Event emission for observability
    - Error recovery and graceful failure

    Non-responsibilities (MUST NOT):
    - Geometry reasoning
    - CAD planning
    - Feature extraction
    """

    def __init__(
        self,
        session_mgr: SessionManager,
        checkpoint_mgr: CheckpointManager,
    ) -> None:
        self._sessions = session_mgr
        self._checkpoints = checkpoint_mgr
        self._log = structlog.get_logger(component="PipelineOrchestrator")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_pipeline(self, session_id: str) -> Dict[str, Any]:
        """
        Execute the full pipeline for a session.

        Returns:
            Dict with final status and output paths.
        """
        self._emit_event(session_id, PipelineStage.PENDING, "pipeline_started",
                         "Pipeline execution started")
        self._sessions.update_session(
            session_id,
            stage=PipelineStage.PENDING,
            status=PipelineStatus.RUNNING,
        )

        session = self._sessions.get_session(session_id)
        stage_data: Dict[str, Any] = {"image_path": session.image_path}

        for stage in _STAGE_ORDER:
            self._log.info("stage.starting", session_id=session_id, stage=stage.value)
            self._emit_event(session_id, stage, "stage_started",
                             f"Starting stage: {stage.value}")
            self._sessions.update_session(
                session_id, stage=stage, status=PipelineStatus.RUNNING,
            )

            t0 = time.monotonic()
            try:
                result = await self._execute_stage_with_retry(
                    session_id, stage, stage_data
                )
                duration_ms = (time.monotonic() - t0) * 1000

                stage_data.update(result)
                self._sessions.update_session(
                    session_id, stage=stage, status=PipelineStatus.RUNNING,
                    stage_output=result,
                )
                self._checkpoints.save_checkpoint(
                    session_id, stage.value, result
                )
                self._emit_event(
                    session_id, stage, "stage_completed",
                    f"Completed {stage.value} in {duration_ms:.0f}ms",
                    data={"duration_ms": duration_ms},
                )
                self._log.info(
                    "stage.completed",
                    session_id=session_id,
                    stage=stage.value,
                    duration_ms=round(duration_ms, 1),
                )

            except Exception as exc:
                duration_ms = (time.monotonic() - t0) * 1000
                error_msg = f"Stage {stage.value} failed: {exc}"
                self._log.error(
                    "stage.failed",
                    session_id=session_id,
                    stage=stage.value,
                    error=str(exc),
                )
                self._emit_event(session_id, stage, "stage_failed", error_msg)
                self._sessions.update_session(
                    session_id,
                    stage=PipelineStage.FAILED,
                    status=PipelineStatus.FAILED,
                    error_message=error_msg,
                )
                return {"status": "failed", "error": error_msg, "stage": stage.value}

        # All stages completed
        self._sessions.update_session(
            session_id,
            stage=PipelineStage.COMPLETED,
            status=PipelineStatus.COMPLETED,
        )
        self._emit_event(session_id, PipelineStage.COMPLETED, "pipeline_completed",
                         "Pipeline completed successfully")
        self._log.info("pipeline.completed", session_id=session_id)
        return {"status": "completed", "outputs": stage_data}

    # ------------------------------------------------------------------
    # Retry logic with exponential backoff
    # ------------------------------------------------------------------

    async def _execute_stage_with_retry(
        self,
        session_id: str,
        stage: PipelineStage,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single stage with retry + exponential backoff."""
        backoff = INITIAL_BACKOFF_S

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return await self._execute_stage(stage, input_data)
            except Exception as exc:
                if attempt == MAX_RETRIES:
                    raise
                self._log.warning(
                    "stage.retry",
                    session_id=session_id,
                    stage=stage.value,
                    attempt=attempt,
                    backoff_s=backoff,
                    error=str(exc),
                )
                self._emit_event(
                    session_id, stage, "retry",
                    f"Retrying stage {stage.value} (attempt {attempt}/{MAX_RETRIES})",
                    data={"attempt": attempt, "backoff_s": backoff},
                )
                await asyncio.sleep(backoff)
                backoff *= BACKOFF_MULTIPLIER

        # Should never reach here
        raise RuntimeError(f"Stage {stage.value} failed after {MAX_RETRIES} retries")

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    async def _execute_stage(
        self,
        stage: PipelineStage,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Dispatch a single stage to the appropriate service.

        For stages that are pure data-flow transitions (GGL_OUTPUT, CAL_OUTPUT,
        CAD_EXECUTION), we pass through the data unchanged.
        """
        if stage in (PipelineStage.GGL_OUTPUT, PipelineStage.CAL_OUTPUT,
                     PipelineStage.CAD_EXECUTION):
            # Data-flow passthrough stages
            return {"status": "passthrough", "stage": stage.value}

        if stage == PipelineStage.VERIFICATION:
            # Verification is a local computation (no external service for V1)
            return {"status": "verification_pending", "stage": stage.value}

        service_url = _SERVICE_URLS.get(stage)
        if not service_url:
            return {"status": "skipped", "stage": stage.value, "reason": "no_service_url"}

        # Map stage to the service endpoint
        endpoint_map = {
            PipelineStage.MODEL_GENERATOR: "/extract_features",
            PipelineStage.GEOMETRY_ENGINE: "/generate_graph",
            PipelineStage.CAD_PLANNER: "/plan",
            PipelineStage.DESKTOP_AGENT: "/execute",
        }
        endpoint = endpoint_map.get(stage, "/process")
        url = f"{service_url}{endpoint}"

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            response = await client.post(url, json=input_data)
            response.raise_for_status()
            return response.json()

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        session_id: str,
        stage: PipelineStage,
        event_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an event in the session log."""
        event = PipelineEvent(
            stage=stage.value,
            event_type=event_type,
            message=message,
            data=data or {},
        )
        try:
            self._sessions.add_event(session_id, event)
        except Exception:
            pass  # Don't let event emission break the pipeline
