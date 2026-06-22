"""
app/main.py — FastAPI application for the Workflow Manager.

Provides pipeline orchestration endpoints, health checks, and metrics.
The workflow manager performs ORCHESTRATION ONLY — it must never perform
geometry reasoning or CAD planning.
"""
from __future__ import annotations

import sys
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure shared-schemas is importable
_WM_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED_SCHEMAS = os.path.normpath(os.path.join(_WM_ROOT, "..", "shared-schemas"))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

from shared_schemas.logging_config import setup_structured_logging

from app.session import (
    SessionManager,
    SessionNotFoundError,
    PipelineStatus,
)
from app.checkpointing import CheckpointManager
from app.orchestrator import PipelineOrchestrator

logger = setup_structured_logging("workflow-manager", log_dir="logs")

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
_start_time: float = 0.0
session_mgr = SessionManager()
checkpoint_mgr = CheckpointManager()
orchestrator = PipelineOrchestrator(session_mgr, checkpoint_mgr)

app = FastAPI(
    title="Workflow Manager — AI CAD OS",
    version="0.1.0",
    description="Central pipeline orchestrator. Orchestration only — no geometry reasoning.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup() -> None:
    global _start_time
    _start_time = time.time()
    logger.info("workflow_manager.startup")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    logger.info("workflow_manager.shutdown")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StartPipelineRequest(BaseModel):
    image_path: str = Field(..., description="Path to the input image")


class StartPipelineResponse(BaseModel):
    session_id: str
    status: str


class SessionStatusResponse(BaseModel):
    session_id: str
    image_path: str
    current_stage: str
    status: str
    events: list
    created_at: str
    updated_at: str
    refinement_iteration: int = 0
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    """Readiness probe for Docker / K8s."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "uptime_seconds": round(time.time() - _start_time, 2) if _start_time else 0,
    }


@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Basic operational metrics."""
    sessions = session_mgr.list_sessions()
    completed = [s for s in sessions if s.status == PipelineStatus.COMPLETED]
    durations = []
    for s in completed:
        try:
            from datetime import datetime
            c = datetime.fromisoformat(s.created_at)
            u = datetime.fromisoformat(s.updated_at)
            durations.append((u - c).total_seconds() * 1000)
        except Exception:
            pass
    return {
        "active_sessions": session_mgr.active_count,
        "completed_pipelines": session_mgr.completed_count,
        "failed_pipelines": session_mgr.failed_count,
        "total_sessions": len(sessions),
        "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
    }


@app.post("/pipeline/start", response_model=StartPipelineResponse)
async def start_pipeline(req: StartPipelineRequest) -> StartPipelineResponse:
    """Start a new pipeline session for the given image."""
    session_id = session_mgr.create_session(req.image_path)
    logger.info("pipeline.start_requested", session_id=session_id, image_path=req.image_path)
    # Launch orchestration in background (non-blocking for V1)
    # In production: use asyncio.create_task or a task queue
    return StartPipelineResponse(session_id=session_id, status="queued")


@app.get("/pipeline/{session_id}/status", response_model=SessionStatusResponse)
async def pipeline_status(session_id: str) -> SessionStatusResponse:
    """Get the current status of a pipeline session."""
    try:
        session = session_mgr.get_session(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return SessionStatusResponse(
        session_id=session.session_id,
        image_path=session.image_path,
        current_stage=session.current_stage.value,
        status=session.status.value,
        events=[e.model_dump() for e in session.events],
        created_at=session.created_at,
        updated_at=session.updated_at,
        refinement_iteration=session.refinement_iteration,
        error_message=session.error_message,
    )


@app.post("/pipeline/{session_id}/cancel")
async def cancel_pipeline(session_id: str) -> Dict[str, str]:
    """Cancel a running pipeline session."""
    try:
        from app.session import PipelineStage
        session_mgr.update_session(
            session_id,
            stage=PipelineStage.CANCELLED,
            status=PipelineStatus.CANCELLED,
        )
        logger.info("pipeline.cancelled", session_id=session_id)
        return {"session_id": session_id, "status": "cancelled"}
    except SessionNotFoundError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
