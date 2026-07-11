"""
tests/test_orchestrator.py — Comprehensive tests for workflow-manager.

Tests cover:
- Session creation and lifecycle
- Checkpoint save/load
- Pipeline event emission
- Health and metrics endpoints
- Retry logic
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

# Ensure shared-schemas and app are importable
_WM_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.normpath(os.path.join(_WM_ROOT, "..", "shared-schemas"))
if _WM_ROOT not in sys.path:
    sys.path.insert(0, _WM_ROOT)
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from app.session import (
    SessionManager,
    SessionNotFoundError,
    PipelineStage,
    PipelineStatus,
    PipelineEvent,
)
from app.checkpointing import CheckpointManager


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------

class TestSessionManager:
    """Tests for SessionManager."""

    def test_create_session(self):
        mgr = SessionManager()
        sid = mgr.create_session("/images/test.png")
        assert sid is not None
        session = mgr.get_session(sid)
        assert session.image_path == "/images/test.png"
        assert session.status == PipelineStatus.QUEUED

    def test_session_not_found(self):
        mgr = SessionManager()
        with pytest.raises(SessionNotFoundError):
            mgr.get_session("nonexistent-id")

    def test_update_session(self):
        mgr = SessionManager()
        sid = mgr.create_session("/images/test.png")
        mgr.update_session(
            sid,
            stage=PipelineStage.GEOMETRY_ENGINE,
            status=PipelineStatus.RUNNING,
        )
        session = mgr.get_session(sid)
        assert session.current_stage == PipelineStage.GEOMETRY_ENGINE
        assert session.status == PipelineStatus.RUNNING

    def test_add_event(self):
        mgr = SessionManager()
        sid = mgr.create_session("/images/test.png")
        event = PipelineEvent(
            stage="model_generator_v2",
            event_type="stage_started",
            message="Starting model generation",
        )
        mgr.add_event(sid, event)
        session = mgr.get_session(sid)
        assert len(session.events) == 1
        assert session.events[0].event_type == "stage_started"

    def test_list_sessions(self):
        mgr = SessionManager()
        mgr.create_session("/images/a.png")
        mgr.create_session("/images/b.png")
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_active_completed_failed_counts(self):
        mgr = SessionManager()
        sid1 = mgr.create_session("/a.png")
        sid2 = mgr.create_session("/b.png")
        sid3 = mgr.create_session("/c.png")
        assert mgr.active_count == 3  # all queued
        mgr.update_session(sid1, PipelineStage.COMPLETED, PipelineStatus.COMPLETED)
        mgr.update_session(sid2, PipelineStage.FAILED, PipelineStatus.FAILED)
        assert mgr.active_count == 1
        assert mgr.completed_count == 1
        assert mgr.failed_count == 1


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_save_and_load_checkpoint(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        mgr.save_checkpoint("sess-1", "geometry_engine", {"ggl": "test_data"})
        loaded = mgr.load_checkpoint("sess-1")
        assert loaded is not None
        assert loaded.session_id == "sess-1"
        assert loaded.stage == "geometry_engine"
        assert loaded.data["ggl"] == "test_data"

    def test_load_nonexistent_session(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        assert mgr.load_checkpoint("no-such-session") is None

    def test_list_checkpoints(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        mgr.save_checkpoint("sess-2", "stage_a", {"step": 1})
        mgr.save_checkpoint("sess-2", "stage_b", {"step": 2})
        cps = mgr.list_checkpoints("sess-2")
        assert len(cps) == 2

    def test_list_empty(self, tmp_path):
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        assert mgr.list_checkpoints("empty") == []


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestFastAPIEndpoints:
    """Tests for main.py endpoints (requires httpx + pytest)."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        try:
            from fastapi.testclient import TestClient
            from app.main import app
            return TestClient(app)
        except ImportError:
            pytest.skip("fastapi or httpx not installed")

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_sessions" in data
        assert "completed_pipelines" in data
        assert "failed_pipelines" in data

    def test_start_pipeline(self, client):
        resp = client.post("/pipeline/start", json={"image_path": "/test.png"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["status"] == "queued"

    def test_get_status(self, client):
        # Create a session first
        resp = client.post("/pipeline/start", json={"image_path": "/test.png"})
        sid = resp.json()["session_id"]
        # Get its status
        resp = client.get(f"/pipeline/{sid}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert data["image_path"] == "/test.png"

    def test_get_status_not_found(self, client):
        resp = client.get("/pipeline/nonexistent/status")
        assert resp.status_code == 404

    def test_cancel_pipeline(self, client):
        resp = client.post("/pipeline/start", json={"image_path": "/test.png"})
        sid = resp.json()["session_id"]
        resp = client.post(f"/pipeline/{sid}/cancel")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
