"""
shared-schemas — Authoritative interface contracts for the AI CAD Operating System.

All inter-repository data flows use schemas defined here.
No repository may define its own GGL/CAL/ExecutionReport schema.

Data Flow:
    Image → geometry-engine → GGL → cad-planner → CAL → desktop-agent → ExecutionReport
                                                                              ↓
                                                                     Verification
                                                                              ↓
                                                              RefinementRequest
                                                                              ↓
                                                              geometry-engine (loop)
"""
# ── Core schemas (original) ────────────────────────────────────────────────
from shared_schemas.ggl_schema import (
    GGLNode, GGLEdge, GGLMetadata, GeometryGraphLanguage
)
from shared_schemas.cal_schema import (
    ActionReasoning, CALActionBase, CreateSketchAction, DrawCircleAction,
    DrawRectangleAction, ExtrudeAction, RevolveAction, FilletAction,
    ChamferAction, CALAction, CALDocument
)
from shared_schemas.execution_report import ExecutionReport, ActionResult
from shared_schemas.refinement_request import RefinementRequest, GeometryDifference

# ── Versioning infrastructure ──────────────────────────────────────────────
from shared_schemas.versioning import SchemaVersion, VersionedSchema

# ── Phase 1 schemas ────────────────────────────────────────────────────────
from shared_schemas.reason_graph import (
    RejectedAlternative, ReasonNode, ReasonEdge, ReasonGraph
)
from shared_schemas.planning_trace import (
    ScoringBreakdown, BeamCandidate, AmbiguityResolution,
    RetrievedMemory, RejectedPlan, PlanningTrace
)
from shared_schemas.events import (
    EventType, PipelineStage, EventPayload, StageResult, PipelineEvent
)
from shared_schemas.verification_report import (
    VerificationMetric, PrimitiveVerification, VerificationReport
)

# ── Logging ────────────────────────────────────────────────────────────────
from shared_schemas.logging_config import (
    setup_structured_logging, get_correlation_id, set_correlation_id
)

__all__ = [
    # Core schemas
    "GGLNode", "GGLEdge", "GGLMetadata", "GeometryGraphLanguage",
    "ActionReasoning", "CALActionBase", "CreateSketchAction", "DrawCircleAction",
    "DrawRectangleAction", "ExtrudeAction", "RevolveAction", "FilletAction",
    "ChamferAction", "CALAction", "CALDocument",
    "ExecutionReport", "ActionResult",
    "RefinementRequest", "GeometryDifference",
    # Versioning
    "SchemaVersion", "VersionedSchema",
    # Phase 1 schemas
    "RejectedAlternative", "ReasonNode", "ReasonEdge", "ReasonGraph",
    "ScoringBreakdown", "BeamCandidate", "AmbiguityResolution",
    "RetrievedMemory", "RejectedPlan", "PlanningTrace",
    "EventType", "PipelineStage", "EventPayload", "StageResult", "PipelineEvent",
    "VerificationMetric", "PrimitiveVerification", "VerificationReport",
    # Logging
    "setup_structured_logging", "get_correlation_id", "set_correlation_id",
]
