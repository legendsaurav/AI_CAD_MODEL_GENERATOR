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
