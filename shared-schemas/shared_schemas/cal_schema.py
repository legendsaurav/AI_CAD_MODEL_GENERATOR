"""
shared_schemas/cal_schema.py — Authoritative CAD Action Language Schema
========================================================================
Single source of truth for CAL. cad-planner produces CAL, desktop-agent consumes it.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from datetime import datetime, timezone


class ActionReasoning(BaseModel):
    """Reason Graph block — provides explainability for each CAD action."""
    purpose: str = Field(..., description="Engineering purpose of the action")
    rationale: str = Field(..., description="Modeling rationale")
    depends_on: List[str] = Field(default_factory=list)
    alternatives_considered: List[str] = Field(default_factory=list)


class CALActionBase(BaseModel):
    """Base class for all CAL actions."""
    action_id: str
    action_type: str
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Propagated from the originating GGL node confidence"
    )
    source_ggl_node_id: Optional[str] = Field(
        default=None,
        description="ID of the GGL node that originated this action (for traceability)"
    )
    reasoning: Optional[ActionReasoning] = None


class CreateSketchAction(CALActionBase):
    action_type: Literal["create_sketch"] = "create_sketch"
    plane: str = Field(..., description="E.g., 'Front', 'Top', 'Right', or a reference face ID")


class DrawCircleAction(CALActionBase):
    action_type: Literal["draw_circle"] = "draw_circle"
    sketch_id: str
    center: List[float] = Field(..., description="[x, y] coordinates in sketch space")
    radius: float


class DrawRectangleAction(CALActionBase):
    action_type: Literal["draw_rectangle"] = "draw_rectangle"
    sketch_id: str
    center: List[float]
    width: float
    height: float


class ExtrudeAction(CALActionBase):
    action_type: Literal["extrude"] = "extrude"
    sketch_id: str
    depth: float
    direction: int = Field(1, description="1 for normal, -1 for reverse, 0 for midplane")
    is_cut: bool = Field(False, description="True if this is a cut-extrude")


class RevolveAction(CALActionBase):
    action_type: Literal["revolve"] = "revolve"
    sketch_id: str
    axis: List[float] = Field(..., description="[x, y, z] axis vector")
    angle: float = 360.0
    is_cut: bool = False


class FilletAction(CALActionBase):
    action_type: Literal["fillet"] = "fillet"
    target_edges: List[str]
    radius: float


class ChamferAction(CALActionBase):
    action_type: Literal["chamfer"] = "chamfer"
    target_edges: List[str]
    distance: float


CALAction = Union[
    CreateSketchAction, DrawCircleAction, DrawRectangleAction,
    ExtrudeAction, RevolveAction, FilletAction, ChamferAction,
]


class PlanningTrace(BaseModel):
    """
    Records the full decision-making trajectory of the CAD Planner.
    Emitted alongside the CAL document for debugging and research analysis.
    """
    ggl_node_count: int = 0
    ggl_edge_count: int = 0
    intents: List[dict] = Field(default_factory=list, description="Intent classifications per node")
    topological_order: List[str] = Field(default_factory=list)
    beam_candidates_count: int = 0
    beam_scores: List[dict] = Field(default_factory=list, description="[{candidate_idx, score}]")
    best_candidate_idx: int = 0
    memory_recalls: List[dict] = Field(default_factory=list, description="[{node_id, pattern}]")
    ambiguity_strategies: List[dict] = Field(default_factory=list)
    manufacturability_score: float = 0.0
    manufacturability_issues: List[str] = Field(default_factory=list)
    total_cal_actions_before_optimization: int = 0
    total_cal_actions_after_optimization: int = 0


class CALDocument(BaseModel):
    """The final software-independent output of the CAD Planning Engine."""
    version: str = "1.0"
    planner_version: str = "0.1"
    generator: str = "cad-planner"
    ggl_version: str = "1.0"
    reason_graph_version: str = "1.0"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    actions: List[CALAction] = Field(default_factory=list)
    planning_trace: Optional[PlanningTrace] = None

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "CALDocument":
        return cls.model_validate_json(json_str)
