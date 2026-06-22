"""
shared_schemas/refinement_request.py — Iterative Refinement Loop Schema
=========================================================================
Schema for the feedback loop:
    CAL → Desktop Agent → SolidWorks → Export Mesh → Verification
    → Geometry Difference → Geometry Engine Refinement → Updated GGL

This enables iterative quality improvement of the CAD reconstruction.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class GeometryDifference(BaseModel):
    """
    Quantified difference between the predicted geometry (GGL) and the
    actual geometry produced by the CAD system (exported mesh).
    """
    node_id: str = Field(..., description="GGL node this difference refers to")
    primitive_type: str = ""
    parameter_diffs: Dict[str, float] = Field(
        default_factory=dict,
        description="E.g., {'radius': -0.5, 'height': 1.2} — signed error per parameter"
    )
    iou_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Intersection-over-Union between predicted and actual volume"
    )
    hausdorff_distance: float = Field(
        default=0.0, ge=0.0,
        description="Maximum surface-to-surface distance (mm)"
    )
    chamfer_distance: float = Field(
        default=0.0, ge=0.0,
        description="Bidirectional Chamfer distance (mm)"
    )
    severity: str = Field(
        default="low",
        description="'low' (cosmetic), 'medium' (functional impact), 'high' (structural failure)"
    )


class RefinementRequest(BaseModel):
    """
    Sent from the verification stage back to the Geometry Engine to
    trigger an iterative refinement cycle.

    Data Flow:
        Image → Geometry Engine → GGL → CAD Planner → CAL → Desktop Agent
        → SolidWorks → Export Mesh → Verification → GeometryDifference
        → RefinementRequest → Geometry Engine → Updated GGL → (repeat)
    """
    version: str = "1.0"
    iteration: int = Field(default=1, ge=1, description="Current refinement iteration (1 = first pass)")
    max_iterations: int = Field(default=5, ge=1)

    original_ggl_path: str = Field(..., description="Path to the original GGL JSON")
    exported_mesh_path: str = Field(..., description="Path to the mesh exported from CAD")
    execution_report_path: Optional[str] = None

    differences: List[GeometryDifference] = Field(default_factory=list)

    # Aggregate metrics
    overall_iou: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_chamfer: float = Field(default=0.0, ge=0.0)
    convergence_threshold: float = Field(
        default=0.95,
        description="IOU threshold above which refinement stops"
    )

    @property
    def has_converged(self) -> bool:
        return self.overall_iou >= self.convergence_threshold

    @property
    def should_continue(self) -> bool:
        return not self.has_converged and self.iteration < self.max_iterations

    def to_json(self) -> str:
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "RefinementRequest":
        return cls.model_validate_json(json_str)
