"""
shared_schemas/execution_report.py — Desktop Agent → Verification
==================================================================
Schema for the execution report produced by the Desktop Agent after
executing CAL actions in a CAD application.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


class ActionResult(BaseModel):
    """Result of executing a single CAL action."""
    action_id: str
    action_type: str
    status: str = Field(..., description="'success', 'failed', 'skipped', 'partial'")
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    ocr_verification: Optional[str] = None
    execution_time_ms: float = 0.0
    cad_feature_id: Optional[str] = Field(
        None, description="The feature ID assigned by the CAD software (e.g., SolidWorks Feature Manager ID)"
    )


class ExecutionReport(BaseModel):
    """
    Full execution report from the Desktop Agent.
    Consumed by the verification/refinement loop and the Clicky Tutor.
    """
    version: str = "1.0"
    cal_version: str = "1.0"
    target_cad_system: str = Field(..., description="E.g., 'SolidWorks', 'FreeCAD', 'Fusion360'")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    action_results: List[ActionResult] = Field(default_factory=list)

    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0

    exported_mesh_path: Optional[str] = Field(
        None, description="Path to the mesh exported from the CAD system for verification"
    )
    final_screenshot_path: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.successful_actions / self.total_actions

    def to_json(self) -> str:
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionReport":
        return cls.model_validate_json(json_str)
