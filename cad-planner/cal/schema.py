"""
cal/schema.py — CAL Schema (imports from shared-schemas)
==========================================================
ARCHITECTURE NOTE: The authoritative CAL schema is defined in
shared-schemas/cal_schema.py. This module re-exports everything
for backward compatibility within the cad-planner.
"""
import sys
import os

_PLANNER_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED_SCHEMAS = os.path.normpath(os.path.join(_PLANNER_ROOT, "..", "shared-schemas"))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

# Re-export everything from the authoritative source
from shared_schemas.cal_schema import (
    ActionReasoning,
    CALActionBase,
    CreateSketchAction,
    DrawCircleAction,
    DrawRectangleAction,
    ExtrudeAction,
    RevolveAction,
    FilletAction,
    ChamferAction,
    CALAction,
    PlanningTrace,
    CALDocument,
)
