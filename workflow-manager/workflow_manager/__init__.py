"""
workflow-manager — Central orchestrator for the AI CAD OS pipeline.

Performs ORCHESTRATION ONLY — never geometry reasoning or CAD planning.

Pipeline:
    Image → MODEL_GENERATOR_V2 → Geometry Engine → GGL
        → CAD Planner → CAL → Desktop Agent → CAD Software
        → Verification → Refinement (loop)
"""

__version__ = "0.1.0"
__package_name__ = "workflow-manager"
