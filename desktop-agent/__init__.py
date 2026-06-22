"""
desktop-agent — CAL Consumer and CAD Automation Module
=======================================================
Consumes CAD Action Language (CAL) documents from the cad-planner
and executes them in target CAD systems (FreeCAD, SolidWorks, Fusion 360).

Pipeline position:
    cad-planner → CAL → desktop-agent → CAD Software → Execution Report

This module is the ONLY component that communicates with CAD software.
Neither geometry-engine nor cad-planner may call CAD APIs.

Data Flow:
    1. Receive CAL JSON from cad-planner
    2. Parse via CALConsumer
    3. Execute via target-specific Executor (FreeCAD, SolidWorks, etc.)
    4. Verify execution via Verifier (screenshot + mesh comparison)
    5. Generate ExecutionReport for the refinement loop
"""
__version__ = "0.1.0"
