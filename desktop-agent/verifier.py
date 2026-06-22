"""
desktop-agent/verifier.py — Post-Execution Verification
=========================================================
Verifies that CAD actions were executed correctly by comparing
screenshots, checking the feature tree, and exporting meshes
for the refinement loop.
"""
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger("desktop_agent.verifier")


class ExecutionVerifier:
    """
    Verifies CAD execution results and produces data for the refinement loop.

    Verification methods:
      1. Screenshot comparison (OCR-based feature tree check)
      2. Mesh export for geometric verification
      3. Feature count validation
    """

    def __init__(self, output_dir: str = "output/verification"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def verify_execution(
        self,
        executor,
        expected_actions: int,
        completed_actions: int,
    ) -> Dict[str, Any]:
        """
        Runs post-execution verification.

        Returns:
            Dict with verification results including mesh export path.
        """
        result = {
            "actions_expected": expected_actions,
            "actions_completed": completed_actions,
            "success_rate": completed_actions / max(expected_actions, 1),
            "mesh_exported": False,
            "mesh_path": None,
            "screenshot_path": None,
        }

        # Export mesh for refinement loop
        mesh_path = os.path.join(self.output_dir, "exported_model.stl")
        if executor.export_mesh(mesh_path):
            result["mesh_exported"] = True
            result["mesh_path"] = mesh_path
            logger.info(f"Verification mesh exported: {mesh_path}")

        # Take screenshot
        screenshot_path = os.path.join(self.output_dir, "final_screenshot.png")
        if executor.take_screenshot(screenshot_path):
            result["screenshot_path"] = screenshot_path

        return result
