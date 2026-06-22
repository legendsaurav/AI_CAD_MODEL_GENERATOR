"""
refinement/loop.py — Iterative Refinement Loop Orchestrator
=============================================================
Orchestrates the complete feedback loop:

    Image → Geometry Engine → GGL → CAD Planner → CAL → Desktop Agent
    → SolidWorks → Export Mesh → Verification → Geometry Difference
    → Geometry Engine Refinement → Updated GGL → (repeat until converged)

This module coordinates the refinement cycle without depending on any
specific CAD software — it communicates exclusively through shared schemas.
"""
import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("geometry_engine.refinement")


class RefinementLoop:
    """
    Orchestrates iterative refinement of GGL predictions by comparing
    against CAD-exported meshes and adjusting parameters.

    Each iteration:
      1. Compare exported mesh against current GGL predictions
      2. Compute geometry differences per primitive
      3. Adjust GGL parameters to reduce error
      4. Export updated GGL for the next CAD planning cycle

    Convergence criteria:
      - Overall IOU ≥ threshold (default 0.95)
      - OR max iterations reached (default 5)
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.95,
        output_dir: str = "output/refinement",
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        ggl_dict: Dict[str, Any],
        exported_mesh_path: str,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """
        Runs one iteration of the refinement loop.

        Args:
            ggl_dict: Current GGL as a dict
            exported_mesh_path: Path to the mesh exported from the CAD system
            iteration: Current iteration number

        Returns:
            Dict with:
                - "updated_ggl": The refined GGL dict
                - "differences": List of geometry differences
                - "overall_iou": Aggregate IOU score
                - "converged": Whether convergence was reached
                - "should_continue": Whether another iteration is warranted
        """
        from refinement.comparator import GeometryComparator

        logger.info(f"═══ Refinement Iteration {iteration}/{self.max_iterations} ═══")

        # Step 1: Compare geometry
        comparator = GeometryComparator()
        differences = comparator.compare(ggl_dict, exported_mesh_path)

        # Step 2: Compute aggregate metrics
        if differences:
            overall_iou = sum(d["iou_score"] for d in differences) / len(differences)
        else:
            overall_iou = 1.0

        logger.info(f"  Overall IOU: {overall_iou:.4f} (threshold: {self.convergence_threshold})")

        converged = overall_iou >= self.convergence_threshold
        should_continue = not converged and iteration < self.max_iterations

        if converged:
            logger.info("  ✅ Converged! Geometry reconstruction meets quality threshold.")
        elif not should_continue:
            logger.warning(f"  ⚠️ Max iterations ({self.max_iterations}) reached without convergence.")
        else:
            logger.info(f"  🔄 Not converged. Applying refinements...")

        # Step 3: Refine GGL parameters
        updated_ggl = self._apply_refinements(ggl_dict, differences) if should_continue else ggl_dict

        # Step 4: Save iteration results
        iteration_path = os.path.join(self.output_dir, f"iteration_{iteration}.json")
        with open(iteration_path, "w") as f:
            json.dump({
                "iteration": iteration,
                "overall_iou": overall_iou,
                "converged": converged,
                "differences": differences,
            }, f, indent=2)

        # Save updated GGL
        ggl_path = os.path.join(self.output_dir, f"ggl_refined_v{iteration}.json")
        with open(ggl_path, "w") as f:
            json.dump(updated_ggl, f, indent=2)

        logger.info(f"  Saved results to {iteration_path}")

        return {
            "updated_ggl": updated_ggl,
            "differences": differences,
            "overall_iou": overall_iou,
            "converged": converged,
            "should_continue": should_continue,
            "iteration": iteration,
        }

    def _apply_refinements(
        self,
        ggl_dict: Dict[str, Any],
        differences: list,
    ) -> Dict[str, Any]:
        """
        Adjusts GGL parameters based on measured geometry differences.

        V1 strategy: Scale parameters proportionally to IOU error.
        Future: Use gradient-based optimization or learned correction networks.
        """
        import copy
        refined = copy.deepcopy(ggl_dict)

        diff_by_id = {d["node_id"]: d for d in differences}

        for node in refined.get("nodes", []):
            node_id = node.get("node_id", "")
            if node_id not in diff_by_id:
                continue

            diff = diff_by_id[node_id]
            iou = diff.get("iou_score", 1.0)

            if iou >= self.convergence_threshold:
                continue  # This primitive is good enough

            # Apply parameter corrections
            params = node.get("parameters", {})
            param_diffs = diff.get("parameter_diffs", {})

            for param_name, error in param_diffs.items():
                if param_name in params:
                    # Simple correction: subtract the error
                    params[param_name] = params[param_name] - error * 0.5  # Damped correction

            # If no parameter-level diffs, apply a scaling heuristic
            if not param_diffs and iou > 0:
                correction_factor = 1.0 + (1.0 - iou) * 0.1  # Small nudge
                for key in ["radius", "width", "height", "depth"]:
                    if key in params:
                        params[key] = abs(params[key] * correction_factor)

            node["parameters"] = params

        return refined

    def run_full_loop(
        self,
        initial_ggl_dict: Dict[str, Any],
        mesh_exporter_fn=None,
    ) -> Dict[str, Any]:
        """
        Runs the complete refinement loop until convergence or max iterations.

        Args:
            initial_ggl_dict: The initial GGL from the geometry engine
            mesh_exporter_fn: Optional callable(ggl_dict) -> mesh_path
                             that runs the full CAD pipeline and returns
                             the exported mesh path.

        Returns:
            Final refinement result dict.
        """
        current_ggl = initial_ggl_dict
        result = None

        for i in range(1, self.max_iterations + 1):
            # In production, mesh_exporter_fn would:
            # 1. Send GGL to cad-planner → get CAL
            # 2. Send CAL to desktop-agent → execute in SolidWorks
            # 3. Export mesh from SolidWorks
            # 4. Return the mesh path
            if mesh_exporter_fn:
                mesh_path = mesh_exporter_fn(current_ggl)
            else:
                logger.warning(f"No mesh_exporter_fn provided. Refinement loop cannot run end-to-end.")
                break

            result = self.run(current_ggl, mesh_path, iteration=i)

            if not result["should_continue"]:
                break

            current_ggl = result["updated_ggl"]

        return result or {"converged": False, "overall_iou": 0.0, "iteration": 0}
