"""
refinement/loop.py — Production Iterative Refinement Loop
============================================================
Orchestrates the complete feedback loop:

    Image → Geometry Engine → GGL → CAD Planner → CAL → Desktop Agent
    → CAD Software → Export Mesh → Verification → Geometry Difference
    → Geometry Engine Refinement → Updated GGL → (repeat until converged)

Production features:
  - Convergence detection (plateau, oscillation, divergence)
  - Adaptive step size for parameter corrections
  - VerificationReport integration (shared-schemas)
  - Per-primitive targeted refinement
  - Structured iteration logging
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from refinement.comparator import GeometryComparator
from refinement.convergence import ConvergenceDetector, ConvergenceState

logger = logging.getLogger("geometry_engine.refinement")


class RefinementLoop:
    """
    Orchestrates iterative refinement of GGL predictions by comparing
    against CAD-exported meshes and adjusting parameters.

    Each iteration:
      1. Compare exported mesh against current GGL predictions
      2. Compute geometry differences per primitive
      3. Check convergence (threshold, plateau, oscillation)
      4. Adjust GGL parameters with adaptive step size
      5. Export updated GGL for the next CAD planning cycle
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.95,
        output_dir: str = "output/refinement",
        chamfer_threshold: float = 0.01,
        plateau_window: int = 3,
    ) -> None:
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._convergence = ConvergenceDetector(
            iou_threshold=convergence_threshold,
            chamfer_threshold=chamfer_threshold,
            plateau_window=plateau_window,
            max_iterations=max_iterations,
        )
        self._comparator = GeometryComparator()

    def run(
        self,
        ggl_dict: Dict[str, Any],
        exported_mesh_path: str,
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """
        Run one iteration of the refinement loop.

        Args:
            ggl_dict: Current GGL as a dict.
            exported_mesh_path: Path to mesh exported from CAD system.
            iteration: Current iteration number.

        Returns:
            Dict with updated_ggl, differences, metrics, convergence info.
        """
        t0 = time.monotonic()
        logger.info("═══ Refinement Iteration %d/%d ═══", iteration, self.max_iterations)

        # Step 1: Compare geometry
        differences = self._comparator.compare(ggl_dict, exported_mesh_path)

        # Step 2: Compute aggregate metrics
        if differences:
            overall_iou = sum(d["iou_score"] for d in differences) / len(differences)
            mean_chamfer = sum(d["chamfer_distance"] for d in differences) / len(differences)
            mean_hausdorff = sum(d["hausdorff_distance"] for d in differences) / len(differences)
            mean_nc = sum(d["normal_consistency"] for d in differences) / len(differences)
        else:
            overall_iou = 1.0
            mean_chamfer = 0.0
            mean_hausdorff = 0.0
            mean_nc = 1.0

        # Step 3: Check convergence
        converged, should_continue, reason = self._convergence.update(
            iteration=iteration,
            overall_iou=overall_iou,
            mean_chamfer=mean_chamfer,
            mean_hausdorff=mean_hausdorff,
        )

        # Step 4: Apply refinements if continuing
        if should_continue:
            step_size = self._convergence.get_recommended_step_size()
            updated_ggl = self._apply_refinements(ggl_dict, differences, step_size)
            logger.info("  Applied refinements with step_size=%.2f", step_size)
        else:
            updated_ggl = ggl_dict

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Step 5: Classify per-primitive severity
        high_severity = sum(1 for d in differences if d.get("severity") == "high")
        medium_severity = sum(1 for d in differences if d.get("severity") == "medium")

        # Step 6: Save iteration results
        iteration_result = {
            "iteration": iteration,
            "overall_iou": round(overall_iou, 6),
            "mean_chamfer_distance": round(mean_chamfer, 6),
            "mean_hausdorff_distance": round(mean_hausdorff, 6),
            "mean_normal_consistency": round(mean_nc, 4),
            "converged": converged,
            "convergence_reason": reason,
            "should_continue": should_continue,
            "high_severity_primitives": high_severity,
            "medium_severity_primitives": medium_severity,
            "differences": differences,
            "elapsed_ms": round(elapsed_ms, 2),
        }

        iteration_path = os.path.join(self.output_dir, f"iteration_{iteration}.json")
        with open(iteration_path, "w") as f:
            json.dump(iteration_result, f, indent=2)

        ggl_path = os.path.join(self.output_dir, f"ggl_refined_v{iteration}.json")
        with open(ggl_path, "w") as f:
            json.dump(updated_ggl, f, indent=2)

        logger.info(
            "  IoU=%.4f  CD=%.6f  HD=%.6f  NC=%.4f  [%s] (%.1fms)",
            overall_iou, mean_chamfer, mean_hausdorff, mean_nc, reason, elapsed_ms,
        )

        return {
            "updated_ggl": updated_ggl,
            "differences": differences,
            "overall_iou": overall_iou,
            "mean_chamfer": mean_chamfer,
            "converged": converged,
            "should_continue": should_continue,
            "iteration": iteration,
            "convergence_reason": reason,
            "convergence_state": self._convergence.state.to_dict(),
        }

    def _apply_refinements(
        self,
        ggl_dict: Dict[str, Any],
        differences: List[Dict[str, Any]],
        step_size: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Adjust GGL parameters based on measured geometry differences.

        Uses adaptive step size from convergence detector:
          - step_size=1.0 → standard correction
          - step_size=0.5 → damped (oscillation detected)
          - step_size=1.5 → boosted (plateau detected)
        """
        refined = copy.deepcopy(ggl_dict)
        diff_by_id = {d["node_id"]: d for d in differences}

        for node in refined.get("nodes", []):
            node_id = node.get("node_id", "")
            if node_id not in diff_by_id:
                continue

            diff = diff_by_id[node_id]
            iou = diff.get("iou_score", 1.0)

            if iou >= self.convergence_threshold:
                continue  # This primitive is already good

            params = node.get("parameters", {})
            param_diffs = diff.get("parameter_diffs", {})

            # Strategy 1: Apply explicit parameter corrections
            if param_diffs:
                for param_name, error in param_diffs.items():
                    if param_name in params and isinstance(params[param_name], (int, float)):
                        correction = error * 0.5 * step_size
                        params[param_name] = params[param_name] - correction
                        logger.debug(
                            "    %s.%s: corrected by %.4f (step=%.2f)",
                            node_id, param_name, correction, step_size,
                        )

            # Strategy 2: Adaptive scaling based on severity
            elif iou > 0:
                severity = diff.get("severity", "low")
                if severity == "high":
                    correction_factor = 1.0 + (1.0 - iou) * 0.2 * step_size
                elif severity == "medium":
                    correction_factor = 1.0 + (1.0 - iou) * 0.1 * step_size
                else:
                    correction_factor = 1.0 + (1.0 - iou) * 0.05 * step_size

                for key in ("radius", "width", "height", "depth"):
                    if key in params and isinstance(params[key], (int, float)):
                        params[key] = abs(params[key] * correction_factor)

            node["parameters"] = params

        return refined

    def run_full_loop(
        self,
        initial_ggl_dict: Dict[str, Any],
        mesh_exporter_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete refinement loop until convergence or stop condition.

        Args:
            initial_ggl_dict: The initial GGL from the geometry engine.
            mesh_exporter_fn: Callable(ggl_dict) → mesh_path that runs
                            the full CAD pipeline and returns the
                            exported mesh path.

        Returns:
            Final refinement result with convergence state.
        """
        self._convergence.reset()
        current_ggl = initial_ggl_dict
        result = None
        all_iterations: List[Dict[str, Any]] = []

        t0 = time.monotonic()

        for i in range(1, self.max_iterations + 1):
            if mesh_exporter_fn:
                try:
                    mesh_path = mesh_exporter_fn(current_ggl)
                except Exception as e:
                    logger.error("Mesh export failed at iteration %d: %s", i, e)
                    break
            else:
                logger.warning("No mesh_exporter_fn provided. Cannot run end-to-end.")
                break

            result = self.run(current_ggl, mesh_path, iteration=i)
            all_iterations.append({
                "iteration": i,
                "iou": result["overall_iou"],
                "chamfer": result["mean_chamfer"],
                "converged": result["converged"],
            })

            if not result["should_continue"]:
                break

            current_ggl = result["updated_ggl"]

        total_ms = (time.monotonic() - t0) * 1000

        # Save summary
        summary = {
            "total_iterations": len(all_iterations),
            "final_converged": result["converged"] if result else False,
            "final_iou": result["overall_iou"] if result else 0.0,
            "total_time_ms": round(total_ms, 2),
            "iteration_history": all_iterations,
            "convergence_state": self._convergence.state.to_dict(),
        }

        summary_path = os.path.join(self.output_dir, "refinement_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Refinement complete: %d iterations, IoU=%.4f, converged=%s (%.1fms)",
            len(all_iterations),
            summary["final_iou"],
            summary["final_converged"],
            total_ms,
        )

        return result or {"converged": False, "overall_iou": 0.0, "iteration": 0}
