"""
verification/orchestrator.py — Production Verification Orchestrator
=====================================================================
Central verification hub that aggregates metrics from:
  - Geometry Engine (Chamfer, Hausdorff, IoU, Normal Consistency)
  - Desktop Agent (execution success rate, feature count)
  - Refinement Loop (convergence state, iteration count)

Produces a final VerificationReport (shared-schemas) that is the
single source of truth for pipeline quality assessment.

Also provides threshold-based pass/fail decisions for CI/CD gates.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

_ENGINE_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.normpath(os.path.join(_ENGINE_ROOT, "..", "shared-schemas"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from shared_schemas.verification_report import (  # noqa: E402
    VerificationMetric,
    PrimitiveVerification,
    VerificationReport,
)

logger = logging.getLogger("geometry_engine.verification.orchestrator")


# ---------------------------------------------------------------------------
# Default quality thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "chamfer_distance": 0.05,       # Max acceptable mean Chamfer distance
    "hausdorff_distance": 0.5,      # Max acceptable Hausdorff distance
    "normal_consistency": 0.85,     # Min normal consistency
    "volume_iou": 0.90,            # Min volume IoU
    "execution_success_rate": 0.95, # Min CAD execution success rate
    "convergence_achieved": True,   # Must converge
    "primitive_coverage": 0.90,     # Min fraction of primitives verified
}


class VerificationOrchestrator:
    """
    Aggregates verification data from all pipeline stages and produces
    a unified VerificationReport.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, Any]] = None,
        output_dir: str = "output/verification",
    ) -> None:
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def build_report(
        self,
        geometry_metrics: Optional[Dict[str, Any]] = None,
        execution_metrics: Optional[Dict[str, Any]] = None,
        refinement_metrics: Optional[Dict[str, Any]] = None,
        per_primitive_data: Optional[List[Dict[str, Any]]] = None,
    ) -> VerificationReport:
        """
        Build a unified VerificationReport from all pipeline stages.

        Args:
            geometry_metrics: From geometry comparator.
                Keys: chamfer_distance, hausdorff_distance,
                      normal_consistency, volume_iou
            execution_metrics: From desktop agent.
                Keys: success_rate, total_actions, successful_actions
            refinement_metrics: From refinement loop.
                Keys: converged, final_iou, iterations,
                      convergence_reason
            per_primitive_data: List of per-primitive verification dicts.

        Returns:
            VerificationReport with all metrics and pass/fail decisions.
        """
        all_metrics: List[VerificationMetric] = []

        # --- Geometry metrics ---
        if geometry_metrics:
            all_metrics.extend(self._geometry_metrics(geometry_metrics))

        # --- Execution metrics ---
        if execution_metrics:
            all_metrics.extend(self._execution_metrics(execution_metrics))

        # --- Refinement metrics ---
        if refinement_metrics:
            all_metrics.extend(self._refinement_metrics(refinement_metrics))

        # --- Per-primitive results ---
        per_prim: List[PrimitiveVerification] = []
        if per_primitive_data:
            for prim in per_primitive_data:
                prim_metrics = []
                for key in ("chamfer_distance", "hausdorff_distance", "iou_score", "normal_consistency"):
                    val = prim.get(key)
                    if val is not None:
                        threshold = self.thresholds.get(
                            key.replace("iou_score", "volume_iou"), 1.0
                        )
                        if key in ("chamfer_distance", "hausdorff_distance"):
                            passed = val <= threshold
                        else:
                            passed = val >= threshold
                        prim_metrics.append(VerificationMetric(
                            metric_name=key,
                            value=float(val),
                            threshold=float(threshold),
                            passed=passed,
                        ))

                per_prim.append(PrimitiveVerification(
                    node_id=prim.get("node_id", "unknown"),
                    primitive_type=prim.get("primitive_type", "unknown"),
                    passed=all(m.passed for m in prim_metrics),
                    metrics=prim_metrics,
                ))

        # --- Overall decision ---
        overall_passed = all(m.passed for m in all_metrics)
        convergence_achieved = (
            refinement_metrics.get("converged", False)
            if refinement_metrics else False
        )

        report = VerificationReport(
            metrics=all_metrics,
            per_primitive_results=per_prim,
            overall_passed=overall_passed,
            convergence_achieved=convergence_achieved,
            chamfer_distance=geometry_metrics.get("chamfer_distance", 0.0) if geometry_metrics else 0.0,
            hausdorff_distance=geometry_metrics.get("hausdorff_distance", 0.0) if geometry_metrics else 0.0,
            normal_consistency=geometry_metrics.get("normal_consistency", 0.0) if geometry_metrics else 0.0,
        )

        logger.info(
            "Verification report: overall_passed=%s, metrics=%d, primitives=%d",
            overall_passed, len(all_metrics), len(per_prim),
        )

        return report

    def save_report(self, report: VerificationReport, filename: str = "verification_report.json") -> str:
        """Save the verification report to disk."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(report.to_json())
        logger.info("Verification report saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Metric builders
    # ------------------------------------------------------------------

    def _geometry_metrics(self, data: Dict[str, Any]) -> List[VerificationMetric]:
        metrics = []

        cd = data.get("chamfer_distance")
        if cd is not None:
            metrics.append(VerificationMetric(
                metric_name="chamfer_distance",
                value=float(cd),
                threshold=self.thresholds["chamfer_distance"],
                passed=cd <= self.thresholds["chamfer_distance"],
            ))

        hd = data.get("hausdorff_distance")
        if hd is not None:
            metrics.append(VerificationMetric(
                metric_name="hausdorff_distance",
                value=float(hd),
                threshold=self.thresholds["hausdorff_distance"],
                passed=hd <= self.thresholds["hausdorff_distance"],
            ))

        nc = data.get("normal_consistency")
        if nc is not None:
            metrics.append(VerificationMetric(
                metric_name="normal_consistency",
                value=float(nc),
                threshold=self.thresholds["normal_consistency"],
                passed=nc >= self.thresholds["normal_consistency"],
            ))

        iou = data.get("volume_iou")
        if iou is not None:
            metrics.append(VerificationMetric(
                metric_name="volume_iou",
                value=float(iou),
                threshold=self.thresholds["volume_iou"],
                passed=iou >= self.thresholds["volume_iou"],
            ))

        return metrics

    def _execution_metrics(self, data: Dict[str, Any]) -> List[VerificationMetric]:
        metrics = []

        sr = data.get("success_rate")
        if sr is not None:
            metrics.append(VerificationMetric(
                metric_name="execution_success_rate",
                value=float(sr),
                threshold=self.thresholds["execution_success_rate"],
                passed=sr >= self.thresholds["execution_success_rate"],
            ))

        return metrics

    def _refinement_metrics(self, data: Dict[str, Any]) -> List[VerificationMetric]:
        metrics = []

        converged = data.get("converged", False)
        metrics.append(VerificationMetric(
            metric_name="convergence_achieved",
            value=1.0 if converged else 0.0,
            threshold=1.0,
            passed=converged,
        ))

        final_iou = data.get("final_iou")
        if final_iou is not None:
            metrics.append(VerificationMetric(
                metric_name="refinement_final_iou",
                value=float(final_iou),
                threshold=self.thresholds["volume_iou"],
                passed=final_iou >= self.thresholds["volume_iou"],
            ))

        iterations = data.get("iterations")
        if iterations is not None:
            # Information-only metric, always passes
            metrics.append(VerificationMetric(
                metric_name="refinement_iterations",
                value=float(iterations),
                threshold=float(data.get("max_iterations", 10)),
                passed=True,
            ))

        return metrics
