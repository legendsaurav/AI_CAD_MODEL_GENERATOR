"""
desktop-agent/verifier.py — Production Post-Execution Verification
=====================================================================
Verifies CAD execution results by:
  1. Checking execution completeness (actions expected vs completed)
  2. Exporting meshes for the refinement loop
  3. Computing VerificationReport metrics (from shared-schemas)
  4. Screenshot capture for visual inspection

Produces a shared-schemas VerificationReport that feeds the
refinement loop's convergence decision.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

_AGENT_ROOT = os.path.normpath(os.path.dirname(__file__))
_SHARED_SCHEMAS = os.path.normpath(os.path.join(_AGENT_ROOT, "..", "shared-schemas"))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

from shared_schemas.verification_report import (
    VerificationMetric,
    PrimitiveVerification,
    VerificationReport,
)

logger = logging.getLogger("desktop_agent.verifier")


class ExecutionVerifier:
    """
    Verifies CAD execution results and produces structured
    VerificationReport data for the refinement loop.
    """

    def __init__(self, output_dir: str = "output/verification") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def verify_execution(
        self,
        executor,
        expected_actions: int,
        completed_actions: int,
        action_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run post-execution verification.

        Args:
            executor: CAD executor instance with export_mesh/take_screenshot.
            expected_actions: Total CAL actions expected.
            completed_actions: Number of actions successfully completed.
            action_results: Per-action results for detailed verification.

        Returns:
            Dict with verification results including mesh path and
            VerificationReport data.
        """
        t0 = time.monotonic()

        result: Dict[str, Any] = {
            "actions_expected": expected_actions,
            "actions_completed": completed_actions,
            "success_rate": completed_actions / max(expected_actions, 1),
            "mesh_exported": False,
            "mesh_path": None,
            "screenshot_path": None,
            "verification_report": None,
        }

        # Step 1: Export mesh for refinement loop
        mesh_path = os.path.join(self.output_dir, "exported_model.stl")
        if executor.export_mesh(mesh_path):
            result["mesh_exported"] = True
            result["mesh_path"] = mesh_path
            logger.info("Verification mesh exported: %s", mesh_path)

        # Step 2: Take screenshot
        screenshot_path = os.path.join(self.output_dir, "final_screenshot.png")
        if executor.take_screenshot(screenshot_path):
            result["screenshot_path"] = screenshot_path

        # Step 3: Build VerificationReport
        report = self._build_verification_report(
            expected_actions, completed_actions, action_results or []
        )
        result["verification_report"] = report.to_json() if report else None

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "Verification complete in %.1fms: %d/%d actions, mesh=%s",
            elapsed_ms, completed_actions, expected_actions,
            result["mesh_exported"],
        )

        return result

    def _build_verification_report(
        self,
        expected_actions: int,
        completed_actions: int,
        action_results: List[Dict[str, Any]],
    ) -> VerificationReport:
        """
        Build a structured VerificationReport from execution results.

        This report integrates with the geometry-engine refinement loop.
        """
        metrics: List[VerificationMetric] = []

        # Execution completeness metric
        success_rate = completed_actions / max(expected_actions, 1)
        metrics.append(VerificationMetric(
            metric_name="execution_completeness",
            value=success_rate,
            threshold=0.95,
            passed=success_rate >= 0.95,
        ))

        # Per-action success metric
        if action_results:
            action_success = sum(
                1 for r in action_results if r.get("success", False)
            ) / max(len(action_results), 1)
            metrics.append(VerificationMetric(
                metric_name="action_success_rate",
                value=action_success,
                threshold=0.90,
                passed=action_success >= 0.90,
            ))

            # Check for critical failures (extrude/revolve failures are blocking)
            critical_ops = {"extrude", "revolve", "create_sketch"}
            critical_failures = sum(
                1 for r in action_results
                if not r.get("success", False)
                and r.get("action_type", "") in critical_ops
            )
            metrics.append(VerificationMetric(
                metric_name="critical_feature_failures",
                value=float(critical_failures),
                threshold=0.0,
                passed=critical_failures == 0,
            ))

        # Build per-primitive results from action data
        per_primitive: List[PrimitiveVerification] = []
        for r in action_results:
            source_ggl = r.get("source_ggl_node_id")
            if source_ggl:
                per_primitive.append(PrimitiveVerification(
                    node_id=source_ggl,
                    primitive_type=r.get("action_type", "unknown"),
                    passed=r.get("success", False),
                    metrics=[VerificationMetric(
                        metric_name="execution_success",
                        value=1.0 if r.get("success") else 0.0,
                        threshold=1.0,
                        passed=r.get("success", False),
                    )],
                ))

        overall_passed = all(m.passed for m in metrics)

        report = VerificationReport(
            metrics=metrics,
            per_primitive_results=per_primitive,
            overall_passed=overall_passed,
            convergence_achieved=overall_passed and success_rate >= 0.95,
        )

        return report

    def verify_mesh_quality(
        self,
        mesh_path: str,
    ) -> Dict[str, Any]:
        """
        Verify exported mesh quality (watertight, manifold, etc.).

        Returns:
            Dict with mesh quality metrics.
        """
        result = {
            "exists": os.path.exists(mesh_path),
            "file_size_bytes": 0,
            "is_watertight": False,
            "vertex_count": 0,
            "face_count": 0,
        }

        if not result["exists"]:
            return result

        result["file_size_bytes"] = os.path.getsize(mesh_path)

        try:
            import trimesh
            mesh = trimesh.load(mesh_path)
            result["is_watertight"] = bool(mesh.is_watertight)
            result["vertex_count"] = len(mesh.vertices)
            result["face_count"] = len(mesh.faces)
            if mesh.is_watertight:
                result["volume"] = float(mesh.volume)
            logger.info(
                "Mesh quality: %d verts, %d faces, watertight=%s",
                result["vertex_count"], result["face_count"],
                result["is_watertight"],
            )
        except ImportError:
            logger.warning("trimesh not installed — skipping mesh quality checks")
        except Exception as e:
            logger.warning("Mesh quality check failed: %s", e)

        return result
