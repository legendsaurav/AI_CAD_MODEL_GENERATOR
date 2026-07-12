"""
desktop-agent/reporter.py — Execution Report Generator
========================================================
Generates ExecutionReport documents for the refinement loop
and the Clicky Tutor.
"""
import logging
import sys
import os
from typing import List, Dict, Any

_AGENT_ROOT = os.path.normpath(os.path.dirname(__file__))
_SHARED_SCHEMAS = os.path.normpath(os.path.join(_AGENT_ROOT, "..", "shared-schemas"))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

from shared_schemas.execution_report import ExecutionReport, ActionResult  # noqa: E402

logger = logging.getLogger("desktop_agent.reporter")


class ExecutionReporter:
    """
    Generates structured ExecutionReport documents from the
    state machine and verification results.
    """

    @staticmethod
    def generate(
        target_system: str,
        action_results: List[Dict[str, Any]],
        verification: Dict[str, Any],
    ) -> ExecutionReport:
        """
        Generates a complete ExecutionReport.

        Args:
            target_system: "FreeCAD", "SolidWorks", etc.
            action_results: List of per-action result dicts
            verification: Verification results from ExecutionVerifier

        Returns:
            ExecutionReport
        """
        results = []
        for ar in action_results:
            results.append(ActionResult(
                action_id=ar.get("action_id", ""),
                action_type=ar.get("action_type", ""),
                status="success" if ar.get("success", False) else "failed",
                error_message=ar.get("error"),
                cad_feature_id=ar.get("feature_id"),
                execution_time_ms=ar.get("execution_time_ms", 0.0),
            ))

        total = len(results)
        successful = sum(1 for r in results if r.status == "success")
        failed = total - successful

        report = ExecutionReport(
            target_cad_system=target_system,
            action_results=results,
            total_actions=total,
            successful_actions=successful,
            failed_actions=failed,
            exported_mesh_path=verification.get("mesh_path"),
            final_screenshot_path=verification.get("screenshot_path"),
        )

        logger.info(
            f"Execution report: {successful}/{total} actions succeeded "
            f"({report.success_rate:.0%})"
        )
        return report
