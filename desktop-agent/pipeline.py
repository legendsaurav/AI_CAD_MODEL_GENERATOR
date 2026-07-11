"""
desktop-agent/pipeline.py — Desktop Agent End-to-End Pipeline
===============================================================
Orchestrates the full desktop agent workflow:

    CAL JSON → Parse → State Machine → Execute → Verify → Report

This is the main entry point for the desktop agent.
"""
import json
import logging
import sys
import os
import time
from typing import Optional

_AGENT_ROOT = os.path.normpath(os.path.dirname(__file__))
_SHARED_SCHEMAS = os.path.normpath(os.path.join(_AGENT_ROOT, "..", "shared-schemas"))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

from cal_consumer import CALConsumer
from state_machine import ExecutionStateMachine
from executor import FreeCADExecutor, BaseExecutor
from verifier import ExecutionVerifier
from reporter import ExecutionReporter

logger = logging.getLogger("desktop_agent.pipeline")


class DesktopAgentPipeline:
    """
    End-to-end desktop agent pipeline.

    Usage:
        agent = DesktopAgentPipeline(target_system="FreeCAD")
        report = agent.run("path/to/output.cal.json")
        # report.exported_mesh_path can be fed to the refinement loop
    """

    def __init__(self, target_system: str = "FreeCAD", output_dir: str = "output"):
        self.target_system = target_system
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.consumer = CALConsumer()
        self.state_machine = ExecutionStateMachine()
        self.verifier = ExecutionVerifier(os.path.join(output_dir, "verification"))

        # Select executor based on target system
        if target_system == "FreeCAD":
            self.executor: BaseExecutor = FreeCADExecutor()
        elif target_system == "SolidWorks":
            from solidworks_executor import SolidWorksExecutor
            self.executor = SolidWorksExecutor()
        else:
            raise ValueError(f"Unsupported target system: {target_system}. Supported: FreeCAD, SolidWorks")

    def run(self, cal_path: str) -> dict:
        """
        Execute a complete CAL document in the target CAD system.

        Args:
            cal_path: Path to the CAL JSON file

        Returns:
            ExecutionReport as a dict
        """
        logger.info(f"{'═'*60}")
        logger.info(f"  Desktop Agent — Executing CAL in {self.target_system}")
        logger.info(f"{'═'*60}")

        # Step 1: Parse CAL
        doc = self.consumer.parse_file(cal_path)
        logger.info(f"Parsed {len(doc.actions)} actions from {cal_path}")

        # Step 2: Initialize state machine
        action_ids = [(a.action_id, a.action_type) for a in doc.actions]
        self.state_machine.load_actions(action_ids)

        # Step 3: Connect to CAD system
        if not self.executor.connect():
            logger.error("Failed to connect to CAD system.")
            return {"error": "Connection failed"}

        # Step 4: Execute actions
        action_results = []
        for action in doc.actions:
            params = action.model_dump()
            t0 = time.time()

            result = self.executor.execute_action(
                action_type=action.action_type,
                action_id=action.action_id,
                params=params,
            )

            result["action_id"] = action.action_id
            result["action_type"] = action.action_type
            result["execution_time_ms"] = (time.time() - t0) * 1000

            if result.get("success"):
                self.state_machine.advance()
            else:
                self.state_machine.fail_current(result.get("error", "Unknown error"))

            action_results.append(result)

        # Step 5: Verify
        summary = self.state_machine.get_summary()
        verification = self.verifier.verify_execution(
            executor=self.executor,
            expected_actions=summary["total"],
            completed_actions=summary["completed"],
            action_results=action_results,
        )

        # Step 6: Generate report
        report = ExecutionReporter.generate(
            target_system=self.target_system,
            action_results=action_results,
            verification=verification,
        )

        # Save report
        report_path = os.path.join(self.output_dir, "execution_report.json")
        with open(report_path, "w") as f:
            f.write(report.to_json())
        logger.info(f"Execution report saved to {report_path}")

        # Step 7: Disconnect
        self.executor.disconnect()

        return report.model_dump()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Desktop Agent — Execute CAL in CAD software")
    parser.add_argument("--cal", required=True, help="Path to CAL JSON file")
    parser.add_argument("--target", default="FreeCAD", help="Target CAD system")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    agent = DesktopAgentPipeline(target_system=args.target, output_dir=args.output)
    result = agent.run(args.cal)

    print(f"\n✅ Execution complete. Success rate: {result.get('successful_actions', 0)}/{result.get('total_actions', 0)}")
