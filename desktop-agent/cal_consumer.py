"""
desktop-agent/cal_consumer.py — CAL Document Parser
=====================================================
Parses and validates incoming CAL JSON documents from the cad-planner.
"""
import json
import logging
import sys
import os
from typing import List

# Add shared-schemas to path
_AGENT_ROOT = os.path.normpath(os.path.dirname(__file__))
_SHARED_SCHEMAS = os.path.normpath(os.path.join(_AGENT_ROOT, "..", "shared-schemas"))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

from shared_schemas.cal_schema import CALDocument, CALAction

logger = logging.getLogger("desktop_agent.consumer")


class CALConsumer:
    """
    Ingests CAL JSON and provides the ordered action sequence
    to the execution engine.
    """

    def parse(self, cal_json: str) -> CALDocument:
        """
        Parses a CAL JSON string into a validated CALDocument.

        Raises:
            ValueError: If the JSON is invalid or schema validation fails.
        """
        try:
            doc = CALDocument.from_json(cal_json)
            logger.info(
                f"Parsed CAL document: version={doc.version}, "
                f"{len(doc.actions)} actions"
            )
            self._validate_action_order(doc)
            return doc
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid CAL JSON: {e}")

    def parse_file(self, cal_path: str) -> CALDocument:
        """Parses a CAL JSON file."""
        with open(cal_path, "r") as f:
            return self.parse(f.read())

    def _validate_action_order(self, doc: CALDocument):
        """Validates that sketches are created before they are referenced."""
        created_sketches = set()
        for action in doc.actions:
            if action.action_type == "create_sketch":
                created_sketches.add(action.action_id)
            elif hasattr(action, "sketch_id"):
                if action.sketch_id not in created_sketches:
                    logger.warning(
                        f"Action {action.action_id} references sketch {action.sketch_id} "
                        f"before it was created."
                    )
