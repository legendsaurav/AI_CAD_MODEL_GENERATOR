"""
reasoning/graph.py — Production Reason Graph Generator
========================================================
Generates a proper ReasonGraph (from shared-schemas) for every
CAD construction operation.

Every CAD action must explain:
  - Purpose: what engineering goal this achieves
  - Rationale: why this approach was chosen
  - Dependencies: what other operations this depends on
  - Alternatives considered and why they were rejected
  - Confidence from the source GGL node
  - Supporting geometry references

The ReasonGraph is emitted alongside the CAL document
to provide full explainability of the design process.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from construction.graph import ConstructionGraph, ConstructionNode

# Import authoritative ReasonGraph from shared-schemas
_PLANNER_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.normpath(os.path.join(_PLANNER_ROOT, "..", "shared-schemas"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from shared_schemas.reason_graph import (  # noqa: E402
    RejectedAlternative,
    ReasonNode,
    ReasonEdge,
    ReasonGraph,
)

# Also import ActionReasoning for backward-compatible API
from cal.schema import ActionReasoning  # noqa: E402

logger = logging.getLogger("cad_planner.reasoning.graph")


# ---------------------------------------------------------------------------
# Engineering knowledge base for reasoning
# ---------------------------------------------------------------------------

_OPERATION_PURPOSE: Dict[str, str] = {
    "create_sketch": "Define 2D profile geometry on a reference plane",
    "extrude": "Generate 3D solid body by extruding a sketch profile along a direction",
    "revolve": "Generate 3D solid of revolution by rotating a sketch profile around an axis",
    "fillet": "Apply smooth radius transition at sharp edges for stress relief and aesthetics",
    "chamfer": "Apply angled edge break for assembly clearance or deburring",
    "pattern": "Replicate feature instances for symmetry or repetitive design intent",
    "mirror": "Create symmetric copy about a reference plane",
    "shell": "Hollow out a solid body to reduce weight while maintaining structural integrity",
    "draft": "Apply taper angle to faces for mold release in injection molding",
    "loft": "Create transitional solid between two or more dissimilar cross-sections",
    "sweep": "Generate solid by moving a profile along a guide curve",
}

_OPERATION_RATIONALE: Dict[str, str] = {
    "create_sketch": "Foundation for feature-based modeling — all features begin with a 2D profile",
    "extrude": "Most robust and editable feature type — preferred for prismatic geometry",
    "revolve": "Natural choice for axisymmetric geometry — single profile captures full revolution",
    "fillet": "Engineering best practice — reduces stress concentration at sharp corners",
    "chamfer": "Standard manufacturing preparation — eases assembly and prevents burrs",
    "pattern": "Maintains design intent — changes propagate to all instances",
    "mirror": "Enforces geometric symmetry — single definition controls both sides",
    "shell": "Weight reduction without structural compromise — uniform wall thickness",
    "draft": "Manufacturing requirement for injection molding — ensures mold release",
    "loft": "Required when cross-section changes along path — more complex than extrude",
    "sweep": "Required when profile follows a curved path — more complex than extrude",
}

_ALTERNATIVES_MAP: Dict[str, List[Dict[str, str]]] = {
    "extrude": [
        {"alternative": "revolve", "reason": "Profile is not axisymmetric"},
        {"alternative": "loft", "reason": "Cross-section does not vary along path"},
    ],
    "revolve": [
        {"alternative": "extrude", "reason": "Geometry is axisymmetric, not prismatic"},
        {"alternative": "sweep", "reason": "Axis is straight, sweep adds unnecessary complexity"},
    ],
    "fillet": [
        {"alternative": "chamfer", "reason": "Rounded transition preferred for stress relief"},
    ],
    "chamfer": [
        {"alternative": "fillet", "reason": "Chamfer preferred for manufacturing simplicity"},
    ],
}


class ReasonGraphGenerator:
    """
    Generates engineering reasoning for CAD operations.

    Provides two APIs:
      - generate_reasoning() → ActionReasoning (backward-compatible)
      - generate_full_graph() → ReasonGraph (production, from shared-schemas)
    """

    # ------------------------------------------------------------------
    # Backward-compatible API
    # ------------------------------------------------------------------

    @staticmethod
    def generate_reasoning(node: ConstructionNode) -> ActionReasoning:
        """
        Generate ActionReasoning for a single construction node.

        Args:
            node: The construction operation to explain.

        Returns:
            ActionReasoning with purpose, rationale, dependencies,
            and alternatives_considered.
        """
        op = node.operation_type
        purpose = _OPERATION_PURPOSE.get(op, f"Execute {op} operation")
        rationale = _OPERATION_RATIONALE.get(op, "Deterministic rule engine selection")

        # Contextual refinement based on parameters
        if op == "extrude" and node.parameters.get("is_cut", False):
            purpose = "Remove material by extruding a cut profile through the body"
            rationale = "Subtractive feature — pocket or through-hole creation"

        # Extract dependencies from the construction graph context
        depends_on = []
        if node.parameters.get("sketch_id"):
            depends_on.append(node.parameters["sketch_id"])
        if node.parameters.get("source_ggl_node_id"):
            depends_on.append(node.parameters["source_ggl_node_id"])

        # Build alternatives considered
        alternatives = []
        for alt_info in _ALTERNATIVES_MAP.get(op, []):
            alternatives.append(
                f"{alt_info['alternative']} (rejected: {alt_info['reason']})"
            )

        return ActionReasoning(
            purpose=purpose,
            rationale=rationale,
            depends_on=depends_on,
            alternatives_considered=alternatives,
        )

    @staticmethod
    def generate_reasoning_for_entity(
        entity: Dict[str, Any], node: ConstructionNode
    ) -> ActionReasoning:
        """Generate reasoning for a sketch entity."""
        entity_type = entity.get("entity_type", "geometry")
        return ActionReasoning(
            purpose=f"Define {entity_type} profile element for {node.operation_type}",
            rationale="Profile boundary geometry derived from GGL primitive parameters",
            depends_on=[node.node_id],
            alternatives_considered=[],
        )

    # ------------------------------------------------------------------
    # Production API: full ReasonGraph
    # ------------------------------------------------------------------

    @classmethod
    def generate_full_graph(
        cls,
        cg: ConstructionGraph,
        ggl_node_map: Optional[Dict[str, Any]] = None,
    ) -> ReasonGraph:
        """
        Generate a complete ReasonGraph for an entire construction sequence.

        Args:
            cg: The construction graph to explain.
            ggl_node_map: Optional mapping of node_id → GGL node data
                         for richer reasoning.

        Returns:
            A shared-schemas ReasonGraph with nodes and edges.
        """
        reason_graph = ReasonGraph()
        sequence = cg.get_sequence()

        for idx, node in enumerate(sequence):
            op = node.operation_type
            confidence = float(node.parameters.get("confidence", 0.8))
            source_ggl = node.parameters.get("source_ggl_node_id")

            # Build supporting geometry references
            supporting = []
            if source_ggl:
                supporting.append(source_ggl)

            # Build rejected alternatives
            rejected_alts = []
            for alt_info in _ALTERNATIVES_MAP.get(op, []):
                rejected_alts.append(RejectedAlternative(
                    alternative=alt_info["alternative"],
                    rejection_reason=alt_info["reason"],
                ))

            # Build dependencies
            deps = []
            if node.parameters.get("sketch_id"):
                deps.append(node.parameters["sketch_id"])

            reason_node = ReasonNode(
                node_id=f"reason_{node.node_id}",
                purpose=_OPERATION_PURPOSE.get(op, f"Execute {op}"),
                rationale=_OPERATION_RATIONALE.get(op, "Rule-based selection"),
                dependencies=deps,
                alternatives_considered=[
                    alt["alternative"]
                    for alt in _ALTERNATIVES_MAP.get(op, [])
                ],
                rejected_alternatives=rejected_alts,
                confidence=confidence,
                supporting_geometry=supporting,
            )
            reason_graph.add_node(reason_node)

            # Add edges from dependencies
            for dep_id in deps:
                dep_reason_id = f"reason_{dep_id}"
                if dep_reason_id in {n.node_id for n in reason_graph.nodes}:
                    reason_graph.add_edge(ReasonEdge(
                        source_id=dep_reason_id,
                        target_id=reason_node.node_id,
                        relation="enables",
                    ))

        logger.info(
            "Generated ReasonGraph: %d nodes, %d edges",
            len(reason_graph.nodes),
            len(reason_graph.edges),
        )

        return reason_graph
