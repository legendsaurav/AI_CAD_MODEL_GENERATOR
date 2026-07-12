"""
cal/generator.py — CAL Generator with Confidence Propagation
===============================================================
Translates the optimized Construction Graph into the universal CAL format.
Now includes:
  - Confidence propagation from source GGL nodes
  - Source GGL node traceability (source_ggl_node_id)
  - PlanningTrace emission
"""
from typing import List
from construction.graph import ConstructionGraph
from cal.schema import (
    CALAction, CreateSketchAction, DrawCircleAction, ExtrudeAction, RevolveAction
)
from reasoning.graph import ReasonGraphGenerator


class CALGenerator:
    """
    Translates the optimized Construction Graph into the universal CAL format.
    Propagates confidence and source GGL node IDs for full traceability.
    """

    @staticmethod
    def generate(cg: ConstructionGraph) -> List[CALAction]:
        actions = []
        sequence = cg.get_sequence()

        for node in sequence:
            reasoning = ReasonGraphGenerator.generate_reasoning(node)

            # Extract confidence and source node from construction node parameters
            confidence = node.parameters.get("confidence", 1.0)
            source_node_id = node.parameters.get("source_ggl_node_id", None)

            if node.operation_type == "create_sketch":
                profile = node.parameters.get("profile", {})
                plane = profile.get("plane", "XY")
                action = CreateSketchAction(
                    action_id=node.node_id,
                    plane=plane,
                    reasoning=reasoning,
                    confidence=confidence,
                    source_ggl_node_id=source_node_id,
                )
                actions.append(action)

                # Create drawing actions for entities
                for entity in profile.get("entities", []):
                    entity_confidence = entity.get("confidence", confidence)
                    if entity.get("entity_type") == "circle":
                        actions.append(DrawCircleAction(
                            action_id=entity["id"],
                            sketch_id=node.node_id,
                            center=entity["center"],
                            radius=entity["radius"],
                            reasoning=ReasonGraphGenerator.generate_reasoning_for_entity(entity, node),
                            confidence=entity_confidence,
                            source_ggl_node_id=source_node_id,
                        ))
                    elif entity.get("entity_type") == "line":
                        pass  # V1: Lines handled as rectangle boundaries

            elif node.operation_type == "extrude":
                actions.append(ExtrudeAction(
                    action_id=node.node_id,
                    sketch_id=node.parameters.get("sketch_id", ""),
                    depth=node.parameters.get("depth", 10.0),
                    is_cut=node.parameters.get("is_cut", False),
                    reasoning=reasoning,
                    confidence=confidence,
                    source_ggl_node_id=source_node_id,
                ))
            elif node.operation_type == "revolve":
                actions.append(RevolveAction(
                    action_id=node.node_id,
                    sketch_id=node.parameters.get("sketch_id", ""),
                    axis=node.parameters.get("axis", [0, 1, 0]),
                    reasoning=reasoning,
                    confidence=confidence,
                    source_ggl_node_id=source_node_id,
                ))

        return actions
