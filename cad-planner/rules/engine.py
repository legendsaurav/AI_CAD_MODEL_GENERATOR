from typing import List, Dict, Any, Optional
from construction.graph import ConstructionNode
from construction.sketch_generator import SketchProfile

class RuleEngine:
    """
    Deterministic mappings of primitives to CAD operations.
    Consumes optimized sketches and intents to produce operation sequences.
    """
    
    @staticmethod
    def apply_rules(primitive_type: str, intent: str, sketch: SketchProfile,
                    ggl_params: Optional[Dict[str, Any]] = None) -> List[ConstructionNode]:
        """
        Maps an intent and a sketch to a specific sequence of construction nodes.
        ggl_params: original GGL node parameters to extract depth/height/angle etc.
        """
        nodes = []
        import uuid
        params = ggl_params or {}
        
        # Every rule starts with creating the sketch
        sketch_node_id = f"op_sketch_{uuid.uuid4().hex[:4]}"
        nodes.append(ConstructionNode(
            node_id=sketch_node_id,
            operation_type="create_sketch",
            parameters={"profile": sketch.model_dump()}
        ))
        
        if intent == "Extrusion":
            depth = params.get("depth", params.get("height", 10.0))
            nodes.append(ConstructionNode(
                node_id=f"op_extrude_{uuid.uuid4().hex[:4]}",
                operation_type="extrude",
                parameters={"sketch_id": sketch_node_id, "depth": depth}
            ))
        elif intent == "Cut Feature":
            depth = params.get("depth", params.get("height", 10.0))
            nodes.append(ConstructionNode(
                node_id=f"op_cut_{uuid.uuid4().hex[:4]}",
                operation_type="extrude",
                parameters={"sketch_id": sketch_node_id, "depth": depth, "is_cut": True}
            ))
        elif intent == "Revolution":
            nodes.append(ConstructionNode(
                node_id=f"op_revolve_{uuid.uuid4().hex[:4]}",
                operation_type="revolve",
                parameters={"sketch_id": sketch_node_id, "axis": [0, 1, 0]}
            ))
        elif intent == "Additive Feature":
            depth = params.get("depth", params.get("height", 10.0))
            nodes.append(ConstructionNode(
                node_id=f"op_add_{uuid.uuid4().hex[:4]}",
                operation_type="extrude",
                parameters={"sketch_id": sketch_node_id, "depth": depth}
            ))
            
        return nodes
