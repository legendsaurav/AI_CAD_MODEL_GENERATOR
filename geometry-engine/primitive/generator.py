"""
primitive/generator.py - Primitive Proposal Generator
BUG FIX: config lookup used wrong key.  The YAML has `primitive.top_k_proposals`
but the constructor was reading `config.get("top_k_proposals")` (missing nesting).
Fixed to fall back gracefully so both flat and nested configs work.
Also: features tensor is accepted but not used (matches real interface contract).
"""
import torch
from typing import Any, Dict, List

from graph.ggl import GGLNode


class PrimitiveProposalGenerator:
    """
    Given a geometry node (e.g. a Part), generates Top-K analytic primitive
    proposals along with initialised parameters.
    """

    SUPPORTED = ["Cylinder", "Cone", "Sphere", "Box", "Plane"]

    def __init__(self, config: Dict[str, Any]):
        # Support both nested `primitive.top_k_proposals` and flat `top_k_proposals`
        prim_cfg = config.get("primitive", config)
        self.top_k = int(prim_cfg.get("top_k_proposals", 3))

    def generate_proposals(
        self, features: torch.Tensor, part_node: GGLNode
    ) -> List[GGLNode]:
        """
        Generates Top-K primitive proposals for the given part node.
        In production a classifier+regressor network replaces the mock below.
        """
        proposals: List[GGLNode] = []
        for i in range(self.top_k):
            prim_type = self.SUPPORTED[i % len(self.SUPPORTED)]
            node = GGLNode(
                node_id=f"prim_{part_node.node_id}_{i}",
                type=prim_type,
                semantic_label=f"Proposal-{i} for {part_node.semantic_label}",
                confidence=round(0.9 - i * 0.1, 2),
                parameters=self._default_parameters(prim_type),
            )
            proposals.append(node)
        return proposals

    # ------------------------------------------------------------------
    @staticmethod
    def _default_parameters(prim_type: str) -> Dict[str, Any]:
        if prim_type == "Cylinder":
            return {"radius": 1.0, "height": 5.0, "axis": [0, 1, 0], "center_x": 0.0, "center_y": 0.0, "center_z": 0.0}
        if prim_type == "Box":
            return {"width": 2.0, "height": 2.0, "depth": 2.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0}
        if prim_type == "Sphere":
            return {"radius": 1.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0}
        if prim_type == "Cone":
            return {"radius": 1.0, "height": 3.0, "center_x": 0.0, "center_y": 0.0, "center_z": 0.0}
        if prim_type == "Plane":
            return {"normal_x": 0.0, "normal_y": 1.0, "normal_z": 0.0, "distance": 0.0}
        return {}
