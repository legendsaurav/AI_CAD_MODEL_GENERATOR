"""
heads/primitive.py - Primitive Geometry Prediction Head
========================================================
Predicts primitive types (Cylinder, Box, Sphere, Cone, Plane, Torus)
and regresses their geometric parameters directly from DiT hidden states.

This is the most critical prediction head in the Geometry Engine - it
transforms latent representations into concrete geometric primitives
that form the foundation of the GGL.

Architecture:
    Stage 1: Type classification (6-way softmax)
    Stage 2: Per-type parameter regression (separate MLP per type)
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from heads.base import GeometryHeadBase
from graph.ggl import GGLNode

logger = logging.getLogger("geometry_engine.heads.primitive")

# -- Primitive type definitions ------------------------------------------ #

PRIMITIVE_TYPES = ["Cylinder", "Box", "Sphere", "Cone", "Plane", "Torus"]

# Number of parameters to regress for each primitive type
PARAM_COUNTS = {
    "Cylinder": 8,   # center(3), axis(3), radius(1), height(1)
    "Box": 10,       # center(3), dimensions(3), rotation_quat(4)
    "Sphere": 4,     # center(3), radius(1)
    "Cone": 8,       # apex(3), axis(3), half_angle(1), height(1)
    "Plane": 6,      # point(3), normal(3)
    "Torus": 8,      # center(3), axis(3), major_radius(1), minor_radius(1)
}

# Parameter name mappings for GGL export
PARAM_NAMES = {
    "Cylinder": ["center_x", "center_y", "center_z", "axis_x", "axis_y", "axis_z", "radius", "height"],
    "Box": ["center_x", "center_y", "center_z", "dim_x", "dim_y", "dim_z", "qw", "qx", "qy", "qz"],
    "Sphere": ["center_x", "center_y", "center_z", "radius"],
    "Cone": ["apex_x", "apex_y", "apex_z", "axis_x", "axis_y", "axis_z", "half_angle", "height"],
    "Plane": ["point_x", "point_y", "point_z", "normal_x", "normal_y", "normal_z"],
    "Torus": ["center_x", "center_y", "center_z", "axis_x", "axis_y", "axis_z", "major_radius", "minor_radius"],
}


class PrimitiveHead(GeometryHeadBase):
    """Predicts primitive types and their geometric parameters from DiT features.

    Input: features [B, N, D] where D = hidden_dim (1024)
    Output: dict with type probabilities and per-type parameter tensors
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        num_types = len(PRIMITIVE_TYPES)

        # Stage 1: Type classification
        self.type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_types),
        )

        # Stage 2: Per-type parameter regression heads
        self.param_heads = nn.ModuleDict()
        for ptype, n_params in PARAM_COUNTS.items():
            self.param_heads[ptype] = nn.Sequential(
                nn.Linear(self.hidden_dim, 256),
                nn.GELU(),
                nn.Dropout(config.get("dropout", 0.1)),
                nn.Linear(256, n_params),
            )

        logger.info(
            "PrimitiveHead initialized: %d types, hidden_dim=%d",
            num_types, self.hidden_dim,
        )

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Predict primitive types and parameters.

        Args:
            features: [B, N, D] - fused DiT hidden state features.

        Returns:
            dict with keys:
                - type_logits: [B, N, num_types]
                - type_probs: [B, N, num_types] (softmax of logits)
                - params: dict[str, Tensor] - per-type parameters [B, N, n_params]
        """
        features = features.float()  # ensure float32 for regression stability

        type_logits = self.type_classifier(features)
        type_probs = F.softmax(type_logits, dim=-1)

        params = {}
        for ptype, head in self.param_heads.items():
            params[ptype] = head(features)

        return {
            "type_logits": type_logits,
            "type_probs": type_probs,
            "params": params,
        }

    def to_ggl_nodes(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> List[GGLNode]:
        """Convert predictions to GGL primitive nodes.

        Groups adjacent high-confidence tokens into clusters, assigns
        each cluster a type and averaged parameters.
        """
        type_probs = predictions["type_probs"]
        params = predictions["params"]
        nodes: List[GGLNode] = []

        B = type_probs.shape[0]

        for b in range(B):
            max_probs, class_idx = torch.max(type_probs[b].detach(), dim=-1)  # [N]

            for c, ptype in enumerate(PRIMITIVE_TYPES):
                mask = (class_idx == c) & (max_probs > threshold)
                if not mask.any():
                    continue

                # Average confidence
                conf = float(max_probs[mask].mean().item())

                # Average the regression parameters for this cluster
                type_params = params[ptype][b].detach()  # [N, n_params]
                avg_params = type_params[mask].mean(dim=0)  # [n_params]

                # Build named parameter dictionary
                param_names = PARAM_NAMES[ptype]
                param_dict = {
                    name: round(float(avg_params[i]), 6)
                    for i, name in enumerate(param_names)
                }

                # Restructure into standard GGL format
                ggl_params = self._structure_params(ptype, param_dict)
                ggl_params["token_count"] = int(mask.sum())

                node = GGLNode(
                    node_id=f"prim_{uuid.uuid4().hex[:8]}",
                    type=ptype,
                    semantic_label=ptype.lower(),
                    confidence=round(conf, 4),
                    parameters=ggl_params,
                )
                nodes.append(node)

        return nodes

    @staticmethod
    def _structure_params(ptype: str, flat: Dict[str, float]) -> Dict[str, Any]:
        """Convert flat named params into structured GGL parameters."""
        result: Dict[str, Any] = {}

        if ptype == "Cylinder":
            result["center"] = [flat["center_x"], flat["center_y"], flat["center_z"]]
            result["axis"] = [flat["axis_x"], flat["axis_y"], flat["axis_z"]]
            result["radius"] = flat["radius"]
            result["height"] = flat["height"]

        elif ptype == "Box":
            result["center"] = [flat["center_x"], flat["center_y"], flat["center_z"]]
            result["dimensions"] = [flat["dim_x"], flat["dim_y"], flat["dim_z"]]
            result["rotation"] = [flat["qw"], flat["qx"], flat["qy"], flat["qz"]]

        elif ptype == "Sphere":
            result["center"] = [flat["center_x"], flat["center_y"], flat["center_z"]]
            result["radius"] = flat["radius"]

        elif ptype == "Cone":
            result["apex"] = [flat["apex_x"], flat["apex_y"], flat["apex_z"]]
            result["axis"] = [flat["axis_x"], flat["axis_y"], flat["axis_z"]]
            result["half_angle"] = flat["half_angle"]
            result["height"] = flat["height"]

        elif ptype == "Plane":
            result["point"] = [flat["point_x"], flat["point_y"], flat["point_z"]]
            result["normal"] = [flat["normal_x"], flat["normal_y"], flat["normal_z"]]

        elif ptype == "Torus":
            result["center"] = [flat["center_x"], flat["center_y"], flat["center_z"]]
            result["axis"] = [flat["axis_x"], flat["axis_y"], flat["axis_z"]]
            result["major_radius"] = flat["major_radius"]
            result["minor_radius"] = flat["minor_radius"]

        result["units"] = "mm"
        return result
