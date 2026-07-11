"""
heads/symmetry.py - Symmetry Detection Head
=============================================
Detects bilateral and rotational symmetry from DiT hidden states.

Symmetry information is critical for engineering intent recovery:
  - Bilateral symmetry -> Mirror features in CAD
  - Rotational symmetry -> Circular patterns in CAD
  - Combined -> Both mirror + pattern

Outputs symmetry nodes and edges for the GGL graph.
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

logger = logging.getLogger("geometry_engine.heads.symmetry")

SYMMETRY_TYPES = ["None", "Bilateral", "Rotational", "Both"]


class SymmetryHead(GeometryHeadBase):
    """Detects symmetry properties from DiT hidden state features.

    Input: features [B, N, D]
    Output: symmetry type probabilities + regressed plane/axis parameters
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Symmetry type classification
        self.type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(256, len(SYMMETRY_TYPES)),
        )

        # Mirror plane regression: point(3) + normal(3) = 6
        self.mirror_regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )

        # Rotation axis regression: point(3) + axis(3) + order(1) = 7
        self.rotation_regressor = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 7),
        )

        logger.info("SymmetryHead initialized: hidden_dim=%d", self.hidden_dim)

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Predict symmetry types and parameters.

        Args:
            features: [B, N, D] - fused DiT hidden state features.

        Returns:
            dict with keys:
                - symmetry_probs: [B, N, 4]
                - mirror_plane: [B, N, 6] - (point_x,y,z, normal_x,y,z)
                - rotation_axis: [B, N, 7] - (point_x,y,z, axis_x,y,z, order)
        """
        features = features.float()

        sym_logits = self.type_classifier(features)
        sym_probs = F.softmax(sym_logits, dim=-1)

        mirror_raw = self.mirror_regressor(features)
        # Normalize the normal vector (last 3 components)
        mirror_plane = torch.cat([
            mirror_raw[..., :3],
            F.normalize(mirror_raw[..., 3:6], dim=-1),
        ], dim=-1)

        rotation_raw = self.rotation_regressor(features)
        # Normalize the axis vector (components 3:6), keep order (component 6)
        rotation_axis = torch.cat([
            rotation_raw[..., :3],
            F.normalize(rotation_raw[..., 3:6], dim=-1),
            rotation_raw[..., 6:7],
        ], dim=-1)

        return {
            "symmetry_probs": sym_probs,
            "mirror_plane": mirror_plane,
            "rotation_axis": rotation_axis,
        }

    def to_ggl_nodes(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> List[GGLNode]:
        """Convert symmetry predictions to GGL nodes.

        Only creates nodes for detected symmetry (not 'None' type).
        Averages parameters across all tokens that agree on symmetry type.
        """
        sym_probs = predictions["symmetry_probs"]
        mirror_plane = predictions["mirror_plane"]
        rotation_axis = predictions["rotation_axis"]
        nodes: List[GGLNode] = []

        B = sym_probs.shape[0]

        for b in range(B):
            max_probs, class_idx = torch.max(sym_probs[b], dim=-1)  # [N]

            for c, sym_type in enumerate(SYMMETRY_TYPES):
                if sym_type == "None":
                    continue  # skip non-symmetry predictions

                mask = (class_idx == c) & (max_probs > threshold)
                if not mask.any():
                    continue

                conf = float(max_probs[mask].mean())
                params: Dict[str, Any] = {"symmetry_type": sym_type}

                # Add mirror plane data for Bilateral or Both
                if sym_type in ("Bilateral", "Both"):
                    avg_mirror = mirror_plane[b][mask].mean(dim=0)
                    params["mirror_point"] = [
                        round(float(avg_mirror[i]), 6) for i in range(3)
                    ]
                    params["mirror_normal"] = [
                        round(float(avg_mirror[i + 3]), 6) for i in range(3)
                    ]

                # Add rotation axis data for Rotational or Both
                if sym_type in ("Rotational", "Both"):
                    avg_rot = rotation_axis[b][mask].mean(dim=0)
                    params["rotation_point"] = [
                        round(float(avg_rot[i]), 6) for i in range(3)
                    ]
                    params["rotation_axis"] = [
                        round(float(avg_rot[i + 3]), 6) for i in range(3)
                    ]
                    params["rotation_order"] = max(2, round(float(avg_rot[6])))

                params["token_count"] = int(mask.sum())

                node = GGLNode(
                    node_id=f"sym_{uuid.uuid4().hex[:8]}",
                    type="Symmetry",
                    semantic_label=sym_type,
                    confidence=round(conf, 4),
                    parameters=params,
                )
                nodes.append(node)

        return nodes
