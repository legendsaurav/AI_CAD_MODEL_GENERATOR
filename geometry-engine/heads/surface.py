"""
heads/surface.py - Surface Patch Prediction Head
==================================================
Predicts continuous surface patches and their differential geometry
from DiT hidden state features. Acts as a mid-level in the hierarchy:

    Part -> Surface -> Primitive

Upgraded in Phase 9 to include:
  - Surface normal regression (unit vector)
  - Principal curvature estimation (k1, k2)
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

logger = logging.getLogger("geometry_engine.heads.surface")

SURFACE_TYPES = ["Planar", "Cylindrical", "Spherical", "Freeform"]


class SurfaceHead(GeometryHeadBase):
    """Predicts surface type, normals, and curvatures from DiT features.

    Input: features [B, N, D]
    Output: surface probabilities, normal vectors, and principal curvatures
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Surface type classification
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, len(SURFACE_TYPES)),
        )

        # Surface normal regression -> unit vector [B, N, 3]
        self.normal_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )

        # Principal curvature estimation -> (k1, k2) [B, N, 2]
        self.curvature_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

        self.surface_types = SURFACE_TYPES
        logger.info(
            "SurfaceHead initialized: %d types, hidden_dim=%d",
            len(self.surface_types), self.hidden_dim,
        )

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Predict surface types, normals, and curvatures.

        Args:
            features: [B, N, D] - fused DiT hidden state features.

        Returns:
            dict with keys:
                - surface_probs: [B, N, 4]
                - normals: [B, N, 3] (unit vectors)
                - curvatures: [B, N, 2] (k1, k2 principal curvatures)
        """
        features = features.float()

        logits = self.net(features)
        probs = F.softmax(logits, dim=-1)

        # Predict and normalize surface normals to unit vectors
        raw_normals = self.normal_head(features)
        normals = F.normalize(raw_normals, dim=-1)

        # Predict principal curvatures
        curvatures = self.curvature_head(features)

        return {
            "surface_probs": probs,
            "normals": normals,
            "curvatures": curvatures,
        }

    def to_ggl_nodes(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> List[GGLNode]:
        """Convert predictions to GGL surface nodes with geometry info."""
        probs = predictions["surface_probs"]
        normals = predictions["normals"]
        curvatures = predictions["curvatures"]
        nodes: List[GGLNode] = []

        B = probs.shape[0]
        for b in range(B):
            max_probs, class_idx = torch.max(probs[b], dim=-1)

            for c in range(len(self.surface_types)):
                mask = (class_idx == c) & (max_probs > threshold)
                if not mask.any():
                    continue

                conf = float(max_probs[mask].mean())

                # Average normals and curvatures over the cluster
                avg_normal = normals[b][mask].mean(dim=0)  # [3]
                # Re-normalize the averaged normal
                avg_normal = F.normalize(avg_normal, dim=0)

                avg_curv = curvatures[b][mask].mean(dim=0)  # [2]

                params: Dict[str, Any] = {
                    "patch_size": int(mask.sum()),
                    "normal": [round(float(avg_normal[i]), 6) for i in range(3)],
                    "curvature_k1": round(float(avg_curv[0]), 6),
                    "curvature_k2": round(float(avg_curv[1]), 6),
                    "mean_curvature": round(float((avg_curv[0] + avg_curv[1]) / 2), 6),
                    "gaussian_curvature": round(float(avg_curv[0] * avg_curv[1]), 6),
                }

                node = GGLNode(
                    node_id=f"surface_{uuid.uuid4().hex[:8]}",
                    type="Surface",
                    semantic_label=self.surface_types[c],
                    confidence=round(conf, 4),
                    parameters=params,
                )
                nodes.append(node)

        return nodes
