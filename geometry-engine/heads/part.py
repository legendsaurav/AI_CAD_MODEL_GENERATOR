"""
heads/part.py - Part Semantic Prediction Head
===============================================
Predicts coarse semantic parts from DiT hidden state features.
Acts as the top level of the hierarchical geometry graph:

    Part -> Surface -> Primitive

Upgraded in Phase 9 to include:
  - Bounding box regression per part
  - Improved GGL node generation with spatial parameters
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

logger = logging.getLogger("geometry_engine.heads.part")

PART_CLASSES = [
    "Main Body", "Handle", "Leg", "Lid", "Base",
    "Mounting", "Shaft", "Flange", "Rib", "Other",
]


class PartHead(GeometryHeadBase):
    """Predicts coarse semantic parts and their bounding boxes from features.

    Input: features [B, N, D] where D = hidden_dim (1024)
    Output: part probabilities [B, N, 10] and bbox params [B, N, 6]
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Part classification
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, len(PART_CLASSES)),
        )

        # Bounding box regression: center(3) + size(3) = 6
        self.bbox_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(256, 6),
        )

        self.classes = PART_CLASSES
        logger.info(
            "PartHead initialized: %d classes, hidden_dim=%d",
            len(self.classes), self.hidden_dim,
        )

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Predict part types and bounding boxes.

        Args:
            features: [B, N, D] - fused DiT hidden state features.

        Returns:
            dict with keys:
                - part_probs: [B, N, num_classes]
                - bbox_params: [B, N, 6] - (cx, cy, cz, sx, sy, sz)
        """
        features = features.float()

        logits = self.net(features)
        probs = F.softmax(logits, dim=-1)

        bbox_params = self.bbox_head(features)
        # Ensure sizes are positive via softplus
        bbox_params = torch.cat([
            bbox_params[..., :3],                          # center (unconstrained)
            F.softplus(bbox_params[..., 3:6]) + 0.001,     # size (positive)
        ], dim=-1)

        return {
            "part_probs": probs,
            "bbox_params": bbox_params,
        }

    def to_ggl_nodes(
        self,
        predictions: Dict[str, torch.Tensor],
        threshold: float = 0.5,
    ) -> List[GGLNode]:
        """Convert predictions to GGL part nodes with spatial information."""
        probs = predictions["part_probs"]
        bbox = predictions["bbox_params"]
        nodes: List[GGLNode] = []

        B = probs.shape[0]
        for b in range(B):
            max_probs, class_idx = torch.max(probs[b], dim=-1)  # [N]

            for c in range(len(self.classes)):
                mask = (class_idx == c) & (max_probs > threshold)
                if not mask.any():
                    continue

                conf = float(max_probs[mask].mean())

                # Average bounding box across tokens in this cluster
                avg_bbox = bbox[b][mask].mean(dim=0)  # [6]

                params: Dict[str, Any] = {
                    "center": [round(float(avg_bbox[i]), 4) for i in range(3)],
                    "size": [round(float(avg_bbox[i + 3]), 4) for i in range(3)],
                    "token_count": int(mask.sum()),
                }

                node = GGLNode(
                    node_id=f"part_{uuid.uuid4().hex[:8]}",
                    type="Part",
                    semantic_label=self.classes[c],
                    confidence=round(conf, 4),
                    parameters=params,
                )
                nodes.append(node)

        return nodes
