"""
heads/topology.py - Topology Head
BUG FIX: `mask.fill_diagonal_(False)` is only valid for 2-D boolean tensors.
If sub_N == 1 this still works, but we add a safety guard.

BUG FIX 2: `mask.nonzero()` returns a tensor; iterating over it gives 1-D
index tensors. `idx[0].item()` and `idx[1].item()` are correct but can fail
if mask is empty.  Added `if len(indices) == 0: continue` guard.

BUG FIX 3: The bilinear + classifier operates on float32.  Added explicit
.float() cast so it doesn't crash on bfloat16 features from real models.
"""
import uuid
from typing import Any, Dict, List

import torch
import torch.nn as nn

from graph.ggl import GGLNode
from heads.base import GeometryHeadBase

RELATION_TYPES = ["Adjacent", "Contains", "Intersects"]


class TopologyHead(GeometryHeadBase):
    """
    Predicts pairwise topological relationships between feature tokens.
    Outputs 'Relation' pseudo-nodes that GraphGenerator converts to GGLEdges.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bilinear = nn.Bilinear(self.hidden_dim, self.hidden_dim, 64)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(64, len(RELATION_TYPES)))
        self.relation_types = RELATION_TYPES

    # ------------------------------------------------------------------
    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """features: [B, N, D]"""
        B, N, D = features.shape
        sub_N = min(N, 64)
        f_sub = features[:, :sub_N, :].float()  # ensure float32

        f1 = f_sub.unsqueeze(2).expand(B, sub_N, sub_N, D)
        f2 = f_sub.unsqueeze(1).expand(B, sub_N, sub_N, D)

        bilinear_out = self.bilinear(f1, f2)          # [B, sub_N, sub_N, 64]
        logits = self.classifier(bilinear_out)         # [B, sub_N, sub_N, 3]
        probs = torch.softmax(logits, dim=-1)
        return {"topology_probs": probs}

    # ------------------------------------------------------------------
    def to_ggl_nodes(
        self, predictions: Dict[str, torch.Tensor], threshold: float = 0.5
    ) -> List[GGLNode]:
        probs = predictions["topology_probs"]
        nodes: List[GGLNode] = []
        B, N1, N2, C = probs.shape

        for b in range(B):
            max_probs, class_idx = torch.max(probs[b], dim=-1)  # [N1, N2]

            for c, rel_name in enumerate(self.relation_types):
                mask = (class_idx == c) & (max_probs > threshold)
                if N1 > 1:
                    mask.fill_diagonal_(False)

                indices = mask.nonzero()
                if len(indices) == 0:
                    continue

                for idx in indices:
                    i, j = int(idx[0]), int(idx[1])
                    conf = float(max_probs[i, j])
                    nodes.append(
                        GGLNode(
                            node_id=f"topology_{uuid.uuid4().hex[:8]}",
                            type="Relation",
                            semantic_label=rel_name,
                            confidence=conf,
                            parameters={"source_idx": i, "target_idx": j},
                        )
                    )
        return nodes
