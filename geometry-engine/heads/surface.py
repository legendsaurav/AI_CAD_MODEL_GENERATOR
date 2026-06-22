import torch
import torch.nn as nn
from typing import Dict, Any, List
import uuid

from heads.base import GeometryHeadBase
from graph.ggl import GGLNode

class SurfaceHead(GeometryHeadBase):
    """
    Predicts distinct continuous surface patches from intermediate features.
    Acts as a lower-level hierarchical representation belonging to Parts.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # E.g., Planar, Cylindrical, Spherical, Freeform
        )
        self.surface_types = ["Planar", "Cylindrical", "Spherical", "Freeform"]

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        logits = self.net(features)
        probs = torch.softmax(logits, dim=-1)
        return {"surface_probs": probs}

    def to_ggl_nodes(self, predictions: Dict[str, torch.Tensor], threshold: float = 0.5) -> List[GGLNode]:
        probs = predictions["surface_probs"]
        nodes = []
        
        B = probs.shape[0]
        for b in range(B):
            max_probs, class_idx = torch.max(probs[b], dim=-1)
            
            for c in range(len(self.surface_types)):
                mask = (class_idx == c) & (max_probs > threshold)
                if mask.any():
                    conf = max_probs[mask].mean().item()
                    
                    node = GGLNode(
                        node_id=f"surface_{uuid.uuid4().hex[:8]}",
                        type="Surface",
                        semantic_label=self.surface_types[c],
                        confidence=conf,
                        parameters={"patch_size": mask.sum().item()}
                    )
                    nodes.append(node)
                    
        return nodes
