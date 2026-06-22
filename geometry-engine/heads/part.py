import torch
import torch.nn as nn
from typing import Dict, Any, List
import uuid

from heads.base import GeometryHeadBase
from graph.ggl import GGLNode

class PartHead(GeometryHeadBase):
    """
    Predicts coarse semantic parts from intermediate features.
    This acts as the top level of the hierarchical geometry graph.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Simple MLP for demonstration. In reality, this might be a 
        # Transformer decoder or a 3D UNet head depending on feature topology.
        self.net = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10) # Assume 10 canonical part classes for now
        )
        self.classes = [
            "Main Body", "Handle", "Leg", "Lid", "Base", 
            "Mounting", "Shaft", "Flange", "Rib", "Other"
        ]

    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Expects features of shape [B, N, hidden_dim]
        Returns part logits of shape [B, N, num_classes]
        """
        logits = self.net(features)
        probs = torch.softmax(logits, dim=-1)
        return {"part_probs": probs}

    def to_ggl_nodes(self, predictions: Dict[str, torch.Tensor], threshold: float = 0.5) -> List[GGLNode]:
        probs = predictions["part_probs"]
        nodes = []
        
        # For demonstration, we just take max across the sequence
        # In a real scenario, we'd cluster tokens to form distinct parts
        B = probs.shape[0]
        for b in range(B):
            max_probs, class_idx = torch.max(probs[b], dim=-1) # [N]
            
            # Very naive clustering: just find unique predicted classes above threshold
            for c in range(len(self.classes)):
                mask = (class_idx == c) & (max_probs > threshold)
                if mask.any():
                    # Average confidence for this part
                    conf = max_probs[mask].mean().item()
                    
                    node = GGLNode(
                        node_id=f"part_{uuid.uuid4().hex[:8]}",
                        type="Part",
                        semantic_label=self.classes[c],
                        confidence=conf,
                        parameters={"token_count": mask.sum().item()}
                    )
                    nodes.append(node)
                    
        return nodes
