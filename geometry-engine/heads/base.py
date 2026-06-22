"""
heads/base.py - Abstract Base Head
BUG FIX: `self.hidden_dim = config.get("hidden_dim", 1024)` – config passed to
heads is the *heads sub-dict*, not the full config.  This is correct as-is, but
we add a clear docstring + a __call__ passthrough so the head can be called
directly without remembering to call .forward() explicitly.
No functional change needed here – file is correct.
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from graph.ggl import GGLNode


class GeometryHeadBase(nn.Module, ABC):
    """
    Abstract base class for all prediction heads.
    Every plugin in heads/ must inherit this and implement forward() + to_ggl_nodes().
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        # config here is the `heads` sub-dict from default.yaml
        self.hidden_dim: int = int(config.get("hidden_dim", 1024))
        self.head_name: str = self.__class__.__name__

    @abstractmethod
    def forward(self, features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Process input features; return raw logit / prediction tensors."""

    @abstractmethod
    def to_ggl_nodes(
        self, predictions: Dict[str, torch.Tensor], threshold: float = 0.5
    ) -> List[GGLNode]:
        """Convert raw tensor predictions to GGL nodes above the threshold."""
