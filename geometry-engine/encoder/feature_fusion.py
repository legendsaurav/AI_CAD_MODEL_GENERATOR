"""
encoder/feature_fusion.py — Multi-Layer Feature Fusion Module
================================================================
Fuses hidden representations from multiple DiT layers and timesteps
into a single, geometry-rich feature tensor suitable for the prediction
heads (PartHead, SurfaceHead, TopologyHead).

Fusion strategies:
  - Weighted mean (learned per-layer attention weights)
  - Concatenation + projection (highest capacity)
  - Gated fusion (element-wise gating per layer)

ARCHITECTURE RULE:
    Input features MUST originate from DiT hidden states captured via
    the HiddenStateBridge. This module must NEVER accept mesh-derived
    features.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("geometry_engine.encoder.feature_fusion")


class LearnedWeightFusion(nn.Module):
    """
    Fuses L layer features by learning a softmax attention weight per layer.

    Input: List of L tensors, each [B, N, D]
    Output: Single tensor [B, N, D]
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        # Learnable layer importance weights (before softmax)
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: List of L tensors [B, N, D].

        Returns:
            Fused tensor [B, N, D].
        """
        assert len(layer_features) == self.num_layers, (
            f"Expected {self.num_layers} layers, got {len(layer_features)}"
        )

        weights = F.softmax(self.layer_weights, dim=0)  # [L]
        stacked = torch.stack(layer_features, dim=0)     # [L, B, N, D]

        # Weighted sum: [L, 1, 1, 1] * [L, B, N, D] → sum over L → [B, N, D]
        weighted = stacked * weights[:, None, None, None]
        return weighted.sum(dim=0)


class ConcatProjectionFusion(nn.Module):
    """
    Concatenates L layer features along the feature dimension
    and projects back to the original dimension.

    Input: List of L tensors, each [B, N, D]
    Output: Single tensor [B, N, D]

    This is the highest-capacity fusion: each layer can contribute
    distinct information through the learned projection.
    """

    def __init__(self, num_layers: int, feature_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_layers * feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: List of L tensors [B, N, D].

        Returns:
            Fused tensor [B, N, D].
        """
        concatenated = torch.cat(layer_features, dim=-1)  # [B, N, L*D]
        return self.projection(concatenated)               # [B, N, D]


class GatedFusion(nn.Module):
    """
    Element-wise gated fusion: learns a gate per layer that selectively
    passes or blocks each feature dimension.

    Gate is a sigmoid over a learned transformation of each layer's features.
    More expressive than weighted mean, more efficient than concat+project.
    """

    def __init__(self, num_layers: int, feature_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid(),
            )
            for _ in range(num_layers)
        ])
        # Final layer norm for stability
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: List of L tensors [B, N, D].

        Returns:
            Fused tensor [B, N, D].
        """
        assert len(layer_features) == self.num_layers

        gated_sum = torch.zeros_like(layer_features[0])
        for feat, gate in zip(layer_features, self.gates):
            g = gate(feat)           # [B, N, D] sigmoid values
            gated_sum = gated_sum + g * feat

        # Normalize by number of layers to keep scale stable
        return self.norm(gated_sum / self.num_layers)


class TimestepFusion(nn.Module):
    """
    Fuses features across multiple diffusion timesteps.

    Different timesteps capture different levels of geometric detail:
      - Early timesteps (t≈1.0): Global structure, coarse shapes
      - Middle timesteps (t≈0.5): Part boundaries, surface types
      - Late timesteps (t≈0.0): Fine detail, edges, corners

    This module learns to combine them optimally.
    """

    def __init__(
        self,
        num_timesteps: int,
        feature_dim: int,
        fusion_type: str = "gated",
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        if fusion_type == "learned_weight":
            self.fuser = LearnedWeightFusion(num_timesteps)
        elif fusion_type == "concat_project":
            self.fuser = ConcatProjectionFusion(num_timesteps, feature_dim)
        elif fusion_type == "gated":
            self.fuser = GatedFusion(num_timesteps, feature_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, timestep_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            timestep_features: List of T tensors [B, N, D], one per timestep.

        Returns:
            Fused tensor [B, N, D].
        """
        return self.fuser(timestep_features)


class MultiScaleFeatureFusion(nn.Module):
    """
    Complete multi-scale fusion pipeline.

    Combines:
      1. Per-timestep: fuse across L layers at each timestep
      2. Cross-timestep: fuse the T timestep representations
      3. Final projection with residual connection

    This is the recommended entry point for the geometry heads.
    """

    def __init__(
        self,
        num_layers: int,
        num_timesteps: int,
        feature_dim: int,
        layer_fusion_type: str = "gated",
        timestep_fusion_type: str = "learned_weight",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps

        # Per-timestep layer fusion
        self.layer_fusers = nn.ModuleList([
            self._make_fuser(layer_fusion_type, num_layers, feature_dim, dropout)
            for _ in range(num_timesteps)
        ])

        # Cross-timestep fusion
        self.timestep_fuser = self._make_fuser(
            timestep_fusion_type, num_timesteps, feature_dim, dropout
        )

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(feature_dim),
        )

    @staticmethod
    def _make_fuser(
        fusion_type: str, num_inputs: int, feature_dim: int, dropout: float
    ) -> nn.Module:
        if fusion_type == "learned_weight":
            return LearnedWeightFusion(num_inputs)
        elif fusion_type == "concat_project":
            return ConcatProjectionFusion(num_inputs, feature_dim, dropout)
        elif fusion_type == "gated":
            return GatedFusion(num_inputs, feature_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        features: Dict[str, Dict[str, torch.Tensor]],
        timestep_order: Optional[List[str]] = None,
        layer_order: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Full multi-scale fusion pipeline.

        Args:
            features: Nested dict from HiddenStateBridge.get_captured_states().
                      Outer key = timestep string, inner key = layer key.
            timestep_order: Ordered list of timestep keys to use.
                           If None, uses sorted keys.
            layer_order: Ordered list of layer keys to use per timestep.
                        If None, uses sorted keys.

        Returns:
            Fused tensor [B, N, D].
        """
        if timestep_order is None:
            timestep_order = sorted(features.keys())[:self.num_timesteps]
        if layer_order is None and timestep_order:
            first_ts = features[timestep_order[0]]
            layer_order = sorted(first_ts.keys())[:self.num_layers]

        timestep_features: List[torch.Tensor] = []

        for t_idx, ts_key in enumerate(timestep_order):
            ts_data = features[ts_key]
            layer_feats = [ts_data[lk] for lk in layer_order if lk in ts_data]

            # Pad if fewer layers than expected
            while len(layer_feats) < self.num_layers:
                layer_feats.append(torch.zeros_like(layer_feats[0]))

            fused_layers = self.layer_fusers[t_idx](layer_feats[:self.num_layers])
            timestep_features.append(fused_layers)

        # Pad timesteps if needed
        while len(timestep_features) < self.num_timesteps:
            timestep_features.append(torch.zeros_like(timestep_features[0]))

        fused_timesteps = self.timestep_fuser(timestep_features[:self.num_timesteps])

        # Final projection with residual
        output = self.final_proj(fused_timesteps) + fused_timesteps
        return output
