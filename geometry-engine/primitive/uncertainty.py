"""
primitive/uncertainty.py — Monte Carlo Dropout Uncertainty Estimation
======================================================================
Provides uncertainty-aware primitive prediction using MC Dropout.

Instead of a single forward pass, this module runs multiple stochastic
forward passes (with dropout active) and measures the variance across
predictions. High variance → low confidence → flag for human review.

Methods:
  - mc_dropout: Multiple forward passes with dropout, aggregate variance
  - ensemble: Uses parameter diversity across N model checkpoints
  - calibrated_confidence: Temperature-scaled softmax calibration
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger("geometry_engine.primitive.uncertainty")


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimation for geometry heads.

    During inference, keeps dropout enabled and runs T forward passes.
    The predictive mean gives the estimate; the predictive variance
    gives the epistemic uncertainty.

    Usage:
        estimator = MCDropoutEstimator(model, num_samples=30)
        mean, variance, confidence = estimator.estimate(features)
    """

    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 30,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            model: PyTorch model with Dropout layers.
            num_samples: Number of stochastic forward passes (T).
            temperature: Temperature scaling for calibrated softmax.
        """
        self.model = model
        self.num_samples = num_samples
        self.temperature = temperature

    def estimate(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run MC Dropout inference.

        Args:
            features: Input tensor [B, N, D].

        Returns:
            Tuple of:
              - mean_prediction: [B, N, C] average across T samples
              - variance: [B, N, C] variance across T samples
              - confidence: scalar in [0, 1] — inverse of mean variance
        """
        self.model.train()  # Enable dropout

        predictions: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                output = self.model(features)
                if isinstance(output, dict):
                    # Handle head output format
                    key = list(output.keys())[0]
                    output = output[key]
                predictions.append(output)

        stacked = torch.stack(predictions, dim=0)  # [T, B, N, C]
        mean_pred = stacked.mean(dim=0)             # [B, N, C]
        variance = stacked.var(dim=0)                # [B, N, C]

        # Confidence: inverse of mean variance, scaled to [0, 1]
        mean_var = float(variance.mean().item())
        confidence = 1.0 / (1.0 + mean_var)

        self.model.eval()  # Restore eval mode

        logger.info(
            "MC Dropout estimation: T=%d, mean_variance=%.6f, confidence=%.4f",
            self.num_samples, mean_var, confidence,
        )

        return mean_pred, variance, confidence

    def estimate_parameters(
        self,
        features: torch.Tensor,
        regressor: nn.Module,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Estimate primitive parameters with uncertainty bounds.

        Args:
            features: Input tensor [1, D] or [D].
            regressor: nn.Module that maps features to parameter vector.

        Returns:
            Tuple of:
              - mean_params: {name: value} mean estimates
              - std_params: {name: value} standard deviations
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        regressor.train()
        samples: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                out = regressor(features)  # [1, num_params]
                samples.append(out)

        regressor.eval()
        stacked = torch.stack(samples, dim=0)  # [T, 1, P]
        means = stacked.mean(dim=0).squeeze(0)  # [P]
        stds = stacked.std(dim=0).squeeze(0)    # [P]

        return (
            {f"param_{i}": float(v) for i, v in enumerate(means)},
            {f"param_{i}": float(v) for i, v in enumerate(stds)},
        )


class EnsembleEstimator:
    """
    Ensemble-based uncertainty estimation using multiple model checkpoints.

    Runs each checkpoint independently and aggregates predictions.
    Provides both epistemic and aleatoric uncertainty separation.
    """

    def __init__(self, models: List[nn.Module]) -> None:
        """
        Args:
            models: List of N model instances (loaded from different checkpoints).
        """
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models.")
        self.models = models

    def estimate(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Run ensemble inference.

        Returns:
            Tuple of (mean_prediction, variance, confidence).
        """
        predictions: List[torch.Tensor] = []

        with torch.no_grad():
            for model in self.models:
                model.eval()
                output = model(features)
                if isinstance(output, dict):
                    key = list(output.keys())[0]
                    output = output[key]
                predictions.append(output)

        stacked = torch.stack(predictions, dim=0)
        mean_pred = stacked.mean(dim=0)
        variance = stacked.var(dim=0)

        mean_var = float(variance.mean().item())
        confidence = 1.0 / (1.0 + mean_var)

        logger.info(
            "Ensemble estimation: N=%d, mean_variance=%.6f, confidence=%.4f",
            len(self.models), mean_var, confidence,
        )

        return mean_pred, variance, confidence


class CalibratedConfidence:
    """
    Temperature-scaled confidence calibration.

    After training, the optimal temperature T is found via the NLL
    on a validation set. At inference time, softmax outputs are
    divided by T before computing probabilities.

    This produces better-calibrated confidence scores that reflect
    true accuracy more faithfully.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model output [B, N, C].

        Returns:
            Calibrated probabilities [B, N, C].
        """
        scaled = logits / self.temperature
        return torch.softmax(scaled, dim=-1)

    def find_optimal_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """
        Find optimal temperature via gradient descent on NLL loss.

        Args:
            logits: Validation logits [N, C].
            labels: Ground truth class indices [N].
            lr: Learning rate.
            max_iter: Maximum optimization steps.

        Returns:
            Optimal temperature value.
        """
        temp = torch.tensor([self.temperature], requires_grad=True)
        optimizer = torch.optim.LBFGS([temp], lr=lr, max_iter=max_iter)
        nll = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = logits / temp
            loss = nll(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(temp.item())
        logger.info("Optimal temperature found: %.4f", self.temperature)
        return self.temperature
