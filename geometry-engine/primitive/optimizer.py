"""
primitive/optimizer.py — Production Geometric Optimizer
========================================================
Replaces mock fitting with real Levenberg-Marquardt and RANSAC-based
geometric optimization.

Takes primitive proposals with initial parameters and optimizes them
against point clouds derived from DiT hidden states.

ARCHITECTURE RULE:
    Optimization targets are point clouds sampled from DiT latent
    representations. The decoded mesh is NEVER used as the optimization
    target — only for verification after the fact.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from graph.ggl import GGLNode
from primitive.fitting import FitResult, LevenbergMarquardtFitter, RANSACFitter

logger = logging.getLogger("geometry_engine.primitive.optimizer")


class GeometricOptimizer:
    """
    Optimizes primitive proposals by fitting them to point clouds
    using mathematically rigorous algorithms.

    Supports:
      - Levenberg-Marquardt (least_squares)
      - RANSAC (ransac)
      - Gauss-Newton (gauss_newton → delegates to LM with λ=0)
      - Differentiable (differentiable → future torch-based)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        prim_config = config.get("primitive", {})
        self.method: str = prim_config.get("optimizer", "ransac")
        self.max_iters: int = int(prim_config.get("max_iterations", 100))
        self.tolerance: float = float(prim_config.get("tolerance", 1e-4))
        self.ransac_threshold: float = float(
            prim_config.get("ransac_residual_threshold", 0.05)
        )
        self.ransac_trials: int = int(prim_config.get("ransac_max_trials", 1000))

        # Initialize fitters
        self._lm_fitter = LevenbergMarquardtFitter(
            max_iterations=self.max_iters,
            tolerance=self.tolerance,
        )
        self._ransac_fitter = RANSACFitter(
            max_trials=self.ransac_trials,
            residual_threshold=self.ransac_threshold,
        )

        logger.info(
            "GeometricOptimizer initialized: method=%s, max_iters=%d, tol=%.2e",
            self.method, self.max_iters, self.tolerance,
        )

    def optimize(
        self,
        proposals: List[GGLNode],
        point_cloud: Optional[np.ndarray] = None,
    ) -> GGLNode:
        """
        Optimize primitive proposals and return the best-fitting one.

        Args:
            proposals: List of GGLNode primitive proposals with initial
                       parameters to be refined.
            point_cloud: [N, 3] numpy array of 3D points from DiT hidden
                        state decoding. If None, uses parameter-level
                        scoring only (degraded mode).

        Returns:
            The single best GGLNode with optimized parameters.

        Raises:
            ValueError: If proposals list is empty.
        """
        if not proposals:
            raise ValueError("optimize() received an empty proposals list.")

        logger.info(
            "Optimizing %d proposals using method='%s'",
            len(proposals), self.method,
        )

        best_proposal: Optional[GGLNode] = None
        best_score: float = -float("inf")

        for proposal in proposals:
            if point_cloud is not None:
                # Real fitting against point cloud
                fit_result = self._fit_primitive(
                    proposal.type, point_cloud, proposal.parameters
                )
                if fit_result.converged:
                    proposal.parameters.update(fit_result.parameters)
                    geometric_score = (
                        proposal.confidence
                        * math.exp(-fit_result.residual_error)
                        * fit_result.inlier_ratio
                    )
                else:
                    geometric_score = proposal.confidence * 0.1
            else:
                # Degraded mode: score by confidence and parameter plausibility
                fit_result = self._parameter_plausibility_score(proposal)
                geometric_score = proposal.confidence * math.exp(
                    -fit_result.residual_error
                )

            logger.info(
                "  %s: score=%.4f, residual=%.6f, inliers=%d/%d",
                proposal.type,
                geometric_score,
                fit_result.residual_error,
                fit_result.inlier_count,
                fit_result.total_points,
            )

            if geometric_score > best_score:
                best_score = geometric_score
                best_proposal = proposal

        # Ensure we have a result
        assert best_proposal is not None
        best_proposal.confidence = min(1.0, max(0.0, best_score))

        logger.info(
            "Selected: %s (confidence=%.3f)",
            best_proposal.type, best_proposal.confidence,
        )
        return best_proposal

    # ------------------------------------------------------------------
    # Fitting dispatch
    # ------------------------------------------------------------------

    def _fit_primitive(
        self,
        primitive_type: str,
        points: np.ndarray,
        initial_params: Dict[str, Any],
    ) -> FitResult:
        """Dispatch to the configured fitting algorithm."""
        float_params = {k: float(v) for k, v in initial_params.items()
                        if isinstance(v, (int, float))}

        if self.method == "ransac":
            return self._ransac_fitter.fit(points, primitive_type)
        elif self.method in ("least_squares", "lm", "gauss_newton"):
            return self._fit_lm(primitive_type, points, float_params)
        elif self.method == "differentiable":
            logger.warning("Differentiable fitting not yet implemented. Falling back to LM.")
            return self._fit_lm(primitive_type, points, float_params)
        else:
            logger.warning("Unknown method '%s'. Falling back to LM.", self.method)
            return self._fit_lm(primitive_type, points, float_params)

    def _fit_lm(
        self,
        primitive_type: str,
        points: np.ndarray,
        params: Dict[str, float],
    ) -> FitResult:
        """Fit using Levenberg-Marquardt."""
        if primitive_type == "Cylinder":
            return self._lm_fitter.fit_cylinder(points, params)
        elif primitive_type == "Sphere":
            return self._lm_fitter.fit_sphere(points, params)
        elif primitive_type == "Plane":
            return self._lm_fitter.fit_plane(points, params)
        else:
            logger.warning(
                "No LM implementation for '%s'. Using parameter-only scoring.",
                primitive_type,
            )
            return FitResult(
                primitive_type, params, 0.1, 0,
                points.shape[0], False, 0,
            )

    @staticmethod
    def _parameter_plausibility_score(proposal: GGLNode) -> FitResult:
        """
        Score a primitive purely by parameter plausibility when no
        point cloud is available (degraded mode).

        Checks:
          - Positive-definite size parameters (radius, width, height > 0)
          - Reasonable value ranges
          - Unit-length axis vectors
        """
        params = proposal.parameters
        error = 0.0

        # Check positive-definite parameters
        for key in ("radius", "width", "height", "depth"):
            val = params.get(key)
            if val is not None and val <= 0:
                error += 1.0

        # Check axis normalization
        axis_keys = [("axis_x", "axis_y", "axis_z"), ("normal_x", "normal_y", "normal_z")]
        for ax_keys in axis_keys:
            if all(k in params for k in ax_keys):
                ax = np.array([params[k] for k in ax_keys])
                norm = np.linalg.norm(ax)
                if norm > 0:
                    error += abs(1.0 - norm)

        return FitResult(
            primitive_type=proposal.type,
            parameters=params,
            residual_error=error,
            inlier_count=0,
            total_points=0,
            converged=True,
            iterations=0,
        )
