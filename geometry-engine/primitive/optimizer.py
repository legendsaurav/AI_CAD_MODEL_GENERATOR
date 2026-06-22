"""
primitive/optimizer.py - Geometric Optimizer
BUG FIX: `tuple[GGLNode, float]` lowercase generic syntax requires Python 3.9+.
Replaced with `Tuple[GGLNode, float]` from typing for Python 3.8 compatibility.
Also fixed: when proposals list is empty we now raise an informative error
instead of AttributeError on `None.confidence`.
"""
from typing import Any, Dict, List, Tuple
import math

from graph.ggl import GGLNode


class GeometricOptimizer:
    """
    Takes primitive proposals and optimizes their continuous parameters
    against geometric constraints (Least Squares, RANSAC).
    """

    def __init__(self, config: Dict[str, Any]):
        prim_config = config.get("primitive", {})
        self.method = prim_config.get("optimizer", "ransac")
        self.max_iters = prim_config.get("max_iterations", 100)
        self.tolerance = float(prim_config.get("tolerance", 1e-4))

    def optimize(self, proposals: List[GGLNode], target_features: Any = None) -> GGLNode:
        """
        Runs the optimization algorithm across the Top-K proposals
        and returns the single winning, mathematically refined primitive.
        """
        if not proposals:
            raise ValueError("optimize() received an empty proposals list.")

        print(f"🔧 Optimizing {len(proposals)} proposals using {self.method}...")

        best_proposal: GGLNode = None
        best_score = -float("inf")

        for proposal in proposals:
            refined_proposal, fit_error = self._simulate_least_squares_fit(proposal)
            geometric_score = proposal.confidence * math.exp(-fit_error)

            if geometric_score > best_score:
                best_score = geometric_score
                best_proposal = refined_proposal

        # Cap confidence at 1.0
        best_proposal.confidence = min(1.0, best_score)
        print(f"   -> Selected {best_proposal.type} (confidence={best_proposal.confidence:.3f})")
        return best_proposal

    # ------------------------------------------------------------------
    def _simulate_least_squares_fit(self, proposal: GGLNode) -> Tuple[GGLNode, float]:
        """
        Mocks a Gauss-Newton / Levenberg-Marquardt solver.
        In production this minimises residuals between the proposed surface
        and a point cloud / SDF sampled from the flow representation.
        """
        fit_error = 0.1  # fixed mock error

        if "radius" in proposal.parameters:
            proposal.parameters["radius"] = proposal.parameters["radius"] * 1.05
        if "center_x" in proposal.parameters:
            proposal.parameters["center_x"] = proposal.parameters["center_x"] + 0.01

        return proposal, fit_error
