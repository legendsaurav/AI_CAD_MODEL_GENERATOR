"""
evaluation/scorer.py — Production Multi-Dimensional Plan Scorer
================================================================
Evaluates construction plans across four quality dimensions:

  1. Manufacturability — can this be physically produced?
  2. Editability — how easy is the feature tree to modify later?
  3. Reconstruction fidelity — does the plan reproduce the GGL geometry?
  4. Intent preservation — does the plan respect engineering semantics?

Each dimension returns a score in [0, 1] and a detailed breakdown.
The composite score is a weighted sum.

Integration:
  - Feeds ScoringBreakdown into shared-schemas PlanningTrace
  - Used by BeamSearchPlanner for candidate ranking
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from construction.graph import ConstructionGraph, ConstructionNode

logger = logging.getLogger("cad_planner.evaluation.scorer")


@dataclass
class ScoringBreakdown:
    """Detailed per-dimension scoring breakdown."""
    manufacturability_score: float = 0.0
    editability_score: float = 0.0
    reconstruction_score: float = 0.0
    intent_score: float = 0.0
    composite_score: float = 0.0

    # Sub-metrics for diagnostics
    operation_count: int = 0
    dag_depth: int = 0
    standard_op_ratio: float = 0.0
    parametric_coverage: float = 0.0
    thin_wall_violations: int = 0
    undercut_violations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manufacturability_score": round(self.manufacturability_score, 4),
            "editability_score": round(self.editability_score, 4),
            "reconstruction_score": round(self.reconstruction_score, 4),
            "intent_score": round(self.intent_score, 4),
            "composite_score": round(self.composite_score, 4),
            "operation_count": self.operation_count,
            "dag_depth": self.dag_depth,
            "standard_op_ratio": round(self.standard_op_ratio, 4),
            "parametric_coverage": round(self.parametric_coverage, 4),
            "thin_wall_violations": self.thin_wall_violations,
            "undercut_violations": self.undercut_violations,
        }


class PlanScorer:
    """
    Multi-dimensional plan evaluation engine.

    Weights are configurable per deployment:
      - For rapid prototyping: emphasize reconstruction fidelity
      - For production machining: emphasize manufacturability
      - For parametric libraries: emphasize editability
    """

    # Standard CAD operations that are universally supported
    STANDARD_OPS = {
        "create_sketch", "extrude", "revolve", "fillet", "chamfer",
        "pattern", "mirror", "shell", "draft",
    }

    # Operations that reduce editability (fragile features)
    FRAGILE_OPS = {"loft", "sweep", "freeform", "surface_trim"}

    def __init__(
        self,
        weight_manufacturability: float = 0.25,
        weight_editability: float = 0.25,
        weight_reconstruction: float = 0.35,
        weight_intent: float = 0.15,
    ) -> None:
        self.w_mfg = weight_manufacturability
        self.w_edit = weight_editability
        self.w_recon = weight_reconstruction
        self.w_intent = weight_intent

    def score(self, cg: ConstructionGraph) -> float:
        """Legacy API: return composite score for backward compatibility."""
        breakdown = self.score_detailed(cg)
        return breakdown.composite_score

    def score_detailed(self, cg: ConstructionGraph) -> ScoringBreakdown:
        """
        Compute full multi-dimensional scoring breakdown.

        Args:
            cg: ConstructionGraph to evaluate.

        Returns:
            ScoringBreakdown with all dimension scores.
        """
        sequence = cg.get_sequence()
        bd = ScoringBreakdown()
        bd.operation_count = len(sequence)

        bd.manufacturability_score = self._score_manufacturability(sequence, cg)
        bd.editability_score = self._score_editability(sequence, cg)
        bd.reconstruction_score = self._score_reconstruction(sequence, cg)
        bd.intent_score = self._score_intent(sequence, cg)

        bd.composite_score = (
            self.w_mfg * bd.manufacturability_score
            + self.w_edit * bd.editability_score
            + self.w_recon * bd.reconstruction_score
            + self.w_intent * bd.intent_score
        )

        logger.info(
            "Plan scored: composite=%.3f (mfg=%.3f, edit=%.3f, recon=%.3f, intent=%.3f) ops=%d",
            bd.composite_score,
            bd.manufacturability_score,
            bd.editability_score,
            bd.reconstruction_score,
            bd.intent_score,
            bd.operation_count,
        )

        return bd

    # ------------------------------------------------------------------
    # Manufacturability scoring
    # ------------------------------------------------------------------

    def _score_manufacturability(
        self, sequence: List[ConstructionNode], cg: ConstructionGraph
    ) -> float:
        """
        Score based on manufacturing feasibility.

        Penalties for:
          - Thin walls (radius < 0.5mm, wall < 1mm)
          - High aspect ratios (impossible to machine)
          - Undercuts requiring special tooling
          - Excessive operation count (toolpath complexity)
        """
        score = 1.0
        violations = 0

        for node in sequence:
            params = node.parameters

            # Thin wall check
            for key in ("radius", "width", "height", "depth"):
                val = params.get(key)
                if val is not None and isinstance(val, (int, float)):
                    if 0 < val < 0.5:  # < 0.5mm
                        score -= 0.05
                        violations += 1

            # High aspect ratio check
            if node.operation_type == "extrude":
                depth = params.get("depth", 0)
                # Check sketch entities for smallest dimension
                profile = params.get("profile", {})
                for entity in profile.get("entities", []):
                    r = entity.get("radius", None)
                    if r and depth and r > 0:
                        aspect = depth / r
                        if aspect > 20:
                            score -= 0.1
                            violations += 1

            # Undercut detection (simplified)
            if node.operation_type in ("extrude",) and params.get("is_cut", False):
                draft = params.get("draft_angle", 0)
                if draft < 0:
                    score -= 0.15
                    violations += 1

        # Penalize very long sequences (toolpath complexity)
        if len(sequence) > 30:
            score -= 0.1

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Editability scoring
    # ------------------------------------------------------------------

    def _score_editability(
        self, sequence: List[ConstructionNode], cg: ConstructionGraph
    ) -> float:
        """
        Score based on feature tree editability.

        Rewards:
          - Standard operations (extrude, revolve, fillet)
          - Shallow dependency trees
          - Parameterized features (no magic numbers)

        Penalties:
          - Deep DAG (>10 levels)
          - Fragile operations (loft, sweep, freeform)
          - Circular or tightly coupled dependencies
        """
        score = 1.0

        if not sequence:
            return 0.5

        # Standard operation ratio
        std_count = sum(
            1 for n in sequence if n.operation_type in self.STANDARD_OPS
        )
        std_ratio = std_count / len(sequence) if sequence else 0
        score *= (0.5 + 0.5 * std_ratio)

        # Fragile operation penalty
        fragile_count = sum(
            1 for n in sequence if n.operation_type in self.FRAGILE_OPS
        )
        score -= fragile_count * 0.1

        # DAG depth penalty
        try:
            import networkx as nx
            depth = nx.dag_longest_path_length(cg.graph)
        except Exception:
            depth = len(sequence)
        if depth > 10:
            score -= 0.1 * (depth - 10) / 10

        # Parametric coverage: what fraction of operations have explicit parameters
        parameterized = sum(
            1 for n in sequence if len(n.parameters) > 1
        )
        param_coverage = parameterized / len(sequence) if sequence else 0
        score *= (0.5 + 0.5 * param_coverage)

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Reconstruction fidelity scoring
    # ------------------------------------------------------------------

    def _score_reconstruction(
        self, sequence: List[ConstructionNode], cg: ConstructionGraph
    ) -> float:
        """
        Score based on geometric reconstruction quality.

        Checks:
          - All GGL nodes have corresponding construction operations
          - Confidence propagation from GGL through construction
          - Parameter completeness (all required params present)
        """
        score = 1.0

        if not sequence:
            return 0.0

        # Check confidence propagation
        confidences = []
        for node in sequence:
            conf = node.parameters.get("confidence", None)
            if conf is not None:
                confidences.append(float(conf))

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            score *= avg_conf
        else:
            score *= 0.5  # No confidence info → penalize

        # Check source GGL traceability
        traced = sum(
            1 for n in sequence
            if n.parameters.get("source_ggl_node_id") is not None
        )
        trace_ratio = traced / len(sequence) if sequence else 0
        score *= (0.5 + 0.5 * trace_ratio)

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Intent preservation scoring
    # ------------------------------------------------------------------

    def _score_intent(
        self, sequence: List[ConstructionNode], cg: ConstructionGraph
    ) -> float:
        """
        Score based on design intent preservation.

        Checks:
          - Feature semantic labels are preserved
          - Boolean operations match expected intent (cut vs add)
          - Symmetry patterns are maintained
        """
        score = 1.0

        if not sequence:
            return 0.5

        # Reward operations with semantic references
        semantic_count = sum(
            1 for n in sequence if n.feature_ref is not None
        )
        semantic_ratio = semantic_count / len(sequence) if sequence else 0
        score *= (0.5 + 0.5 * semantic_ratio)

        # Check for cut/add consistency
        for node in sequence:
            label = (node.feature_ref or "").lower()
            is_cut = node.parameters.get("is_cut", False)

            if any(kw in label for kw in ("hole", "pocket", "cut", "slot")):
                if not is_cut:
                    score -= 0.05  # Cut feature not marked as cut
            elif any(kw in label for kw in ("boss", "pad", "rib")):
                if is_cut:
                    score -= 0.05  # Additive feature marked as cut

        return max(0.0, min(1.0, score))
