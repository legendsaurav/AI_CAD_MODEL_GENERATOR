"""
refinement/convergence.py — Production Convergence Detection
==============================================================
Monitors refinement loop progress and decides when to stop iterating.

Convergence strategies:
  1. Absolute threshold — stop when all metrics exceed thresholds
  2. Plateau detection — stop when improvement stalls across N iterations
  3. Oscillation detection — stop if metrics bounce (step size too large)
  4. Cost-benefit — stop when marginal improvement < cost of another iteration

Integrates with shared-schemas VerificationReport to produce structured
convergence decisions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("geometry_engine.refinement.convergence")


@dataclass
class ConvergenceState:
    """Tracks convergence metrics across iterations."""
    iteration: int = 0
    converged: bool = False
    reason: str = ""
    iou_history: List[float] = field(default_factory=list)
    chamfer_history: List[float] = field(default_factory=list)
    hausdorff_history: List[float] = field(default_factory=list)
    improvement_rates: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "converged": self.converged,
            "reason": self.reason,
            "iou_history": [round(v, 6) for v in self.iou_history],
            "chamfer_history": [round(v, 6) for v in self.chamfer_history],
            "improvement_rates": [round(v, 6) for v in self.improvement_rates],
        }


class ConvergenceDetector:
    """
    Detects convergence of the refinement loop using multiple strategies.

    Configuration:
        iou_threshold: Target IoU for absolute convergence (default 0.95)
        chamfer_threshold: Target Chamfer distance (default 0.01)
        plateau_window: Number of iterations to check for plateau (default 3)
        plateau_min_improvement: Minimum improvement rate to not be a plateau
        max_iterations: Hard stop limit
    """

    def __init__(
        self,
        iou_threshold: float = 0.95,
        chamfer_threshold: float = 0.01,
        plateau_window: int = 3,
        plateau_min_improvement: float = 0.005,
        max_iterations: int = 10,
        oscillation_window: int = 4,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.chamfer_threshold = chamfer_threshold
        self.plateau_window = plateau_window
        self.plateau_min_improvement = plateau_min_improvement
        self.max_iterations = max_iterations
        self.oscillation_window = oscillation_window

        self._state = ConvergenceState()

    @property
    def state(self) -> ConvergenceState:
        return self._state

    def update(
        self,
        iteration: int,
        overall_iou: float,
        mean_chamfer: float = 0.0,
        mean_hausdorff: float = 0.0,
    ) -> Tuple[bool, bool, str]:
        """
        Update convergence state with new iteration metrics.

        Args:
            iteration: Current iteration number.
            overall_iou: Mean IoU across all primitives.
            mean_chamfer: Mean Chamfer distance.
            mean_hausdorff: Mean Hausdorff distance.

        Returns:
            Tuple of (converged, should_continue, reason_string).
        """
        self._state.iteration = iteration
        self._state.iou_history.append(overall_iou)
        self._state.chamfer_history.append(mean_chamfer)
        self._state.hausdorff_history.append(mean_hausdorff)

        # Compute improvement rate
        if len(self._state.iou_history) >= 2:
            prev = self._state.iou_history[-2]
            improvement = overall_iou - prev
            self._state.improvement_rates.append(improvement)
        else:
            self._state.improvement_rates.append(0.0)

        # Check convergence strategies in order of priority
        converged, reason = self._check_convergence(overall_iou, mean_chamfer)

        if converged:
            self._state.converged = True
            self._state.reason = reason
            logger.info("✅ Converged at iteration %d: %s", iteration, reason)
            return True, False, reason

        # Check if we should stop without convergence
        should_stop, stop_reason = self._check_early_stop(iteration)
        if should_stop:
            self._state.reason = stop_reason
            logger.warning("⚠️ Stopping at iteration %d: %s", iteration, stop_reason)
            return False, False, stop_reason

        # Continue iterating
        logger.info(
            "🔄 Iteration %d: IoU=%.4f, CD=%.6f — continuing",
            iteration, overall_iou, mean_chamfer,
        )
        return False, True, "continuing"

    def _check_convergence(
        self, iou: float, chamfer: float
    ) -> Tuple[bool, str]:
        """Check absolute convergence thresholds."""
        # Strategy 1: IoU threshold
        if iou >= self.iou_threshold:
            return True, f"IoU {iou:.4f} ≥ threshold {self.iou_threshold}"

        # Strategy 2: Chamfer distance threshold
        if 0 < chamfer <= self.chamfer_threshold:
            return True, f"Chamfer {chamfer:.6f} ≤ threshold {self.chamfer_threshold}"

        return False, ""

    def _check_early_stop(self, iteration: int) -> Tuple[bool, str]:
        """Check if iteration should stop early without convergence."""
        # Hard iteration limit
        if iteration >= self.max_iterations:
            return True, f"Max iterations ({self.max_iterations}) reached"

        # Plateau detection: no meaningful improvement for N iterations
        if len(self._state.improvement_rates) >= self.plateau_window:
            recent = self._state.improvement_rates[-self.plateau_window:]
            max_improvement = max(abs(r) for r in recent)
            if max_improvement < self.plateau_min_improvement:
                return True, (
                    f"Plateau detected: max improvement {max_improvement:.6f} "
                    f"< {self.plateau_min_improvement} over {self.plateau_window} iterations"
                )

        # Oscillation detection: IoU alternating up/down
        if len(self._state.iou_history) >= self.oscillation_window:
            recent = self._state.iou_history[-self.oscillation_window:]
            diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
            sign_changes = sum(
                1 for i in range(len(diffs) - 1)
                if diffs[i] * diffs[i + 1] < 0
            )
            if sign_changes >= len(diffs) - 1:
                return True, (
                    f"Oscillation detected over {self.oscillation_window} iterations"
                )

        # Divergence: IoU getting worse
        if len(self._state.iou_history) >= 3:
            recent_3 = self._state.iou_history[-3:]
            if recent_3[-1] < recent_3[-2] < recent_3[-3]:
                return True, "Divergence detected: IoU declining for 3 consecutive iterations"

        return False, ""

    def reset(self) -> None:
        """Reset convergence state for a new refinement run."""
        self._state = ConvergenceState()

    def get_recommended_step_size(self) -> float:
        """
        Recommend a correction step size based on convergence history.

        If improving steadily → maintain step size (1.0)
        If plateauing → increase step size (1.5)
        If oscillating → decrease step size (0.5)
        """
        if len(self._state.improvement_rates) < 2:
            return 1.0

        recent = self._state.improvement_rates[-2:]

        # Oscillation: signs differ
        if recent[-1] * recent[-2] < 0:
            return 0.5

        # Plateau: both near zero
        if all(abs(r) < self.plateau_min_improvement for r in recent):
            return 1.5

        # Steady improvement
        return 1.0
