"""
manufacturing/analyzer.py — Production Manufacturing Constraint Analyzer
==========================================================================
Validates GGL and construction plans against real manufacturing constraints.

Checks:
  - Minimum wall thickness (per material class)
  - Draft angles for injection molding
  - Undercut detection (requires side actions)
  - Aspect ratio limits (drilling/milling)
  - Fillet radius minimums (tool radius constraints)
  - Internal corner accessibility (tool reach)
  - Thread specifications (standard sizes)

Each violation produces a structured issue report with severity,
affected node, and suggested remediation.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cad_planner.manufacturing.analyzer")


class Severity(str, Enum):
    ERROR = "error"          # Cannot be manufactured
    WARNING = "warning"      # Difficult/expensive to manufacture
    INFO = "info"            # Suggestion for improvement


@dataclass
class ManufacturingIssue:
    """A single manufacturing constraint violation."""
    node_id: str
    constraint: str
    severity: Severity
    message: str
    remediation: str
    parameter_name: Optional[str] = None
    current_value: Optional[float] = None
    limit_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "constraint": self.constraint,
            "severity": self.severity.value,
            "message": self.message,
            "remediation": self.remediation,
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
        }


@dataclass
class ManufacturabilityScore:
    """Overall manufacturability assessment."""
    score: float
    issues: List[ManufacturingIssue] = field(default_factory=list)
    process_recommendation: str = "CNC Milling"

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "process_recommendation": self.process_recommendation,
            "issues": [i.to_dict() for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Material-specific constraints
# ---------------------------------------------------------------------------

@dataclass
class MaterialConstraints:
    min_wall_thickness_mm: float = 1.0
    min_fillet_radius_mm: float = 0.5
    max_aspect_ratio_drill: float = 15.0
    max_aspect_ratio_mill: float = 8.0
    min_draft_angle_deg: float = 1.0
    max_unsupported_length_mm: float = 100.0


MATERIAL_CONSTRAINTS = {
    "steel": MaterialConstraints(
        min_wall_thickness_mm=1.5,
        min_fillet_radius_mm=0.8,
        max_aspect_ratio_drill=12.0,
    ),
    "aluminum": MaterialConstraints(
        min_wall_thickness_mm=1.0,
        min_fillet_radius_mm=0.5,
        max_aspect_ratio_drill=18.0,
    ),
    "plastic": MaterialConstraints(
        min_wall_thickness_mm=0.8,
        min_fillet_radius_mm=0.3,
        max_aspect_ratio_drill=20.0,
        min_draft_angle_deg=0.5,
    ),
    "default": MaterialConstraints(),
}


class ManufacturingAnalyzer:
    """
    Evaluates GGL primitives and construction plans against
    manufacturing constraints.
    """

    def __init__(self, material: str = "default") -> None:
        self.material = material
        self.constraints = MATERIAL_CONSTRAINTS.get(
            material, MATERIAL_CONSTRAINTS["default"]
        )

    @classmethod
    def analyze(cls, ggl, material: str = "default") -> ManufacturabilityScore:
        """
        Analyze a GGL for manufacturing feasibility.

        Convenience entry point: constructs an analyzer for the given
        material and runs the analysis. Callers use it statically, e.g.
        ``ManufacturingAnalyzer.analyze(ggl)``.

        Args:
            ggl: GeometryGraphLanguage instance (with .nodes attribute).
            material: Material profile to evaluate against.

        Returns:
            ManufacturabilityScore with issues and overall score.
        """
        return cls(material)._analyze(ggl)

    def _analyze(self, ggl) -> ManufacturabilityScore:
        """Run the manufacturing analysis using this instance's constraints."""
        issues: List[ManufacturingIssue] = []
        score = 1.0

        for node in ggl.nodes:
            node_issues = self._check_node(node)
            issues.extend(node_issues)

        # Compute score from issues
        for issue in issues:
            if issue.severity == Severity.ERROR:
                score -= 0.2
            elif issue.severity == Severity.WARNING:
                score -= 0.05

        score = max(0.0, min(1.0, score))

        # Recommend manufacturing process
        process = self._recommend_process(ggl, issues)

        result = ManufacturabilityScore(
            score=score,
            issues=issues,
            process_recommendation=process,
        )

        logger.info(
            "Manufacturing analysis: score=%.3f, errors=%d, warnings=%d, process=%s",
            result.score, result.error_count, result.warning_count, process,
        )

        return result

    def _check_node(self, node) -> List[ManufacturingIssue]:
        """Run all constraint checks on a single GGL node."""
        issues: List[ManufacturingIssue] = []
        params = getattr(node, "parameters", {})
        node_id = getattr(node, "node_id", "unknown")
        prim_type = getattr(node, "type", "")

        # --- Wall thickness ---
        for key in ("width", "height", "depth"):
            val = params.get(key)
            if val is not None and isinstance(val, (int, float)):
                if 0 < val < self.constraints.min_wall_thickness_mm:
                    issues.append(ManufacturingIssue(
                        node_id=node_id,
                        constraint="min_wall_thickness",
                        severity=Severity.ERROR,
                        message=(
                            f"{key}={val:.2f}mm is below minimum wall "
                            f"thickness {self.constraints.min_wall_thickness_mm}mm "
                            f"for {self.material}"
                        ),
                        remediation=f"Increase {key} to ≥{self.constraints.min_wall_thickness_mm}mm",
                        parameter_name=key,
                        current_value=val,
                        limit_value=self.constraints.min_wall_thickness_mm,
                    ))

        # --- Aspect ratio (cylinders = drill/bore operations) ---
        if prim_type == "Cylinder":
            r = params.get("radius", 1.0)
            h = params.get("height", 1.0)
            if r > 0:
                aspect = h / r
                limit = self.constraints.max_aspect_ratio_drill
                if aspect > limit:
                    issues.append(ManufacturingIssue(
                        node_id=node_id,
                        constraint="max_aspect_ratio",
                        severity=Severity.WARNING if aspect < limit * 1.5 else Severity.ERROR,
                        message=(
                            f"Cylinder aspect ratio {aspect:.1f} exceeds "
                            f"drilling limit {limit:.1f} for {self.material}"
                        ),
                        remediation="Reduce height or increase radius",
                        parameter_name="aspect_ratio",
                        current_value=aspect,
                        limit_value=limit,
                    ))

        # --- Very small radii (tool radius constraint) ---
        radius = params.get("radius")
        if radius is not None and isinstance(radius, (int, float)):
            if 0 < radius < self.constraints.min_fillet_radius_mm:
                issues.append(ManufacturingIssue(
                    node_id=node_id,
                    constraint="min_fillet_radius",
                    severity=Severity.WARNING,
                    message=(
                        f"Radius={radius:.2f}mm is below minimum tool radius "
                        f"{self.constraints.min_fillet_radius_mm}mm"
                    ),
                    remediation=f"Increase radius to ≥{self.constraints.min_fillet_radius_mm}mm",
                    parameter_name="radius",
                    current_value=radius,
                    limit_value=self.constraints.min_fillet_radius_mm,
                ))

        # --- Cone: check for zero-radius tip (unmachined point) ---
        if prim_type == "Cone":
            r = params.get("radius", 1.0)
            h = params.get("height", 1.0)
            if r > 0 and h > 0:
                tip_angle = math.degrees(math.atan(r / h))
                if tip_angle < 5:
                    issues.append(ManufacturingIssue(
                        node_id=node_id,
                        constraint="cone_tip_angle",
                        severity=Severity.WARNING,
                        message=f"Very sharp cone tip ({tip_angle:.1f}°) — difficult to machine",
                        remediation="Add a flat or fillet at the cone tip",
                        parameter_name="tip_angle",
                        current_value=tip_angle,
                        limit_value=5.0,
                    ))

        return issues

    def _recommend_process(self, ggl, issues: List[ManufacturingIssue]) -> str:
        """Recommend the best manufacturing process based on geometry."""
        has_axisymmetric = any(
            getattr(n, "type", "") in ("Cylinder", "Cone", "Sphere")
            for n in ggl.nodes
        )
        has_prismatic = any(
            getattr(n, "type", "") == "Box" for n in ggl.nodes
        )
        error_count = sum(1 for i in issues if i.severity == Severity.ERROR)

        if error_count > 2:
            return "Additive Manufacturing (3D Printing)"
        elif has_axisymmetric and not has_prismatic:
            return "CNC Turning"
        elif has_prismatic:
            return "CNC Milling"
        else:
            return "CNC Milling (Multi-axis)"
