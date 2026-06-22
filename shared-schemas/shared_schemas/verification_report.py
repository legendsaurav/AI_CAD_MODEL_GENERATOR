"""
shared_schemas/verification_report.py — Authoritative Verification Report Schema
==================================================================================
Stores the numeric outcome of geometry verification: distance metrics,
normal consistency, curvature error, topology checks, and per-primitive
breakdowns.

Consumed by the refinement loop to decide whether the reconstructed CAD
model is acceptable or needs another iteration.

This is the SINGLE SOURCE OF TRUTH. No other repository may redefine
these models.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from shared_schemas.versioning import SchemaVersion, VersionedSchema

T = TypeVar("T", bound="VerificationReport")


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


class VerificationMetric(BaseModel):
    """A single named metric with a threshold-based pass/fail gate."""

    metric_name: str = Field(
        ..., description="Human-readable name of the metric."
    )
    value: float = Field(
        ..., description="Measured value of the metric."
    )
    threshold: float = Field(
        ..., description="Pass/fail threshold for this metric."
    )
    passed: bool = Field(
        ..., description="True if the metric value satisfies the threshold."
    )
    unit: Optional[str] = Field(
        None, description="Optional unit of measurement (e.g. 'mm', 'radians')."
    )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "VerificationMetric":
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)


class PrimitiveVerification(BaseModel):
    """Verification results scoped to a single geometric primitive."""

    primitive_id: str = Field(
        ..., description="GGL node ID of the primitive being verified."
    )
    primitive_type: str = Field(
        ..., description="Type of primitive (e.g. 'Cylinder', 'Box', 'Sphere')."
    )
    metrics: List[VerificationMetric] = Field(
        default_factory=list,
        description="All metrics computed for this primitive.",
    )
    passed: bool = Field(
        ..., description="True if ALL metrics for this primitive passed."
    )
    error_summary: Optional[str] = Field(
        None,
        description="Human-readable summary of failures, if any.",
    )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "PrimitiveVerification":
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)


# ---------------------------------------------------------------------------
# Top-level report model
# ---------------------------------------------------------------------------


class VerificationReport(VersionedSchema):
    """Comprehensive verification report for a single reconstruction attempt.

    Aggregates global distance metrics and per-primitive breakdowns.
    The refinement loop inspects :pyattr:`overall_passed` and
    :pyattr:`convergence_achieved` to decide whether to iterate.
    """

    _SCHEMA_NAME: str = "verification_report"
    _CURRENT_VERSION: SchemaVersion = SchemaVersion(major=1, minor=0, patch=0)

    schema_name: str = Field(
        default="verification_report",
        description="Fixed discriminator for this schema type.",
    )

    # -- Global metrics -------------------------------------------------------

    chamfer_distance: VerificationMetric = Field(
        ..., description="Chamfer distance between target and reconstructed geometry."
    )
    hausdorff_distance: VerificationMetric = Field(
        ..., description="Hausdorff distance (worst-case point deviation)."
    )
    normal_consistency: VerificationMetric = Field(
        ..., description="Surface normal consistency score."
    )
    curvature_error: VerificationMetric = Field(
        ..., description="Mean curvature error across matched surface patches."
    )
    topology_consistency: VerificationMetric = Field(
        ..., description="Topological consistency check (genus, holes, connectivity)."
    )
    primitive_parameter_error: VerificationMetric = Field(
        ..., description="Aggregate parameter error across all fitted primitives."
    )

    # -- Per-primitive breakdown ----------------------------------------------

    per_primitive_results: List[PrimitiveVerification] = Field(
        default_factory=list,
        description="Per-primitive verification breakdowns.",
    )

    # -- Aggregate verdicts ---------------------------------------------------

    overall_passed: bool = Field(
        ...,
        description="True if ALL global metrics and ALL primitives passed.",
    )
    convergence_achieved: bool = Field(
        ...,
        description=(
            "True if the iterative refinement loop has converged "
            "(metrics improving below threshold)."
        ),
    )
    iteration_index: int = Field(
        default=0,
        ge=0,
        description="Zero-based refinement iteration that produced this report.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (e.g. mesh paths, timing).",
    )

    # -- Serialization --------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the full report to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)
