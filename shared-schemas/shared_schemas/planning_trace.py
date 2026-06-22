"""
shared_schemas/planning_trace.py — Authoritative Planning Trace Schema
========================================================================
Captures *every* decision the CAD Planner makes: beam-search candidates,
scoring breakdowns, ambiguity resolutions, retrieved memories, and
rejected plans.

This is the SINGLE SOURCE OF TRUTH. No other repository may redefine
these models.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from shared_schemas.versioning import SchemaVersion, VersionedSchema

T = TypeVar("T", bound="PlanningTrace")


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


class ScoringBreakdown(BaseModel):
    """Per-candidate scoring breakdown across quality dimensions."""

    manufacturability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How feasible the plan is to manufacture in a real shop.",
    )
    editability_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How easy the resulting CAD model is to edit later.",
    )
    reconstruction_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How faithfully the plan reconstructs the target geometry.",
    )
    intent_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How well the plan satisfies the user's stated intent.",
    )

    @property
    def total_score(self) -> float:
        """Unweighted mean of all four sub-scores."""
        return (
            self.manufacturability_score
            + self.editability_score
            + self.reconstruction_score
            + self.intent_score
        ) / 4.0


class BeamCandidate(BaseModel):
    """A single candidate produced during beam-search planning."""

    candidate_index: int = Field(
        ..., ge=0, description="Zero-based index of this candidate in the beam."
    )
    construction_sequence: List[str] = Field(
        ...,
        description=(
            "Ordered list of CAL action IDs or textual descriptions "
            "representing the construction steps."
        ),
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate score assigned to this candidate.",
    )
    scoring_breakdown: ScoringBreakdown = Field(
        ..., description="Detailed per-dimension scoring for this candidate."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for extensibility.",
    )


class AmbiguityResolution(BaseModel):
    """Records how a planning-time ambiguity was resolved."""

    ambiguity_description: str = Field(
        ..., description="What was ambiguous (e.g., 'Fillet radius unclear')."
    )
    resolution: str = Field(
        ..., description="How the ambiguity was resolved."
    )
    resolution_source: str = Field(
        ...,
        description=(
            "Where the resolution came from: 'user_input', 'memory', "
            "'heuristic', 'default'."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this resolution.",
    )


class RetrievedMemory(BaseModel):
    """A memory item retrieved from the experience store during planning."""

    memory_id: str = Field(
        ..., description="Unique identifier for the memory entry."
    )
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Cosine / relevance similarity score.",
    )
    source_session_id: Optional[str] = Field(
        None, description="Session that produced the original memory."
    )
    summary: str = Field(
        ..., description="Human-readable summary of the retrieved memory."
    )
    applied: bool = Field(
        ..., description="Whether this memory was actually used in the plan."
    )


class RejectedPlan(BaseModel):
    """A plan that was fully evaluated but ultimately rejected."""

    plan_index: int = Field(
        ..., ge=0, description="Index of this plan among all candidates."
    )
    construction_sequence: List[str] = Field(
        ..., description="The construction steps of the rejected plan."
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate score of the rejected plan.",
    )
    rejection_reason: str = Field(
        ..., description="Why this plan was rejected."
    )


# ---------------------------------------------------------------------------
# Top-level trace model
# ---------------------------------------------------------------------------


class PlanningTrace(VersionedSchema):
    """Full audit trail of a single CAD planning session.

    Contains beam-search candidates, scoring breakdowns, ambiguity
    resolutions, retrieved memories, and rejected plans.
    """

    _SCHEMA_NAME: str = "planning_trace"
    _CURRENT_VERSION: SchemaVersion = SchemaVersion(major=1, minor=0, patch=0)

    schema_name: str = Field(
        default="planning_trace",
        description="Fixed discriminator for this schema type.",
    )
    beam_candidates: List[BeamCandidate] = Field(
        default_factory=list,
        description="All beam-search candidates evaluated during planning.",
    )
    ambiguity_resolutions: List[AmbiguityResolution] = Field(
        default_factory=list,
        description="Ambiguities encountered and how they were resolved.",
    )
    retrieved_memories: List[RetrievedMemory] = Field(
        default_factory=list,
        description="Memories retrieved from the experience store.",
    )
    rejected_plans: List[RejectedPlan] = Field(
        default_factory=list,
        description="Plans that were evaluated and rejected.",
    )
    selected_plan_index: int = Field(
        ...,
        ge=0,
        description="Index into beam_candidates of the chosen plan.",
    )
    total_planning_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock time spent planning, in milliseconds.",
    )

    # -- Serialization --------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the full trace to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)
