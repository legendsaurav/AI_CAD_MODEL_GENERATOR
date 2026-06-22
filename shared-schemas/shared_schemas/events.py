"""
shared_schemas/events.py — Authoritative Pipeline Event Schemas
================================================================
Typed event envelopes consumed by the Workflow Manager.

Every stage transition, failure, checkpoint, or refinement request in
the AI CAD OS pipeline is communicated through a :class:`PipelineEvent`.

This is the SINGLE SOURCE OF TRUTH. No other repository may redefine
these models.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from shared_schemas.versioning import SchemaVersion, VersionedSchema

T = TypeVar("T", bound="PipelineEvent")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """Discriminator for pipeline event kinds."""

    PIPELINE_START = "PIPELINE_START"
    STAGE_START = "STAGE_START"
    STAGE_COMPLETE = "STAGE_COMPLETE"
    STAGE_FAILED = "STAGE_FAILED"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    REFINEMENT_REQUESTED = "REFINEMENT_REQUESTED"


class PipelineStage(str, Enum):
    """Ordered stages of the AI CAD OS pipeline."""

    IMAGE_PREPROCESSING = "IMAGE_PREPROCESSING"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    GEOMETRY_ENGINE = "GEOMETRY_ENGINE"
    CAD_PLANNING = "CAD_PLANNING"
    DESKTOP_EXECUTION = "DESKTOP_EXECUTION"
    VERIFICATION = "VERIFICATION"
    REFINEMENT = "REFINEMENT"


# ---------------------------------------------------------------------------
# Payload models
# ---------------------------------------------------------------------------


class EventPayload(BaseModel):
    """Base model for event-specific payloads.

    Subclass this to create strongly-typed payloads for individual
    event types. The base version is a transparent bag of key-value data.
    """

    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary payload data associated with this event.",
    )


class StageResult(BaseModel):
    """Outcome of a single pipeline stage execution."""

    stage: PipelineStage = Field(
        ..., description="Which pipeline stage produced this result."
    )
    status: str = Field(
        ...,
        description="Execution status: 'success', 'failed', 'skipped', 'partial'.",
    )
    duration_ms: float = Field(
        ...,
        ge=0.0,
        description="Wall-clock duration of this stage in milliseconds.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Filesystem path to the stage's output artifact, if any.",
    )
    error_message: Optional[str] = Field(
        None,
        description="Human-readable error message when status is 'failed'.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary stage-specific metadata.",
    )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "StageResult":
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)


# ---------------------------------------------------------------------------
# Top-level event envelope
# ---------------------------------------------------------------------------


class PipelineEvent(VersionedSchema):
    """Immutable event envelope emitted by every pipeline stage transition.

    Workflow Manager, logging, and the Clicky Tutor all consume these
    events for orchestration and observability.
    """

    _SCHEMA_NAME: str = "pipeline_event"
    _CURRENT_VERSION: SchemaVersion = SchemaVersion(major=1, minor=0, patch=0)

    schema_name: str = Field(
        default="pipeline_event",
        description="Fixed discriminator for this schema type.",
    )
    event_id: UUID = Field(
        default_factory=uuid4,
        description="Globally unique identifier for this event.",
    )
    event_type: EventType = Field(
        ..., description="The kind of event being reported."
    )
    stage: Optional[PipelineStage] = Field(
        None,
        description=(
            "Pipeline stage associated with this event. "
            "None for pipeline-level events (START / COMPLETE / FAILED)."
        ),
    )
    session_id: str = Field(
        ..., description="Session ID grouping all events in one pipeline run."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of event creation.",
    )
    payload: EventPayload = Field(
        default_factory=EventPayload,
        description="Event-specific payload.",
    )
    stage_result: Optional[StageResult] = Field(
        None,
        description="Populated for STAGE_COMPLETE and STAGE_FAILED events.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary top-level metadata (correlation IDs, etc.).",
    )

    # -- Serialization --------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the event to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)
