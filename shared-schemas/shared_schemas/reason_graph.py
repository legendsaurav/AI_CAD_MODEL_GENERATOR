"""
shared_schemas/reason_graph.py — Authoritative Reason Graph Schema
====================================================================
Every CAD operation in the pipeline MUST carry a Reason Graph that
explains *why* it was chosen, *what* alternatives were considered,
and *how confident* the system is.

This is the SINGLE SOURCE OF TRUTH. No other repository may redefine
these models.
"""
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from shared_schemas.versioning import SchemaVersion, VersionedSchema

T = TypeVar("T", bound="ReasonGraph")


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


class RejectedAlternative(BaseModel):
    """An alternative that was explicitly considered and rejected."""

    alternative: str = Field(
        ..., description="Short description of the rejected alternative approach."
    )
    rejection_reason: str = Field(
        ..., description="Why this alternative was not selected."
    )


class ReasonNode(BaseModel):
    """A single reasoning node — one logical step or decision point."""

    node_id: str = Field(
        ..., description="Unique identifier for this reason node."
    )
    purpose: str = Field(
        ..., description="What this operation intends to achieve."
    )
    rationale: str = Field(
        ..., description="Why this approach was chosen over others."
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of other ReasonNodes this node depends on.",
    )
    alternatives_considered: List[str] = Field(
        default_factory=list,
        description="Short descriptions of alternative approaches that were evaluated.",
    )
    rejected_alternatives: List[RejectedAlternative] = Field(
        default_factory=list,
        description="Alternatives that were explicitly rejected, with reasons.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision (0 = no confidence, 1 = certain).",
    )
    supporting_geometry: List[str] = Field(
        default_factory=list,
        description="GGL node IDs that support / motivate this decision.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for extensibility.",
    )


class ReasonEdge(BaseModel):
    """Directed edge linking two :class:`ReasonNode` instances."""

    source_id: str = Field(
        ..., description="ID of the source ReasonNode."
    )
    target_id: str = Field(
        ..., description="ID of the target ReasonNode."
    )
    relation: str = Field(
        ...,
        description=(
            "Type of relationship, e.g. 'depends_on', 'supersedes', "
            "'refines', 'contradicts'."
        ),
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Strength or importance of this relationship.",
    )


# ---------------------------------------------------------------------------
# Top-level graph model
# ---------------------------------------------------------------------------


class ReasonGraph(VersionedSchema):
    """Complete reasoning trace for a CAD planning decision.

    A directed graph of :class:`ReasonNode` objects connected by
    :class:`ReasonEdge` relationships, versioned via :class:`VersionedSchema`.
    """

    _SCHEMA_NAME: str = "reason_graph"
    _CURRENT_VERSION: SchemaVersion = SchemaVersion(major=1, minor=0, patch=0)

    schema_name: str = Field(
        default="reason_graph",
        description="Fixed discriminator for this schema type.",
    )
    nodes: List[ReasonNode] = Field(
        default_factory=list,
        description="All reasoning nodes in the graph.",
    )
    edges: List[ReasonEdge] = Field(
        default_factory=list,
        description="Directed edges between reasoning nodes.",
    )

    # -- Convenience API ------------------------------------------------------

    def add_node(self, node: ReasonNode) -> None:
        """Append a :class:`ReasonNode` to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: ReasonEdge) -> None:
        """Append a :class:`ReasonEdge` to the graph."""
        self.edges.append(edge)

    def get_node(self, node_id: str) -> Optional[ReasonNode]:
        """Return the node with *node_id*, or ``None``."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def root_nodes(self) -> List[ReasonNode]:
        """Return nodes that have no incoming edges."""
        targets = {e.target_id for e in self.edges}
        return [n for n in self.nodes if n.node_id not in targets]

    # -- Serialization (inherited, re-declared for type hints) ----------------

    def to_json(self) -> str:
        """Serialize the full graph to a JSON string."""
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize from a JSON string."""
        return cls.model_validate_json(json_str)
