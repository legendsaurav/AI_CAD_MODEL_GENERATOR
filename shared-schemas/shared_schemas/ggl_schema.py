"""
shared_schemas/ggl_schema.py — Authoritative Geometry Graph Language Schema
============================================================================
This is the SINGLE SOURCE OF TRUTH for GGL. Both geometry-engine and
cad-planner must import from here. No duplication allowed.
"""
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class GGLNode(BaseModel):
    """Represents a geometric entity in the GGL (Part, Surface, or Primitive)."""
    node_id: str
    type: str = Field(..., description="E.g., Cylinder, Box, Part, Surface, Plane")
    semantic_label: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Geometric parameters: radius, height, axis, center, etc."
    )


class GGLEdge(BaseModel):
    """Represents a relationship between two nodes in the GGL."""
    source_id: str
    target_id: str
    relation: str = Field(..., description="E.g., Contains, Adjacent, Symmetric, Instantiates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class GGLMetadata(BaseModel):
    """Metadata tracing the GGL back to its source."""
    generator: str = "geometry-engine-v1.0"
    original_image: Optional[str] = None
    timestep_extracted: Optional[float] = None
    layers_used: List[int] = Field(default_factory=list)
    hunyuan_model_version: str = "2.1"
    source_type: str = Field(
        default="dit_hidden_states",
        description="MUST be 'dit_hidden_states'. If set to 'mesh' this is an architecture violation."
    )


class GeometryGraphLanguage(BaseModel):
    """
    The universal intermediate representation between the Geometry Engine
    and the CAD Planner. Software-independent, versioned, JSON-serializable.

    INVARIANT: GGL is always derived from Hunyuan3D Flow DiT intermediate
    representations, never from the decoded mesh.
    """
    version: str = "1.0"
    schema_name: str = "geometry_graph_language"
    metadata: GGLMetadata = Field(default_factory=GGLMetadata)
    nodes: List[GGLNode] = Field(default_factory=list)
    edges: List[GGLEdge] = Field(default_factory=list)

    def add_node(self, node: GGLNode):
        self.nodes.append(node)

    def add_edge(self, edge: GGLEdge):
        self.edges.append(edge)

    def to_json(self) -> str:
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> 'GeometryGraphLanguage':
        return cls.model_validate_json(json_str)

    def validate_source_integrity(self):
        """
        Runtime assertion: confirms this GGL was derived from DiT hidden states,
        not from a mesh. Raises if violated.
        """
        if self.metadata.source_type != "dit_hidden_states":
            raise ValueError(
                f"ARCHITECTURE VIOLATION: GGL source_type is '{self.metadata.source_type}', "
                f"expected 'dit_hidden_states'. The system must never reconstruct CAD "
                f"from meshes. GGL must be derived from Hunyuan3D Flow DiT intermediate "
                f"representations."
            )
