"""
ggl_parser/ggl_parser.py — GGL Parser (uses shared-schemas)
==============================================================
Stage 1 — Ingests and validates Geometry Graph Language (GGL) JSON.
Checks schema version, missing parameters, confidence bounds, and
source integrity (must be derived from DiT hidden states).

ARCHITECTURE NOTE: GGL schema is defined in shared-schemas/ggl_schema.py.
This module only adds parsing + business rule validation.
"""
import json
import sys
import os

from pydantic import ValidationError

# Import from shared-schemas (authoritative source)
_PLANNER_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED_SCHEMAS = os.path.normpath(os.path.join(_PLANNER_ROOT, "..", "shared-schemas"))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

from shared_schemas.ggl_schema import GeometryGraphLanguage  # noqa: E402


class GGLParser:
    """
    Stage 1 — GGL Parser
    Ingests and validates Geometry Graph Language (GGL) JSON.
    Checks schema version, missing parameters, confidence bounds,
    and source integrity.
    """

    SUPPORTED_VERSIONS = ["1.0"]

    @staticmethod
    def parse(json_str: str) -> GeometryGraphLanguage:
        try:
            ggl = GeometryGraphLanguage.model_validate_json(json_str)
            GGLParser._validate_version(ggl)
            GGLParser._validate_source_integrity(ggl)
            GGLParser._validate_business_rules(ggl)
            return ggl
        except ValidationError as e:
            raise ValueError(f"GGL Schema Validation Failed:\n{e}")
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string provided.")

    @staticmethod
    def _validate_version(ggl: GeometryGraphLanguage):
        """Check that the GGL version is supported."""
        if ggl.version not in GGLParser.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported GGL version '{ggl.version}'. "
                f"Supported versions: {GGLParser.SUPPORTED_VERSIONS}"
            )

    @staticmethod
    def _validate_source_integrity(ggl: GeometryGraphLanguage):
        """
        Verify that the GGL was derived from DiT hidden states, not meshes.
        This is the most critical architectural invariant.
        """
        ggl.validate_source_integrity()

    @staticmethod
    def _validate_business_rules(ggl: GeometryGraphLanguage):
        """
        Validates higher level consistency rules of GGL beyond type checking.
        """
        node_ids = {n.node_id for n in ggl.nodes}

        for n in ggl.nodes:
            if not (0.0 <= n.confidence <= 1.0):
                raise ValueError(f"Node {n.node_id} confidence {n.confidence} out of bounds [0, 1].")

        for e in ggl.edges:
            if not (0.0 <= e.confidence <= 1.0):
                raise ValueError(f"Edge from {e.source_id} to {e.target_id} confidence out of bounds.")
            if e.source_id not in node_ids:
                raise ValueError(f"Edge references unknown source_id: {e.source_id}")
            if e.target_id not in node_ids:
                raise ValueError(f"Edge references unknown target_id: {e.target_id}")
