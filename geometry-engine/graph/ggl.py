"""
geometry-engine/graph/ggl.py — Re-export from shared-schemas.

This file previously duplicated the GGL schema.  It now re-exports the
authoritative definitions from ``shared-schemas`` so that downstream code
continues to work with ``from graph.ggl import GeometryGraphLanguage``
while all changes are made in exactly one place.
"""

import os
import sys

_ENGINE_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SHARED = os.path.normpath(os.path.join(_ENGINE_ROOT, "..", "shared-schemas"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)

from shared_schemas.ggl_schema import (  # noqa: E402, F401
    GGLEdge,
    GGLMetadata,
    GGLNode,
    GeometryGraphLanguage,
)

__all__ = [
    "GGLNode",
    "GGLEdge",
    "GGLMetadata",
    "GeometryGraphLanguage",
]


# ── Quick validation ───────────────────────────────────────────────────
if __name__ == "__main__":
    ggl = GeometryGraphLanguage()
    ggl.add_node(GGLNode(
        node_id="n1",
        type="Body",
        semantic_label="Main Body",
    ))
    ggl.add_node(GGLNode(
        node_id="n2",
        type="Cylinder",
        semantic_label="Mounting Hole",
        confidence=0.91,
        parameters={"radius": 20.1, "height": 60.0, "axis": [0, 0, 1]},
    ))
    ggl.add_edge(GGLEdge(
        source_id="n1",
        target_id="n2",
        relation="Contains",
    ))

    print(ggl.to_json())  # noqa: T201 — CLI test harness
