"""
scripts/run_ggl_export.py  –  Version 4: GGL Export & Validation
=================================================================
Validates the GGL JSON schema (round-trip parse) and writes the
universal ggl_final.json.

ARCHITECTURE NOTE: CAD macro generation has been removed.
CAD reconstruction is the exclusive responsibility of:
  - cad-planner (GGL → CAL)
  - desktop-agent (CAL → SolidWorks / FreeCAD / Fusion 360)
"""
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.logger import ExperimentLogger  # noqa: E402
from graph.ggl import GeometryGraphLanguage, GGLNode, GGLEdge  # noqa: E402


# ── Build a sample fully-parameterised GGL ───────────────────────────────

def build_sample_parameterised_ggl() -> GeometryGraphLanguage:
    """
    Represents the GGL after Version 3 has assigned analytic primitives
    with full parameter sets to every Part node.
    """
    ggl = GeometryGraphLanguage()

    # Part nodes
    body  = GGLNode(node_id="part_body",  type="Part", semantic_label="Main Body",     confidence=0.92)
    shaft = GGLNode(node_id="part_shaft", type="Part", semantic_label="Shaft",         confidence=0.85)
    hole  = GGLNode(node_id="part_hole",  type="Part", semantic_label="Mounting Hole", confidence=0.79)
    ggl.add_node(body)
    ggl.add_node(shaft)
    ggl.add_node(hole)

    # Analytic primitives assigned by Version 3
    prim_body = GGLNode(
        node_id="prim_body_0", type="Box", semantic_label="Body Box",
        confidence=0.91,
        parameters={"width": 80.0, "height": 40.0, "depth": 20.0,
                    "center_x": 0.0, "center_y": 0.0, "center_z": 0.0},
    )
    prim_shaft = GGLNode(
        node_id="prim_shaft_0", type="Cylinder", semantic_label="Shaft Cylinder",
        confidence=0.88,
        parameters={"radius": 5.0, "height": 60.0,
                    "center_x": 0.0, "center_y": 0.0, "center_z": 20.0,
                    "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0},
    )
    prim_hole = GGLNode(
        node_id="prim_hole_0", type="Cylinder", semantic_label="Hole Cylinder",
        confidence=0.80,
        parameters={"radius": 3.0, "height": 20.0,
                    "center_x": 30.0, "center_y": 0.0, "center_z": 0.0,
                    "axis_x": 0.0, "axis_y": 0.0, "axis_z": 1.0},
    )
    ggl.add_node(prim_body)
    ggl.add_node(prim_shaft)
    ggl.add_node(prim_hole)

    # Hierarchy edges
    ggl.add_edge(GGLEdge(source_id="part_body",  target_id="prim_body_0",  relation="Instantiates", confidence=0.91))
    ggl.add_edge(GGLEdge(source_id="part_shaft", target_id="prim_shaft_0", relation="Instantiates", confidence=0.88))
    ggl.add_edge(GGLEdge(source_id="part_hole",  target_id="prim_hole_0",  relation="Instantiates", confidence=0.80))
    ggl.add_edge(GGLEdge(source_id="part_body",  target_id="part_shaft",   relation="Adjacent",     confidence=0.75))
    ggl.add_edge(GGLEdge(source_id="part_body",  target_id="part_hole",    relation="Contains",     confidence=0.82))

    return ggl


def main():
    print("\n" + "=" * 60)
    print("  VERSION 4 – GGL Universal Representation Validation")
    print("=" * 60)

    logger = ExperimentLogger()

    # ── VERSION 4: Build + validate GGL ──────────────────────────────────
    ggl = build_sample_parameterised_ggl()

    # Schema validation: round-trip JSON serialisation
    json_str  = ggl.to_json()
    reloaded  = GeometryGraphLanguage.from_json(json_str)

    assert len(reloaded.nodes) == len(ggl.nodes), "Node count mismatch after round-trip!"
    assert len(reloaded.edges) == len(ggl.edges), "Edge count mismatch after round-trip!"
    assert reloaded.version == "1.0", "Schema version mismatch!"
    print("✅ GGL JSON round-trip validation passed.")

    # Check all primitive nodes have parameters
    primitive_nodes = [n for n in reloaded.nodes if n.type in {"Cylinder", "Box", "Sphere"}]
    for pn in primitive_nodes:
        assert len(pn.parameters) > 0, f"Primitive {pn.node_id} has no parameters!"
    print(f"✅ All {len(primitive_nodes)} primitive nodes have parameters.")

    # Save the validated GGL
    logger.save_ggl(ggl.model_dump(), filename="ggl_final.json")

    ggl_completeness = {
        "total_nodes":         len(ggl.nodes),
        "total_edges":         len(ggl.edges),
        "primitive_nodes":     len(primitive_nodes),
        "json_schema_valid":   True,
        "round_trip_pass":     True,
    }
    logger.log_metrics(ggl_completeness, step=4)
    print(f"📋 GGL Completeness: {ggl_completeness}")

    print(f"\n✅ Version 4 complete — GGL exported to: {logger.get_exp_dir()}")
    print("📌 NOTE: CAD macro generation removed. Use cad-planner → desktop-agent pipeline.")
    return ggl


if __name__ == "__main__":
    main()
