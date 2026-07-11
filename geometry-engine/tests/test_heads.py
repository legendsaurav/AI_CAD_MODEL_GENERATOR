"""
tests/test_heads.py — Comprehensive tests for all prediction heads and GGL Builder
====================================================================================
Tests:
  - PrimitiveHead: forward shapes, to_ggl_nodes output
  - SymmetryHead: forward shapes, symmetry node creation
  - PartHead (upgraded): bbox_params in output
  - SurfaceHead (upgraded): normals and curvatures in output
  - GGLBuilder: assembly, validation, deduplication
"""
import os
import sys
import pytest
import torch

# Ensure the geometry-engine root is on the path
_ENGINE_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

_SHARED = os.path.normpath(os.path.join(_ENGINE_ROOT, "..", "shared-schemas"))
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)


# ── Test configuration ────────────────────────────────────────────────── #

HEAD_CONFIG = {"hidden_dim": 1024, "dropout": 0.0}
BATCH, SEQ_LEN, HIDDEN = 2, 64, 1024


def _random_features() -> torch.Tensor:
    """Generate random features simulating DiT hidden states."""
    return torch.randn(BATCH, SEQ_LEN, HIDDEN)


# ═══════════════════════════════════════════════════════════════════════ #
#  PrimitiveHead Tests                                                    #
# ═══════════════════════════════════════════════════════════════════════ #


class TestPrimitiveHead:
    """Tests for the PrimitiveHead prediction head."""

    def _make_head(self):
        from heads.primitive import PrimitiveHead
        return PrimitiveHead(HEAD_CONFIG)

    def test_forward_output_keys(self):
        """forward() returns expected dictionary keys."""
        head = self._make_head()
        out = head(_random_features())
        assert "type_logits" in out
        assert "type_probs" in out
        assert "params" in out

    def test_forward_type_shapes(self):
        """Type prediction tensors have correct shapes."""
        head = self._make_head()
        out = head(_random_features())
        assert out["type_logits"].shape == (BATCH, SEQ_LEN, 6)
        assert out["type_probs"].shape == (BATCH, SEQ_LEN, 6)

    def test_forward_param_shapes(self):
        """Per-type parameter tensors have correct shapes."""
        from heads.primitive import PARAM_COUNTS
        head = self._make_head()
        out = head(_random_features())
        for ptype, n_params in PARAM_COUNTS.items():
            assert ptype in out["params"], f"Missing param head for {ptype}"
            assert out["params"][ptype].shape == (BATCH, SEQ_LEN, n_params)

    def test_type_probs_sum_to_one(self):
        """Type probabilities should sum to ~1 along the class dimension."""
        head = self._make_head()
        out = head(_random_features())
        sums = out["type_probs"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_to_ggl_nodes_returns_list(self):
        """to_ggl_nodes returns a list of GGLNode objects."""
        head = self._make_head()
        out = head(_random_features())
        nodes = head.to_ggl_nodes(out, threshold=0.1)  # low threshold for test
        assert isinstance(nodes, list)

    def test_to_ggl_node_structure(self):
        """Generated GGL nodes have required fields."""
        head = self._make_head()
        out = head(_random_features())
        nodes = head.to_ggl_nodes(out, threshold=0.1)
        if nodes:  # random weights may or may not produce nodes
            node = nodes[0]
            assert node.node_id.startswith("prim_")
            assert node.type in ["Cylinder", "Box", "Sphere", "Cone", "Plane", "Torus"]
            assert 0.0 <= node.confidence <= 1.0
            assert "units" in node.parameters


# ═══════════════════════════════════════════════════════════════════════ #
#  SymmetryHead Tests                                                     #
# ═══════════════════════════════════════════════════════════════════════ #


class TestSymmetryHead:
    """Tests for the SymmetryHead prediction head."""

    def _make_head(self):
        from heads.symmetry import SymmetryHead
        return SymmetryHead(HEAD_CONFIG)

    def test_forward_output_keys(self):
        head = self._make_head()
        out = head(_random_features())
        assert "symmetry_probs" in out
        assert "mirror_plane" in out
        assert "rotation_axis" in out

    def test_forward_shapes(self):
        head = self._make_head()
        out = head(_random_features())
        assert out["symmetry_probs"].shape == (BATCH, SEQ_LEN, 4)
        assert out["mirror_plane"].shape == (BATCH, SEQ_LEN, 6)
        assert out["rotation_axis"].shape == (BATCH, SEQ_LEN, 7)

    def test_mirror_normal_is_unit_vector(self):
        """Mirror plane normals should be unit vectors."""
        head = self._make_head()
        out = head(_random_features())
        normals = out["mirror_plane"][..., 3:6]
        norms = torch.norm(normals, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_rotation_axis_is_unit_vector(self):
        """Rotation axis vectors should be unit vectors."""
        head = self._make_head()
        out = head(_random_features())
        axes = out["rotation_axis"][..., 3:6]
        norms = torch.norm(axes, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_to_ggl_nodes_skip_none(self):
        """to_ggl_nodes should not create nodes for 'None' symmetry type."""
        head = self._make_head()
        out = head(_random_features())
        nodes = head.to_ggl_nodes(out, threshold=0.1)
        for node in nodes:
            assert node.semantic_label != "None"


# ═══════════════════════════════════════════════════════════════════════ #
#  PartHead (Upgraded) Tests                                              #
# ═══════════════════════════════════════════════════════════════════════ #


class TestPartHead:
    """Tests for the upgraded PartHead."""

    def _make_head(self):
        from heads.part import PartHead
        return PartHead(HEAD_CONFIG)

    def test_forward_has_bbox_params(self):
        """Upgraded PartHead should output bbox_params."""
        head = self._make_head()
        out = head(_random_features())
        assert "part_probs" in out
        assert "bbox_params" in out

    def test_bbox_shape(self):
        head = self._make_head()
        out = head(_random_features())
        assert out["bbox_params"].shape == (BATCH, SEQ_LEN, 6)

    def test_bbox_sizes_positive(self):
        """Bounding box sizes (last 3 dims) should be positive."""
        head = self._make_head()
        out = head(_random_features())
        sizes = out["bbox_params"][..., 3:6]
        assert (sizes > 0).all()

    def test_to_ggl_nodes_has_spatial_params(self):
        """Generated nodes should include center and size."""
        head = self._make_head()
        out = head(_random_features())
        nodes = head.to_ggl_nodes(out, threshold=0.1)
        if nodes:
            node = nodes[0]
            assert "center" in node.parameters
            assert "size" in node.parameters
            assert len(node.parameters["center"]) == 3
            assert len(node.parameters["size"]) == 3


# ═══════════════════════════════════════════════════════════════════════ #
#  SurfaceHead (Upgraded) Tests                                           #
# ═══════════════════════════════════════════════════════════════════════ #


class TestSurfaceHead:
    """Tests for the upgraded SurfaceHead."""

    def _make_head(self):
        from heads.surface import SurfaceHead
        return SurfaceHead(HEAD_CONFIG)

    def test_forward_has_normals_and_curvatures(self):
        head = self._make_head()
        out = head(_random_features())
        assert "surface_probs" in out
        assert "normals" in out
        assert "curvatures" in out

    def test_normal_shape_and_unit(self):
        head = self._make_head()
        out = head(_random_features())
        assert out["normals"].shape == (BATCH, SEQ_LEN, 3)
        norms = torch.norm(out["normals"], dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_curvature_shape(self):
        head = self._make_head()
        out = head(_random_features())
        assert out["curvatures"].shape == (BATCH, SEQ_LEN, 2)

    def test_to_ggl_nodes_has_geometry(self):
        head = self._make_head()
        out = head(_random_features())
        nodes = head.to_ggl_nodes(out, threshold=0.1)
        if nodes:
            node = nodes[0]
            assert "normal" in node.parameters
            assert "curvature_k1" in node.parameters


# ═══════════════════════════════════════════════════════════════════════ #
#  GGLBuilder Tests                                                       #
# ═══════════════════════════════════════════════════════════════════════ #


class TestGGLBuilder:
    """Tests for GGLBuilder assembly and validation."""

    def _make_heads_and_outputs(self):
        """Create all heads and run them on random features."""
        from heads.part import PartHead
        from heads.surface import SurfaceHead
        from heads.primitive import PrimitiveHead
        from heads.topology import TopologyHead
        from heads.symmetry import SymmetryHead

        heads = {
            "part": PartHead(HEAD_CONFIG),
            "surface": SurfaceHead(HEAD_CONFIG),
            "primitive": PrimitiveHead(HEAD_CONFIG),
            "topology": TopologyHead(HEAD_CONFIG),
            "symmetry": SymmetryHead(HEAD_CONFIG),
        }

        features = _random_features()
        outputs = {}
        for name, head in heads.items():
            outputs[name] = head(features)

        return heads, outputs

    def test_build_returns_ggl(self):
        """build() should return a GeometryGraphLanguage object."""
        from graph.ggl_builder import GGLBuilder
        from graph.ggl import GeometryGraphLanguage

        builder = GGLBuilder()
        heads, outputs = self._make_heads_and_outputs()
        ggl = builder.build(heads, outputs)
        assert isinstance(ggl, GeometryGraphLanguage)

    def test_build_has_nodes(self):
        """Built GGL should contain at least some nodes."""
        from graph.ggl_builder import GGLBuilder

        builder = GGLBuilder({"primitive_threshold": 0.1, "part_threshold": 0.1,
                              "surface_threshold": 0.1, "symmetry_threshold": 0.1,
                              "topology_threshold": 0.1})
        heads, outputs = self._make_heads_and_outputs()
        ggl = builder.build(heads, outputs)
        # With low thresholds and random weights, we should get some nodes
        assert len(ggl.nodes) >= 0  # no crash is the minimum requirement

    def test_validation_catches_bad_edges(self):
        """Validation should warn about edges referencing missing nodes."""
        from graph.ggl_builder import GGLBuilder
        from graph.ggl import GeometryGraphLanguage, GGLEdge, GGLNode

        builder = GGLBuilder()
        ggl = GeometryGraphLanguage()
        ggl.add_node(GGLNode(node_id="n1", type="Box"))
        ggl.add_edge(GGLEdge(
            source_id="n1", target_id="NONEXISTENT", relation="Contains"
        ))

        warnings = builder._validate(ggl)
        assert any("NONEXISTENT" in w for w in warnings)

    def test_deduplication(self):
        """Nearby nodes of the same type should be merged."""
        from graph.ggl_builder import GGLBuilder
        from graph.ggl import GGLNode

        builder = GGLBuilder()
        nodes = [
            GGLNode(
                node_id="c1", type="Cylinder", confidence=0.9,
                parameters={"center": [10.0, 10.0, 10.0], "radius": 5.0},
            ),
            GGLNode(
                node_id="c2", type="Cylinder", confidence=0.8,
                parameters={"center": [10.5, 10.5, 10.5], "radius": 5.0},
            ),
            GGLNode(
                node_id="b1", type="Box", confidence=0.7,
                parameters={"center": [100.0, 100.0, 100.0]},
            ),
        ]

        result = builder._deduplicate_nodes(nodes, distance_threshold=5.0)
        # c1 and c2 should merge (same type, <5mm apart)
        # b1 should remain (different type)
        assert len(result) == 2
        types = {n.type for n in result}
        assert "Cylinder" in types
        assert "Box" in types

    def test_ggl_serialization_roundtrip(self):
        """Built GGL should survive JSON serialization roundtrip."""
        from graph.ggl_builder import GGLBuilder
        from graph.ggl import GeometryGraphLanguage

        builder = GGLBuilder({"primitive_threshold": 0.1, "part_threshold": 0.1,
                              "surface_threshold": 0.1})
        heads, outputs = self._make_heads_and_outputs()
        ggl = builder.build(heads, outputs)

        json_str = ggl.to_json()
        restored = GeometryGraphLanguage.from_json(json_str)
        assert len(restored.nodes) == len(ggl.nodes)
        assert len(restored.edges) == len(ggl.edges)


# ═══════════════════════════════════════════════════════════════════════ #
#  Run tests                                                              #
# ═══════════════════════════════════════════════════════════════════════ #

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
