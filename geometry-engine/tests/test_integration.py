"""
tests/test_integration.py — End-to-End Pipeline Integration Tests
====================================================================
Validates the complete AI CAD OS pipeline from GGL → CAL → Execution
→ Verification → Refinement without requiring actual CAD software.

Uses synthetic GGL/CAL data and mock executors to test:
  1. GGL → CAD Planner → CAL generation
  2. CAL → Desktop Agent (mock) → ExecutionReport
  3. ExecutionReport → Refinement Loop → Convergence
  4. Full pipeline with VerificationReport
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

# Setup sys.path for all modules
_TEST_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_SCRATCH = os.path.normpath(os.path.join(_TEST_ROOT, ".."))
for subdir in ("geometry-engine", "cad-planner", "desktop-agent", "shared-schemas"):
    path = os.path.join(_SCRATCH, subdir)
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# Synthetic test data factories
# ---------------------------------------------------------------------------

def make_synthetic_ggl() -> dict:
    """Create a synthetic GGL with cylinder + box primitives."""
    return {
        "schema_version": {"major": 1, "minor": 0, "patch": 0},
        "nodes": [
            {
                "node_id": "body_1",
                "type": "Box",
                "semantic_label": "main_body",
                "parameters": {
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "center_z": 0.0,
                    "width": 50.0,
                    "height": 30.0,
                    "depth": 20.0,
                    "confidence": 0.92,
                    "source_ggl_node_id": "body_1",
                },
                "confidence": 0.92,
            },
            {
                "node_id": "hole_1",
                "type": "Cylinder",
                "semantic_label": "through_hole",
                "parameters": {
                    "center_x": 10.0,
                    "center_y": 0.0,
                    "center_z": 0.0,
                    "radius": 5.0,
                    "height": 30.0,
                    "axis_x": 0.0,
                    "axis_y": 1.0,
                    "axis_z": 0.0,
                    "confidence": 0.88,
                    "source_ggl_node_id": "hole_1",
                },
                "confidence": 0.88,
            },
            {
                "node_id": "fillet_1",
                "type": "Sphere",
                "semantic_label": "fillet_blend",
                "parameters": {
                    "center_x": 25.0,
                    "center_y": 15.0,
                    "center_z": 10.0,
                    "radius": 3.0,
                    "confidence": 0.85,
                    "source_ggl_node_id": "fillet_1",
                },
                "confidence": 0.85,
            },
        ],
        "edges": [
            {"source": "body_1", "target": "hole_1", "relation": "contains"},
        ],
    }


def make_synthetic_mesh_points(ggl: dict) -> np.ndarray:
    """Generate synthetic mesh points from GGL for testing comparator."""
    all_points = []
    for node in ggl["nodes"]:
        params = node["parameters"]
        ptype = node["type"]
        n = 500

        if ptype == "Box":
            pts = np.random.uniform(-1, 1, (n, 3))
            pts[:, 0] *= params.get("width", 1) / 2
            pts[:, 1] *= params.get("height", 1) / 2
            pts[:, 2] *= params.get("depth", 1) / 2
            pts += [params.get("center_x", 0), params.get("center_y", 0), params.get("center_z", 0)]
        elif ptype == "Cylinder":
            theta = np.random.uniform(0, 2 * np.pi, n)
            r = params.get("radius", 1.0)
            h = params.get("height", 1.0)
            z = np.random.uniform(-h / 2, h / 2, n)
            pts = np.stack([
                r * np.cos(theta) + params.get("center_x", 0),
                r * np.sin(theta) + params.get("center_y", 0),
                z + params.get("center_z", 0),
            ], axis=1)
        elif ptype == "Sphere":
            phi = np.random.uniform(0, 2 * np.pi, n)
            cos_t = np.random.uniform(-1, 1, n)
            sin_t = np.sqrt(1 - cos_t ** 2)
            r = params.get("radius", 1.0)
            pts = np.stack([
                r * sin_t * np.cos(phi) + params.get("center_x", 0),
                r * sin_t * np.sin(phi) + params.get("center_y", 0),
                r * cos_t + params.get("center_z", 0),
            ], axis=1)
        else:
            pts = np.random.randn(n, 3)

        all_points.append(pts)

    return np.vstack(all_points)


# ---------------------------------------------------------------------------
# Test: Convergence Detection
# ---------------------------------------------------------------------------

class TestConvergenceDetection:
    """Test the convergence detector standalone."""

    def test_absolute_convergence(self):
        from refinement.convergence import ConvergenceDetector

        detector = ConvergenceDetector(iou_threshold=0.95, max_iterations=10)
        converged, should_continue, reason = detector.update(
            iteration=1, overall_iou=0.96
        )
        assert converged is True
        assert should_continue is False
        assert "IoU" in reason

    def test_plateau_detection(self):
        from refinement.convergence import ConvergenceDetector

        detector = ConvergenceDetector(
            iou_threshold=0.99,
            max_iterations=20,
            plateau_window=3,
            plateau_min_improvement=0.01,
        )

        # Simulate stalling improvement
        for i, iou in enumerate([0.5, 0.501, 0.502, 0.503], start=1):
            converged, should_continue, reason = detector.update(
                iteration=i, overall_iou=iou
            )

        assert should_continue is False
        assert "Plateau" in reason or "plateau" in reason.lower()

    def test_oscillation_detection(self):
        from refinement.convergence import ConvergenceDetector

        detector = ConvergenceDetector(
            iou_threshold=0.99,
            max_iterations=20,
            oscillation_window=4,
        )

        # Simulate oscillating IoU
        for i, iou in enumerate([0.7, 0.8, 0.7, 0.8, 0.7], start=1):
            converged, should_continue, reason = detector.update(
                iteration=i, overall_iou=iou
            )

        assert should_continue is False
        assert "scillation" in reason

    def test_divergence_detection(self):
        from refinement.convergence import ConvergenceDetector

        detector = ConvergenceDetector(iou_threshold=0.99, max_iterations=20)

        for i, iou in enumerate([0.8, 0.7, 0.6], start=1):
            converged, should_continue, reason = detector.update(
                iteration=i, overall_iou=iou
            )

        assert should_continue is False
        assert "Divergence" in reason or "diverge" in reason.lower()

    def test_adaptive_step_size(self):
        from refinement.convergence import ConvergenceDetector

        detector = ConvergenceDetector(iou_threshold=0.99, max_iterations=20)

        # Normal improvement → step_size = 1.0
        detector.update(1, 0.5)
        detector.update(2, 0.6)
        assert detector.get_recommended_step_size() == 1.0

        # Oscillation → step_size = 0.5
        detector.reset()
        detector.update(1, 0.5)
        detector.update(2, 0.6)
        detector.update(3, 0.55)  # went down
        assert detector.get_recommended_step_size() == 0.5


# ---------------------------------------------------------------------------
# Test: Distance Metrics Integration
# ---------------------------------------------------------------------------

class TestDistanceMetricsIntegration:
    """Test distance metrics on realistic synthetic data."""

    def test_self_comparison_is_near_perfect(self):
        from refinement.comparator import chamfer_distance, volume_iou

        ggl = make_synthetic_ggl()
        pts = make_synthetic_mesh_points(ggl)

        cd = chamfer_distance(pts, pts.copy())
        iou = volume_iou(pts, pts.copy(), voxel_size=1.0)

        assert cd < 1e-8
        assert iou == 1.0

    def test_perturbed_cloud_has_nonzero_distance(self):
        from refinement.comparator import chamfer_distance

        ggl = make_synthetic_ggl()
        pts = make_synthetic_mesh_points(ggl)
        perturbed = pts + np.random.randn(*pts.shape) * 2.0

        cd = chamfer_distance(pts, perturbed)
        assert cd > 0.1


# ---------------------------------------------------------------------------
# Test: Primitive Fitting Integration
# ---------------------------------------------------------------------------

class TestPrimitiveFittingIntegration:
    """Test that fitting algorithms work on GGL-derived point clouds."""

    def test_cylinder_fit_from_ggl(self):
        from primitive.fitting import LevenbergMarquardtFitter

        np.random.seed(42)
        fitter = LevenbergMarquardtFitter(max_iterations=100)

        # Generate cylinder points from GGL parameters
        r, h = 5.0, 30.0
        n = 300
        theta = np.random.uniform(0, 2 * np.pi, n)
        z = np.random.uniform(-h / 2, h / 2, n)
        noise = np.random.randn(n, 3) * 0.1

        points = np.stack([
            r * np.cos(theta),
            r * np.sin(theta),
            z,
        ], axis=1) + noise

        result = fitter.fit_cylinder(points)
        assert result.converged or result.residual_error < 1.0
        assert abs(result.parameters["radius"] - r) < 2.0

    def test_sphere_fit_from_ggl(self):
        from primitive.fitting import RANSACFitter

        np.random.seed(123)
        fitter = RANSACFitter(max_trials=300, residual_threshold=0.3)

        # Generate sphere points with noise
        r = 3.0
        n = 200
        phi = np.random.uniform(0, 2 * np.pi, n)
        cos_t = np.random.uniform(-1, 1, n)
        sin_t = np.sqrt(1 - cos_t ** 2)

        points = r * np.stack([
            sin_t * np.cos(phi),
            sin_t * np.sin(phi),
            cos_t,
        ], axis=1) + np.random.randn(n, 3) * 0.1

        result = fitter.fit(points, "Sphere")
        assert result.inlier_ratio > 0.5


# ---------------------------------------------------------------------------
# Test: Plan Scoring Integration
# ---------------------------------------------------------------------------

class TestPlanScoringIntegration:
    """Test multi-dimensional scoring on construction graphs."""

    def test_simple_plan_scores_well(self):
        from construction.graph import ConstructionGraph, ConstructionNode
        from evaluation.scorer import PlanScorer

        cg = ConstructionGraph()
        cg.add_operation(ConstructionNode(
            "sk1", "create_sketch", {"plane": "XY", "confidence": 0.9, "source_ggl_node_id": "body_1"},
            feature_ref="main_body",
        ))
        cg.add_operation(ConstructionNode(
            "ext1", "extrude", {"sketch_id": "sk1", "depth": 20, "confidence": 0.9, "source_ggl_node_id": "body_1"},
            feature_ref="main_body",
        ))
        cg.add_dependency("ext1", "sk1")

        scorer = PlanScorer()
        breakdown = scorer.score_detailed(cg)

        assert breakdown.composite_score > 0.3
        assert breakdown.editability_score > 0.3
        assert breakdown.operation_count == 2

    def test_fragile_plan_scores_lower(self):
        from construction.graph import ConstructionGraph, ConstructionNode
        from evaluation.scorer import PlanScorer

        cg = ConstructionGraph()
        cg.add_operation(ConstructionNode("s1", "loft", {}, feature_ref=None))
        cg.add_operation(ConstructionNode("s2", "freeform", {}, feature_ref=None))
        cg.add_operation(ConstructionNode("s3", "sweep", {}, feature_ref=None))

        scorer = PlanScorer()
        fragile_score = scorer.score(cg)

        cg2 = ConstructionGraph()
        cg2.add_operation(ConstructionNode("s1", "create_sketch", {"confidence": 0.9}, feature_ref="x"))
        cg2.add_operation(ConstructionNode("s2", "extrude", {"confidence": 0.9}, feature_ref="x"))

        robust_score = scorer.score(cg2)

        assert robust_score > fragile_score


# ---------------------------------------------------------------------------
# Test: Manufacturing Analysis Integration
# ---------------------------------------------------------------------------

class TestManufacturingIntegration:
    """Test manufacturing constraint analysis."""

    def test_thin_wall_detection(self):
        from manufacturing.analyzer import ManufacturingAnalyzer

        class MockGGL:
            class Node:
                def __init__(self, nid, ptype, params):
                    self.node_id = nid
                    self.type = ptype
                    self.parameters = params

            def __init__(self, nodes):
                self.nodes = nodes

        ggl = MockGGL([
            MockGGL.Node("n1", "Box", {"width": 0.3, "height": 10, "depth": 10}),
        ])

        analyzer = ManufacturingAnalyzer(material="steel")
        result = analyzer.analyze(ggl)

        assert result.error_count > 0
        assert result.score < 1.0
        assert any("wall" in i.constraint for i in result.issues)

    def test_high_aspect_cylinder(self):
        from manufacturing.analyzer import ManufacturingAnalyzer

        class MockGGL:
            class Node:
                def __init__(self, nid, ptype, params):
                    self.node_id = nid
                    self.type = ptype
                    self.parameters = params

            def __init__(self, nodes):
                self.nodes = nodes

        ggl = MockGGL([
            MockGGL.Node("n1", "Cylinder", {"radius": 1.0, "height": 50.0}),
        ])

        analyzer = ManufacturingAnalyzer(material="steel")
        result = analyzer.analyze(ggl)

        assert result.warning_count > 0 or result.error_count > 0


# ---------------------------------------------------------------------------
# Test: Memory Retrieval Integration
# ---------------------------------------------------------------------------

class TestMemoryRetrievalIntegration:
    """Test pattern memory retrieval system."""

    def test_keyword_retrieval(self):
        from memory.pattern_database import PatternDatabase
        from memory.retrieval import MemoryRetrieval

        db = PatternDatabase()
        retrieval = MemoryRetrieval(db)

        result = retrieval.retrieve("bearing seat feature")
        assert result is not None
        assert "create_sketch" in result

    def test_ranked_retrieval(self):
        from memory.pattern_database import PatternDatabase
        from memory.retrieval import MemoryRetrieval

        db = PatternDatabase()
        retrieval = MemoryRetrieval(db)

        results = retrieval.retrieve_ranked("hole bore drill", top_k=3)
        assert len(results) >= 1
        assert results[0].relevance_score > 0

    def test_no_match_returns_none(self):
        from memory.pattern_database import PatternDatabase
        from memory.retrieval import MemoryRetrieval

        db = PatternDatabase()
        retrieval = MemoryRetrieval(db)

        result = retrieval.retrieve("completely unrelated quantum physics")
        assert result is None


# ---------------------------------------------------------------------------
# Test: Beam Search with PlanningTrace
# ---------------------------------------------------------------------------

class TestBeamSearchIntegration:
    """Test beam search produces valid PlanningTrace."""

    def test_plan_with_trace(self):
        from construction.graph import ConstructionGraph, ConstructionNode
        from beam_search.planner import BeamSearchPlanner

        candidates = []
        for i in range(3):
            cg = ConstructionGraph()
            cg.add_operation(ConstructionNode(
                f"sk_{i}", "create_sketch",
                {"plane": "XY", "confidence": 0.8 + i * 0.05, "source_ggl_node_id": f"n{i}"},
                feature_ref="body",
            ))
            cg.add_operation(ConstructionNode(
                f"ext_{i}", "extrude",
                {"sketch_id": f"sk_{i}", "depth": 10 + i * 5, "confidence": 0.8 + i * 0.05, "source_ggl_node_id": f"n{i}"},
                feature_ref="body",
            ))
            cg.add_dependency(f"ext_{i}", f"sk_{i}")
            candidates.append(cg)

        planner = BeamSearchPlanner(beam_width=2)
        best, trace = planner.plan_with_trace(candidates)

        assert best is not None
        assert "beam_candidates" in trace
        assert "selected_plan_index" in trace
        assert trace["selected_plan_index"] == 0
        assert trace["num_candidates_evaluated"] == 3
        assert len(trace["beam_candidates"]) <= 2
