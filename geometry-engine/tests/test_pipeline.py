"""
tests/test_pipeline.py
======================
End-to-end integration tests covering every pipeline version (0–5).
All tests use mock models/features — no Hunyuan3D weights required.
"""
import os
import sys
import json
import tempfile
import unittest

import torch
import torch.nn as nn
import numpy as np

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.config import ConfigManager  # noqa: E402
from utils.logger import ExperimentLogger  # noqa: E402
from hooks.feature_extractor import DiTFeatureExtractor  # noqa: E402
from graph.ggl import GeometryGraphLanguage, GGLNode, GGLEdge  # noqa: E402
from graph.generator import GraphGenerator  # noqa: E402
from probing.analyzer import FeatureAnalyzer  # noqa: E402
from primitive.generator import PrimitiveProposalGenerator  # noqa: E402
from primitive.estimator import ParameterEstimator  # noqa: E402
from primitive.optimizer import GeometricOptimizer  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Shared mock DiT model (reused across tests)
# ═══════════════════════════════════════════════════════════════════════════

class MockDoubleBlock(nn.Module):
    def forward(self, img, txt, vec=None, pe=None):
        return img * 2.0, txt * 2.0

class MockSingleBlock(nn.Module):
    def forward(self, x, vec=None, pe=None):
        return x * 3.0

class MockHunyuanDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.double_blocks = nn.ModuleList([MockDoubleBlock() for _ in range(4)])
        self.single_blocks  = nn.ModuleList([MockSingleBlock() for _ in range(8)])

    def forward(self, img, txt, timestep=None):
        for block in self.double_blocks:
            img, txt = block(img, txt)
        x = torch.cat([txt, img], dim=1)
        for block in self.single_blocks:
            x = block(x)
        return x


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION 0 – Hook Infrastructure
# ═══════════════════════════════════════════════════════════════════════════

class TestVersion0Hooks(unittest.TestCase):
    """Validates DiTFeatureExtractor hook registration and capture."""

    def setUp(self):
        ConfigManager.reset()
        self.model     = MockHunyuanDiT()
        self.extractor = DiTFeatureExtractor(self.model)

    def tearDown(self):
        self.extractor.clear_hooks()

    def test_timestep_is_captured(self):
        """The pre-hook must capture the timestep kwarg correctly."""
        self.extractor.register_hooks(double_indices=[0], single_indices=[0])
        self.extractor.set_target_timesteps([0.5])

        img = torch.randn(1, 64, 1024)
        txt = torch.randn(1, 32, 1024)
        self.model(img, txt, timestep=0.5)

        self.assertEqual(self.extractor.current_timestep, 0.5,
                         "Timestep was not captured by pre-hook")

    def test_features_captured_at_target_timestep(self):
        """Features are saved only at the requested timestep."""
        self.extractor.register_hooks(double_indices=[1, 3], single_indices=[0, 7])
        self.extractor.set_target_timesteps([0.5])

        img = torch.randn(2, 3072, 1024)
        txt = torch.randn(2, 1370, 1024)
        self.model(img, txt, timestep=0.5)

        feats = self.extractor.features
        self.assertIn(0.5, feats["double_blocks"], "double_blocks missing t=0.5")
        self.assertIn(0.5, feats["single_blocks"],  "single_blocks missing t=0.5")

        db = feats["double_blocks"][0.5]
        self.assertIn(1, db)
        self.assertIn(3, db)
        self.assertNotIn(0, db)
        self.assertEqual(db[1]["img"].shape, (2, 3072, 1024))
        self.assertEqual(db[1]["txt"].shape, (2, 1370, 1024))

        sb = feats["single_blocks"][0.5]
        self.assertIn(0, sb)
        self.assertIn(7, sb)
        self.assertEqual(sb[0].shape, (2, 4442, 1024))  # 3072+1370

    def test_no_capture_at_wrong_timestep(self):
        """Nothing is saved when the forward pass uses a non-target timestep."""
        self.extractor.register_hooks(double_indices=[0], single_indices=[0])
        self.extractor.set_target_timesteps([0.5])
        self.extractor.clear_features()

        img = torch.randn(1, 64, 1024)
        txt = torch.randn(1, 32, 1024)
        self.model(img, txt, timestep=0.8)

        self.assertEqual(len(self.extractor.features["double_blocks"]), 0,
                         "Should not capture at non-target t=0.8")

    def test_clear_hooks(self):
        self.extractor.register_hooks(double_indices=[0], single_indices=[0])
        self.extractor.clear_hooks()
        self.assertEqual(len(self.extractor.hooks), 0)

    def test_capture_without_target_filter(self):
        """If no target_timesteps set, all timesteps are captured."""
        self.extractor.register_hooks(double_indices=[0], single_indices=[])
        # No set_target_timesteps call — default is []
        img = torch.randn(1, 64, 1024)
        txt = torch.randn(1, 32, 1024)
        self.model(img, txt, timestep=0.3)
        self.model(img, txt, timestep=0.7)
        db = self.extractor.features["double_blocks"]
        self.assertIn(0.3, db)
        self.assertIn(0.7, db)


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION 1 – Probing & Ranking
# ═══════════════════════════════════════════════════════════════════════════

class TestVersion1Probing(unittest.TestCase):

    def setUp(self):
        ConfigManager.reset()
        # 3 mini feature tensors: [1, 16, 64]
        self.feats = {
            "L0_t0.10":  np.random.randn(1, 16, 64).astype(np.float32),
            "L12_t0.50": np.random.randn(1, 16, 64).astype(np.float32) * 2,
            "L12_t0.90": np.random.randn(1, 16, 64).astype(np.float32) * 0.5,
        }
        self.analyzer = FeatureAnalyzer(log_dir=".")

    def test_pca_shape_and_variance(self):
        feat = self.feats["L12_t0.50"]
        proj, evr = self.analyzer.run_pca(feat, n_components=3)
        self.assertEqual(proj.shape, (1, 16, 3))
        self.assertEqual(len(evr), 3)
        self.assertAlmostEqual(float(np.sum(evr)), sum(evr), places=5)
        self.assertGreater(float(evr[0]), 0.0)

    def test_correlation_matrix_shape(self):
        keys, mat = self.analyzer.compute_correlation_matrix(self.feats)
        n = len(self.feats)
        self.assertEqual(mat.shape, (n, n))
        # Diagonal must be 1.0 (self-similarity)
        for i in range(n):
            self.assertAlmostEqual(mat[i, i], 1.0, places=4)

    def test_temporal_stability(self):
        series = [self.feats["L12_t0.50"], self.feats["L12_t0.90"]]
        drifts = self.analyzer.temporal_stability_analysis(series)
        self.assertEqual(len(drifts), 1)
        self.assertGreaterEqual(drifts[0], 0.0)

    def test_rank_layers_returns_sorted(self):
        rankings = self.analyzer.rank_layers(self.feats)
        self.assertEqual(len(rankings), len(self.feats))
        scores = [s for _, s in rankings]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION 2 – Hierarchical Graph Extraction
# ═══════════════════════════════════════════════════════════════════════════

class TestVersion2GraphExtraction(unittest.TestCase):

    def setUp(self):
        ConfigManager.reset()
        self.config = ConfigManager.get_all()

    def test_graph_generation_produces_nodes_and_edges(self):
        generator = GraphGenerator(self.config)
        features  = torch.randn(1, 64, 1024)
        ggl       = generator.generate_graph(features)
        # Should produce at least some nodes when threshold is permissive
        self.assertIsInstance(ggl, GeometryGraphLanguage)
        self.assertEqual(ggl.version, "1.0")

    def test_metadata_layers_used_is_list(self):
        generator = GraphGenerator(self.config)
        features  = torch.randn(1, 64, 1024)
        ggl       = generator.generate_graph(
            features, metadata_kwargs={"layers_used": 5}   # int → must be coerced
        )
        self.assertIsInstance(ggl.metadata.layers_used, list)

    def test_ggl_json_round_trip(self):
        ggl = GeometryGraphLanguage()
        ggl.add_node(GGLNode(node_id="n1", type="Part", semantic_label="Body", confidence=0.9))
        ggl.add_node(GGLNode(node_id="n2", type="Cylinder",
                             parameters={"radius": 5.0, "height": 20.0}, confidence=0.85))
        ggl.add_edge(GGLEdge(source_id="n1", target_id="n2",
                             relation="Contains", confidence=0.9))

        json_str = ggl.to_json()
        restored = GeometryGraphLanguage.from_json(json_str)
        self.assertEqual(len(restored.nodes), 2)
        self.assertEqual(len(restored.edges), 1)
        self.assertAlmostEqual(restored.nodes[1].parameters["radius"], 5.0)


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION 3 – Primitive Recovery
# ═══════════════════════════════════════════════════════════════════════════

class TestVersion3PrimitiveRecovery(unittest.TestCase):

    def setUp(self):
        ConfigManager.reset()
        self.config    = ConfigManager.get_all()
        self.feat      = torch.randn(1, 1024)
        self.part_node = GGLNode(node_id="p1", type="Part", semantic_label="Body")

    def test_proposer_returns_top_k(self):
        proposer  = PrimitiveProposalGenerator(self.config)
        proposals = proposer.generate_proposals(self.feat, self.part_node)
        top_k     = self.config.get("primitive", {}).get("top_k_proposals", 3)
        self.assertEqual(len(proposals), top_k)

    def test_proposals_have_decreasing_confidence(self):
        proposer  = PrimitiveProposalGenerator(self.config)
        proposals = proposer.generate_proposals(self.feat, self.part_node)
        confs     = [p.confidence for p in proposals]
        self.assertEqual(confs, sorted(confs, reverse=True))

    def test_estimator_populates_parameters(self):
        proposer  = PrimitiveProposalGenerator(self.config)
        estimator = ParameterEstimator(self.config)
        proposals = proposer.generate_proposals(self.feat, self.part_node)
        for prop in proposals:
            result = estimator.estimate(self.feat.clone(), prop)
            if result.type in estimator.PARAM_MAP:
                expected_params = estimator.PARAM_MAP[result.type]
                for param in expected_params:
                    self.assertIn(param, result.parameters,
                                  f"Missing param '{param}' in {result.type}")

    def test_optimizer_returns_single_best(self):
        proposer  = PrimitiveProposalGenerator(self.config)
        estimator = ParameterEstimator(self.config)
        optimizer = GeometricOptimizer(self.config)
        proposals = proposer.generate_proposals(self.feat, self.part_node)
        parameterised = [estimator.estimate(self.feat.clone(), p) for p in proposals]
        best = optimizer.optimize(parameterised)
        self.assertIsInstance(best, GGLNode)
        self.assertGreater(best.confidence, 0.0)
        self.assertLessEqual(best.confidence, 1.0)

    def test_optimizer_raises_on_empty_proposals(self):
        optimizer = GeometricOptimizer(self.config)
        with self.assertRaises(ValueError):
            optimizer.optimize([])


# ═══════════════════════════════════════════════════════════════════════════
#  VERSION 4 – GGL Universal Representation
# ═══════════════════════════════════════════════════════════════════════════

class TestVersion4GGLSchema(unittest.TestCase):

    def test_confidence_upper_bound(self):
        with self.assertRaises(Exception):
            GGLNode(node_id="x", type="Y", confidence=1.5)

    def test_confidence_lower_bound(self):
        with self.assertRaises(Exception):
            GGLNode(node_id="x", type="Y", confidence=-0.1)

    def test_full_schema_valid_json(self):
        ggl = GeometryGraphLanguage()
        ggl.add_node(GGLNode(node_id="n1", type="Box",
                             parameters={"width": 10.0, "height": 5.0, "depth": 3.0},
                             confidence=0.95))
        data = json.loads(ggl.to_json())
        self.assertIn("version", data)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertEqual(data["version"], "1.0")

    def test_metadata_defaults(self):
        ggl = GeometryGraphLanguage()
        self.assertEqual(ggl.metadata.generator, "geometry-engine-v1.0")
        self.assertIsInstance(ggl.metadata.layers_used, list)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIG & LOGGER
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigAndLogger(unittest.TestCase):

    def setUp(self):
        ConfigManager.reset()

    def test_config_loads(self):
        cfg = ConfigManager.get_all()
        self.assertIn("heads", cfg)
        self.assertIn("extraction", cfg)
        self.assertIn("primitive", cfg)
        self.assertIn("logging", cfg)

    def test_dot_notation(self):
        val = ConfigManager.get("heads.confidence_threshold")
        self.assertIsNotNone(val)
        self.assertIsInstance(val, (float, int))

    def test_missing_key_default(self):
        val = ConfigManager.get("does.not.exist", default="FALLBACK")
        self.assertEqual(val, "FALLBACK")

    def test_logger_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_base = os.path.join(tmpdir, "experiments")
            log_base = os.path.join(tmpdir, "logs")
            logger   = ExperimentLogger(base_experiments_dir=exp_base,
                                        log_base_dir=log_base)
            self.assertTrue(os.path.isdir(logger.get_exp_dir()))
            self.assertTrue(os.path.isdir(os.path.join(logger.get_exp_dir(), "plots")))
            self.assertTrue(os.path.isdir(os.path.join(log_base, "features")))

    def test_logger_save_ggl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(
                base_experiments_dir=os.path.join(tmpdir, "exp"),
                log_base_dir=os.path.join(tmpdir, "logs"),
            )
            ggl = GeometryGraphLanguage()
            ggl.add_node(GGLNode(node_id="n1", type="Part"))
            logger.save_ggl(ggl.model_dump(), filename="test.json")
            saved = os.path.join(logger.get_exp_dir(), "test.json")
            self.assertTrue(os.path.exists(saved))
            with open(saved) as f:
                data = json.load(f)
            self.assertIn("nodes", data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
