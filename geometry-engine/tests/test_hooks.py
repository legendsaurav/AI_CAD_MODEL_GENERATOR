"""
tests/test_hooks.py  –  Version 0 Unit Tests (backward-compatible)
===================================================================
Tests the DiTFeatureExtractor hook mechanism, GGL schema, and ConfigManager.
Kept minimal here since test_pipeline.py has exhaustive coverage.
"""
import os
import sys
import unittest

import torch
import torch.nn as nn

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hooks.feature_extractor import DiTFeatureExtractor
from graph.ggl import GeometryGraphLanguage, GGLNode, GGLEdge
from utils.config import ConfigManager


# ── Mock DiT ──────────────────────────────────────────────────────────────

class MockDoubleBlock(nn.Module):
    def forward(self, img, txt, vec=None, pe=None):
        return img * 2, txt * 2

class MockSingleBlock(nn.Module):
    def forward(self, x, vec=None, pe=None):
        return x * 3

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


# ── Tests ──────────────────────────────────────────────────────────────────

class TestDiTFeatureExtractor(unittest.TestCase):

    def setUp(self):
        ConfigManager.reset()
        self.model     = MockHunyuanDiT()
        self.extractor = DiTFeatureExtractor(self.model)

    def tearDown(self):
        self.extractor.clear_hooks()

    def test_feature_capture(self):
        self.extractor.register_hooks(double_indices=[1, 3], single_indices=[0, 7])
        self.extractor.set_target_timesteps([0.5])

        img_in = torch.randn(2, 3072, 1024)
        txt_in = torch.randn(2, 1370, 1024)
        self.model(img_in, txt_in, timestep=0.5)

        feats = self.extractor.features
        self.assertIn(0.5, feats["double_blocks"])
        self.assertIn(0.5, feats["single_blocks"])

        db = feats["double_blocks"][0.5]
        self.assertIn(1, db); self.assertIn(3, db); self.assertNotIn(0, db)
        self.assertEqual(db[1]["img"].shape, (2, 3072, 1024))
        self.assertEqual(db[1]["txt"].shape, (2, 1370, 1024))

        sb = feats["single_blocks"][0.5]
        self.assertIn(0, sb); self.assertIn(7, sb)
        self.assertEqual(sb[0].shape, (2, 4442, 1024))

    def test_no_capture_at_wrong_timestep(self):
        self.extractor.register_hooks(double_indices=[0], single_indices=[0])
        self.extractor.set_target_timesteps([0.5])
        self.extractor.clear_features()

        img = torch.randn(2, 3072, 1024)
        txt = torch.randn(2, 1370, 1024)
        self.model(img, txt, timestep=0.8)
        self.assertEqual(len(self.extractor.features["double_blocks"]), 0)

    def test_clear_hooks(self):
        self.extractor.register_hooks(double_indices=[0], single_indices=[0])
        self.extractor.clear_hooks()
        self.assertEqual(len(self.extractor.hooks), 0)


class TestGGLSchema(unittest.TestCase):

    def test_round_trip_json(self):
        ggl = GeometryGraphLanguage()
        ggl.add_node(GGLNode(node_id="n1", type="Part"))
        ggl.add_node(GGLNode(node_id="n2", type="Cylinder",
                             confidence=0.91, parameters={"radius": 5.0}))
        ggl.add_edge(GGLEdge(source_id="n1", target_id="n2", relation="Contains"))
        restored = GeometryGraphLanguage.from_json(ggl.to_json())
        self.assertEqual(len(restored.nodes), 2)
        self.assertEqual(len(restored.edges), 1)
        self.assertAlmostEqual(restored.nodes[1].parameters["radius"], 5.0)

    def test_confidence_bounds(self):
        with self.assertRaises(Exception):
            GGLNode(node_id="bad", type="X", confidence=1.5)


class TestConfigManager(unittest.TestCase):

    def setUp(self):
        ConfigManager.reset()

    def test_loads_default_config(self):
        cfg = ConfigManager.get_all()
        self.assertIn("heads", cfg)
        self.assertIn("extraction", cfg)

    def test_dot_notation(self):
        val = ConfigManager.get("heads.confidence_threshold")
        self.assertIsNotNone(val)

    def test_missing_key_default(self):
        self.assertEqual(ConfigManager.get("no.such.key", default=99), 99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
