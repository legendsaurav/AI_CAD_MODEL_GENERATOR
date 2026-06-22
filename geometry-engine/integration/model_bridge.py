"""
integration/model_bridge.py — Connects MODEL_GENERATOR_V2's DiT to the Geometry Engine
=========================================================================================
This is the critical bridge between the Hunyuan3D inference pipeline and
the geometry extraction pipeline. It:
  1. Loads the Hunyuan3D DiT model from MODEL_GENERATOR_V2
  2. Registers DiTFeatureExtractor hooks on the DiT's double_blocks/single_blocks
  3. Runs diffusion inference while hooks capture intermediate hidden states
  4. Returns the captured features for downstream geometry head processing

ARCHITECTURE INVARIANT:
  The geometry engine ALWAYS operates on DiT hidden representations.
  The decoded mesh is NEVER used as a source for CAD reconstruction.
  Meshes may only be used for visualization or verification.
"""
import os
import sys
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("geometry_engine.integration")


class ModelBridge:
    """
    Bridges MODEL_GENERATOR_V2 (Hunyuan3D DiT) with the Geometry Engine.

    Usage:
        bridge = ModelBridge(model_path="tencent/Hunyuan3D-2")
        features = bridge.extract_features("input.png",
                                            target_timesteps=[0.1, 0.5, 0.9],
                                            double_indices=[0, 4, 8, 12],
                                            single_indices=[0, 4, 8])
        # features is a dict matching DiTFeatureExtractor.features schema
    """

    def __init__(self, model_path: str = "tencent/Hunyuan3D-2", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.extractor = None
        self._dit_model = None

    def load(self):
        """
        Loads the Hunyuan3D pipeline and attaches feature extraction hooks.
        Must be called before extract_features().
        """
        try:
            # Import MODEL_GENERATOR_V2's pipeline
            # The parent directory of MODEL_GENERATOR_V2 must be on sys.path
            from MODEL_GENERATOR_V2.generation import GeometryPipeline
            from hooks.feature_extractor import DiTFeatureExtractor

            logger.info(f"Loading Hunyuan3D pipeline from {self.model_path}...")
            self.pipeline = GeometryPipeline.from_pretrained(
                model_path=self.model_path,
                preset='balanced',
            )

            # Access the DiT model inside the pipeline
            # The exact attribute name depends on MODEL_GENERATOR_V2's implementation
            if hasattr(self.pipeline, 'dit_model'):
                self._dit_model = self.pipeline.dit_model
            elif hasattr(self.pipeline, 'dit'):
                self._dit_model = self.pipeline.dit
            elif hasattr(self.pipeline, 'model'):
                self._dit_model = self.pipeline.model
            else:
                raise AttributeError(
                    "Cannot find DiT model in GeometryPipeline. "
                    "Expected attribute: dit_model, dit, or model."
                )

            # Verify DiT architecture has the expected blocks
            if not hasattr(self._dit_model, 'double_blocks') and not hasattr(self._dit_model, 'single_blocks'):
                raise AttributeError(
                    "DiT model does not have 'double_blocks' or 'single_blocks'. "
                    "Ensure this is a Hunyuan3D Flow DiT model."
                )

            # Create the feature extractor with hooks
            self.extractor = DiTFeatureExtractor(self._dit_model)
            logger.info("ModelBridge loaded successfully.")

        except ImportError as e:
            logger.warning(
                f"MODEL_GENERATOR_V2 not available: {e}. "
                f"Using mock mode. Install MODEL_GENERATOR_V2 for real inference."
            )
            self.pipeline = None
            self.extractor = None

    def extract_features(
        self,
        image_path: str,
        target_timesteps: List[float] = None,
        double_indices: List[int] = None,
        single_indices: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs Hunyuan3D DiT inference on the image and captures intermediate
        hidden states at specified layers and timesteps.

        Returns:
            Dict with keys:
                - "double_blocks": {timestep: {layer_idx: {"img": Tensor, "txt": Tensor}}}
                - "single_blocks": {timestep: {layer_idx: Tensor}}
                - "final_layer": Tensor or None
                - "source_type": "dit_hidden_states" (architecture invariant)
        """
        if target_timesteps is None:
            target_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]
        if double_indices is None:
            double_indices = [0, 4, 8, 12]
        if single_indices is None:
            single_indices = [0, 4, 8]

        if self.extractor is None:
            logger.warning("No DiT model loaded. Returning mock features.")
            return self._generate_mock_features(target_timesteps, double_indices, single_indices)

        # Configure which timesteps and layers to capture
        self.extractor.set_target_timesteps(target_timesteps)
        self.extractor.register_hooks(
            double_indices=double_indices,
            single_indices=single_indices,
        )
        self.extractor.clear_features()
        self.extractor.enabled = True

        # Run inference — hooks automatically capture features during the forward pass
        logger.info(f"Running DiT inference on {image_path}...")
        _ = self.pipeline(image_path)  # Mesh output is discarded

        # Retrieve captured features
        features = self.extractor.features.copy()
        features["source_type"] = "dit_hidden_states"  # Architecture invariant

        # Clean up hooks
        self.extractor.clear_hooks()

        logger.info(
            f"Captured features: "
            f"{len(features.get('double_blocks', {}))} timesteps, "
            f"{sum(len(v) for v in features.get('double_blocks', {}).values())} double block snapshots"
        )
        return features

    def _generate_mock_features(
        self,
        timesteps: List[float],
        double_indices: List[int],
        single_indices: List[int],
    ) -> Dict[str, Any]:
        """Generates mock features matching the real schema for testing."""
        import numpy as np

        features = {
            "double_blocks": {},
            "single_blocks": {},
            "final_layer": None,
            "source_type": "dit_hidden_states",
        }

        for t in timesteps:
            features["double_blocks"][t] = {}
            features["single_blocks"][t] = {}
            for idx in double_indices:
                import torch
                feat = torch.randn(1, 256, 1024) * (1.0 - t)
                features["double_blocks"][t][idx] = {"img": feat, "txt": feat * 0.5}
            for idx in single_indices:
                import torch
                features["single_blocks"][t][idx] = torch.randn(1, 256, 1024) * (1.0 - t)

        return features
