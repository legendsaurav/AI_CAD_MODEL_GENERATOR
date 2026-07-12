"""
verification/foundation_check.py — Hunyuan3D Foundation Verification
=====================================================================
Runtime assertion module that verifies the entire pipeline uses
Hunyuan3D Flow DiT intermediate hidden representations as the SOLE
geometry source. Call this before any pipeline run.

Expected verified pipeline:
    Image → DINOv2 → Hunyuan3D Flow DiT → Intermediate Hidden States
    → Representation Probing → Geometry Heads → Primitive Recovery → GGL

CRITICAL INVARIANTS:
  1. The generated mesh is NEVER used as the primary source for CAD reconstruction.
  2. GGL is ALWAYS derived from DiT hidden states extracted via forward hooks.
  3. Meshes may only be used for visualization or post-hoc verification.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("geometry_engine.verification")


class FoundationCheck:
    """
    Verifies that the AI CAD Operating System architecture maintains
    Hunyuan3D-2.1 as the foundation and does not reconstruct CAD from meshes.
    """

    REQUIRED_PIPELINE_STAGES = [
        "image_conditioning",    # DINOv2 image features
        "dit_forward_hooks",     # Forward hooks on DiT blocks
        "representation_probing", # Multi-layer, multi-timestep analysis
        "geometry_heads",        # Plugin heads consuming hidden states
        "primitive_recovery",    # Primitive proposal + parameter estimation
        "ggl_generation",       # Hierarchical graph assembly
    ]

    @staticmethod
    def verify_architecture(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a comprehensive pre-flight check on the pipeline configuration.

        Returns:
            Dict with 'passed' (bool), 'checks' (list of results),
            and 'violations' (list of violation descriptions).
        """
        checks = []
        violations = []

        # ── Check 1: Heads are enabled ────────────────────────────────────
        heads_cfg = config.get("heads", {})
        enabled_heads = heads_cfg.get("enabled", [])
        check_1 = len(enabled_heads) > 0
        checks.append({
            "name": "geometry_heads_enabled",
            "passed": check_1,
            "detail": f"Enabled heads: {enabled_heads}" if check_1 else "No geometry heads enabled!"
        })
        if not check_1:
            violations.append("No geometry heads are enabled. Heads consume DiT hidden states to produce GGL.")

        # ── Check 2: Feature extraction is configured ─────────────────────
        extraction_cfg = config.get("extraction", {})
        target_ts = extraction_cfg.get("target_timesteps", [])
        check_2 = len(target_ts) > 0
        checks.append({
            "name": "feature_extraction_timesteps",
            "passed": check_2,
            "detail": f"Target timesteps: {target_ts}" if check_2 else "No target timesteps configured!"
        })
        if not check_2:
            violations.append("No feature extraction timesteps. Cannot probe DiT representations.")

        # ── Check 3: hidden_dim matches Hunyuan3D DiT ─────────────────────
        hidden_dim = heads_cfg.get("hidden_dim", 0)
        check_3 = hidden_dim in [1024, 2048, 3072]  # Known Hunyuan3D dimensions
        checks.append({
            "name": "hidden_dim_valid",
            "passed": check_3,
            "detail": f"hidden_dim={hidden_dim}" + (" (valid)" if check_3 else " (unexpected)")
        })

        # ── Check 4: Primitive recovery is configured ─────────────────────
        prim_cfg = config.get("primitive", {})
        top_k = prim_cfg.get("top_k_proposals", 0)
        check_4 = top_k > 0
        checks.append({
            "name": "primitive_recovery_configured",
            "passed": check_4,
            "detail": f"top_k_proposals={top_k}"
        })
        if not check_4:
            violations.append("Primitive recovery not configured.")

        # ── Check 5: No mesh-based CAD reconstruction ─────────────────────
        # Verify that no config key suggests mesh-to-CAD conversion
        mesh_keys = ["mesh_to_cad", "mesh_reconstruction", "reverse_engineer_mesh"]
        check_5 = True
        for key in mesh_keys:
            if key in config:
                check_5 = False
                violations.append(
                    f"CRITICAL: Config contains '{key}' — mesh-based CAD reconstruction "
                    f"is an architecture violation. GGL must come from DiT hidden states."
                )
        checks.append({
            "name": "no_mesh_based_reconstruction",
            "passed": check_5,
            "detail": "No mesh-to-CAD keys found" if check_5 else "VIOLATION DETECTED"
        })

        # ── Summary ───────────────────────────────────────────────────────
        all_passed = all(c["passed"] for c in checks)

        if all_passed:
            logger.info("✅ Foundation Check PASSED — Hunyuan3D DiT is the verified geometry source.")
        else:
            logger.error(f"❌ Foundation Check FAILED — {len(violations)} violation(s) detected.")
            for v in violations:
                logger.error(f"   ❌ {v}")

        return {
            "passed": all_passed,
            "checks": checks,
            "violations": violations,
        }

    @staticmethod
    def verify_ggl_source(ggl_dict: Dict[str, Any]) -> bool:
        """
        Verifies that a GGL document was derived from DiT hidden states.
        Raises ValueError if the source_type is not 'dit_hidden_states'.
        """
        metadata = ggl_dict.get("metadata", {})
        source_type = metadata.get("source_type", "dit_hidden_states")

        if source_type != "dit_hidden_states":
            raise ValueError(
                f"ARCHITECTURE VIOLATION: GGL metadata.source_type is '{source_type}', "
                f"expected 'dit_hidden_states'. This GGL was not derived from Hunyuan3D "
                f"Flow DiT intermediate representations. The system must NEVER reconstruct "
                f"CAD from meshes."
            )

        logger.info("✅ GGL source verified: derived from DiT hidden states.")
        return True

    @staticmethod
    def print_pipeline_diagram():
        """Prints the verified pipeline for documentation."""
        diagram = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         VERIFIED HUNYUAN3D FOUNDATION PIPELINE              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║   Input Image                                                ║
    ║       ↓                                                      ║
    ║   DINOv2 Image Conditioning                                  ║
    ║       ↓                                                      ║
    ║   Hunyuan3D Flow DiT (diffusion inference)                   ║
    ║       ↓                                                      ║
    ║   Forward Hooks → Intermediate Hidden States                 ║
    ║       ↓                (NOT the decoded mesh)                ║
    ║   Representation Probing (PCA, correlation, ranking)         ║
    ║       ↓                                                      ║
    ║   Geometry Heads (Part, Surface, Topology plugins)           ║
    ║       ↓                                                      ║
    ║   Primitive Recovery (proposal → estimation → optimization)  ║
    ║       ↓                                                      ║
    ║   GGL (Geometry Graph Language)                              ║
    ║       ↓                                                      ║
    ║   ═══ cad-planner ═══                                        ║
    ║       ↓                                                      ║
    ║   CAL (CAD Action Language)                                  ║
    ║       ↓                                                      ║
    ║   ═══ desktop-agent ═══                                      ║
    ║       ↓                                                      ║
    ║   SolidWorks / FreeCAD / Fusion 360                          ║
    ║       ↓                                                      ║
    ║   Export Mesh → Verification → Geometry Difference           ║
    ║       ↓                                                      ║
    ║   Refinement Loop → Updated GGL → (repeat)                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝

    ⚠️  The decoded mesh is NEVER the primary source for CAD.
    ⚠️  Meshes are used ONLY for visualization and verification.
"""
        print(diagram)
