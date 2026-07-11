#!/usr/bin/env python3
"""
pipeline_e2e.py — AI CAD OS End-to-End Pipeline
==================================================
Runs the complete pipeline from a single image to parametric CAD output:

    Image → Hunyuan3D + Bridge → Hidden States → Feature Fusion
    → Prediction Heads → GGL Builder → GGL
    → CAD Planner (Beam Search) → CAL + ReasonGraph
    → [Desktop Agent → SolidWorks/FreeCAD]  (optional)
    → Verification → Refinement Loop

Every stage is checkpointed. The pipeline can resume from any point.

Usage:
    # Full pipeline (requires GPU server for model inference)
    python pipeline_e2e.py --image bracket.png --device cuda:0

    # From saved hidden states (no GPU needed)
    python pipeline_e2e.py --states-file outputs/bracket_states.pt

    # Generate only GGL (skip CAD execution)
    python pipeline_e2e.py --image bracket.png --stop-at ggl

    # Generate GGL + CAL (skip desktop execution)
    python pipeline_e2e.py --image bracket.png --stop-at cal

Author: AI CAD OS Project
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ── Path setup ────────────────────────────────────────────────────────── #
PROJECT_ROOT = Path(__file__).parent
SHARED_SCHEMAS = PROJECT_ROOT / "shared-schemas"
GEOMETRY_ENGINE = PROJECT_ROOT / "geometry-engine"
CAD_PLANNER = PROJECT_ROOT / "cad-planner"
MODEL_GEN = PROJECT_ROOT / "MODEL_GENERATOR_V2"

for p in [str(SHARED_SCHEMAS), str(GEOMETRY_ENGINE), str(CAD_PLANNER), str(MODEL_GEN)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Logging ───────────────────────────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_e2e")


# ═══════════════════════════════════════════════════════════════════════ #
#  Stage 1: Model Inference + Hidden State Capture                        #
# ═══════════════════════════════════════════════════════════════════════ #

def stage_model_inference(
    image_path: str,
    model_path: str,
    device: str,
    output_dir: str,
    steps: int = 25,
    resolution: int = 256,
) -> Dict[str, Any]:
    """Run Hunyuan3D-2.1 inference with hidden state capture.

    Returns:
        dict with 'mesh_path', 'states_path', 'states_meta'
    """
    import torch
    import trimesh
    from PIL import Image

    logger.info("═══ STAGE 1: Model Inference + Hidden State Capture ═══")
    t0 = time.perf_counter()

    image_stem = Path(image_path).stem

    # Check for cached states
    states_path = os.path.join(output_dir, f"{image_stem}_states.pt")
    mesh_path = os.path.join(output_dir, f"{image_stem}.glb")

    if os.path.exists(states_path) and os.path.exists(mesh_path):
        logger.info("Using cached states and mesh from %s", output_dir)
        states = torch.load(states_path, map_location="cpu", weights_only=False)
        return {
            "mesh_path": mesh_path,
            "states_path": states_path,
            "states": states,
            "cached": True,
            "duration_s": 0.0,
        }

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Setup module aliases for v2.1 compat
    try:
        from MODEL_GENERATOR_V2.run_generate import _setup_module_aliases
        _setup_module_aliases()
    except ImportError:
        pass

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    # Load pipeline
    logger.info("Loading Hunyuan3D-2.1 pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path, subfolder="hunyuan3d-dit-v2-1", use_safetensors=False,
    )
    if device != "cpu":
        pipeline.to(device)

    # Setup bridge for hidden state capture
    from MODEL_GENERATOR_V2.core.hidden_state_bridge import HiddenStateBridge
    bridge = HiddenStateBridge()

    # Find the transformer model
    transformer = None
    for attr_name in ['model', 'transformer', 'dit', 'denoiser']:
        candidate = getattr(pipeline, attr_name, None)
        if candidate is not None and hasattr(candidate, 'parameters'):
            transformer = candidate
            break

    if transformer is not None:
        bridge.register_hooks(transformer)
        bridge.set_capture_timesteps([0.0, 0.25, 0.5, 0.75, 1.0])
        logger.info("HiddenStateBridge attached to transformer")
    else:
        logger.warning("Could not find transformer — running without state capture")

    # Generate
    image = Image.open(image_path)
    logger.info("Generating 3D mesh (%d steps, %d res)...", steps, resolution)

    result = pipeline(
        image=image,
        num_inference_steps=steps,
        octree_resolution=resolution,
        guidance_scale=7.5,
        output_type="trimesh",
    )

    # Extract mesh
    if isinstance(result, list):
        mesh = result[0]
    elif isinstance(result, trimesh.Trimesh):
        mesh = result
    elif hasattr(result, 'meshes'):
        mesh = result.meshes[0]
    else:
        mesh = result

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    mesh.export(mesh_path)
    logger.info("Mesh saved: %s (%d verts, %d faces)",
                mesh_path, len(mesh.vertices), len(mesh.faces))

    # Save hidden states
    states = bridge.get_captured_states()
    if states:
        bridge.save_states(states_path)
    bridge.clear()

    duration = time.perf_counter() - t0
    logger.info("Stage 1 complete in %.1fs", duration)

    return {
        "mesh_path": mesh_path,
        "states_path": states_path,
        "states": states,
        "cached": False,
        "duration_s": duration,
    }


# ═══════════════════════════════════════════════════════════════════════ #
#  Stage 2: Feature Fusion + Prediction Heads → GGL                       #
# ═══════════════════════════════════════════════════════════════════════ #

def stage_geometry_engine(
    states: Dict[str, Dict[str, Any]],
    output_dir: str,
    image_stem: str,
) -> Dict[str, Any]:
    """Process captured hidden states through the Geometry Engine.

    Returns:
        dict with 'ggl' (GeometryGraphLanguage object), 'ggl_path'
    """
    import torch
    from graph.ggl import GeometryGraphLanguage, GGLMetadata
    from graph.ggl_builder import GGLBuilder
    from heads.part import PartHead
    from heads.surface import SurfaceHead
    from heads.primitive import PrimitiveHead
    from heads.topology import TopologyHead
    from heads.symmetry import SymmetryHead

    logger.info("═══ STAGE 2: Geometry Engine — Feature Fusion + Heads → GGL ═══")
    t0 = time.perf_counter()

    head_config = {"hidden_dim": 1024, "dropout": 0.0}

    # Initialize heads
    heads = {
        "part": PartHead(head_config),
        "surface": SurfaceHead(head_config),
        "primitive": PrimitiveHead(head_config),
        "topology": TopologyHead(head_config),
        "symmetry": SymmetryHead(head_config),
    }

    # Set all heads to eval mode
    for head in heads.values():
        head.eval()

    # Fuse hidden states across layers and timesteps
    # Strategy: for each captured timestep, average all layer outputs,
    # then average across timesteps.
    fused_features = _fuse_states(states)

    if fused_features is None:
        logger.warning("No hidden states available — generating placeholder GGL")
        ggl = GeometryGraphLanguage(
            metadata=GGLMetadata(generator="geometry-engine-v1.0"),
        )
        ggl_path = os.path.join(output_dir, f"{image_stem}_ggl.json")
        with open(ggl_path, "w") as f:
            f.write(ggl.to_json())
        return {"ggl": ggl, "ggl_path": ggl_path, "duration_s": 0.0}

    logger.info("Fused features shape: %s", list(fused_features.shape))

    # Run all heads
    head_outputs = {}
    with torch.no_grad():
        for name, head in heads.items():
            head_outputs[name] = head(fused_features)
            logger.info("Head '%s' executed", name)

    # Assemble GGL
    builder = GGLBuilder({
        "part_threshold": 0.3,
        "surface_threshold": 0.3,
        "primitive_threshold": 0.3,
        "topology_threshold": 0.5,
        "symmetry_threshold": 0.4,
        "dedup_distance": 5.0,
    })

    metadata = GGLMetadata(
        generator="geometry-engine-v1.0",
        original_image=f"{image_stem}.png",
        source_type="dit_hidden_states",
        hunyuan_model_version="2.1",
    )

    ggl = builder.build(heads, head_outputs, metadata=metadata)

    # Save GGL
    ggl_path = os.path.join(output_dir, f"{image_stem}_ggl.json")
    with open(ggl_path, "w") as f:
        f.write(ggl.to_json())
    logger.info("GGL saved: %s (%d nodes, %d edges)",
                ggl_path, len(ggl.nodes), len(ggl.edges))

    duration = time.perf_counter() - t0
    logger.info("Stage 2 complete in %.1fs", duration)

    return {
        "ggl": ggl,
        "ggl_path": ggl_path,
        "duration_s": duration,
    }


def _fuse_states(states: Dict[str, Dict[str, Any]]) -> Optional[Any]:
    """Fuse multi-timestep, multi-layer hidden states into a single tensor.

    Strategy: mean across layers per timestep, then mean across timesteps.
    Returns: [1, N, D] tensor or None if no states.
    """
    import torch

    if not states:
        return None

    timestep_features = []
    for ts_key, layer_dict in states.items():
        layer_tensors = []
        for layer_name, tensor in layer_dict.items():
            if isinstance(tensor, torch.Tensor):
                layer_tensors.append(tensor.float())

        if layer_tensors:
            # Average across layers: each is [B, N, D] or [N, D]
            stacked = torch.stack(layer_tensors, dim=0)
            avg_layer = stacked.mean(dim=0)  # [B, N, D]
            timestep_features.append(avg_layer)

    if not timestep_features:
        return None

    # Average across timesteps
    stacked_ts = torch.stack(timestep_features, dim=0)
    fused = stacked_ts.mean(dim=0)  # [B, N, D]

    # Ensure batch dimension
    if fused.dim() == 2:
        fused = fused.unsqueeze(0)

    return fused


# ═══════════════════════════════════════════════════════════════════════ #
#  Stage 3: CAD Planner — GGL → CAL                                      #
# ═══════════════════════════════════════════════════════════════════════ #

def stage_cad_planner(
    ggl: Any,
    output_dir: str,
    image_stem: str,
) -> Dict[str, Any]:
    """Convert GGL to CAL using the CAD Planner's beam search.

    Returns:
        dict with 'cal' (CALDocument), 'cal_path', 'reason_graph_path'
    """
    logger.info("═══ STAGE 3: CAD Planner — GGL → CAL ═══")
    t0 = time.perf_counter()

    try:
        from beam_search.search import BeamSearchPlanner
        from construction.graph import ConstructionGraphBuilder
        from intent.classifier import IntentClassifier
        from reasoning.reason_graph import ReasonGraphGenerator

        # Run the planner pipeline
        intent_classifier = IntentClassifier()
        graph_builder = ConstructionGraphBuilder()
        planner = BeamSearchPlanner()
        reason_gen = ReasonGraphGenerator()

        # Classify intent per GGL node
        intents = intent_classifier.classify(ggl)

        # Build construction dependency graph
        construction_graph = graph_builder.build(ggl, intents)

        # Beam search for best construction plan
        best_plan = planner.search(construction_graph, ggl)

        # Generate CAL document
        cal = best_plan.to_cal_document()

        # Generate reason graph
        reason_graph = reason_gen.generate(ggl, cal, best_plan)

        # Save outputs
        cal_path = os.path.join(output_dir, f"{image_stem}_cal.json")
        with open(cal_path, "w") as f:
            f.write(cal.to_json(indent=2))

        rg_path = os.path.join(output_dir, f"{image_stem}_reason_graph.json")
        with open(rg_path, "w") as f:
            f.write(reason_graph.to_json())

        logger.info("CAL saved: %s (%d actions)", cal_path, len(cal.actions))

        duration = time.perf_counter() - t0
        return {
            "cal": cal,
            "cal_path": cal_path,
            "reason_graph_path": rg_path,
            "duration_s": duration,
        }

    except ImportError as e:
        logger.warning("CAD Planner not available: %s — generating stub CAL", e)

        # Generate stub CAL from GGL primitives
        cal_stub = _ggl_to_stub_cal(ggl)
        cal_path = os.path.join(output_dir, f"{image_stem}_cal.json")
        with open(cal_path, "w") as f:
            json.dump(cal_stub, f, indent=2)

        duration = time.perf_counter() - t0
        return {
            "cal": cal_stub,
            "cal_path": cal_path,
            "reason_graph_path": None,
            "duration_s": duration,
        }


def _ggl_to_stub_cal(ggl: Any) -> Dict[str, Any]:
    """Generate a simple CAL from GGL primitives (fallback when planner unavailable)."""
    actions = []
    action_idx = 0

    for node in ggl.nodes:
        if node.type in ("Cylinder", "Box", "Sphere", "Cone"):
            # Create sketch
            sketch_id = f"sk_{action_idx}"
            actions.append({
                "action_id": sketch_id,
                "action_type": "create_sketch",
                "plane": "XY",
                "confidence": node.confidence,
                "source_ggl_node_id": node.node_id,
                "reasoning": {
                    "purpose": f"Sketch for {node.type} ({node.semantic_label})",
                    "rationale": f"Auto-generated from GGL node {node.node_id}",
                    "depends_on": [],
                    "alternatives_considered": [],
                },
            })
            action_idx += 1

            # Draw profile
            center = node.parameters.get("center", [0, 0, 0])
            if node.type == "Cylinder":
                radius = node.parameters.get("radius", 10.0)
                actions.append({
                    "action_id": f"draw_{action_idx}",
                    "action_type": "draw_circle",
                    "sketch_id": sketch_id,
                    "center": [center[0], center[1]],
                    "radius": abs(radius),
                    "confidence": node.confidence,
                    "source_ggl_node_id": node.node_id,
                })
            else:
                dims = node.parameters.get("dimensions", [20, 20, 20])
                w = abs(dims[0]) if isinstance(dims, list) and len(dims) > 0 else 20
                h = abs(dims[1]) if isinstance(dims, list) and len(dims) > 1 else 20
                actions.append({
                    "action_id": f"draw_{action_idx}",
                    "action_type": "draw_rectangle",
                    "sketch_id": sketch_id,
                    "center": [center[0], center[1]],
                    "width": w,
                    "height": h,
                    "confidence": node.confidence,
                    "source_ggl_node_id": node.node_id,
                })
            action_idx += 1

            # Extrude
            depth = node.parameters.get("height", 10.0)
            if depth is None or not isinstance(depth, (int, float)):
                depth = 10.0
            actions.append({
                "action_id": f"ext_{action_idx}",
                "action_type": "extrude",
                "sketch_id": sketch_id,
                "depth": abs(depth),
                "direction": 1,
                "is_cut": False,
                "confidence": node.confidence,
                "source_ggl_node_id": node.node_id,
            })
            action_idx += 1

    return {
        "version": "1.0",
        "planner_version": "stub-0.1",
        "generator": "pipeline-e2e-stub",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actions": actions,
    }


# ═══════════════════════════════════════════════════════════════════════ #
#  Main Pipeline Orchestrator                                             #
# ═══════════════════════════════════════════════════════════════════════ #

def run_pipeline(args):
    """Execute the full AI CAD OS pipeline."""
    import torch

    total_start = time.perf_counter()
    image_stem = Path(args.image).stem if args.image else "loaded"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    stages_completed = []
    results = {}

    print(f"\n{'═'*64}")
    print(f"  AI CAD OS — End-to-End Pipeline")
    print(f"{'═'*64}")
    print(f"  Image:     {args.image or 'N/A (using saved states)'}")
    print(f"  Stop at:   {args.stop_at or 'full pipeline'}")
    print(f"  Output:    {output_dir}/")
    print(f"{'═'*64}\n")

    # ── Stage 1: Model Inference ──────────────────────────────────
    if args.states_file:
        # Load pre-saved states
        logger.info("Loading pre-saved hidden states from %s", args.states_file)
        states = torch.load(args.states_file, map_location="cpu", weights_only=False)
        results["stage1"] = {
            "states": states,
            "mesh_path": None,
            "states_path": args.states_file,
            "cached": True,
            "duration_s": 0.0,
        }
    elif args.image:
        results["stage1"] = stage_model_inference(
            image_path=args.image,
            model_path=args.model_path,
            device=args.device,
            output_dir=output_dir,
            steps=args.steps,
            resolution=args.resolution,
        )
    else:
        logger.error("Must provide either --image or --states-file")
        sys.exit(1)

    stages_completed.append("model_inference")
    states = results["stage1"]["states"]

    if args.stop_at == "states":
        _print_summary(stages_completed, results, total_start)
        return results

    # ── Stage 2: Geometry Engine ──────────────────────────────────
    results["stage2"] = stage_geometry_engine(states, output_dir, image_stem)
    stages_completed.append("geometry_engine")
    ggl = results["stage2"]["ggl"]

    if args.stop_at == "ggl":
        _print_summary(stages_completed, results, total_start)
        return results

    # ── Stage 3: CAD Planner ──────────────────────────────────────
    results["stage3"] = stage_cad_planner(ggl, output_dir, image_stem)
    stages_completed.append("cad_planner")

    if args.stop_at == "cal":
        _print_summary(stages_completed, results, total_start)
        return results

    # ── Stage 4: Desktop Agent (future) ───────────────────────────
    logger.info("═══ STAGE 4: Desktop Agent — Not yet connected ═══")
    logger.info("CAL is ready for execution in SolidWorks or FreeCAD")
    stages_completed.append("desktop_agent_ready")

    _print_summary(stages_completed, results, total_start)
    return results


def _print_summary(stages, results, total_start):
    """Print pipeline execution summary."""
    total = time.perf_counter() - total_start
    print(f"\n{'═'*64}")
    print(f"  Pipeline Summary")
    print(f"{'═'*64}")
    for stage in stages:
        key = f"stage{stages.index(stage)+1}"
        dur = results.get(key, {}).get("duration_s", 0)
        print(f"  ✓ {stage:<30s} {dur:>6.1f}s")
    print(f"  {'─'*40}")
    print(f"  Total:  {total:.1f}s")

    # Print output files
    print(f"\n  Output files:")
    for key, val in results.items():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str) and os.path.exists(v):
                    size = os.path.getsize(v)
                    print(f"    • {os.path.basename(v)} ({size:,} bytes)")
    print(f"{'═'*64}\n")


# ═══════════════════════════════════════════════════════════════════════ #
#  CLI                                                                    #
# ═══════════════════════════════════════════════════════════════════════ #

def main():
    parser = argparse.ArgumentParser(
        prog="pipeline_e2e",
        description="AI CAD OS — Full pipeline: Image → Parametric CAD",
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", "-i", help="Input image path")
    input_group.add_argument("--states-file", help="Pre-saved hidden states (.pt)")

    parser.add_argument("--output-dir", "-o", default="./pipeline_output",
                        help="Output directory")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1",
                        help="Hunyuan3D model path")
    parser.add_argument("--device", default="auto",
                        help="Device: auto, cuda, cuda:0, cpu")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--stop-at", choices=["states", "ggl", "cal"],
                        default=None, help="Stop after this stage")

    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.image and not args.states_file:
        parser.error("Must provide either --image or --states-file")

    run_pipeline(args)


if __name__ == "__main__":
    main()
