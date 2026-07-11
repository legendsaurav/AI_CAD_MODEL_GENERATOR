#!/usr/bin/env python3
"""
generate_cad.py - Advanced Image-to-Parametric-CAD Pipeline (V2)
================================================================
Generates precise SolidWorks-compatible parametric CAD data from images
using Hunyuan3D-2.1 mesh generation + advanced geometry analysis.

Key improvements over V1:
  - Surface segmentation (region growing on face normals)
  - Cylinder/hole detection with precise radius + depth
  - Pocket and slot detection via cross-section profiling
  - Step/shoulder detection from area change analysis
  - Fillet detection on curved transition surfaces
  - Precise dimension extraction from fitted primitives
  - Ordered SolidWorks construction tree (base -> boss -> cut -> fillet)
  - SolidWorks API call mapping for each operation

Usage:
    python3 generate_cad.py --image input.png --device cuda:0
    python3 generate_cad.py --mesh outputs/input.glb --skip-inference
"""
import argparse, json, logging, os, sys, time, math, collections
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

PROJECT_ROOT = Path(__file__).parent
MODEL_GEN = PROJECT_ROOT / "MODEL_GENERATOR_V2"
for p in [str(PROJECT_ROOT), str(MODEL_GEN)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_cad")


# ================================================================ #
#  Data Structures                                                   #
# ================================================================ #

@dataclass
class SurfacePatch:
    """A segmented surface region of the mesh."""
    patch_id: int
    face_indices: Any  # np.ndarray
    avg_normal: List[float] = field(default_factory=list)
    area: float = 0.0
    centroid: List[float] = field(default_factory=list)
    surface_type: str = "unknown"  # planar, cylindrical, spherical, conical, freeform
    fit_params: Dict[str, Any] = field(default_factory=dict)
    fit_error: float = 1.0


@dataclass
class DetectedFeature:
    """A detected manufacturing feature with precise dimensions."""
    feature_id: str
    feature_type: str  # base_extrude, boss, pocket, through_hole, blind_hole,
                        # counterbore, slot, fillet, chamfer, rib, shell
    primitive: str      # box, cylinder, sphere, cone, torus
    position: List[float] = field(default_factory=lambda: [0, 0, 0])
    dimensions: Dict[str, float] = field(default_factory=dict)
    axis: List[float] = field(default_factory=lambda: [0, 0, 1])
    confidence: float = 0.0
    parent_feature: Optional[str] = None
    sketch_plane: str = "XY"
    is_subtractive: bool = False
    notes: str = ""


# ================================================================ #
#  Stage 1: Model Inference (unchanged from V1)                      #
# ================================================================ #

def stage1_model_inference(image_path, model_path, device, output_dir,
                           steps=25, resolution=256):
    """Generate 3D mesh from image using Hunyuan3D-2.1."""
    import torch
    logger.info("=== STAGE 1: Model Inference + Hidden State Capture ===")
    t0 = time.perf_counter()
    image_stem = Path(image_path).stem
    mesh_path = os.path.join(output_dir, f"{image_stem}.glb")
    states_path = os.path.join(output_dir, f"{image_stem}_states.pt")

    if os.path.exists(mesh_path):
        logger.info("Found cached mesh: %s", mesh_path)
        states = {}
        if os.path.exists(states_path):
            states = torch.load(states_path, map_location="cpu", weights_only=False)
        return {"mesh_path": mesh_path, "states_path": states_path,
                "states": states, "cached": True, "duration_s": 0.0}

    try:
        from MODEL_GENERATOR_V2.run_generate import _setup_module_aliases
        _setup_module_aliases()
    except ImportError:
        pass

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    logger.info("Loading Hunyuan3D-2.1 pipeline...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path, subfolder="hunyuan3d-dit-v2-1", use_safetensors=False)
    if device != "cpu":
        pipeline.to(device)

    from MODEL_GENERATOR_V2.core.hidden_state_bridge import HiddenStateBridge
    bridge = HiddenStateBridge()
    transformer = None
    for name in ['model', 'transformer', 'dit', 'denoiser']:
        c = getattr(pipeline, name, None)
        if c is not None and hasattr(c, 'parameters'):
            transformer = c
            break
    if transformer:
        bridge.register_hooks(transformer)
        bridge.set_capture_timesteps([0.0, 0.25, 0.5, 0.75, 1.0])

    logger.info("Generating mesh (%d steps, %d res)...", steps, resolution)
    from PIL import Image
    img = Image.open(image_path).convert("RGBA")
    mesh = pipeline(img, num_inference_steps=steps, octree_resolution=resolution)[0]
    mesh.export(mesh_path)
    nv = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0
    nf = len(mesh.faces) if hasattr(mesh, 'faces') else 0
    logger.info("Mesh: %s (%d verts, %d faces)", mesh_path, nv, nf)

    states = bridge.get_captured_states()
    if states:
        bridge.save_states(states_path)
    bridge.clear()
    dur = time.perf_counter() - t0
    logger.info("Stage 1: %.1fs", dur)
    return {"mesh_path": mesh_path, "states_path": states_path,
            "states": states, "cached": False, "duration_s": dur}


# ================================================================ #
#  Stage 2: Advanced Mesh Analysis Engine                            #
# ================================================================ #

class AdvancedMeshAnalyzer:
    """
    Comprehensive mesh analysis engine that detects manufacturing features.

    Pipeline:
        1. Preprocess (center, align, scale)
        2. Segment surfaces by face normal region growing
        3. Classify each patch (planar, cylindrical, spherical)
        4. Detect features (holes, pockets, bosses, steps, fillets)
        5. Extract precise dimensions
        6. Detect symmetry and patterns
    """

    NORMAL_ANGLE_THRESHOLD = 15.0   # degrees for region growing
    MIN_PATCH_FACES = 20            # ignore tiny patches
    CYLINDER_FIT_TOLERANCE = 0.12   # relative radius std for cylinder
    HOLE_MAX_RADIUS_RATIO = 0.45    # max hole radius vs part size
    FILLET_MAX_RADIUS_RATIO = 0.08  # max fillet radius vs part size

    def __init__(self, mesh_path: str):
        import trimesh
        import numpy as np
        self.np = np
        self.trimesh = trimesh

        logger.info("Loading mesh: %s", mesh_path)
        self.mesh = trimesh.load(mesh_path, force='mesh')
        self.features: List[DetectedFeature] = []
        self.patches: List[SurfacePatch] = []
        self._feature_counter = 0

        self._preprocess()

    def _next_id(self, prefix: str) -> str:
        self._feature_counter += 1
        return f"{prefix}_{self._feature_counter:03d}"

    # ---- Preprocessing ----

    def _preprocess(self):
        """Center mesh, compute oriented bounding box, normalize scale."""
        np = self.np
        mesh = self.mesh

        # Center at origin
        mesh.vertices -= mesh.centroid

        # Try oriented bounding box alignment
        try:
            obb_transform = mesh.principal_inertia_transform
            mesh.apply_transform(obb_transform)
        except Exception:
            pass

        self.bounds = mesh.bounds.copy()
        self.extents = mesh.extents.copy()
        self.max_extent = float(max(self.extents))

        # Compute volume metrics
        try:
            self.volume = float(mesh.volume) if mesh.is_watertight else float(mesh.convex_hull.volume)
        except Exception:
            self.volume = float(self.extents[0] * self.extents[1] * self.extents[2])

        self.surface_area = float(mesh.area)

        # Convex hull volume ratio (indicates how many cuts/holes exist)
        try:
            ch_vol = float(mesh.convex_hull.volume)
            self.convexity_ratio = self.volume / max(ch_vol, 1e-10)
        except Exception:
            self.convexity_ratio = 1.0

        logger.info("  Extents: %.3f x %.3f x %.3f", *self.extents)
        logger.info("  Volume: %.4f, Surface area: %.4f", self.volume, self.surface_area)
        logger.info("  Convexity ratio: %.3f (1.0=fully convex, <1.0=has cuts/holes)",
                     self.convexity_ratio)

    # ---- Surface Segmentation ----

    def _segment_surfaces(self):
        """Segment mesh faces into patches using region growing on normals."""
        np = self.np
        mesh = self.mesh
        normals = mesh.face_normals
        n_faces = len(normals)

        # Build face adjacency list
        adj = [[] for _ in range(n_faces)]
        for f1, f2 in mesh.face_adjacency:
            adj[f1].append(f2)
            adj[f2].append(f1)

        visited = np.zeros(n_faces, dtype=bool)
        cos_thresh = np.cos(np.radians(self.NORMAL_ANGLE_THRESHOLD))

        patch_id = 0
        for seed in range(n_faces):
            if visited[seed]:
                continue
            # Region growing from seed
            patch_faces = []
            queue = collections.deque([seed])
            visited[seed] = True
            seed_normal = normals[seed].copy()

            while queue:
                face = queue.popleft()
                patch_faces.append(face)
                for nb in adj[face]:
                    if not visited[nb]:
                        # Check angle with SEED normal (not previous face)
                        # This prevents drift
                        if np.dot(seed_normal, normals[nb]) > cos_thresh:
                            visited[nb] = True
                            queue.append(nb)

            if len(patch_faces) >= self.MIN_PATCH_FACES:
                fi = np.array(patch_faces)
                avg_n = normals[fi].mean(axis=0)
                norm_len = np.linalg.norm(avg_n)
                if norm_len > 0:
                    avg_n /= norm_len

                patch_area = float(mesh.area_faces[fi].sum())
                patch_centroid = mesh.triangles_center[fi].mean(axis=0).tolist()

                self.patches.append(SurfacePatch(
                    patch_id=patch_id,
                    face_indices=fi,
                    avg_normal=avg_n.tolist(),
                    area=patch_area,
                    centroid=patch_centroid,
                ))
                patch_id += 1

        # Sort patches by area (largest first)
        self.patches.sort(key=lambda p: p.area, reverse=True)
        logger.info("  Segmented into %d surface patches", len(self.patches))

    # ---- Primitive Fitting ----

    def _classify_patches(self):
        """Classify each surface patch as planar, cylindrical, spherical, or freeform."""
        np = self.np
        mesh = self.mesh

        for patch in self.patches:
            fi = patch.face_indices
            faces = mesh.faces[fi]
            vert_ids = np.unique(faces.ravel())
            verts = mesh.vertices[vert_ids]
            normals = mesh.face_normals[fi]

            if len(verts) < 6:
                patch.surface_type = "freeform"
                continue

            # ---- Test planarity ----
            centroid = verts.mean(axis=0)
            centered = verts - centroid
            try:
                _, s, vh = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                patch.surface_type = "freeform"
                continue

            # Planarity = ratio of smallest singular value to largest
            planarity = s[2] / max(s[0], 1e-10)

            if planarity < 0.03:
                patch.surface_type = "planar"
                plane_normal = vh[2]
                # Ensure normal points same way as face normals
                if np.dot(plane_normal, np.array(patch.avg_normal)) < 0:
                    plane_normal = -plane_normal
                patch.fit_params = {
                    "normal": plane_normal.tolist(),
                    "point": centroid.tolist(),
                    "offset": float(np.dot(centroid, plane_normal)),
                }
                patch.fit_error = planarity
                continue

            # ---- Test cylindricity ----
            # For a cylinder, face normals span a plane (perpendicular to axis)
            # The direction with smallest normal variance = cylinder axis
            normal_cov = np.cov(normals.T)
            eigvals, eigvecs = np.linalg.eigh(normal_cov)
            axis = eigvecs[:, np.argmin(eigvals)]

            # Project face centers onto plane perpendicular to axis
            centers = mesh.triangles_center[fi]
            along_axis = np.outer(centers @ axis, axis)
            projected = centers - along_axis
            proj_center = projected.mean(axis=0)
            distances = np.linalg.norm(projected - proj_center, axis=1)
            radius = float(distances.mean())
            radius_std = float(distances.std())
            radius_rel_err = radius_std / max(radius, 1e-10)

            if radius_rel_err < self.CYLINDER_FIT_TOLERANCE:
                patch.surface_type = "cylindrical"
                # Compute height along axis
                axis_proj = centers @ axis
                height = float(axis_proj.max() - axis_proj.min())
                mid_axis = float((axis_proj.max() + axis_proj.min()) / 2)
                cyl_center = proj_center + mid_axis * axis

                patch.fit_params = {
                    "axis": axis.tolist(),
                    "center": cyl_center.tolist(),
                    "radius": radius,
                    "height": height,
                    "is_concave": bool(np.dot(
                        np.array(patch.avg_normal),
                        (np.array(patch.centroid) - cyl_center.tolist())
                    ) < 0),
                }
                patch.fit_error = radius_rel_err
                continue

            # ---- Test sphericity ----
            # For a sphere, all normals point away from a common center
            # Use least squares: minimize sum(|p_i - c| - R)^2
            try:
                # Simple approach: center estimate
                A = 2 * (verts[1:] - verts[0])
                b = np.sum(verts[1:]**2 - verts[0]**2, axis=1)
                sph_center = np.linalg.lstsq(A, b, rcond=None)[0]
                sph_dists = np.linalg.norm(verts - sph_center, axis=1)
                sph_radius = float(sph_dists.mean())
                sph_err = float(sph_dists.std()) / max(sph_radius, 1e-10)

                if sph_err < 0.08:
                    patch.surface_type = "spherical"
                    patch.fit_params = {
                        "center": sph_center.tolist(),
                        "radius": sph_radius,
                    }
                    patch.fit_error = sph_err
                    continue
            except Exception:
                pass

            patch.surface_type = "freeform"
            patch.fit_error = min(planarity, radius_rel_err)

        # Log classification results
        type_counts = collections.Counter(p.surface_type for p in self.patches)
        logger.info("  Patch classification: %s", dict(type_counts))

    # ---- Feature Detection ----

    def _detect_base_body(self):
        """Identify the base/stock body from the largest planar patches."""
        np = self.np

        # Find the two largest opposing planar patches = top and bottom faces
        planar_patches = [p for p in self.patches if p.surface_type == "planar"]
        if len(planar_patches) < 2:
            # Fallback: use bounding box
            self.features.append(DetectedFeature(
                feature_id=self._next_id("base"),
                feature_type="base_extrude",
                primitive="box",
                position=[0, 0, 0],
                dimensions={
                    "width": round(float(self.extents[0]), 4),
                    "height": round(float(self.extents[1]), 4),
                    "depth": round(float(self.extents[2]), 4),
                },
                confidence=0.5,
                sketch_plane="XY",
                notes="Bounding box fallback - no clear planar faces detected",
            ))
            return

        # Find opposing pairs (normals pointing in opposite directions)
        best_pair = None
        best_area = 0
        for i, p1 in enumerate(planar_patches):
            for p2 in planar_patches[i+1:]:
                n1 = np.array(p1.avg_normal)
                n2 = np.array(p2.avg_normal)
                if np.dot(n1, n2) < -0.9:
                    pair_area = p1.area + p2.area
                    if pair_area > best_area:
                        best_area = pair_area
                        best_pair = (p1, p2)

        if best_pair:
            p1, p2 = best_pair
            # The normal of these faces defines the extrusion direction
            extrude_axis = np.array(p1.avg_normal)
            c1 = np.array(p1.centroid)
            c2 = np.array(p2.centroid)
            depth = float(abs(np.dot(c2 - c1, extrude_axis)))

            # Determine sketch plane from extrusion direction
            abs_axis = np.abs(extrude_axis)
            dominant = int(np.argmax(abs_axis))
            plane_names = ["YZ", "XZ", "XY"]
            sketch_plane = plane_names[dominant]

            # Width and height = extents in the other two directions
            dims = list(self.extents)
            width = float(dims[(dominant + 1) % 3])
            height = float(dims[(dominant + 2) % 3])

            self.features.append(DetectedFeature(
                feature_id=self._next_id("base"),
                feature_type="base_extrude",
                primitive="box",
                position=[0, 0, 0],
                dimensions={
                    "width": round(width, 4),
                    "height": round(height, 4),
                    "depth": round(depth, 4),
                },
                axis=extrude_axis.tolist(),
                confidence=min(best_area / max(self.surface_area, 1e-10) * 3, 0.95),
                sketch_plane=sketch_plane,
                notes=f"Base body from opposing planar faces (area ratio: {best_area/self.surface_area:.2f})",
            ))
        else:
            # Use largest planar face as base
            p = planar_patches[0]
            n = np.array(p.avg_normal)
            abs_n = np.abs(n)
            dominant = int(np.argmax(abs_n))
            plane_names = ["YZ", "XZ", "XY"]

            self.features.append(DetectedFeature(
                feature_id=self._next_id("base"),
                feature_type="base_extrude",
                primitive="box",
                position=[0, 0, 0],
                dimensions={
                    "width": round(float(self.extents[(dominant + 1) % 3]), 4),
                    "height": round(float(self.extents[(dominant + 2) % 3]), 4),
                    "depth": round(float(self.extents[dominant]), 4),
                },
                axis=n.tolist(),
                confidence=0.6,
                sketch_plane=plane_names[dominant],
            ))

    def _detect_cylindrical_features(self):
        """Detect holes, pins, and cylindrical bosses from cylindrical patches."""
        np = self.np

        cyl_patches = [p for p in self.patches if p.surface_type == "cylindrical"]
        logger.info("  Analyzing %d cylindrical patches...", len(cyl_patches))

        for patch in cyl_patches:
            params = patch.fit_params
            radius = params["radius"]
            height = params["height"]
            center = params["center"]
            axis = np.array(params["axis"])
            is_concave = params.get("is_concave", False)

            # Skip very large cylinders (likely part of the base body)
            if radius > self.max_extent * self.HOLE_MAX_RADIUS_RATIO:
                continue

            # Check if it's a fillet (small radius, curved transition)
            if radius < self.max_extent * self.FILLET_MAX_RADIUS_RATIO:
                self.features.append(DetectedFeature(
                    feature_id=self._next_id("fillet"),
                    feature_type="fillet",
                    primitive="torus",
                    position=[round(c, 4) for c in center],
                    dimensions={
                        "radius": round(radius, 4),
                        "length": round(height, 4),
                    },
                    axis=axis.tolist(),
                    confidence=round(1.0 - patch.fit_error, 3),
                    is_subtractive=False,
                    notes=f"Fillet/round edge (r={radius:.3f})",
                ))
                continue

            # Determine hole vs boss
            if is_concave:
                # Concave cylinder = HOLE
                # Check if through or blind
                is_through = height > self.extents.min() * 0.85
                hole_type = "through_hole" if is_through else "blind_hole"

                # Determine axis orientation
                abs_axis = np.abs(axis)
                dominant = int(np.argmax(abs_axis))
                plane_names = ["YZ", "XZ", "XY"]

                self.features.append(DetectedFeature(
                    feature_id=self._next_id("hole"),
                    feature_type=hole_type,
                    primitive="cylinder",
                    position=[round(c, 4) for c in center],
                    dimensions={
                        "radius": round(radius, 4),
                        "diameter": round(radius * 2, 4),
                        "depth": round(height, 4),
                        "is_through": is_through,
                    },
                    axis=axis.tolist(),
                    confidence=round(1.0 - patch.fit_error, 3),
                    parent_feature="base_001",
                    sketch_plane=plane_names[dominant],
                    is_subtractive=True,
                    notes=f"{'Through' if is_through else 'Blind'} hole d={radius*2:.3f}",
                ))
            else:
                # Convex cylinder = boss/pin
                abs_axis = np.abs(axis)
                dominant = int(np.argmax(abs_axis))
                plane_names = ["YZ", "XZ", "XY"]

                self.features.append(DetectedFeature(
                    feature_id=self._next_id("boss"),
                    feature_type="boss",
                    primitive="cylinder",
                    position=[round(c, 4) for c in center],
                    dimensions={
                        "radius": round(radius, 4),
                        "diameter": round(radius * 2, 4),
                        "height": round(height, 4),
                    },
                    axis=axis.tolist(),
                    confidence=round(1.0 - patch.fit_error, 3),
                    parent_feature="base_001",
                    sketch_plane=plane_names[dominant],
                    is_subtractive=False,
                    notes=f"Cylindrical boss d={radius*2:.3f} h={height:.3f}",
                ))

    def _detect_pockets_via_cross_sections(self):
        """Detect pockets and steps by analyzing cross-section area changes."""
        np = self.np
        mesh = self.mesh

        primary_axis = int(np.argmax(self.extents))
        normal = np.zeros(3)
        normal[primary_axis] = 1.0

        z_min = float(self.bounds[0][primary_axis])
        z_max = float(self.bounds[1][primary_axis])
        total_h = z_max - z_min

        n_sections = min(40, max(10, int(total_h * 20)))
        heights = np.linspace(z_min + total_h * 0.02, z_max - total_h * 0.02, n_sections)

        section_data = []
        for h in heights:
            origin = np.zeros(3)
            origin[primary_axis] = h
            try:
                section = mesh.section(plane_origin=origin, plane_normal=normal)
                if section is not None:
                    path_2d, to_2d = section.to_planar()
                    if path_2d is not None and hasattr(path_2d, 'area'):
                        area = float(path_2d.area)
                        # Count distinct closed polygons (inner loops = holes)
                        n_polygons = len(path_2d.polygons_full) if hasattr(path_2d, 'polygons_full') else 0
                        section_data.append({
                            "height": float(h),
                            "area": area,
                            "n_polygons": n_polygons,
                            "path_2d": path_2d,
                            "transform": to_2d,
                        })
                    else:
                        section_data.append({"height": float(h), "area": 0, "n_polygons": 0})
                else:
                    section_data.append({"height": float(h), "area": 0, "n_polygons": 0})
            except Exception:
                section_data.append({"height": float(h), "area": 0, "n_polygons": 0})

        if not section_data:
            return

        areas = [s["area"] for s in section_data]
        max_area = max(areas) if areas else 1
        if max_area < 1e-10:
            return

        # Detect significant area transitions = feature boundaries
        transitions = []
        for i in range(1, len(areas)):
            change_ratio = (areas[i] - areas[i-1]) / max_area
            if abs(change_ratio) > 0.10:
                transitions.append({
                    "index": i,
                    "height": section_data[i]["height"],
                    "change": change_ratio,
                    "area_before": areas[i-1],
                    "area_after": areas[i],
                })

        # Group consecutive same-direction transitions
        for t in transitions:
            if t["change"] < -0.10:
                # Area DECREASED = step down or start of pocket
                abs_axis = np.abs(normal)
                dominant = int(np.argmax(abs_axis))
                plane_names = ["YZ", "XZ", "XY"]

                # Estimate pocket dimensions from area difference
                area_diff = t["area_before"] - t["area_after"]
                side_length = math.sqrt(abs(area_diff)) if area_diff > 0 else 0

                if side_length > self.max_extent * 0.05:
                    self.features.append(DetectedFeature(
                        feature_id=self._next_id("pocket"),
                        feature_type="pocket",
                        primitive="box",
                        position=[0, 0, round(t["height"], 4)],
                        dimensions={
                            "width": round(side_length, 4),
                            "height": round(side_length, 4),
                            "depth": round(total_h * 0.1, 4),
                        },
                        axis=normal.tolist(),
                        confidence=min(abs(t["change"]) * 2, 0.85),
                        parent_feature="base_001",
                        sketch_plane=plane_names[dominant],
                        is_subtractive=True,
                        notes=f"Pocket at h={t['height']:.3f} (area change: {t['change']*100:.1f}%)",
                    ))

            elif t["change"] > 0.10:
                # Area INCREASED = step up or flange/boss
                abs_axis = np.abs(normal)
                dominant = int(np.argmax(abs_axis))
                plane_names = ["YZ", "XZ", "XY"]
                area_diff = t["area_after"] - t["area_before"]
                side_length = math.sqrt(abs(area_diff)) if area_diff > 0 else 0

                if side_length > self.max_extent * 0.05:
                    self.features.append(DetectedFeature(
                        feature_id=self._next_id("step"),
                        feature_type="boss",
                        primitive="box",
                        position=[0, 0, round(t["height"], 4)],
                        dimensions={
                            "width": round(side_length, 4),
                            "height": round(side_length, 4),
                            "depth": round(total_h * 0.1, 4),
                        },
                        axis=normal.tolist(),
                        confidence=min(abs(t["change"]) * 2, 0.80),
                        parent_feature="base_001",
                        sketch_plane=plane_names[dominant],
                        is_subtractive=False,
                        notes=f"Step/flange at h={t['height']:.3f} (area change: {t['change']*100:.1f}%)",
                    ))

        logger.info("  Cross-section analysis: %d transitions, %d features added",
                     len(transitions),
                     sum(1 for f in self.features if "pocket" in f.feature_id or "step" in f.feature_id))

    def _detect_holes_from_sections(self):
        """Detect holes by finding inner loops in cross-sections."""
        np = self.np
        mesh = self.mesh

        for axis_idx in range(3):
            normal = np.zeros(3)
            normal[axis_idx] = 1.0
            origin = np.zeros(3)

            try:
                section = mesh.section(plane_origin=origin, plane_normal=normal)
                if section is None:
                    continue
                path_2d, to_2d = section.to_planar()
                if path_2d is None:
                    continue

                # Look for inner polygons (holes show as inner contours)
                if hasattr(path_2d, 'polygons_full'):
                    for poly in path_2d.polygons_full:
                        # Check for interior rings (holes in the cross-section)
                        if hasattr(poly, 'interiors'):
                            for interior in poly.interiors:
                                coords = np.array(interior.coords)
                                if len(coords) < 4:
                                    continue
                                # Check circularity
                                center_2d = coords.mean(axis=0)
                                dists = np.linalg.norm(coords - center_2d, axis=1)
                                mean_r = float(dists.mean())
                                std_r = float(dists.std())
                                circularity = 1.0 - std_r / max(mean_r, 1e-10)

                                if circularity > 0.80 and mean_r < self.max_extent * 0.4:
                                    # Check if already detected
                                    duplicate = False
                                    for f in self.features:
                                        if f.feature_type in ("through_hole", "blind_hole"):
                                            fr = f.dimensions.get("radius", 0)
                                            if abs(fr - mean_r) < mean_r * 0.2:
                                                duplicate = True
                                                break
                                    if duplicate:
                                        continue

                                    plane_names = ["YZ", "XZ", "XY"]
                                    depth = float(self.extents[axis_idx])

                                    pos_3d = [0.0, 0.0, 0.0]
                                    other_axes = [(axis_idx + 1) % 3, (axis_idx + 2) % 3]
                                    pos_3d[other_axes[0]] = float(center_2d[0])
                                    pos_3d[other_axes[1]] = float(center_2d[1])

                                    self.features.append(DetectedFeature(
                                        feature_id=self._next_id("hole"),
                                        feature_type="through_hole",
                                        primitive="cylinder",
                                        position=[round(p, 4) for p in pos_3d],
                                        dimensions={
                                            "radius": round(mean_r, 4),
                                            "diameter": round(mean_r * 2, 4),
                                            "depth": round(depth, 4),
                                            "is_through": True,
                                        },
                                        axis=normal.tolist(),
                                        confidence=round(circularity, 3),
                                        parent_feature="base_001",
                                        sketch_plane=plane_names[axis_idx],
                                        is_subtractive=True,
                                        notes=f"Hole from section analysis (circularity={circularity:.2f})",
                                    ))
            except Exception:
                continue

    def _detect_symmetry(self):
        """Detect symmetry planes for the overall part."""
        np = self.np
        mesh = self.mesh
        self.symmetry_planes = []

        for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
            verts = mesh.vertices.copy()
            mirrored = verts.copy()
            mirrored[:, axis_idx] *= -1

            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                dists, _ = tree.query(mirrored, k=1)
                avg_dist = float(dists.mean())
                score = 1.0 - min(avg_dist / max(self.max_extent * 0.005, 1e-10), 1.0)
                if score > 0.85:
                    self.symmetry_planes.append(axis_name)
                    logger.info("  Symmetry: %s-plane (score=%.2f)", axis_name, score)
            except ImportError:
                # scipy not available - skip symmetry detection
                break

    def _merge_duplicate_features(self):
        """Remove duplicate features (same type, similar position and size)."""
        np = self.np
        if len(self.features) < 2:
            return

        unique = [self.features[0]]
        for f in self.features[1:]:
            is_dup = False
            for u in unique:
                if f.feature_type == u.feature_type:
                    pos_dist = np.linalg.norm(
                        np.array(f.position) - np.array(u.position))
                    if pos_dist < self.max_extent * 0.05:
                        # Check dimension similarity
                        f_r = f.dimensions.get("radius", f.dimensions.get("width", 0))
                        u_r = u.dimensions.get("radius", u.dimensions.get("width", 0))
                        if abs(f_r - u_r) < max(f_r, u_r, 1e-10) * 0.25:
                            is_dup = True
                            break
            if not is_dup:
                unique.append(f)

        removed = len(self.features) - len(unique)
        if removed > 0:
            logger.info("  Removed %d duplicate features", removed)
        self.features = unique

    # ---- Main Analysis Pipeline ----

    def analyze(self) -> List[DetectedFeature]:
        """Run the full analysis pipeline."""
        logger.info("Running advanced mesh analysis...")

        self._segment_surfaces()
        self._classify_patches()
        self._detect_base_body()
        self._detect_cylindrical_features()
        self._detect_pockets_via_cross_sections()
        self._detect_holes_from_sections()
        self._detect_symmetry()
        self._merge_duplicate_features()

        # Summary
        type_counts = collections.Counter(f.feature_type for f in self.features)
        logger.info("  Detected %d features: %s", len(self.features), dict(type_counts))

        return self.features


def stage2_geometry_analysis(mesh_path, output_dir, image_stem):
    """Stage 2: Analyze mesh geometry to produce feature list + GGL."""
    logger.info("=== STAGE 2: Advanced Geometry Analysis ===")
    t0 = time.perf_counter()

    analyzer = AdvancedMeshAnalyzer(mesh_path)
    features = analyzer.analyze()

    # Build GGL document
    nodes = []
    edges = []
    for f in features:
        nodes.append({
            "node_id": f.feature_id,
            "node_type": f.primitive,
            "feature_type": f.feature_type,
            "semantic_label": f.notes,
            "confidence": f.confidence,
            "position": f.position,
            "dimensions": f.dimensions,
            "axis": f.axis,
            "sketch_plane": f.sketch_plane,
            "is_subtractive": f.is_subtractive,
        })
        if f.parent_feature:
            edges.append({
                "edge_id": f"edge_{f.feature_id}",
                "source_id": f.parent_feature,
                "target_id": f.feature_id,
                "edge_type": "contains",
            })

    ggl_doc = {
        "version": "2.0",
        "generator": "generate_cad_v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_image": f"{image_stem}.png",
        "mesh_file": os.path.basename(mesh_path),
        "analysis_method": "surface_segmentation + cross_section + primitive_fitting",
        "symmetry_planes": getattr(analyzer, 'symmetry_planes', []),
        "mesh_stats": {
            "extents": [round(float(e), 4) for e in analyzer.extents],
            "volume": round(analyzer.volume, 6),
            "surface_area": round(analyzer.surface_area, 6),
            "convexity_ratio": round(analyzer.convexity_ratio, 4),
            "n_patches": len(analyzer.patches),
        },
        "nodes": nodes,
        "edges": edges,
    }

    ggl_path = os.path.join(output_dir, f"{image_stem}_ggl.json")
    with open(ggl_path, "w") as f:
        json.dump(ggl_doc, f, indent=2)
    logger.info("GGL saved: %s (%d nodes, %d edges)", ggl_path, len(nodes), len(edges))

    dur = time.perf_counter() - t0
    logger.info("Stage 2: %.1fs", dur)
    return {"features": features, "ggl": ggl_doc, "ggl_path": ggl_path, "duration_s": dur}


# ================================================================ #
#  Stage 3: CAL Generation (SolidWorks Construction Tree)            #
# ================================================================ #

def _feature_sort_key(f: DetectedFeature) -> Tuple[int, float]:
    """Sort features into SolidWorks construction order."""
    order = {
        "base_extrude": 0,
        "boss": 1,
        "pocket": 2,
        "slot": 2,
        "through_hole": 3,
        "blind_hole": 3,
        "counterbore": 3,
        "fillet": 4,
        "chamfer": 4,
        "rib": 1,
        "shell": 5,
    }
    return (order.get(f.feature_type, 9), -f.confidence)


def stage3_cal_generation(features, output_dir, image_stem, ggl_doc):
    """Generate SolidWorks-compatible CAL from detected features."""
    logger.info("=== STAGE 3: CAL Generation (SolidWorks Construction Tree) ===")
    t0 = time.perf_counter()

    # Sort features into construction order
    features_sorted = sorted(features, key=_feature_sort_key)

    construction_tree = []
    act_idx = 0

    for feat in features_sorted:
        # ---- BASE EXTRUSION ----
        if feat.feature_type == "base_extrude":
            sketch_id = f"sketch_{act_idx:03d}"
            dims = feat.dimensions

            construction_tree.append({
                "step": act_idx,
                "feature_id": feat.feature_id,
                "feature_name": "Base Extrude",
                "feature_type": "extrude_boss",
                "solidworks_feature": "Boss-Extrude",
                "sketch": {
                    "sketch_id": sketch_id,
                    "plane": feat.sketch_plane,
                    "plane_offset": 0,
                    "profile": {
                        "type": "rectangle",
                        "center": [0, 0],
                        "width": dims.get("width", 20),
                        "height": dims.get("height", 20),
                    },
                    "constraints": [
                        {"type": "centered_at_origin"},
                        {"type": "fully_defined"},
                    ],
                    "dimensions": [
                        {"type": "horizontal", "value": dims.get("width", 20), "units": "mm"},
                        {"type": "vertical", "value": dims.get("height", 20), "units": "mm"},
                    ],
                },
                "operation": {
                    "type": "mid_plane",
                    "depth": dims.get("depth", 20),
                    "depth_units": "mm",
                    "direction": feat.axis,
                    "draft": False,
                },
                "solidworks_api": {
                    "method": "FeatureManager.FeatureExtrusion3",
                    "params": {
                        "sd": True,
                        "flip": False,
                        "dir": True,
                        "d1": dims.get("depth", 20) / 2,
                        "d2": dims.get("depth", 20) / 2,
                        "dchk1": False,
                        "dchk2": False,
                        "ddir1": 0,
                        "ddir2": 0,
                        "dang1": 0,
                        "dang2": 0,
                        "offReverse1": False,
                        "offReverse2": False,
                        "translateSurface1": False,
                        "translateSurface2": False,
                        "merge": True,
                        "useFeatScope": True,
                        "useAutoSelect": True,
                    },
                    "sketch_api": "SketchManager.CreateCenterRectangle",
                },
                "confidence": feat.confidence,
                "notes": feat.notes,
            })
            act_idx += 1

        # ---- BOSS (cylindrical or rectangular) ----
        elif feat.feature_type == "boss":
            sketch_id = f"sketch_{act_idx:03d}"
            dims = feat.dimensions

            if feat.primitive == "cylinder":
                profile = {
                    "type": "circle",
                    "center": [feat.position[0], feat.position[1]],
                    "radius": dims.get("radius", 5),
                    "diameter": dims.get("diameter", 10),
                }
                sketch_api = "SketchManager.CreateCircle"
            else:
                profile = {
                    "type": "rectangle",
                    "center": [feat.position[0], feat.position[1]],
                    "width": dims.get("width", 10),
                    "height": dims.get("height", 10),
                }
                sketch_api = "SketchManager.CreateCenterRectangle"

            construction_tree.append({
                "step": act_idx,
                "feature_id": feat.feature_id,
                "feature_name": f"Boss-Extrude ({feat.primitive})",
                "feature_type": "extrude_boss",
                "solidworks_feature": "Boss-Extrude",
                "sketch": {
                    "sketch_id": sketch_id,
                    "plane": feat.sketch_plane,
                    "plane_offset": feat.position[2] if len(feat.position) > 2 else 0,
                    "profile": profile,
                    "dimensions": [
                        {"type": "radius" if feat.primitive == "cylinder" else "width",
                         "value": dims.get("radius", dims.get("width", 10)),
                         "units": "mm"},
                    ],
                },
                "operation": {
                    "type": "blind",
                    "depth": dims.get("height", dims.get("depth", 10)),
                    "depth_units": "mm",
                    "direction": feat.axis,
                    "draft": False,
                },
                "solidworks_api": {
                    "method": "FeatureManager.FeatureExtrusion3",
                    "sketch_api": sketch_api,
                },
                "confidence": feat.confidence,
                "notes": feat.notes,
            })
            act_idx += 1

        # ---- THROUGH HOLE ----
        elif feat.feature_type == "through_hole":
            dims = feat.dimensions
            construction_tree.append({
                "step": act_idx,
                "feature_id": feat.feature_id,
                "feature_name": f"Through Hole (d={dims.get('diameter', 0):.3f}mm)",
                "feature_type": "hole_wizard",
                "solidworks_feature": "Hole Wizard",
                "hole_spec": {
                    "hole_type": "straight_through",
                    "standard": "ANSI Metric",
                    "diameter": dims.get("diameter", 10),
                    "radius": dims.get("radius", 5),
                    "depth_type": "through_all",
                    "depth": dims.get("depth", 0),
                },
                "position": {
                    "face_ref": f"{features_sorted[0].feature_id}:top_face",
                    "x": feat.position[0],
                    "y": feat.position[1],
                    "z": feat.position[2],
                },
                "axis": feat.axis,
                "solidworks_api": {
                    "method": "FeatureManager.HoleWizard5",
                    "params": {
                        "GenericHoleType": 0,
                        "StandardIndex": 2,
                        "FastenerTypeIndex": 0,
                        "SSize": f"M{max(1, round(dims.get('diameter', 10)))}",
                        "EndType": 1,
                        "Depth": dims.get("depth", 0),
                        "Value1": dims.get("diameter", 10) / 1000,
                    },
                    "alternative_method": "FeatureManager.FeatureCut4",
                    "alternative_sketch": {
                        "method": "SketchManager.CreateCircle",
                        "center": [feat.position[0], feat.position[1]],
                        "radius": dims.get("radius", 5),
                    },
                },
                "confidence": feat.confidence,
                "notes": feat.notes,
            })
            act_idx += 1

        # ---- BLIND HOLE ----
        elif feat.feature_type == "blind_hole":
            dims = feat.dimensions
            construction_tree.append({
                "step": act_idx,
                "feature_id": feat.feature_id,
                "feature_name": f"Blind Hole (d={dims.get('diameter', 0):.3f}mm, depth={dims.get('depth', 0):.3f}mm)",
                "feature_type": "hole_wizard",
                "solidworks_feature": "Hole Wizard",
                "hole_spec": {
                    "hole_type": "straight_blind",
                    "diameter": dims.get("diameter", 10),
                    "radius": dims.get("radius", 5),
                    "depth_type": "blind",
                    "depth": dims.get("depth", 10),
                },
                "position": {
                    "x": feat.position[0],
                    "y": feat.position[1],
                    "z": feat.position[2],
                },
                "axis": feat.axis,
                "solidworks_api": {
                    "method": "FeatureManager.HoleWizard5",
                    "alternative_method": "FeatureManager.FeatureCut4",
                },
                "confidence": feat.confidence,
                "notes": feat.notes,
            })
            act_idx += 1

        # ---- POCKET (extrude cut) ----
        elif feat.feature_type == "pocket":
            sketch_id = f"sketch_{act_idx:03d}"
            dims = feat.dimensions

            construction_tree.append({
                "step": act_idx,
                "feature_id": feat.feature_id,
                "feature_name": f"Pocket ({dims.get('width', 0):.2f} x {dims.get('height', 0):.2f} x {dims.get('depth', 0):.2f}mm)",
                "feature_type": "extrude_cut",
                "solidworks_feature": "Cut-Extrude",
                "sketch": {
                    "sketch_id": sketch_id,
                    "plane": feat.sketch_plane,
                    "plane_offset": feat.position[2] if len(feat.position) > 2 else 0,
                    "profile": {
                        "type": "rectangle",
                        "center": [feat.position[0], feat.position[1]],
                        "width": dims.get("width", 10),
                        "height": dims.get("height", 10),
                    },
                },
                "operation": {
                    "type": "blind",
                    "depth": dims.get("depth", 5),
                    "depth_units": "mm",
                    "direction": [-a for a in feat.axis],
                    "draft": False,
                },
                "solidworks_api": {
                    "method": "FeatureManager.FeatureCut4",
                    "params": {
                        "sd": True,
                        "flip": False,
                        "dir": True,
                        "d1": dims.get("depth", 5),
                        "dchk1": False,
                        "ddir1": 0,
                        "dang1": 0,
                        "NormalCut": True,
                    },
                    "sketch_api": "SketchManager.CreateCenterRectangle",
                },
                "confidence": feat.confidence,
                "notes": feat.notes,
            })
            act_idx += 1

        # ---- FILLET ----
        elif feat.feature_type == "fillet":
            dims = feat.dimensions
            construction_tree.append({
                "step": act_idx,
                "feature_id": feat.feature_id,
                "feature_name": f"Fillet (R={dims.get('radius', 1):.3f}mm)",
                "feature_type": "fillet",
                "solidworks_feature": "Fillet",
                "fillet_spec": {
                    "radius": dims.get("radius", 1.0),
                    "radius_units": "mm",
                    "type": "constant_radius",
                    "propagate_along_tangent": True,
                },
                "edge_selection": {
                    "method": "by_radius_match",
                    "target_radius": dims.get("radius", 1.0),
                    "position": feat.position,
                },
                "solidworks_api": {
                    "method": "FeatureManager.FeatureFillet3",
                    "params": {
                        "Radius": dims.get("radius", 1.0) / 1000,
                        "FeatureOptions": 1,
                        "OverflowType": 0,
                    },
                },
                "confidence": feat.confidence,
                "notes": feat.notes,
            })
            act_idx += 1

    # Build final CAL document
    cal_document = {
        "version": "2.0",
        "format": "CAD_Action_Language",
        "generator": "generate_cad_v2",
        "target_cad": "SolidWorks 2024+",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "units": {
            "length": "mm",
            "angle": "degrees",
            "mass": "kg",
        },
        "material": {
            "name": "1060 Alloy",
            "density": 2700,
            "units": "kg/m^3",
        },
        "mesh_reference": ggl_doc.get("mesh_file", ""),
        "construction_tree": construction_tree,
        "summary": {
            "total_operations": len(construction_tree),
            "base_features": sum(1 for c in construction_tree if c["feature_type"] == "extrude_boss" and "base" in c.get("feature_id", "").lower()),
            "boss_features": sum(1 for c in construction_tree if c["feature_type"] == "extrude_boss" and "base" not in c.get("feature_id", "").lower()),
            "cut_features": sum(1 for c in construction_tree if c["feature_type"] == "extrude_cut"),
            "holes": sum(1 for c in construction_tree if c["feature_type"] == "hole_wizard"),
            "fillets": sum(1 for c in construction_tree if c["feature_type"] == "fillet"),
            "chamfers": sum(1 for c in construction_tree if c["feature_type"] == "chamfer"),
        },
        "solidworks_macro_hint": (
            "Use SolidWorks VBA/C# API: "
            "For each step in construction_tree, call the solidworks_api.method "
            "with the specified params. Create sketches on the specified planes "
            "using the profile geometry, then apply the operation."
        ),
    }

    # Save CAL
    cal_path = os.path.join(output_dir, f"{image_stem}_cal.json")
    with open(cal_path, "w") as f:
        json.dump(cal_document, f, indent=2)
    logger.info("CAL saved: %s (%d operations)", cal_path, len(construction_tree))

    dur = time.perf_counter() - t0
    logger.info("Stage 3: %.1fs", dur)
    return {"cal": cal_document, "cal_path": cal_path, "duration_s": dur}


# ================================================================ #
#  Main Pipeline Orchestrator                                        #
# ================================================================ #

def run_pipeline(args):
    total_start = time.perf_counter()
    image_stem = Path(args.image).stem if args.image else Path(args.mesh).stem
    os.makedirs(args.output_dir, exist_ok=True)
    results = {}

    print(f"\n{'='*64}")
    print(f"  AI CAD OS - Advanced Image-to-CAD Pipeline V2")
    print(f"{'='*64}")
    print(f"  Image:    {args.image or 'N/A (using existing mesh)'}")
    print(f"  Mesh:     {args.mesh or 'will be generated'}")
    print(f"  Device:   {args.device}")
    print(f"  Output:   {args.output_dir}/")
    print(f"{'='*64}\n")

    # Stage 1: Model Inference
    mesh_path = args.mesh
    if not args.skip_inference and args.image:
        results["stage1"] = stage1_model_inference(
            image_path=args.image,
            model_path=args.model_path,
            device=args.device,
            output_dir=args.output_dir,
            steps=args.steps,
            resolution=args.resolution,
        )
        mesh_path = results["stage1"]["mesh_path"]
    else:
        results["stage1"] = {"duration_s": 0.0, "cached": True}
        if not mesh_path:
            # Try to find existing mesh
            for ext in [".glb", ".obj", ".ply", ".stl"]:
                candidate = os.path.join(args.output_dir, f"{image_stem}{ext}")
                if os.path.exists(candidate):
                    mesh_path = candidate
                    break
        if not mesh_path or not os.path.exists(mesh_path):
            logger.error("No mesh found! Provide --image or --mesh")
            sys.exit(1)
        logger.info("Using existing mesh: %s", mesh_path)

    # Stage 2: Advanced Geometry Analysis
    results["stage2"] = stage2_geometry_analysis(mesh_path, args.output_dir, image_stem)
    features = results["stage2"]["features"]
    ggl_doc = results["stage2"]["ggl"]

    # Stage 3: CAL Generation
    results["stage3"] = stage3_cal_generation(features, args.output_dir, image_stem, ggl_doc)

    # ---- Summary ----
    total = time.perf_counter() - total_start
    cal = results["stage3"]["cal"]

    print(f"\n{'='*64}")
    print(f"  PIPELINE COMPLETE - V2 Advanced Analysis")
    print(f"{'='*64}")
    for i, (key, val) in enumerate(results.items(), 1):
        dur = val.get("duration_s", 0)
        print(f"  Stage {i}: {dur:>7.1f}s")
    print(f"  {'_'*40}")
    print(f"  Total:  {total:.1f}s\n")

    print(f"  OUTPUT FILES:")
    for key, val in results.items():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str) and os.path.exists(v):
                    sz = os.path.getsize(v)
                    print(f"    {os.path.basename(v):40s} {sz:>10,} bytes")

    print(f"\n  DETECTED FEATURES:")
    for f in features:
        sub = " [CUT]" if f.is_subtractive else ""
        dims_str = ", ".join(f"{k}={v:.3f}" for k, v in f.dimensions.items()
                            if isinstance(v, (int, float)))
        print(f"    {f.feature_type:20s} {f.primitive:10s} {dims_str}{sub}")

    print(f"\n  SOLIDWORKS CONSTRUCTION TREE:")
    for step in cal["construction_tree"]:
        print(f"    {step['step']:2d}. {step['feature_name']:40s} [{step['solidworks_feature']}]")

    print(f"\n  CAL SUMMARY:")
    s = cal["summary"]
    print(f"    Total operations: {s['total_operations']}")
    print(f"    Base features:    {s['base_features']}")
    print(f"    Boss features:    {s['boss_features']}")
    print(f"    Cut features:     {s['cut_features']}")
    print(f"    Holes:            {s['holes']}")
    print(f"    Fillets:          {s['fillets']}")
    print(f"{'='*64}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI CAD OS - Advanced Image-to-Parametric-CAD Pipeline V2")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--image", "-i", help="Input image path")
    input_group.add_argument("--mesh", "-m", help="Pre-existing mesh file (.glb/.obj/.stl)")

    parser.add_argument("--output-dir", "-o", default="./outputs")
    parser.add_argument("--model-path", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip model inference, use existing mesh")

    args = parser.parse_args()

    if not args.image and not args.mesh:
        parser.error("Provide --image or --mesh")

    if args.mesh:
        args.skip_inference = True

    run_pipeline(args)


if __name__ == "__main__":
    main()
