"""
refinement/comparator.py — Mesh-vs-GGL Geometry Comparator
============================================================
Compares an exported mesh from the CAD system against the original
GGL predictions to quantify reconstruction error.
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("geometry_engine.refinement")


class GeometryComparator:
    """
    Compares the CAD-exported mesh against the original GGL to produce
    a list of GeometryDifference objects for the refinement loop.

    In production this uses:
      - Trimesh for mesh loading
      - Point sampling for Chamfer/Hausdorff distance
      - Voxelization for IOU computation
      - Analytic fitting for parameter error estimation

    V1 uses simplified heuristic comparison.
    """

    def compare(
        self,
        ggl_dict: Dict[str, Any],
        exported_mesh_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Compares GGL primitives against the exported mesh.

        Args:
            ggl_dict: The original GGL as a dict (from ggl.model_dump())
            exported_mesh_path: Path to the mesh file (.obj, .stl, .glb)

        Returns:
            List of GeometryDifference dicts, one per primitive node.
        """
        differences = []

        primitive_types = {"Cylinder", "Box", "Sphere", "Cone"}
        primitives = [n for n in ggl_dict.get("nodes", []) if n.get("type") in primitive_types]

        if not primitives:
            logger.warning("No primitive nodes found in GGL to compare.")
            return differences

        # V1: Load mesh and compute bounding box comparison
        mesh_bbox = self._load_mesh_bbox(exported_mesh_path)

        for prim in primitives:
            params = prim.get("parameters", {})
            prim_type = prim.get("type", "")
            diff = {
                "node_id": prim.get("node_id", ""),
                "primitive_type": prim_type,
                "parameter_diffs": {},
                "iou_score": 0.0,
                "hausdorff_distance": 0.0,
                "chamfer_distance": 0.0,
                "severity": "low",
            }

            if mesh_bbox is not None:
                # Simplified comparison: check if bounding dimensions are close
                predicted_volume = self._estimate_volume(prim_type, params)
                mesh_volume = mesh_bbox.get("volume", 1.0)

                if predicted_volume > 0 and mesh_volume > 0:
                    overlap = min(predicted_volume, mesh_volume)
                    union = max(predicted_volume, mesh_volume)
                    diff["iou_score"] = round(overlap / union, 4)

                    if diff["iou_score"] < 0.5:
                        diff["severity"] = "high"
                    elif diff["iou_score"] < 0.8:
                        diff["severity"] = "medium"

            differences.append(diff)

        return differences

    def _load_mesh_bbox(self, mesh_path: str) -> Optional[Dict[str, Any]]:
        """Loads a mesh and returns bounding box info."""
        try:
            import trimesh
            mesh = trimesh.load(mesh_path)
            extents = mesh.bounding_box.extents
            return {
                "width": float(extents[0]),
                "height": float(extents[1]),
                "depth": float(extents[2]),
                "volume": float(mesh.volume) if mesh.is_watertight else float(extents.prod()),
            }
        except Exception as e:
            logger.warning(f"Could not load mesh {mesh_path}: {e}")
            return None

    @staticmethod
    def _estimate_volume(prim_type: str, params: Dict[str, Any]) -> float:
        """Estimates the volume of a primitive from its parameters."""
        import math
        if prim_type == "Box":
            return params.get("width", 1) * params.get("height", 1) * params.get("depth", 1)
        elif prim_type == "Cylinder":
            r = params.get("radius", 1)
            h = params.get("height", 1)
            return math.pi * r * r * h
        elif prim_type == "Sphere":
            r = params.get("radius", 1)
            return (4/3) * math.pi * r ** 3
        elif prim_type == "Cone":
            r = params.get("radius", 1)
            h = params.get("height", 1)
            return (1/3) * math.pi * r * r * h
        return 1.0
