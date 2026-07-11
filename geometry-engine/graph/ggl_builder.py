"""
graph/ggl_builder.py - GGL Assembly from Prediction Head Outputs
=================================================================
Takes raw prediction outputs from all heads (PartHead, SurfaceHead,
PrimitiveHead, TopologyHead, SymmetryHead) and assembles a complete,
validated GeometryGraphLanguage document.

Responsibilities:
  1. Call to_ggl_nodes() on each head's predictions
  2. Build hierarchy: Part -> Surface -> Primitive (Contains edges)
  3. Convert topology predictions to GGLEdge objects
  4. Add symmetry edges
  5. Deduplicate overlapping nodes
  6. Validate the final graph

ARCHITECTURE RULE:
    All nodes MUST trace back to DiT hidden state predictions.
    This builder must NEVER accept mesh-derived inputs.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from graph.ggl import (
    GGLEdge,
    GGLMetadata,
    GGLNode,
    GeometryGraphLanguage,
)

logger = logging.getLogger("geometry_engine.graph.ggl_builder")


class GGLBuilder:
    """Assembles complete GGL documents from prediction head outputs.

    Parameters
    ----------
    config : dict
        Configuration containing thresholds for each head:
            - part_threshold (float): default 0.5
            - surface_threshold (float): default 0.5
            - primitive_threshold (float): default 0.5
            - topology_threshold (float): default 0.5
            - symmetry_threshold (float): default 0.5
            - dedup_distance (float): default 5.0 (mm)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.part_threshold = config.get("part_threshold", 0.5)
        self.surface_threshold = config.get("surface_threshold", 0.5)
        self.primitive_threshold = config.get("primitive_threshold", 0.5)
        self.topology_threshold = config.get("topology_threshold", 0.5)
        self.symmetry_threshold = config.get("symmetry_threshold", 0.5)
        self.dedup_distance = config.get("dedup_distance", 5.0)

        logger.info(
            "GGLBuilder initialized: part=%.2f, surface=%.2f, primitive=%.2f, "
            "topology=%.2f, symmetry=%.2f, dedup=%.1fmm",
            self.part_threshold, self.surface_threshold,
            self.primitive_threshold, self.topology_threshold,
            self.symmetry_threshold, self.dedup_distance,
        )

    def build(
        self,
        heads: Dict[str, Any],
        head_outputs: Dict[str, Dict[str, torch.Tensor]],
        metadata: Optional[GGLMetadata] = None,
    ) -> GeometryGraphLanguage:
        """Build a complete GGL from all head predictions.

        Parameters
        ----------
        heads : dict
            Mapping of head_name -> head_instance (with to_ggl_nodes method).
        head_outputs : dict
            Mapping of head_name -> forward() output dict.
        metadata : GGLMetadata, optional
            Source metadata. If None, uses defaults.

        Returns
        -------
        GeometryGraphLanguage
            Fully assembled and validated GGL document.
        """
        # -- Step 1: Extract nodes from each head --------------------
        part_nodes: List[GGLNode] = []
        surface_nodes: List[GGLNode] = []
        primitive_nodes: List[GGLNode] = []
        topology_pseudo_nodes: List[GGLNode] = []
        symmetry_nodes: List[GGLNode] = []

        thresholds = {
            "part": self.part_threshold,
            "surface": self.surface_threshold,
            "primitive": self.primitive_threshold,
            "topology": self.topology_threshold,
            "symmetry": self.symmetry_threshold,
        }

        for name, head in heads.items():
            if name not in head_outputs:
                logger.warning("Head '%s' has no output - skipping", name)
                continue

            threshold = thresholds.get(name, 0.5)
            predictions = head_outputs[name]
            nodes = head.to_ggl_nodes(predictions, threshold=threshold)

            if name == "part":
                part_nodes = nodes
            elif name == "surface":
                surface_nodes = nodes
            elif name == "primitive":
                primitive_nodes = nodes
            elif name == "topology":
                topology_pseudo_nodes = nodes
            elif name == "symmetry":
                symmetry_nodes = nodes
            else:
                logger.warning("Unknown head '%s' - nodes added as-is", name)
                primitive_nodes.extend(nodes)

            logger.info(
                "Head '%s' produced %d nodes (threshold=%.2f)",
                name, len(nodes), threshold,
            )

        # -- Step 2: Deduplicate nodes with similar geometry ---------
        primitive_nodes = self._deduplicate_nodes(
            primitive_nodes, distance_threshold=self.dedup_distance
        )

        # -- Step 3: Build hierarchy edges (Contains) ----------------
        hierarchy_edges = self._build_hierarchy(
            part_nodes, surface_nodes, primitive_nodes
        )

        # -- Step 4: Convert topology pseudo-nodes to edges ----------
        topology_edges = self._topology_nodes_to_edges(
            topology_pseudo_nodes, primitive_nodes
        )

        # -- Step 5: Create symmetry edges ---------------------------
        symmetry_edges = self._symmetry_nodes_to_edges(
            symmetry_nodes, primitive_nodes
        )

        # -- Step 6: Assemble GGL ------------------------------------
        ggl = GeometryGraphLanguage(
            metadata=metadata or GGLMetadata(),
        )

        all_nodes = part_nodes + surface_nodes + primitive_nodes + symmetry_nodes
        all_edges = hierarchy_edges + topology_edges + symmetry_edges

        for node in all_nodes:
            ggl.add_node(node)
        for edge in all_edges:
            ggl.add_edge(edge)

        # -- Step 7: Validate ----------------------------------------
        warnings = self._validate(ggl)
        for w in warnings:
            logger.warning("GGL validation: %s", w)

        logger.info(
            "GGL assembled: %d nodes, %d edges, %d warnings",
            len(ggl.nodes), len(ggl.edges), len(warnings),
        )

        return ggl

    # -- Hierarchy builder ----------------------------------------------- #

    def _build_hierarchy(
        self,
        parts: List[GGLNode],
        surfaces: List[GGLNode],
        primitives: List[GGLNode],
    ) -> List[GGLEdge]:
        """Create Contains edges: Part -> Surface -> Primitive.

        Assignment is based on spatial proximity of node centers.
        """
        edges: List[GGLEdge] = []

        # Assign primitives to surfaces (or directly to parts)
        for prim in primitives:
            prim_center = self._get_center(prim)
            if prim_center is None:
                continue

            # Try to find closest surface
            best_surface = self._find_nearest(prim_center, surfaces)
            if best_surface is not None:
                edges.append(GGLEdge(
                    source_id=best_surface.node_id,
                    target_id=prim.node_id,
                    relation="Contains",
                    confidence=min(best_surface.confidence, prim.confidence),
                ))
            else:
                # Assign directly to closest part
                best_part = self._find_nearest(prim_center, parts)
                if best_part is not None:
                    edges.append(GGLEdge(
                        source_id=best_part.node_id,
                        target_id=prim.node_id,
                        relation="Contains",
                        confidence=min(best_part.confidence, prim.confidence),
                    ))

        # Assign surfaces to parts
        for surface in surfaces:
            surf_center = self._get_center(surface)
            if surf_center is None:
                continue

            best_part = self._find_nearest(surf_center, parts)
            if best_part is not None:
                edges.append(GGLEdge(
                    source_id=best_part.node_id,
                    target_id=surface.node_id,
                    relation="Contains",
                    confidence=min(best_part.confidence, surface.confidence),
                ))

        return edges

    # -- Topology conversion --------------------------------------------- #

    def _topology_nodes_to_edges(
        self,
        topo_nodes: List[GGLNode],
        primitives: List[GGLNode],
    ) -> List[GGLEdge]:
        """Convert topology pseudo-nodes (from TopologyHead) to GGLEdges.

        TopologyHead outputs 'Relation' nodes with source_idx and target_idx
        in their parameters. We map these indices to actual primitive node IDs.
        """
        edges: List[GGLEdge] = []

        for topo_node in topo_nodes:
            src_idx = topo_node.parameters.get("source_idx")
            tgt_idx = topo_node.parameters.get("target_idx")

            if src_idx is None or tgt_idx is None:
                continue

            # Map indices to primitive nodes (best effort)
            if src_idx < len(primitives) and tgt_idx < len(primitives):
                edges.append(GGLEdge(
                    source_id=primitives[src_idx].node_id,
                    target_id=primitives[tgt_idx].node_id,
                    relation=topo_node.semantic_label or "Adjacent",
                    confidence=topo_node.confidence,
                ))

        return edges

    # -- Symmetry conversion --------------------------------------------- #

    def _symmetry_nodes_to_edges(
        self,
        sym_nodes: List[GGLNode],
        primitives: List[GGLNode],
    ) -> List[GGLEdge]:
        """Create Symmetric edges between primitives that share symmetry.

        For each symmetry node, finds all primitives on opposite sides
        of the mirror plane and links them.
        """
        edges: List[GGLEdge] = []

        for sym in sym_nodes:
            sym_type = sym.parameters.get("symmetry_type", "")

            if sym_type in ("Bilateral", "Both"):
                mirror_normal = sym.parameters.get("mirror_normal")
                mirror_point = sym.parameters.get("mirror_point")

                if mirror_normal and mirror_point and len(primitives) >= 2:
                    # Find pairs of primitives symmetric about the plane
                    paired = self._find_symmetric_pairs(
                        primitives, mirror_point, mirror_normal
                    )
                    for p1_id, p2_id, dist in paired:
                        edges.append(GGLEdge(
                            source_id=p1_id,
                            target_id=p2_id,
                            relation="Symmetric",
                            confidence=sym.confidence,
                            parameters={
                                "symmetry_type": "bilateral",
                                "mirror_distance": round(dist, 4),
                            },
                        ))

        return edges

    # -- Deduplication --------------------------------------------------- #

    def _deduplicate_nodes(
        self,
        nodes: List[GGLNode],
        distance_threshold: float = 5.0,
    ) -> List[GGLNode]:
        """Merge nodes with the same type and nearby centers.

        Keeps the node with higher confidence when merging.
        """
        if len(nodes) <= 1:
            return nodes

        merged: List[GGLNode] = []
        used = [False] * len(nodes)

        for i, node_i in enumerate(nodes):
            if used[i]:
                continue

            center_i = self._get_center(node_i)

            for j in range(i + 1, len(nodes)):
                if used[j]:
                    continue

                node_j = nodes[j]

                # Same type required
                if node_i.type != node_j.type:
                    continue

                center_j = self._get_center(node_j)

                if center_i is not None and center_j is not None:
                    dist = math.sqrt(sum(
                        (a - b) ** 2 for a, b in zip(center_i, center_j)
                    ))
                    if dist < distance_threshold:
                        # Mark j as used (keep i as it's processed first)
                        used[j] = True
                        # Update confidence to the higher one
                        if node_j.confidence > node_i.confidence:
                            node_i = node_j  # swap to keep higher confidence

            merged.append(node_i)
            used[i] = True

        removed = len(nodes) - len(merged)
        if removed > 0:
            logger.info(
                "Deduplicated: %d -> %d nodes (%d merged)",
                len(nodes), len(merged), removed,
            )

        return merged

    # -- Validation ------------------------------------------------------ #

    def _validate(self, ggl: GeometryGraphLanguage) -> List[str]:
        """Validate the assembled GGL. Returns list of warnings."""
        warnings: List[str] = []

        node_ids = {n.node_id for n in ggl.nodes}

        # Check for duplicate node IDs
        if len(node_ids) != len(ggl.nodes):
            warnings.append(
                f"Duplicate node IDs detected: {len(ggl.nodes)} nodes "
                f"but only {len(node_ids)} unique IDs"
            )

        # Check all edge references
        for edge in ggl.edges:
            if edge.source_id not in node_ids:
                warnings.append(
                    f"Edge references missing source node: {edge.source_id}"
                )
            if edge.target_id not in node_ids:
                warnings.append(
                    f"Edge references missing target node: {edge.target_id}"
                )

        # Check for orphan nodes (no edges)
        connected = set()
        for edge in ggl.edges:
            connected.add(edge.source_id)
            connected.add(edge.target_id)

        orphans = node_ids - connected
        if orphans:
            # Symmetry and single-component objects may have orphans
            logger.debug("Orphan nodes (no edges): %s", orphans)

        return warnings

    # -- Spatial helpers -------------------------------------------------- #

    @staticmethod
    def _get_center(node: GGLNode) -> Optional[List[float]]:
        """Extract the center coordinate from a node's parameters."""
        center = node.parameters.get("center")
        if center and isinstance(center, list) and len(center) >= 3:
            return center[:3]

        point = node.parameters.get("point")
        if point and isinstance(point, list) and len(point) >= 3:
            return point[:3]

        apex = node.parameters.get("apex")
        if apex and isinstance(apex, list) and len(apex) >= 3:
            return apex[:3]

        return None

    @staticmethod
    def _euclidean_dist(a: List[float], b: List[float]) -> float:
        """Euclidean distance between two 3D points."""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a[:3], b[:3])))

    def _find_nearest(
        self,
        query: List[float],
        candidates: List[GGLNode],
    ) -> Optional[GGLNode]:
        """Find the candidate node whose center is closest to the query point."""
        best_node = None
        best_dist = float("inf")

        for node in candidates:
            center = self._get_center(node)
            if center is None:
                continue

            dist = self._euclidean_dist(query, center)
            if dist < best_dist:
                best_dist = dist
                best_node = node

        return best_node

    @staticmethod
    def _find_symmetric_pairs(
        primitives: List[GGLNode],
        mirror_point: List[float],
        mirror_normal: List[float],
    ) -> List[Tuple[str, str, float]]:
        """Find pairs of primitives that are symmetric about a mirror plane.

        Returns list of (id_a, id_b, distance_to_mirror) tuples.
        """
        pairs: List[Tuple[str, str, float]] = []
        used = set()

        # Compute signed distance of each primitive to the mirror plane
        def signed_dist(center: List[float]) -> float:
            d = sum(
                (c - p) * n
                for c, p, n in zip(center, mirror_point, mirror_normal)
            )
            return d

        # Collect primitives with their signed distances
        items = []
        for prim in primitives:
            center = None
            c = prim.parameters.get("center")
            if c and isinstance(c, list) and len(c) >= 3:
                center = c[:3]
            if center is not None:
                items.append((prim.node_id, center, signed_dist(center)))

        # Find pairs on opposite sides of the plane
        for i, (id_a, center_a, dist_a) in enumerate(items):
            if id_a in used:
                continue
            for j in range(i + 1, len(items)):
                id_b, center_b, dist_b = items[j]
                if id_b in used:
                    continue

                # Opposite sides: signs differ
                if dist_a * dist_b < 0:
                    # Check that the distances are roughly equal (symmetric)
                    if abs(abs(dist_a) - abs(dist_b)) < 5.0:  # 5mm tolerance
                        pairs.append((id_a, id_b, abs(dist_a)))
                        used.add(id_a)
                        used.add(id_b)
                        break

        return pairs
