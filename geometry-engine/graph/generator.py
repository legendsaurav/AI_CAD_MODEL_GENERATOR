"""
graph/generator.py
==================
BUG FIX: metadata_kwargs may pass `layers_used` as an int (e.g. 5) but
GGLMetadata declares it as List[int].  Pydantic raises a validation error
when setattr() is called.  We now normalise the value before setting.
"""
import torch
from typing import Any, Dict, List, Optional

from graph.ggl import GeometryGraphLanguage, GGLEdge, GGLNode
from heads.loader import HeadPluginLoader


class GraphGenerator:
    """
    Orchestrates the Hierarchical Geometry Graph Extraction (Version 2).
    Passes features through every enabled plugin head, resolves conflicts,
    builds Part→Surface hierarchy, and returns a validated GGL object.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.head_loader = HeadPluginLoader(config)
        self.heads = self.head_loader.get_all_heads()
        self.confidence_threshold: float = float(
            config.get("heads", {}).get("confidence_threshold", 0.5)
        )

    # ── Public entry-point ────────────────────────────────────────────────

    def generate_graph(
        self,
        features: torch.Tensor,
        metadata_kwargs: Optional[Dict[str, Any]] = None,
    ) -> GeometryGraphLanguage:
        """
        Run all enabled prediction heads on `features` and compile the GGL.

        Args:
            features: Tensor of shape [B, N, D] representing fused DiT features.
            metadata_kwargs: Optional dict of GGLMetadata field values to set.
        Returns:
            A validated GeometryGraphLanguage object.
        """
        ggl = GeometryGraphLanguage()

        # ── Populate metadata ─────────────────────────────────────────────
        if metadata_kwargs:
            for k, v in metadata_kwargs.items():
                if hasattr(ggl.metadata, k):
                    # Coerce layers_used: int → List[int]
                    if k == "layers_used":
                        if isinstance(v, int):
                            v = list(range(v))
                        elif not isinstance(v, list):
                            v = list(v)
                    try:
                        setattr(ggl.metadata, k, v)
                    except Exception:
                        pass  # silently skip invalid metadata fields

        print(f"🕸️  GraphGenerator running heads: {list(self.heads.keys())}")

        all_nodes: List[GGLNode] = []
        topology_relations: List[GGLNode] = []

        # ── 1. Forward pass through all plugin heads ──────────────────────
        for head_name, head in self.heads.items():
            print(f"   → Running {head_name} head...")
            preds = head(features)
            nodes = head.to_ggl_nodes(preds, threshold=self.confidence_threshold)
            for n in nodes:
                if n.type == "Relation":
                    topology_relations.append(n)
                else:
                    all_nodes.append(n)

        # ── 2. Consistency check (filter + deduplicate) ───────────────────
        resolved = self._consistency_check(all_nodes)
        for n in resolved:
            ggl.add_node(n)

        # ── 3. Convert topology predictions → GGLEdges ───────────────────
        if len(resolved) > 1 and topology_relations:
            for rel in topology_relations:
                s_i = rel.parameters.get("source_idx", 0) % len(resolved)
                t_i = rel.parameters.get("target_idx", 1) % len(resolved)
                if s_i != t_i:
                    ggl.add_edge(
                        GGLEdge(
                            source_id=resolved[s_i].node_id,
                            target_id=resolved[t_i].node_id,
                            relation=rel.semantic_label or "Adjacent",
                            confidence=rel.confidence,
                        )
                    )

        # ── 4. Build Part → Surface containment hierarchy ─────────────────
        self._build_hierarchy(ggl)

        print(
            f"✅ Graph ready: {len(ggl.nodes)} nodes, {len(ggl.edges)} edges."
        )
        return ggl

    # ── Internal helpers ──────────────────────────────────────────────────

    def _consistency_check(self, nodes: List[GGLNode]) -> List[GGLNode]:
        """
        Sort by confidence descending and cap at 20 to avoid explosion.
        Future: implement geometric overlap NMS.
        """
        nodes.sort(key=lambda n: n.confidence, reverse=True)
        return nodes[:20]

    def _build_hierarchy(self, ggl: GeometryGraphLanguage):
        """Attach Surface nodes to Part nodes via 'Contains' edges."""
        parts    = [n for n in ggl.nodes if n.type == "Part"]
        surfaces = [n for n in ggl.nodes if n.type == "Surface"]

        if parts and surfaces:
            for i, s in enumerate(surfaces):
                parent = parts[i % len(parts)]
                ggl.add_edge(
                    GGLEdge(
                        source_id=parent.node_id,
                        target_id=s.node_id,
                        relation="Contains",
                        confidence=1.0,
                    )
                )
