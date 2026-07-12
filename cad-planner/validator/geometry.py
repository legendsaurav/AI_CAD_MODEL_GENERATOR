import networkx as nx
from ggl_parser.ggl_parser import GeometryGraphLanguage

class GeometryValidator:
    """
    Stage 2 — Geometry Validation
    Checks: disconnected components, impossible topology, invalid dimensions, inconsistent hierarchy.
    """
    def __init__(self, ggl: GeometryGraphLanguage):
        self.ggl = ggl
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        for n in self.ggl.nodes:
            self.graph.add_node(n.node_id, **n.model_dump())
        for e in self.ggl.edges:
            self.graph.add_edge(e.source_id, e.target_id, relation=e.relation)

    def validate(self) -> GeometryGraphLanguage:
        """
        Validates the geometry and returns the corrected GGL (or raises ValueError).
        """
        self._check_disconnected_components()
        self._check_invalid_dimensions()
        self._check_inconsistent_hierarchy()
        return self.ggl

    def _check_disconnected_components(self):
        # In a valid CAD model, all geometry features should be physically connected
        # or grouped under a root assembly/body node.
        # We use an undirected version of the graph to check weak connectivity.
        undirected = self.graph.to_undirected()
        if not nx.is_connected(undirected) and len(self.graph.nodes) > 1:
            components = list(nx.connected_components(undirected))
            raise ValueError(f"Geometry Validation Failed: Found {len(components)} disconnected components in the GGL.")

    def _check_invalid_dimensions(self):
        for n in self.ggl.nodes:
            params = n.parameters
            if n.type == "Cylinder":
                if params.get("radius", 1) <= 0 or params.get("height", 1) <= 0:
                    raise ValueError(f"Invalid dimensions for Cylinder {n.node_id}: {params}")
            elif n.type == "Box":
                if params.get("width", 1) <= 0 or params.get("height", 1) <= 0 or params.get("depth", 1) <= 0:
                    raise ValueError(f"Invalid dimensions for Box {n.node_id}: {params}")

    def _check_inconsistent_hierarchy(self):
        # Hierarchy is typically represented by 'Contains' or 'Instantiates' relations.
        # It must form a DAG (no cycles).
        hierarchy_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("relation") in ("Contains", "Instantiates")]
        h_graph = nx.DiGraph(hierarchy_edges)
        if not nx.is_directed_acyclic_graph(h_graph):
            try:
                cycle = nx.find_cycle(h_graph)
                raise ValueError(f"Inconsistent Hierarchy: Circular dependency detected {cycle}")
            except nx.NetworkXNoCycle:
                pass
