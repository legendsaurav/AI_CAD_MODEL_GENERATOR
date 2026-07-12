import networkx as nx
from typing import List
from ggl_parser.ggl_parser import GeometryGraphLanguage

class DependencyGraph:
    """
    Transforms the spatial GGL into a Directed Acyclic Graph (DAG) of feature dependencies.
    Represents dependencies like: Base Body -> Pocket -> Hole.
    """
    def __init__(self):
        self.dag = nx.DiGraph()

    def build(self, ggl: GeometryGraphLanguage) -> nx.DiGraph:
        """
        Builds the feature dependency DAG from the parsed GGL.
        """
        # Add all primitive nodes as feature nodes
        primitive_types = {"Box", "Cylinder", "Sphere", "Cone", "Torus", "Plane"}
        for node in ggl.nodes:
            if node.type in primitive_types or node.type == "Part":
                self.dag.add_node(node.node_id, **node.model_dump())

        # Establish dependencies based on GGL relations
        # E.g., 'Contains' implies the target depends on the source
        # 'Adjacent' might imply a dependency if one is a base and the other is a child.
        
        for edge in ggl.edges:
            source = edge.source_id
            target = edge.target_id
            
            if edge.relation in ["Contains", "Instantiates"]:
                # Contains means source is parent, target is child. Target depends on Source.
                self.dag.add_edge(source, target, relation=edge.relation)
            elif edge.relation == "Adjacent":
                # For adjacent, we might need heuristics. 
                # E.g., largest volume feature is the parent. For now, assume a symmetric constraint 
                # or a simple sequential dependency if not cyclic.
                if not nx.has_path(self.dag, target, source):
                    self.dag.add_edge(source, target, relation=edge.relation)

        # Ensure DAG
        if not nx.is_directed_acyclic_graph(self.dag):
            # Attempt to resolve cycles by removing weakest edges (confidence)
            try:
                cycles = list(nx.simple_cycles(self.dag))
                for cycle in cycles:
                    # simplistic cycle breaking: remove the last edge in the cycle list
                    u, v = cycle[-1], cycle[0]
                    if self.dag.has_edge(u, v):
                        self.dag.remove_edge(u, v)
            except nx.NetworkXNoCycle:
                pass

        if not nx.is_directed_acyclic_graph(self.dag):
            raise ValueError("Failed to resolve cyclic dependencies in the feature graph.")

        return self.dag

    def get_topological_order(self) -> List[str]:
        return list(nx.topological_sort(self.dag))

