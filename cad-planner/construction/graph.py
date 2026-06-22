import networkx as nx
from typing import List, Dict, Any, Optional

class ConstructionNode:
    def __init__(self, node_id: str, operation_type: str, parameters: Dict[str, Any], feature_ref: Optional[str] = None):
        self.node_id = node_id
        self.operation_type = operation_type
        self.parameters = parameters
        self.feature_ref = feature_ref # ID of the GGL feature this helps construct

class ConstructionGraph:
    """
    Represents the modeling strategy (e.g., Sketch -> Extrude -> Cut -> Fillet).
    Distinct from the Dependency Graph. Allows exploring multiple alternative histories.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def add_operation(self, node: ConstructionNode):
        self.graph.add_node(node.node_id, obj=node)
        
    def add_dependency(self, source_id: str, target_id: str, relation: str = "depends_on"):
        self.graph.add_edge(source_id, target_id, relation=relation)
        
    def get_sequence(self) -> List[ConstructionNode]:
        """Returns a valid linear execution sequence of construction operations."""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Construction Graph must be a DAG.")
        order = list(nx.topological_sort(self.graph))
        return [self.graph.nodes[node_id]['obj'] for node_id in order]
        
    def clone(self) -> "ConstructionGraph":
        new_cg = ConstructionGraph()
        new_cg.graph = self.graph.copy()
        return new_cg
