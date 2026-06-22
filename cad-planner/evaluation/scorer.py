from construction.graph import ConstructionGraph

class PlanScorer:
    """
    Evaluates plans based on reconstruction quality, operation count, 
    manufacturability, rebuild stability, and design intent preservation.
    """
    
    def score(self, cg: ConstructionGraph) -> float:
        """
        Calculates a composite heuristic score for a construction graph.
        Higher is better.
        """
        score = 100.0
        
        sequence = cg.get_sequence()
        
        # Penalize overly long sequences (Action count / CAL optimization ratio)
        score -= len(sequence) * 2.0
        
        # Penalize deep/complex dependencies
        # In a real implementation, we would check the depth of the DAG.
        
        # Reward design intent preservation
        # E.g. using standard operations like Extrude/Revolve over complex sweeps
        for op in sequence:
            if op.operation_type in ["extrude", "revolve", "create_sketch"]:
                score += 1.0 # Reward simple, robust operations
                
        return score
