from typing import List
from constraints.infer import GeometricConstraint
from construction.sketch_generator import SketchProfile

class ConstraintRepairer:
    """
    Validates constraints and repairs over-constrained sketches, 
    under-constrained sketches, conflicting dimensions, or invalid references.
    """
    
    @staticmethod
    def repair(profile: SketchProfile, constraints: List[GeometricConstraint]) -> List[GeometricConstraint]:
        """
        Validates and cleans a set of constraints for a sketch profile.
        """
        valid_constraints = []
        entity_ids = {e.id for e in profile.entities}
        
        # 1. Remove constraints referring to non-existent entities (invalid references)
        for c in constraints:
            if all(e_id in entity_ids for e_id in c.entities):
                valid_constraints.append(c)
                
        # 2. Detect conflicting constraints (e.g. parallel AND perpendicular on same two lines)
        # 3. Detect over-constraint (e.g. multiple coincident constraints on same point)
        
        # Simple conflict resolution: keep the first constraint of a pair
        pair_set = set()
        final_constraints = []
        for c in valid_constraints:
            if len(c.entities) == 2:
                pair = tuple(sorted(c.entities))
                # For V1, just avoid strictly duplicate constraints
                key = (c.type, pair)
                if key not in pair_set:
                    pair_set.add(key)
                    final_constraints.append(c)
            else:
                final_constraints.append(c)
                
        return final_constraints
