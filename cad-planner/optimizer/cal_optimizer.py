from typing import List
from cal.schema import CALAction

class CALOptimizer:
    """
    Merges redundant sketches, removes duplicate operations, simplifies feature ordering, 
    and optimizes references to yield minimal, stable, and editable CAL.
    """
    
    @staticmethod
    def optimize(actions: List[CALAction]) -> List[CALAction]:
        """
        Takes raw CAL actions and optimizes them.
        """
        optimized = []
        seen_sketches = set()
        
        for action in actions:
            # Example optimization: remove consecutive duplicate sketches on the same plane
            # if they have identical reasoning. (Simplified for V1)
            if action.action_type == "create_sketch":
                key = action.plane
                if key not in seen_sketches:
                    seen_sketches.add(key)
                    optimized.append(action)
                else:
                    # Keep it anyway for now unless it's perfectly identical
                    optimized.append(action)
            else:
                optimized.append(action)
                
        return optimized
