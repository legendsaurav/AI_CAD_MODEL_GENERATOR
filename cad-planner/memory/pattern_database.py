from typing import List, Dict

class PatternDatabase:
    """
    Database of common construction strategies.
    Stores reusable engineering construction patterns.
    """
    
    def __init__(self):
        # Hardcoded for V1, could be loaded from JSON/DB later
        self.patterns = {
            "bearing_seat": ["create_sketch", "extrude", "fillet", "pattern"],
            "counterbore_hole": ["create_sketch", "extrude_cut", "extrude_cut_counterbore", "chamfer"],
            "flange": ["create_sketch", "revolve", "create_sketch", "extrude_cut", "pattern", "fillet"],
            "standard_hole": ["create_sketch", "extrude_cut"]
        }
        
    def get_pattern(self, name: str) -> List[str]:
        return self.patterns.get(name, [])
        
    def get_all_patterns(self) -> Dict[str, List[str]]:
        return self.patterns
