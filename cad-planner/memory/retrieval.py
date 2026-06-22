from typing import List, Optional
from memory.pattern_database import PatternDatabase

class MemoryRetrieval:
    """
    Rule-based (and later semantic) retrieval of patterns from the database.
    """
    
    def __init__(self, db: PatternDatabase):
        self.db = db
        
    def retrieve(self, feature_semantics: str) -> Optional[List[str]]:
        """
        Retrieves a construction pattern based on semantic label or intent.
        """
        feature_lower = feature_semantics.lower()
        
        # Simple rule-based retrieval
        if "bearing" in feature_lower or "seat" in feature_lower:
            return self.db.get_pattern("bearing_seat")
        elif "counterbore" in feature_lower:
            return self.db.get_pattern("counterbore_hole")
        elif "flange" in feature_lower:
            return self.db.get_pattern("flange")
        elif "hole" in feature_lower:
            return self.db.get_pattern("standard_hole")
            
        return None
