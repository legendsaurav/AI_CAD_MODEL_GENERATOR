from typing import Optional, List
from memory.pattern_database import PatternDatabase
from memory.retrieval import MemoryRetrieval
from construction.graph import ConstructionNode

class PlannerMemory:
    """
    Stores and recalls reusable engineering construction patterns to improve 
    planning speed, consistency, and design intent preservation.
    """
    
    def __init__(self):
        self.db = PatternDatabase()
        self.retrieval = MemoryRetrieval(self.db)
        
    def recall_strategy(self, semantic_label: str) -> Optional[List[str]]:
        """
        Attempts to recall a known construction strategy for a feature.
        """
        if not semantic_label:
            return None
        return self.retrieval.retrieve(semantic_label)
