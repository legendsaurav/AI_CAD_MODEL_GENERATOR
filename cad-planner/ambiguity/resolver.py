from typing import List, Dict
from intent.classifier import IntentClassification

class AmbiguityResolver:
    """
    Identifies branching paths (e.g., Extrude vs. Revolve for a cylinder) 
    and generates alternative strategies.
    """
    
    @staticmethod
    def resolve(intent: IntentClassification) -> List[str]:
        """
        Returns a list of all valid modeling strategies for a given intent classification.
        """
        strategies = [intent.primary_intent]
        strategies.extend(intent.alternatives)
        return strategies
