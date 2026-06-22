from typing import Dict, List, Any
from ggl_parser.ggl_parser import GeometryGraphLanguage, GGLNode

class IntentClassification:
    def __init__(self, primary_intent: str, alternatives: List[str] = None):
        self.primary_intent = primary_intent
        self.alternatives = alternatives or []

class IntentClassifier:
    """
    Construction Intent Layer.
    Transforms geometric primitives into engineering design intent before planning.
    """
    
    # Simple deterministic mapping for V1.
    # Future versions can use machine learning based on geometric context.
    INTENT_MAP = {
        "Cylinder": ["Extrusion", "Revolution", "Sweep"],
        "Box": ["Extrusion"],
        "Sphere": ["Revolution"],
        "Cone": ["Revolution", "Extrusion", "Loft"],
        "Torus": ["Sweep", "Revolution"],
    }

    def classify(self, ggl: GeometryGraphLanguage) -> Dict[str, IntentClassification]:
        """
        Takes validated GGL and generates an Intent Classification for each primitive node.
        """
        intents = {}
        for node in ggl.nodes:
            if node.type in self.INTENT_MAP:
                possible = self.INTENT_MAP[node.type]
                
                # Contextual overrides (e.g. if semantic_label says "Pocket" -> Cut Feature)
                label = node.semantic_label.lower()
                if "pocket" in label or "hole" in label or "cut" in label:
                    primary = "Cut Feature"
                    alts = [p for p in possible]
                elif "boss" in label or "pad" in label:
                    primary = "Additive Feature"
                    alts = [p for p in possible]
                else:
                    primary = possible[0]
                    alts = possible[1:]

                intents[node.node_id] = IntentClassification(primary_intent=primary, alternatives=alts)
                
        return intents
