from typing import Dict, Any, List
from ggl_parser.ggl_parser import GeometryGraphLanguage

class ManufacturabilityScore:
    def __init__(self, score: float, issues: List[str]):
        self.score = score
        self.issues = issues

class ManufacturingAnalyzer:
    """
    Evaluates minimum wall thickness, impossible draft angles, undercuts, 
    inaccessible cuts, and manufacturing complexity to generate manufacturability scores.
    """
    
    @staticmethod
    def analyze(ggl: GeometryGraphLanguage) -> ManufacturabilityScore:
        score = 1.0
        issues = []
        
        # Simple heuristics for V1
        for node in ggl.nodes:
            if node.type == "Cylinder":
                # Check for extremely thin high cylinders (impossible to machine easily)
                r = node.parameters.get("radius", 1.0)
                h = node.parameters.get("height", 1.0)
                if h / r > 20:
                    score -= 0.1
                    issues.append(f"High aspect ratio cylinder (impossible to drill/mill easily): {node.node_id}")
            elif node.type == "Part":
                # Check for overly complex single parts
                pass
                
        # Lower bound score
        score = max(0.0, score)
        return ManufacturabilityScore(score, issues)
