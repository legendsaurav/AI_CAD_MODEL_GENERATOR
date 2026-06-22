from typing import List, Dict, Any
from pydantic import BaseModel, Field
from construction.sketch_generator import SketchProfile, SketchEntity

class GeometricConstraint(BaseModel):
    type: str  # "coincident", "parallel", "perpendicular", "concentric", "tangent", "equal", "symmetry", "midpoint"
    entities: List[str]  # IDs of entities involved
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ConstraintInferer:
    """
    Automatically infers geometric constraints from sketch entities and parameters.
    """
    
    @staticmethod
    def infer(profile: SketchProfile) -> List[GeometricConstraint]:
        constraints = []
        
        # Simple heuristic: if we have a rectangle, infer horizontal/vertical/perpendicular constraints
        # Real implementation would do geometric analysis of the entities.
        
        lines = [e for e in profile.entities if e.entity_type == "line"]
        if len(lines) == 4:
            # Assume it's a generated box sketch
            # Add coincident constraints at corners
            constraints.append(GeometricConstraint(type="coincident", entities=[lines[0].id, lines[1].id]))
            constraints.append(GeometricConstraint(type="coincident", entities=[lines[1].id, lines[2].id]))
            constraints.append(GeometricConstraint(type="coincident", entities=[lines[2].id, lines[3].id]))
            constraints.append(GeometricConstraint(type="coincident", entities=[lines[3].id, lines[0].id]))
            
            # Add parallel constraints
            constraints.append(GeometricConstraint(type="parallel", entities=[lines[0].id, lines[2].id]))
            constraints.append(GeometricConstraint(type="parallel", entities=[lines[1].id, lines[3].id]))
            
            # Add perpendicular constraints
            constraints.append(GeometricConstraint(type="perpendicular", entities=[lines[0].id, lines[1].id]))

        return constraints
