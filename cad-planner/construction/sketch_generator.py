from typing import Dict, Any, List, Union, Annotated, Literal
from pydantic import BaseModel, Field

class SketchEntity(BaseModel):
    entity_type: str
    id: str

class SketchCircle(SketchEntity):
    entity_type: Literal["circle"] = "circle"
    center: List[float]
    radius: float

class SketchLine(SketchEntity):
    entity_type: Literal["line"] = "line"
    start: List[float]
    end: List[float]

# Discriminated union ensures Pydantic serializes/deserializes the correct subclass
SketchEntityUnion = Annotated[Union[SketchCircle, SketchLine], Field(discriminator='entity_type')]

class SketchProfile(BaseModel):
    id: str
    plane: str
    entities: List[SketchEntityUnion] = Field(default_factory=list)

class SketchGenerator:
    """
    Dedicated module for profile reconstruction and dimension generation.
    """
    
    @staticmethod
    def generate_for_primitive(primitive_type: str, parameters: Dict[str, Any], plane: str = "XY") -> SketchProfile:
        """
        Generates the base sketch for a standard geometric primitive.
        """
        import uuid
        sketch_id = f"sketch_{uuid.uuid4().hex[:6]}"
        profile = SketchProfile(id=sketch_id, plane=plane)
        
        if primitive_type == "Cylinder":
            # Profile is a circle
            r = parameters.get("radius", 10.0)
            cx = parameters.get("center_x", 0.0)
            cy = parameters.get("center_y", 0.0)
            profile.entities.append(
                SketchCircle(id=f"circ_{uuid.uuid4().hex[:4]}", center=[cx, cy], radius=r)
            )
            
        elif primitive_type == "Box":
            # Profile is a rectangle (4 lines)
            w = parameters.get("width", 10.0)
            h = parameters.get("height", 10.0)
            cx = parameters.get("center_x", 0.0)
            cy = parameters.get("center_y", 0.0)
            
            x1, x2 = cx - w/2, cx + w/2
            y1, y2 = cy - h/2, cy + h/2
            
            profile.entities.extend([
                SketchLine(id=f"l1_{uuid.uuid4().hex[:4]}", start=[x1, y1], end=[x2, y1]),
                SketchLine(id=f"l2_{uuid.uuid4().hex[:4]}", start=[x2, y1], end=[x2, y2]),
                SketchLine(id=f"l3_{uuid.uuid4().hex[:4]}", start=[x2, y2], end=[x1, y2]),
                SketchLine(id=f"l4_{uuid.uuid4().hex[:4]}", start=[x1, y2], end=[x1, y1]),
            ])
            
        return profile
