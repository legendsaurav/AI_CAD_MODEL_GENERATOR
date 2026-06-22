from construction.sketch_generator import SketchProfile

class SketchOptimizer:
    """
    Simplifies and optimizes generated sketch profiles.
    Merges collinear segments, removes zero-length lines, and optimizes geometry.
    """
    
    @staticmethod
    def optimize(profile: SketchProfile, tolerance: float = 1e-6) -> SketchProfile:
        """
        Optimizes a sketch profile in-place or returns a new one.
        """
        optimized_entities = []
        
        for entity in profile.entities:
            # Use entity_type string field instead of isinstance() because
            # Pydantic deserialization returns the base SketchEntity, not subclasses.
            if entity.entity_type == "line":
                start = entity.start  # type: ignore[attr-defined]
                end = entity.end      # type: ignore[attr-defined]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if (dx**2 + dy**2) > tolerance**2:
                    optimized_entities.append(entity)
            elif entity.entity_type == "circle":
                if entity.radius > tolerance:  # type: ignore[attr-defined]
                    optimized_entities.append(entity)
            else:
                optimized_entities.append(entity)
                
        # Future: Implement collinear line merging
        # Future: Implement arc fitting for highly tessellated line segments
        
        profile.entities = optimized_entities
        return profile
