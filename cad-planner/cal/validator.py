from typing import List
from cal.schema import CALDocument

class CALValidator:
    """
    Validates the final CAL sequence for parameter completeness and operational ordering.
    """
    
    @staticmethod
    def validate(document: CALDocument) -> bool:
        """
        Ensures the CAL is structurally sound and dependencies are correct.
        """
        # Pydantic already handles base schema validation.
        # Here we check business logic (e.g. extrude references an existing sketch)
        
        seen_sketches = set()
        
        for action in document.actions:
            if action.action_type == "create_sketch":
                seen_sketches.add(action.action_id)
            elif action.action_type in ["draw_circle", "draw_rectangle", "extrude", "revolve"]:
                if hasattr(action, 'sketch_id') and action.sketch_id not in seen_sketches:
                    raise ValueError(f"Action {action.action_id} references unknown sketch: {action.sketch_id}")
                    
        return True
