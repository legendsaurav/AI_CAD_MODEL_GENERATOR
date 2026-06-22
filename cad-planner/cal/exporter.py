import json
from cal.schema import CALDocument

class CALExporter:
    """
    Outputs the validated, software-independent CAL + Reason Graph.
    """
    
    @staticmethod
    def export(document: CALDocument, filepath: str):
        """
        Writes the CAL document to disk as JSON.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(document.to_json(indent=2))
