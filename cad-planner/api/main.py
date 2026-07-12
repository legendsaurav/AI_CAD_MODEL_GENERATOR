from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ggl_parser.ggl_parser import GGLParser
from validator.geometry import GeometryValidator
from intent.classifier import IntentClassifier
from dependency.graph import DependencyGraph
from construction.graph import ConstructionGraph
from construction.sketch_generator import SketchGenerator
from construction.sketch_optimizer import SketchOptimizer
from rules.engine import RuleEngine
from beam_search.planner import BeamSearchPlanner
from optimizer.feature_tree_optimizer import FeatureTreeOptimizer
from cal.generator import CALGenerator
from optimizer.cal_optimizer import CALOptimizer
from cal.schema import CALDocument
from cal.validator import CALValidator

app = FastAPI(title="CAD Planning Engine API", version="1.0.0")

class ParseRequest(BaseModel):
    ggl_json: str

@app.post("/parse_ggl")
def parse_ggl(req: ParseRequest):
    try:
        ggl = GGLParser.parse(req.ggl_json)
        return {"status": "success", "nodes": len(ggl.nodes), "edges": len(ggl.edges)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/validate_geometry")
def validate_geometry(req: ParseRequest):
    try:
        ggl = GGLParser.parse(req.ggl_json)
        validator = GeometryValidator(ggl)
        validator.validate()
        return {"status": "success", "message": "Geometry is valid"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate_construction_graph")
def generate_construction_graph(req: ParseRequest):
    """Generates a Construction Graph from validated GGL."""
    try:
        ggl = GGLParser.parse(req.ggl_json)
        GeometryValidator(ggl).validate()
        intents = IntentClassifier().classify(ggl)
        dep = DependencyGraph()
        dep.build(ggl)
        topo_order = dep.get_topological_order()
        
        cg = ConstructionGraph()
        for nid in topo_order:
            node = next((n for n in ggl.nodes if n.node_id == nid), None)
            intent = intents.get(nid)
            if node and intent:
                sketch = SketchGenerator.generate_for_primitive(node.type, node.parameters)
                sketch = SketchOptimizer.optimize(sketch)
                ops = RuleEngine.apply_rules(node.type, intent.primary_intent, sketch, ggl_params=node.parameters)
                for op in ops:
                    cg.add_operation(op)
        
        sequence = cg.get_sequence()
        return {
            "status": "success",
            "operations": [{"id": op.node_id, "type": op.operation_type} for op in sequence]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/plan")
def plan(req: ParseRequest):
    """Runs the full planning pipeline and returns the best CAL."""
    try:
        ggl = GGLParser.parse(req.ggl_json)
        GeometryValidator(ggl).validate()
        intents = IntentClassifier().classify(ggl)
        dep = DependencyGraph()
        dep.build(ggl)
        topo_order = dep.get_topological_order()
        
        sketches = {}
        for node in ggl.nodes:
            sketch = SketchGenerator.generate_for_primitive(node.type, node.parameters)
            sketch = SketchOptimizer.optimize(sketch)
            sketches[node.node_id] = sketch
            
        candidates = []
        cg = ConstructionGraph()
        for nid in topo_order:
            node = next((n for n in ggl.nodes if n.node_id == nid), None)
            intent = intents.get(nid)
            if node and intent:
                ops = RuleEngine.apply_rules(node.type, intent.primary_intent, sketches[nid], ggl_params=node.parameters)
                for op in ops:
                    cg.add_operation(op)
        candidates.append(cg)
        
        best = BeamSearchPlanner().plan(candidates)
        optimized = FeatureTreeOptimizer.optimize(best)
        actions = CALGenerator.generate(optimized)
        optimized_actions = CALOptimizer.optimize(actions)
        doc = CALDocument(actions=optimized_actions)
        CALValidator.validate(doc)
        
        return {"status": "success", "cal": doc.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate_cal")
def generate_cal(req: ParseRequest):
    """Alias for /plan — generates the final CAL from GGL."""
    return plan(req)

@app.post("/validate_cal")
def validate_cal(doc: CALDocument):
    try:
        CALValidator.validate(doc)
        return {"status": "success", "message": "CAL is valid"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
