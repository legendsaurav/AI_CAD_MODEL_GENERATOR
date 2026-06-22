"""
Comprehensive test suite for the CAD Planning Engine.
Tests every stage of the 20-step pipeline from GGL parsing to CAL export.
"""
import json
import os
import sys
import tempfile

# Ensure cad-planner root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

MOCK_GGL = {
    "version": "1.0",
    "nodes": [
        {"node_id": "n1", "type": "Box", "semantic_label": "Base Body",
         "parameters": {"width": 100, "height": 100, "depth": 20}},
        {"node_id": "n2", "type": "Cylinder", "semantic_label": "Through Hole",
         "parameters": {"radius": 10, "height": 20, "center_x": 0, "center_y": 0}}
    ],
    "edges": [
        {"source_id": "n1", "target_id": "n2", "relation": "Contains"}
    ]
}

MOCK_GGL_STR = json.dumps(MOCK_GGL)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: GGL Parser
# ─────────────────────────────────────────────────────────────────────────────

def test_ggl_parser_valid():
    from ggl_parser.ggl_parser import GGLParser
    ggl = GGLParser.parse(MOCK_GGL_STR)
    assert len(ggl.nodes) == 2
    assert len(ggl.edges) == 1
    assert ggl.nodes[0].node_id == "n1"

def test_ggl_parser_invalid_json():
    from ggl_parser.ggl_parser import GGLParser
    try:
        GGLParser.parse("not valid json")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_ggl_parser_bad_confidence():
    from ggl_parser.ggl_parser import GGLParser
    bad = {
        "version": "1.0",
        "nodes": [{"node_id": "x", "type": "Box", "confidence": 5.0}],
        "edges": []
    }
    try:
        GGLParser.parse(json.dumps(bad))
        assert False, "Should have raised ValueError for out-of-range confidence"
    except ValueError:
        pass

def test_ggl_parser_dangling_edge():
    from ggl_parser.ggl_parser import GGLParser
    bad = {
        "version": "1.0",
        "nodes": [{"node_id": "a", "type": "Box"}],
        "edges": [{"source_id": "a", "target_id": "nonexistent", "relation": "Contains"}]
    }
    try:
        GGLParser.parse(json.dumps(bad))
        assert False, "Should have raised ValueError for dangling edge"
    except ValueError:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Geometry Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_geometry_validation_valid():
    from ggl_parser.ggl_parser import GGLParser
    from validator.geometry import GeometryValidator
    ggl = GGLParser.parse(MOCK_GGL_STR)
    validated = GeometryValidator(ggl).validate()
    assert validated is not None

def test_geometry_validation_disconnected():
    from ggl_parser.ggl_parser import GGLParser
    from validator.geometry import GeometryValidator
    disconnected = {
        "version": "1.0",
        "nodes": [
            {"node_id": "a", "type": "Box"},
            {"node_id": "b", "type": "Cylinder"}
        ],
        "edges": []
    }
    ggl = GGLParser.parse(json.dumps(disconnected))
    try:
        GeometryValidator(ggl).validate()
        assert False, "Should have raised for disconnected components"
    except ValueError:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Intent Classification
# ─────────────────────────────────────────────────────────────────────────────

def test_intent_classification():
    from ggl_parser.ggl_parser import GGLParser
    from intent.classifier import IntentClassifier
    ggl = GGLParser.parse(MOCK_GGL_STR)
    intents = IntentClassifier().classify(ggl)
    # Box "Base Body" -> Extrusion
    assert intents["n1"].primary_intent == "Extrusion"
    # Cylinder "Through Hole" -> Cut Feature (because "hole" is in the label)
    assert intents["n2"].primary_intent == "Cut Feature"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Dependency Graph
# ─────────────────────────────────────────────────────────────────────────────

def test_dependency_graph():
    from ggl_parser.ggl_parser import GGLParser
    from dependency.graph import DependencyGraph
    ggl = GGLParser.parse(MOCK_GGL_STR)
    dep = DependencyGraph()
    dag = dep.build(ggl)
    order = dep.get_topological_order()
    # n1 must come before n2 (Box contains Cylinder)
    assert order.index("n1") < order.index("n2")

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Sketch Generation & Optimization
# ─────────────────────────────────────────────────────────────────────────────

def test_sketch_generator_cylinder():
    from construction.sketch_generator import SketchGenerator
    sketch = SketchGenerator.generate_for_primitive("Cylinder", {"radius": 15.0, "center_x": 5.0, "center_y": 3.0})
    assert len(sketch.entities) == 1
    assert sketch.entities[0].entity_type == "circle"

def test_sketch_generator_box():
    from construction.sketch_generator import SketchGenerator
    sketch = SketchGenerator.generate_for_primitive("Box", {"width": 50, "height": 30})
    assert len(sketch.entities) == 4
    for e in sketch.entities:
        assert e.entity_type == "line"

def test_sketch_optimizer_removes_zero_length():
    from construction.sketch_generator import SketchGenerator, SketchProfile, SketchLine
    from construction.sketch_optimizer import SketchOptimizer
    profile = SketchProfile(id="test", plane="XY")
    profile.entities = [
        SketchLine(id="ok", start=[0, 0], end=[10, 10]),
        SketchLine(id="zero", start=[5, 5], end=[5, 5]),  # zero length
    ]
    optimized = SketchOptimizer.optimize(profile)
    assert len(optimized.entities) == 1
    assert optimized.entities[0].id == "ok"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Constraints
# ─────────────────────────────────────────────────────────────────────────────

def test_constraint_inference_rectangle():
    from construction.sketch_generator import SketchGenerator
    from constraints.infer import ConstraintInferer
    sketch = SketchGenerator.generate_for_primitive("Box", {"width": 50, "height": 30})
    constraints = ConstraintInferer.infer(sketch)
    # 4 coincident + 2 parallel + 1 perpendicular = 7
    assert len(constraints) == 7

def test_constraint_repair_removes_invalid():
    from construction.sketch_generator import SketchGenerator
    from constraints.infer import GeometricConstraint
    from constraints.repair import ConstraintRepairer
    sketch = SketchGenerator.generate_for_primitive("Cylinder", {"radius": 10})
    # Create a constraint referencing a non-existent entity
    bad_constraint = GeometricConstraint(type="coincident", entities=["nonexistent1", "nonexistent2"])
    valid = ConstraintRepairer.repair(sketch, [bad_constraint])
    assert len(valid) == 0

# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: Manufacturing Analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_manufacturing_analyzer_passes():
    from ggl_parser.ggl_parser import GGLParser
    from manufacturing.analyzer import ManufacturingAnalyzer
    ggl = GGLParser.parse(MOCK_GGL_STR)
    score = ManufacturingAnalyzer.analyze(ggl)
    assert score.score == 1.0
    assert len(score.issues) == 0

def test_manufacturing_analyzer_high_aspect_ratio():
    from ggl_parser.ggl_parser import GGLParser
    from manufacturing.analyzer import ManufacturingAnalyzer
    bad = {
        "version": "1.0",
        "nodes": [{"node_id": "c", "type": "Cylinder", "parameters": {"radius": 1, "height": 100}}],
        "edges": []
    }
    ggl = GGLParser.parse(json.dumps(bad))
    score = ManufacturingAnalyzer.analyze(ggl)
    assert score.score < 1.0
    assert len(score.issues) > 0

# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Rule Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_rule_engine_extrusion():
    from construction.sketch_generator import SketchGenerator
    from rules.engine import RuleEngine
    sketch = SketchGenerator.generate_for_primitive("Box", {"width": 50, "height": 30})
    ops = RuleEngine.apply_rules("Box", "Extrusion", sketch, ggl_params={"depth": 25})
    assert len(ops) == 2
    assert ops[0].operation_type == "create_sketch"
    assert ops[1].operation_type == "extrude"
    assert ops[1].parameters["depth"] == 25

def test_rule_engine_cut_feature():
    from construction.sketch_generator import SketchGenerator
    from rules.engine import RuleEngine
    sketch = SketchGenerator.generate_for_primitive("Cylinder", {"radius": 10})
    ops = RuleEngine.apply_rules("Cylinder", "Cut Feature", sketch, ggl_params={"height": 20})
    assert len(ops) == 2
    assert ops[1].parameters["is_cut"] is True
    assert ops[1].parameters["depth"] == 20

# ─────────────────────────────────────────────────────────────────────────────
# Stage 9: Planner Memory
# ─────────────────────────────────────────────────────────────────────────────

def test_planner_memory_recall():
    from memory.planner_memory import PlannerMemory
    mem = PlannerMemory()
    assert mem.recall_strategy("Bearing Seat") is not None
    assert mem.recall_strategy("Through Hole") is not None
    assert mem.recall_strategy("Unknown Feature") is None

# ─────────────────────────────────────────────────────────────────────────────
# Stage 10: Ambiguity Resolver
# ─────────────────────────────────────────────────────────────────────────────

def test_ambiguity_resolver():
    from intent.classifier import IntentClassification
    from ambiguity.resolver import AmbiguityResolver
    ic = IntentClassification(primary_intent="Extrusion", alternatives=["Revolution", "Sweep"])
    strategies = AmbiguityResolver.resolve(ic)
    assert strategies == ["Extrusion", "Revolution", "Sweep"]

# ─────────────────────────────────────────────────────────────────────────────
# Stage 11: Construction Graph & Beam Search
# ─────────────────────────────────────────────────────────────────────────────

def test_construction_graph_sequence():
    from construction.graph import ConstructionGraph, ConstructionNode
    cg = ConstructionGraph()
    cg.add_operation(ConstructionNode(node_id="a", operation_type="create_sketch", parameters={}))
    cg.add_operation(ConstructionNode(node_id="b", operation_type="extrude", parameters={}))
    seq = cg.get_sequence()
    assert len(seq) == 2

def test_beam_search_returns_best():
    from construction.graph import ConstructionGraph, ConstructionNode
    from beam_search.planner import BeamSearchPlanner
    
    cg1 = ConstructionGraph()
    cg1.add_operation(ConstructionNode(node_id="a", operation_type="create_sketch", parameters={}))
    cg1.add_operation(ConstructionNode(node_id="b", operation_type="extrude", parameters={}))
    
    cg2 = ConstructionGraph()
    for i in range(10):
        cg2.add_operation(ConstructionNode(node_id=f"x{i}", operation_type="create_sketch", parameters={}))
    
    planner = BeamSearchPlanner()
    best = planner.plan([cg1, cg2])
    # cg1 should win because fewer operations and better score
    assert len(best.get_sequence()) == 2

# ─────────────────────────────────────────────────────────────────────────────
# Stage 12: Feature Tree Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_tree_optimizer():
    from construction.graph import ConstructionGraph, ConstructionNode
    from optimizer.feature_tree_optimizer import FeatureTreeOptimizer
    cg = ConstructionGraph()
    cg.add_operation(ConstructionNode(node_id="a", operation_type="create_sketch", parameters={}))
    optimized = FeatureTreeOptimizer.optimize(cg)
    # Should return a clone, not the same object
    assert optimized is not cg
    assert len(optimized.get_sequence()) == 1

# ─────────────────────────────────────────────────────────────────────────────
# Stage 13: CAL Schema & Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_cal_document_schema():
    from cal.schema import CALDocument
    doc = CALDocument()
    assert doc.version == "1.0"
    assert doc.ggl_version == "1.0"
    assert doc.reason_graph_version == "1.0"
    assert doc.generator == "cad-planner"
    assert doc.timestamp != ""

def test_cal_document_roundtrip():
    from cal.schema import CALDocument, CreateSketchAction
    doc = CALDocument(actions=[
        CreateSketchAction(action_id="s1", plane="Front")
    ])
    json_str = doc.to_json()
    restored = CALDocument.from_json(json_str)
    assert len(restored.actions) == 1
    assert restored.version == "1.0"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 14: CAL Validator
# ─────────────────────────────────────────────────────────────────────────────

def test_cal_validator_valid():
    from cal.schema import CALDocument, CreateSketchAction, ExtrudeAction
    from cal.validator import CALValidator
    doc = CALDocument(actions=[
        CreateSketchAction(action_id="s1", plane="Front"),
        ExtrudeAction(action_id="e1", sketch_id="s1", depth=10)
    ])
    assert CALValidator.validate(doc) is True

def test_cal_validator_missing_sketch():
    from cal.schema import CALDocument, ExtrudeAction
    from cal.validator import CALValidator
    doc = CALDocument(actions=[
        ExtrudeAction(action_id="e1", sketch_id="nonexistent", depth=10)
    ])
    try:
        CALValidator.validate(doc)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# Stage 15: CAL Exporter
# ─────────────────────────────────────────────────────────────────────────────

def test_cal_exporter():
    from cal.schema import CALDocument, CreateSketchAction
    from cal.exporter import CALExporter
    doc = CALDocument(actions=[
        CreateSketchAction(action_id="s1", plane="Top")
    ])
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        CALExporter.export(doc, path)
        with open(path, 'r') as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert len(data["actions"]) == 1
    finally:
        os.unlink(path)

# ─────────────────────────────────────────────────────────────────────────────
# End-to-End Pipeline Test
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline():
    """Runs the complete pipeline from GGL to CAL export and verifies the output."""
    from ggl_parser.ggl_parser import GGLParser
    from validator.geometry import GeometryValidator
    from intent.classifier import IntentClassifier
    from dependency.graph import DependencyGraph
    from construction.graph import ConstructionGraph
    from construction.sketch_generator import SketchGenerator
    from construction.sketch_optimizer import SketchOptimizer
    from constraints.infer import ConstraintInferer
    from constraints.repair import ConstraintRepairer
    from manufacturing.analyzer import ManufacturingAnalyzer
    from rules.engine import RuleEngine
    from beam_search.planner import BeamSearchPlanner
    from optimizer.feature_tree_optimizer import FeatureTreeOptimizer
    from cal.generator import CALGenerator
    from optimizer.cal_optimizer import CALOptimizer
    from cal.schema import CALDocument
    from cal.validator import CALValidator

    ggl = GGLParser.parse(MOCK_GGL_STR)
    ggl = GeometryValidator(ggl).validate()
    intents = IntentClassifier().classify(ggl)
    dep = DependencyGraph()
    dag = dep.build(ggl)
    topo = dep.get_topological_order()
    
    cg = ConstructionGraph()
    for nid in topo:
        node = next(n for n in ggl.nodes if n.node_id == nid)
        intent = intents.get(nid)
        if intent:
            sketch = SketchGenerator.generate_for_primitive(node.type, node.parameters)
            sketch = SketchOptimizer.optimize(sketch)
            ops = RuleEngine.apply_rules(node.type, intent.primary_intent, sketch, ggl_params=node.parameters)
            for op in ops:
                cg.add_operation(op)
    
    best = BeamSearchPlanner().plan([cg])
    optimized = FeatureTreeOptimizer.optimize(best)
    actions = CALGenerator.generate(optimized)
    optimized_actions = CALOptimizer.optimize(actions)
    doc = CALDocument(actions=optimized_actions)
    CALValidator.validate(doc)
    
    # Verify the output
    assert len(doc.actions) > 0
    assert doc.version == "1.0"
    
    # Verify we have sketch + extrude for the Box
    action_types = [a.action_type for a in doc.actions]
    assert "create_sketch" in action_types
    assert "extrude" in action_types


if __name__ == "__main__":
    # Simple test runner
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS  {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed:
        exit(1)
