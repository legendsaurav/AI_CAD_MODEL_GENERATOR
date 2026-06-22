import json
import os
import sys

# Add parent dir to sys path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, _ROOT_DIR)

# Also add shared-schemas
_SHARED_SCHEMAS = os.path.abspath(os.path.join(_ROOT_DIR, '..', 'shared-schemas'))
if _SHARED_SCHEMAS not in sys.path:
    sys.path.insert(0, _SHARED_SCHEMAS)

from ggl_parser.ggl_parser import GGLParser
from validator.geometry import GeometryValidator
from intent.classifier import IntentClassifier
from dependency.graph import DependencyGraph
from construction.graph import ConstructionGraph, ConstructionNode
from construction.sketch_generator import SketchGenerator
from construction.sketch_optimizer import SketchOptimizer
from constraints.infer import ConstraintInferer
from constraints.repair import ConstraintRepairer
from manufacturing.analyzer import ManufacturingAnalyzer
from rules.engine import RuleEngine
from memory.planner_memory import PlannerMemory
from ambiguity.resolver import AmbiguityResolver
from beam_search.planner import BeamSearchPlanner
from optimizer.feature_tree_optimizer import FeatureTreeOptimizer
from cal.generator import CALGenerator
from optimizer.cal_optimizer import CALOptimizer
from cal.schema import CALDocument, PlanningTrace
from cal.validator import CALValidator
from cal.exporter import CALExporter
from utils.logger import setup_logger

logger = setup_logger("pipeline_runner")

def run():
    logger.info("=" * 60)
    logger.info("Starting CAD Planning Engine Pipeline")
    logger.info("=" * 60)
    
    # ─────────────────────────────────────────────────────────
    # 1. Mock GGL Input (would come from Geometry Engine)
    #    CRITICAL: metadata.source_type MUST be "dit_hidden_states"
    # ─────────────────────────────────────────────────────────
    mock_ggl = {
        "version": "1.0",
        "schema_name": "geometry_graph_language",
        "metadata": {
            "generator": "geometry-engine-v1.0",
            "source_type": "dit_hidden_states",
            "hunyuan_model_version": "2.1",
            "layers_used": [0, 4, 8, 12],
            "timestep_extracted": 0.5,
        },
        "nodes": [
            {"node_id": "n1", "type": "Box", "semantic_label": "Base Body", "confidence": 0.92, "parameters": {"width": 100, "height": 100, "depth": 20}},
            {"node_id": "n2", "type": "Cylinder", "semantic_label": "Through Hole", "confidence": 0.85, "parameters": {"radius": 10, "height": 20, "center_x": 0, "center_y": 0}}
        ],
        "edges": [
            {"source_id": "n1", "target_id": "n2", "relation": "Contains", "confidence": 0.88}
        ]
    }
    
    # Initialize PlanningTrace
    trace = PlanningTrace(
        ggl_node_count=len(mock_ggl["nodes"]),
        ggl_edge_count=len(mock_ggl["edges"]),
    )
    
    # ─────────────────────────────────────────────────────────
    # Stage 1: Parse GGL (with source integrity check)
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 1: Parsing GGL (with source integrity verification)")
    ggl = GGLParser.parse(json.dumps(mock_ggl))
    logger.info(f"  Parsed {len(ggl.nodes)} nodes, {len(ggl.edges)} edges")
    logger.info(f"  Source type: {ggl.metadata.source_type} ✓")
    
    # ─────────────────────────────────────────────────────────
    # Stage 2: Geometry Validation
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 2: Geometry Validation")
    ggl = GeometryValidator(ggl).validate()
    logger.info("  Geometry is valid")
    
    # ─────────────────────────────────────────────────────────
    # Stage 3: Construction Intent Classification
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 3: Intent Classification")
    intents = IntentClassifier().classify(ggl)
    for nid, ic in intents.items():
        logger.info(f"  {nid}: primary={ic.primary_intent}, alternatives={ic.alternatives}")
        trace.intents.append({"node_id": nid, "primary": ic.primary_intent, "alternatives": ic.alternatives})
    
    # ─────────────────────────────────────────────────────────
    # Stage 4: Dependency Graph Generation
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 4: Dependency Graph Generation")
    dep_builder = DependencyGraph()
    dag = dep_builder.build(ggl)
    topo_order = dep_builder.get_topological_order()
    logger.info(f"  Topological order: {topo_order}")
    trace.topological_order = list(topo_order)
    
    # ─────────────────────────────────────────────────────────
    # Stage 5: Sketch Generation + Constraint Inference/Repair
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 5: Sketch Generation & Constraints")
    sketches = {}
    for node in ggl.nodes:
        sketch = SketchGenerator.generate_for_primitive(node.type, node.parameters)
        sketch = SketchOptimizer.optimize(sketch)
        constraints = ConstraintInferer.infer(sketch)
        valid_constraints = ConstraintRepairer.repair(sketch, constraints)
        sketches[node.node_id] = sketch
        logger.info(f"  {node.node_id}: {len(sketch.entities)} entities, {len(valid_constraints)} constraints")
    
    # ─────────────────────────────────────────────────────────
    # Stage 6: Manufacturability Analysis
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 6: Manufacturability Analysis")
    mfg_score = ManufacturingAnalyzer.analyze(ggl)
    logger.info(f"  Manufacturability score: {mfg_score.score}")
    trace.manufacturability_score = mfg_score.score
    for issue in mfg_score.issues:
        logger.info(f"  Issue: {issue}")
        trace.manufacturability_issues.append(issue)
    
    # ─────────────────────────────────────────────────────────
    # Stage 7: Planner Memory Retrieval
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 7: Planner Memory Retrieval")
    memory = PlannerMemory()
    for node in ggl.nodes:
        pattern = memory.recall_strategy(node.semantic_label)
        if pattern:
            logger.info(f"  {node.node_id}: recalled pattern {pattern}")
            trace.memory_recalls.append({"node_id": node.node_id, "pattern": pattern})
        else:
            logger.info(f"  {node.node_id}: no stored pattern, using rule engine")

    # ─────────────────────────────────────────────────────────
    # Stage 8: Rule Engine + Ambiguity Resolution
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 8: Rule Engine & Ambiguity Resolution")
    candidates = []
    
    for node_id in topo_order:
        node = next((n for n in ggl.nodes if n.node_id == node_id), None)
        intent = intents.get(node_id)
        if not node or not intent:
            continue

        all_strategies = AmbiguityResolver.resolve(intent)
        logger.info(f"  {node_id}: strategies = {all_strategies}")
        trace.ambiguity_strategies.append({"node_id": node_id, "strategies": all_strategies})
        
        for strategy in all_strategies:
            cg = ConstructionGraph()
            for nid in topo_order:
                n = next((nn for nn in ggl.nodes if nn.node_id == nid), None)
                ic = intents.get(nid)
                if not n or not ic:
                    continue
                    
                active_intent = strategy if nid == node_id else ic.primary_intent
                sketch = sketches[nid]
                ops = RuleEngine.apply_rules(n.type, active_intent, sketch, ggl_params=n.parameters)
                
                # Propagate confidence and source node to construction operations
                for op in ops:
                    op.parameters["confidence"] = n.confidence
                    op.parameters["source_ggl_node_id"] = n.node_id
                    cg.add_operation(op)
                    
            candidates.append(cg)
    
    if not candidates:
        logger.error("No candidates generated!")
        return
    logger.info(f"  Generated {len(candidates)} candidate construction graphs")
    trace.beam_candidates_count = len(candidates)
    
    # ─────────────────────────────────────────────────────────
    # Stage 9: Beam Search Planning + Design Intent Scoring
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 9: Beam Search Planning")
    planner = BeamSearchPlanner()
    
    # Score all candidates for trace
    from evaluation.scorer import PlanScorer
    scorer = PlanScorer()
    for idx, cg in enumerate(candidates):
        score = scorer.score(cg)
        trace.beam_scores.append({"candidate_idx": idx, "score": score})
    
    best_plan = planner.plan(candidates)
    best_idx = next((s["candidate_idx"] for s in trace.beam_scores 
                     if s["score"] == max(s["score"] for s in trace.beam_scores)), 0)
    trace.best_candidate_idx = best_idx
    logger.info(f"  Best plan has {len(best_plan.get_sequence())} operations")
    
    # ─────────────────────────────────────────────────────────
    # Stage 10: Feature Tree Optimization
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 10: Feature Tree Optimization")
    optimized_plan = FeatureTreeOptimizer.optimize(best_plan)
    
    # ─────────────────────────────────────────────────────────
    # Stage 11: CAL Generation (with confidence propagation)
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 11: CAL Generation (with confidence propagation)")
    actions = CALGenerator.generate(optimized_plan)
    logger.info(f"  Generated {len(actions)} CAL actions")
    trace.total_cal_actions_before_optimization = len(actions)
    
    # Log confidence values propagated to CAL
    for a in actions:
        logger.info(f"    {a.action_id}: confidence={a.confidence:.2f}, source={a.source_ggl_node_id}")
    
    # ─────────────────────────────────────────────────────────
    # Stage 12: CAL Optimization
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 12: CAL Optimization")
    optimized_actions = CALOptimizer.optimize(actions)
    logger.info(f"  Optimized to {len(optimized_actions)} CAL actions")
    trace.total_cal_actions_after_optimization = len(optimized_actions)
    
    # ─────────────────────────────────────────────────────────
    # Stage 13: Assemble CAL Document + Validate + Attach Trace
    # ─────────────────────────────────────────────────────────
    logger.info("Stage 13: Assembling & Validating CAL Document (with PlanningTrace)")
    doc = CALDocument(actions=optimized_actions, planning_trace=trace)
    CALValidator.validate(doc)
    logger.info("  CAL document is valid")
    logger.info(f"  PlanningTrace attached: {trace.beam_candidates_count} candidates, best={trace.best_candidate_idx}")
    
    # ─────────────────────────────────────────────────────────
    # Stage 14: Export
    # ─────────────────────────────────────────────────────────
    output_dir = os.path.join(_ROOT_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "final.cal.json")
    CALExporter.export(doc, out_path)
    
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — CAL exported to: {out_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run()
