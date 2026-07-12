"""
scripts/run_primitive_recovery.py  –  Version 3: Primitive Recovery
====================================================================
Takes the un-parameterised GGL from Version 2, proposes Top-K analytic
primitives for each Part node, regresses their exact parameters, and
selects the best fit via geometric optimisation.
"""
import os
import sys
import torch

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.config import ConfigManager  # noqa: E402
from utils.logger import ExperimentLogger  # noqa: E402
from graph.ggl import GeometryGraphLanguage, GGLNode, GGLEdge  # noqa: E402
from primitive.generator import PrimitiveProposalGenerator  # noqa: E402
from primitive.estimator import ParameterEstimator  # noqa: E402
from primitive.optimizer import GeometricOptimizer  # noqa: E402


def build_mock_ggl() -> GeometryGraphLanguage:
    """Mocks the un-parameterised Part-graph output of Version 2."""
    ggl = GeometryGraphLanguage()
    ggl.add_node(GGLNode(node_id="part_1", type="Part", semantic_label="Main Body",    confidence=0.92))
    ggl.add_node(GGLNode(node_id="part_2", type="Part", semantic_label="Mounting Hole", confidence=0.87))
    ggl.add_node(GGLNode(node_id="part_3", type="Part", semantic_label="Shaft",         confidence=0.78))
    return ggl


def main():
    print("\n" + "=" * 60)
    print("  VERSION 3 – Primitive Recovery")
    print("=" * 60)

    config = ConfigManager.get_all()
    logger = ExperimentLogger()

    # ── Inputs ────────────────────────────────────────────────────────────
    print("🧪 Loading mock Part-graph from Version 2...")
    ggl = build_mock_ggl()

    hidden_dim   = config.get("heads", {}).get("hidden_dim", 1024)
    mock_feature = torch.randn(1, hidden_dim)   # [1, D] – single pooled feature

    # ── Pipeline modules ──────────────────────────────────────────────────
    proposer  = PrimitiveProposalGenerator(config)
    estimator = ParameterEstimator(config)
    optimizer = GeometricOptimizer(config)

    optimised_primitives = []
    part_nodes = [n for n in ggl.nodes if n.type == "Part"]

    print(f"\n⚙️  Processing {len(part_nodes)} Part nodes...")

    for node in part_nodes:
        print(f"\n   Part: {node.semantic_label}  ({node.node_id})")

        # Step A: Top-K proposals
        proposals = proposer.generate_proposals(mock_feature, node)
        print(f"   Proposals: {[f'{p.type}({p.confidence:.2f})' for p in proposals]}")

        # Step B: Regress parameters for each proposal
        param_proposals = []
        for prop in proposals:
            parameterised = estimator.estimate(mock_feature.clone(), prop)
            param_proposals.append(parameterised)

        # Step C: Geometric optimisation → pick winner
        best = optimizer.optimize(param_proposals, target_features=mock_feature)
        print(f"   Winner: {best.type}  params={list(best.parameters.keys())[:4]}")
        optimised_primitives.append((node.node_id, best))

    # ── Inject optimised primitives back into the GGL ────────────────────
    for part_id, prim in optimised_primitives:
        ggl.add_node(prim)
        ggl.add_edge(GGLEdge(
            source_id=part_id,
            target_id=prim.node_id,
            relation="Instantiates",
            confidence=prim.confidence,
        ))

    # ── Save ──────────────────────────────────────────────────────────────
    logger.save_ggl(ggl.model_dump(), filename="optimised_primitive_graph.json")

    metrics = {
        "parts_processed":           len(part_nodes),
        "primitives_added":          len(optimised_primitives),
        "total_nodes_after":         len(ggl.nodes),
        "total_edges_after":         len(ggl.edges),
        "primitive_confidence_mean": round(
            sum(p.confidence for _, p in optimised_primitives) / max(len(optimised_primitives), 1), 3
        ),
    }
    logger.log_metrics(metrics, step=3)

    print("\n📋 Primitive Recovery Summary:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    print(f"\n✅ Version 3 complete  →  {logger.get_exp_dir()}")
    return ggl


if __name__ == "__main__":
    main()
