"""
scripts/run_probing.py  –  Version 1: Temporal Representation Probing
======================================================================
Generates mock DiT features (or loads real ones from logs/features/),
performs PCA, correlation-matrix, temporal-stability, and layer-ranking
analysis, saves plots, and writes a Markdown research report.
"""
import os
import sys
import numpy as np

# ── ensure the repo root is always on sys.path ───────────────────────────
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from probing.analyzer import FeatureAnalyzer  # noqa: E402
from visualization.plotter import VisualizationPlotter  # noqa: E402
from utils.logger import ExperimentLogger  # noqa: E402


# ── Mock feature generator ────────────────────────────────────────────────

def generate_mock_features(log_dir: str):
    """
    Generates synthetic tensors that mimic Hunyuan3D DiT hidden states.
    Shape: [1, 256, 1024]  →  batch=1, sequence_len=256, hidden_dim=1024
    Saved as single_L{layer}_t{timestep:.2f}.npy in log_dir.
    """
    print("🧪 Generating mock DiT feature tensors...")
    os.makedirs(log_dir, exist_ok=True)

    timesteps = [0.1, 0.3, 0.5, 0.7, 0.9]
    layers    = [0, 4, 8, 12]
    features  = {}

    for t in timesteps:
        base = np.random.randn(1, 256, 1024).astype(np.float32) * (1.0 - t)
        for layer in layers:
            feat = base + np.random.randn(1, 256, 1024).astype(np.float32) * 0.1
            # Inject structured signal into layer 12 so rankings work meaningfully
            if layer == 12:
                signal = np.sin(np.linspace(0, 10, 256))[:, None].astype(np.float32)
                feat[0, :, :5] += signal * 2.0
            key      = f"L{layer}_t{t:.2f}"
            filename = f"single_{key}.npy"
            np.save(os.path.join(log_dir, filename), feat)
            features[key] = feat

    print(f"   Saved {len(features)} feature tensors → {log_dir}")
    return features, timesteps


# ── Report writer ─────────────────────────────────────────────────────────

def write_markdown_report(exp_dir: str, rankings, timesteps, drifts):
    lines = [
        "# Version 1 – Temporal Representation Probing Report\n",
        "## 1. Feature Importance Ranking",
        "_Ranked by geometric distinctness (sum of top-3 PCA explained variance)._\n",
        "| Rank | Layer / Timestep | Score |",
        "|------|-----------------|-------|",
    ]
    for i, (name, score) in enumerate(rankings, 1):
        lines.append(f"| {i} | {name} | {score:.4f} |")

    lines += [
        "",
        "## 2. Temporal Stability (L2 Feature Drift)",
        "_Average L2 distance between feature tensors at consecutive timesteps._\n",
    ]
    for i, drift in enumerate(drifts):
        lines.append(f"- **t={timesteps[i]:.2f} → t={timesteps[i+1]:.2f}**: {drift:.4f}")

    lines += [
        "",
        "## 3. Visualisations",
        "See `plots/` for correlation heatmap, temporal drift chart, and PCA variance.",
    ]

    report_path = os.path.join(exp_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"📝 Research report → {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  VERSION 1 – Temporal Representation Probing")
    print("=" * 60)

    logger       = ExperimentLogger()
    features_dir = logger.get_log_dir("features")

    # 1. Load or generate features
    features, timesteps = generate_mock_features(features_dir)

    # 2. Analysis
    analyzer = FeatureAnalyzer(features_dir)

    print("\n📊 Computing Layer–Timestep Correlation Matrix...")
    keys, corr_matrix = analyzer.compute_correlation_matrix(features)

    print("📊 Computing Temporal Stability (L12)...")
    l12_series = [features[f"L12_t{t:.2f}"] for t in timesteps]
    drifts     = analyzer.temporal_stability_analysis(l12_series)

    print("📊 Ranking Layers by Geometric Distinctness...")
    rankings = analyzer.rank_layers(features)

    print("\n🏆 Top-5 Layer/Timestep combinations:")
    for i, (name, score) in enumerate(rankings[:5], 1):
        print(f"   {i}. {name}  →  {score:.4f}")

    # 3. Visualise
    plotter = VisualizationPlotter(logger.get_exp_dir())

    print("\n🎨 Saving plots...")
    plotter.plot_correlation_matrix(keys, corr_matrix)
    plotter.plot_temporal_stability(timesteps, drifts)

    best_key     = rankings[0][0]
    _, best_evr  = analyzer.run_pca(features[best_key], n_components=5)
    plotter.plot_pca_variance(best_evr, title=f"Best: {best_key}", filename="best_layer_pca.png")

    # 4. Write report
    write_markdown_report(logger.get_exp_dir(), rankings, timesteps, drifts)

    print(f"\n✅ Version 1 complete  →  {logger.get_exp_dir()}")
    return rankings  # return for run_all.py chaining


if __name__ == "__main__":
    main()
