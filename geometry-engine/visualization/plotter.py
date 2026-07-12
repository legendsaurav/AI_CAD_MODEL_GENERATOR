import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List

class VisualizationPlotter:
    """
    Generates research plots for Version 1 Representation Probing.
    """
    def __init__(self, exp_dir: str):
        self.exp_dir = exp_dir
        self.plots_dir = os.path.join(self.exp_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_correlation_matrix(self, keys: List[str], matrix: np.ndarray, filename: str = "correlation_matrix.png"):
        """Plots a heatmap of the cosine similarity between layer/timestep features."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, xticklabels=keys, yticklabels=keys, annot=True, fmt=".2f", cmap="viridis")
        plt.title("Layer-Timestep Feature Correlation Matrix")
        plt.tight_layout()
        path = os.path.join(self.plots_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"📈 Saved correlation matrix to {path}")

    def plot_temporal_stability(self, timesteps: List[float], drifts: List[float], filename: str = "temporal_stability.png"):
        """Plots the L2 feature drift between consecutive timesteps."""
        plt.figure(figsize=(8, 5))
        # drifts array has length N-1, where N is len(timesteps)
        x_labels = [f"t={timesteps[i]}->{timesteps[i+1]}" for i in range(len(drifts))]
        
        plt.plot(x_labels, drifts, marker='o', linestyle='-', color='b', linewidth=2)
        plt.title("Temporal Representation Stability (Feature Drift)")
        plt.xlabel("Diffusion Timestep Transition")
        plt.ylabel("Mean L2 Feature Drift")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"📈 Saved temporal stability plot to {path}")

    def plot_pca_variance(self, evr: np.ndarray, title: str, filename: str):
        """Plots the explained variance ratio of PCA components."""
        plt.figure(figsize=(6, 4))
        plt.bar(range(1, len(evr) + 1), evr, alpha=0.7, color='g')
        plt.step(range(1, len(evr) + 1), np.cumsum(evr), where='mid', color='r', label='Cumulative')
        plt.title(f"PCA Explained Variance - {title}")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.legend()
        plt.tight_layout()
        
        path = os.path.join(self.plots_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"📈 Saved PCA variance plot to {path}")
