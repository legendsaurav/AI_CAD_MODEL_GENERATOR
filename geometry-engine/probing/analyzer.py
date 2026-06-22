import numpy as np
import os
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

class FeatureAnalyzer:
    """
    Analyzes extracted hidden states to prove whether they 
    contain geometric information. Implements PCA, stability analysis,
    and cross-layer/timestep correlation.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir

    def load_feature(self, filename: str) -> np.ndarray:
        path = os.path.join(self.log_dir, filename)
        if os.path.exists(path):
            return np.load(path)
        raise FileNotFoundError(f"{path} not found.")

    def run_pca(self, feature: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs PCA on the feature map to visualize its principal components.
        Expected feature shape: [Batch, Sequence_Length, Hidden_Dim]
        Returns:
            - projected: [Batch, Sequence_Length, n_components]
            - explained_variance_ratio: [n_components]
        """
        B, N, D = feature.shape
        # Flatten batch and sequence dimensions to run PCA across all tokens
        flattened = feature.reshape(-1, D)
        
        pca = PCA(n_components=n_components)
        projected_flat = pca.fit_transform(flattened)
        
        projected = projected_flat.reshape(B, N, n_components)
        return projected, pca.explained_variance_ratio_

    def compute_correlation_matrix(self, features: Dict[str, np.ndarray]) -> Tuple[List[str], np.ndarray]:
        """
        Computes the pairwise cosine similarity between all provided layers/timesteps.
        Returns the keys and the correlation matrix.
        """
        keys = list(features.keys())
        n = len(keys)
        matrix = np.zeros((n, n))
        
        # We compute correlation by comparing the mean feature vectors across the sequence
        mean_vectors = {}
        for k in keys:
            feat = features[k] # [B, N, D]
            # Average over Sequence and Batch to get a global signature [1, D]
            mean_vectors[k] = feat.mean(axis=(0, 1)).reshape(1, -1)
            
        for i in range(n):
            for j in range(n):
                sim = cosine_similarity(mean_vectors[keys[i]], mean_vectors[keys[j]])
                matrix[i, j] = sim[0, 0]
                
        return keys, matrix

    def temporal_stability_analysis(self, features_over_time: List[np.ndarray]) -> List[float]:
        """
        Given a list of feature tensors [B, N, D] ordered by timestep,
        computes the point-wise feature drift (L2 distance) between consecutive steps.
        Returns a list of average drift values.
        """
        drifts = []
        for i in range(1, len(features_over_time)):
            prev = features_over_time[i-1]
            curr = features_over_time[i]
            # Mean L2 distance across tokens
            drift = np.linalg.norm(curr - prev, axis=-1).mean()
            drifts.append(float(drift))
        return drifts

    def rank_layers(self, all_features: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
        """
        Calculates the feature importance score based on geometric distinctness.
        Here we proxy distinctness by the total variance explained by the Top 3 PCA components.
        Higher variance means the representation is highly structured (likely geometric).
        """
        scores = []
        for name, feat in all_features.items():
            _, evr = self.run_pca(feat, n_components=3)
            # Sum of explained variance of top 3 components
            score = float(np.sum(evr))
            scores.append((name, score))
            
        # Sort descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
