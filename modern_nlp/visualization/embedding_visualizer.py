import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Union

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from modern_nlp.core.utils import get_logger
from modern_nlp.visualization.base_visualizer import BaseVisualizer

logger = get_logger(__name__)

class EmbeddingVisualizer(BaseVisualizer):
    """
    Concrete visualizer for dense embeddings.
    Supports dimensionality reduction (PCA, t-SNE, UMAP), similarity heatmaps, 
    distance matrices, clustering, and nearest neighbors charting.
    """
    def __init__(self, output_dir: str) -> None:
        super().__init__(output_dir)

    def _plot_scatter(self, reduced_embeddings: np.ndarray, labels: np.ndarray, method_name: str, title: str) -> None:
        """Helper to plot 2D scatter plots mapping labels to distinct clusters."""
        plt.figure()
        unique_labels = np.unique(labels)
        colors = sns.color_palette("husl", len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            idx = labels == label
            # Try to map common binary labels to semantic terms if possible
            if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
                label_name = "Duplicate/Similar" if label == 1 else "Non-Duplicate/Dissimilar"
            else:
                label_name = f"Cluster {label}"
                
            plt.scatter(
                reduced_embeddings[idx, 0],
                reduced_embeddings[idx, 1],
                color=colors[i],
                label=label_name,
                alpha=0.6,
                s=25
            )
            
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.3)
        self.save_figure(f"embeddings_{method_name.lower()}")

    def plot_pca(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        logger.info("Generating PCA visualization...")
        reduced = PCA(n_components=2).fit_transform(embeddings)
        self._plot_scatter(reduced, labels, "PCA", "PCA Embedding Projection")

    def plot_tsne(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        logger.info("Generating t-SNE visualization...")
        reduced = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
        self._plot_scatter(reduced, labels, "TSNE", "t-SNE Embedding Projection")

    def plot_umap(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available. Skipping plot_umap.")
            return
        logger.info("Generating UMAP visualization...")
        reduced = umap.UMAP(n_components=2, random_state=42).fit_transform(embeddings)
        self._plot_scatter(reduced, labels, "UMAP", "UMAP Embedding Projection")
        
    def plot_similarity_heatmap(self, embeddings: np.ndarray) -> None:
        logger.info("Generating Similarity Heatmap...")
        plt.figure(figsize=(8, 7))
        sim_matrix = cosine_similarity(embeddings)
        sns.heatmap(sim_matrix, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
        plt.title("Cosine Similarity Heatmap")
        self.save_figure("similarity_heatmap")

    def plot_distance_matrix(self, embeddings: np.ndarray) -> None:
        logger.info("Generating Distance Matrix...")
        plt.figure(figsize=(8, 7))
        dist_matrix = euclidean_distances(embeddings)
        sns.heatmap(dist_matrix, cmap="mako_r", cbar=True, xticklabels=False, yticklabels=False)
        plt.title("Euclidean Distance Matrix")
        self.save_figure("distance_matrix")

    def plot_clusters(self, embeddings: np.ndarray, n_clusters: int = 5) -> None:
        logger.info(f"Generating Semantic Cluster Visualization (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Use PCA for base projection to visualize clusters
        reduced = PCA(n_components=2).fit_transform(embeddings)
        self._plot_scatter(reduced, cluster_labels, "CLUSTERS", f"Semantic Clusters (k={n_clusters}) projected via PCA")

    def generate_all(self, embeddings: np.ndarray, labels: np.ndarray, methods: Union[str, List[str]] = "all", max_samples: int = 500) -> None:
        """
        Orchestrates all requested visualization plots.
        Subsamples to `max_samples` to prevent memory exhaustion on heatmaps/tsne.
        """
        logger.info(f"EmbeddingVisualizer: Generating visual suite in {self.output_dir}")
        
        if len(embeddings) > max_samples:
            logger.info(f"Subsampling embeddings from {len(embeddings)} to {max_samples} for visualization.")
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
            
        valid_methods = ["pca", "tsne", "umap", "heatmap", "distance", "clusters"]
        if methods == "all":
            methods_to_run = valid_methods
        elif isinstance(methods, str):
            methods_to_run = [methods.lower()]
        else:
            methods_to_run = [m.lower() for m in methods]
            
        if "pca" in methods_to_run:
            self.plot_pca(embeddings, labels)
        if "tsne" in methods_to_run:
            self.plot_tsne(embeddings, labels)
        if "umap" in methods_to_run:
            self.plot_umap(embeddings, labels)
        if "heatmap" in methods_to_run:
            self.plot_similarity_heatmap(embeddings)
        if "distance" in methods_to_run:
            self.plot_distance_matrix(embeddings)
        if "clusters" in methods_to_run:
            self.plot_clusters(embeddings, n_clusters=min(5, len(np.unique(labels)) + 2))
