import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)


def reduce_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduces the dimensionality of embeddings using Principal Component Analysis (PCA).
    """
    logger.info(f"Applying PCA reduction to {n_components} components...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced


def reduce_tsne(embeddings: np.ndarray, n_components: int = 2, perplexity: float = 30.0, max_iter: int = 1000) -> np.ndarray:
    """
    Reduces the dimensionality of embeddings using t-SNE.
    """
    logger.info(f"Applying t-SNE reduction to {n_components} components...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    return reduced


def reduce_umap(embeddings: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """
    Reduces the dimensionality of embeddings using UMAP.
    """
    if not UMAP_AVAILABLE:
        raise ImportError("The 'umap-learn' package is required for UMAP reduction. Please install it using 'pip install umap-learn'.")
        
    logger.info(f"Applying UMAP reduction to {n_components} components...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    return reduced


def plot_embeddings(
    reduced_embeddings: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    output_dir: str,
    title: Optional[str] = None
) -> None:
    """
    Plots the 2D reduced embeddings using matplotlib.
    Colors the points based on whether they belong to a duplicate pair or not.
    Saves the figure as both PNG and SVG.
    """
    if reduced_embeddings.shape[1] != 2:
        raise ValueError("plot_embeddings currently only supports 2D embeddings.")

    logger.info(f"Plotting {method_name} embeddings...")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Identify unique labels to map to colors
    unique_labels = np.unique(labels)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, label in enumerate(unique_labels):
        idx = labels == label
        # In a duplicate detection context, label 1 might be duplicates, 0 non-duplicates.
        label_name = f"Class {label}"
        if label == 1:
            label_name = "Duplicate"
        elif label == 0:
            label_name = "Non-Duplicate"
            
        plt.scatter(
            reduced_embeddings[idx, 0],
            reduced_embeddings[idx, 1],
            c=colors[i % len(colors)],
            label=label_name,
            alpha=0.6,
            s=20
        )
        
    plot_title = title if title else f"Embedding Visualization ({method_name})"
    plt.title(plot_title, fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Save PNG
    png_path = os.path.join(output_dir, f"embeddings_{method_name.lower()}.png")
    plt.savefig(png_path, format="png", dpi=300)
    logger.info(f"Saved PNG to {png_path}")

    # Save SVG
    svg_path = os.path.join(output_dir, f"embeddings_{method_name.lower()}.svg")
    plt.savefig(svg_path, format="svg")
    logger.info(f"Saved SVG to {svg_path}")

    plt.close()


def generate_visualizations(
    embeddings: np.ndarray, 
    labels: np.ndarray, 
    output_dir: str,
    methods: Union[str, List[str]] = "all"
) -> None:
    """
    Orchestrates the generation of PCA, t-SNE, and UMAP visualizations.
    
    Args:
        embeddings: The raw high-dimensional embeddings.
        labels: Array of labels (e.g., 0 for non-duplicates, 1 for duplicates).
        output_dir: Directory where the images will be saved.
        methods: List of methods to run ("pca", "tsne", "umap") or "all".
    """
    valid_methods = ["pca", "tsne", "umap"]
    
    if methods == "all":
        methods_to_run = valid_methods
    elif isinstance(methods, str):
        methods_to_run = [methods.lower()]
    else:
        methods_to_run = [m.lower() for m in methods]
        
    for method in methods_to_run:
        if method not in valid_methods:
            logger.warning(f"Unknown reduction method: {method}. Skipping.")
            continue
            
        try:
            if method == "pca":
                reduced = reduce_pca(embeddings, n_components=2)
            elif method == "tsne":
                reduced = reduce_tsne(embeddings, n_components=2)
            elif method == "umap":
                if not UMAP_AVAILABLE:
                    logger.warning("UMAP is not installed. Skipping UMAP visualization.")
                    continue
                reduced = reduce_umap(embeddings, n_components=2)
                
            plot_embeddings(
                reduced_embeddings=reduced,
                labels=labels,
                method_name=method.upper(),
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"Failed to generate {method.upper()} visualization: {e}")
