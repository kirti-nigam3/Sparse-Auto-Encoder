"""UMAP / t-SNE / HDBSCAN visualization of sparse codes."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def run_umap(
    z: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
        verbose=True,
    )
    return reducer.fit_transform(z)


def run_tsne(z: np.ndarray, perplexity: float = 50.0, random_state: int = 42) -> np.ndarray:
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca")
    return tsne.fit_transform(z)


def run_hdbscan(
    embedding: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster the (UMAP-projected) embedding with HDBSCAN.

    Returns (labels, probabilities).
    """
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embedding)
    probs = clusterer.probabilities_
    return labels, probs


def cluster_metrics(z: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute Calinski-Harabasz, Davies-Bouldin, Silhouette (excluding noise)."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    mask = labels >= 0
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return {"calinski_harabasz": float("nan"), "davies_bouldin": float("nan"), "silhouette": float("nan")}
    z_eff, lab_eff = z[mask], labels[mask]
    return {
        "calinski_harabasz": float(calinski_harabasz_score(z_eff, lab_eff)),
        "davies_bouldin": float(davies_bouldin_score(z_eff, lab_eff)),
        "silhouette": float(silhouette_score(z_eff, lab_eff, metric="euclidean", sample_size=min(10000, z_eff.shape[0]))),
    }


def save_scatter(
    embedding: np.ndarray,
    labels: np.ndarray | None,
    out_path: str | Path,
    title: str = "UMAP",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=2, alpha=0.4)
    else:
        unique = np.unique(labels)
        cmap = plt.get_cmap("tab20")
        for i, lab in enumerate(unique):
            mask = labels == lab
            color = "lightgray" if lab == -1 else cmap(i % 20)
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=2, alpha=0.5, color=color, label=str(lab))
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_aspect("equal", adjustable="datalim")
    if labels is not None and len(np.unique(labels)) <= 20:
        ax.legend(markerscale=4, loc="best", fontsize=6)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
